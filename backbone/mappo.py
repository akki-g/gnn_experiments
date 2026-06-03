"""
MAPPO model — top-level module composing encoder, comm slot, actor, and critic.

Architecture pipeline
---------------------
Actor path (CTDE execution path):
    raw_obs  →  SharedMLPEncoder  →  CommModule  →  Actor  →  action / log_prob / entropy
                                        ↑
                              comm slot (default: IdentityComm)

Critic path (training only, centralised):
    raw global state  OR  concat(raw_obs)  →  CentralCritic  →  [B, N] values

CRITICAL (Q5 isolation):
    The comm slot output does NOT feed the critic.
    The critic always reads RAW state.
    This ensures that swapping comm modules changes ONLY the actor path,
    making the ablation comparison meaningful.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from backbone.encoder import SharedMLPEncoder
from backbone.actor import Actor
from backbone.critic import CentralCritic
from comm.identity import IdentityComm
from comm.base import CommModule


class MAPPO(nn.Module):
    """
    Top-level MAPPO model.

    Composes:
        encoder  : SharedMLPEncoder (obs → hidden embedding)
        comm     : CommModule       (hidden → comm-modulated hidden; default IdentityComm)
        actor    : Actor            (hidden → Categorical actions)
        critic   : CentralCritic   (raw state → per-agent values)

    The encoder is shared between the actor path (obs → encoder → comm → actor)
    and, when share_encoder=True, provides the embedding used before comm.
    The critic has its own separate raw-state input path — it never sees comm output.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int,
        comm_module: Optional[CommModule] = None,
        share_encoder: bool = True,
        encoder_layers: int = 2,
        actor_layers: int = 1,
        critic_layers: int = 2,
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        obs_dim        : per-agent local observation dimension.
        state_dim      : global state dimension fed to the critic.
                         If the environment has no global state, pass
                         num_agents * obs_dim here and use build_state_from_obs().
        action_dim     : number of discrete actions.
        num_agents     : N.
        hidden_dim     : shared MLP hidden width D.
        comm_module    : CommModule instance.  If None, defaults to IdentityComm().
                         Future modules (Broadcast, Attention, Graph) are injected here.
        share_encoder  : If True (default), one encoder is used for all agents.
        encoder_layers : number of MLP layers in the shared encoder (default 2).
        actor_layers   : number of trunk layers in the actor head (default 1).
        critic_layers  : number of hidden layers in the centralized critic (default 2).
        device         : "cpu" or "cuda".
        """
        super().__init__()

        # If no comm_module is provided, default to IdentityComm — do NOT put
        # IdentityComm() as a default argument (mutable default anti-pattern).
        if comm_module is None:
            comm_module = IdentityComm()

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.share_encoder = share_encoder
        self.encoder_layers = encoder_layers
        self.actor_layers = actor_layers
        self.critic_layers = critic_layers
        self.device = torch.device(device)

        # Actor-path components
        self.encoder = SharedMLPEncoder(obs_dim, hidden_dim, num_layers=encoder_layers)
        self.comm = comm_module
        self.actor = Actor(hidden_dim, action_dim, num_layers=actor_layers)

        # Critic reads RAW state — completely separate from comm output (Q5)
        self.critic = CentralCritic(state_dim, hidden_dim, num_agents, num_layers=critic_layers)

        self.to(self.device)

    # ------------------------------------------------------------------
    # Actor path
    # ------------------------------------------------------------------

    def encode(
        self,
        obs: Tensor,
        mask: Optional[Tensor] = None,
        class_id: Optional[Tensor] = None,
    ) -> Tensor:
        """
        obs → encoder → comm(mask, class_id) → h_comm [B, N, D]

        Applies the shared encoder then the comm slot to produce
        comm-modulated per-agent embeddings.  These feed the actor only.

        Parameters
        ----------
        obs      : [B, N, obs_dim] or [T, B, N, obs_dim]
        mask     : optional [B, N, N] or [T, B, N, N] bool comm mask
                   passed straight through to comm.forward(); IdentityComm ignores it.
        class_id : optional [N] long agent-class ids
                   passed straight through to comm.forward(); IdentityComm ignores it.

        Returns
        -------
        h_comm : [B, N, hidden_dim] or [T, B, N, hidden_dim]
        """
        h = self.encoder(obs)
        return self.comm(h, mask=mask, class_id=class_id)

    def act(
        self,
        obs: Tensor,
        state: Tensor,
        deterministic: bool = False,
        mask: Optional[Tensor] = None,
        class_id: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Collect one step of experience.

        Actor path  : obs → encode(mask, class_id) → actor.act()
        Critic path : state (RAW) → critic.forward()   [Q5: no comm here]

        Parameters
        ----------
        obs           : [B, N, obs_dim]     local observations.
        state         : [B, state_dim]      raw global state (or concat fallback).
        deterministic : greedy if True.
        mask          : optional [B, N, N] comm mask forwarded to the comm slot.
        class_id      : optional [N] agent-class ids forwarded to the comm slot.

        Returns dict with keys:
            "action"   : [B, N] long
            "log_prob" : [B, N] float  (detached — stored in buffer as-is)
            "entropy"  : [B, N] float
            "value"    : [B, N] float  (detached — stored in buffer as-is)
        """
        with torch.no_grad():
            h_comm = self.encode(obs, mask=mask, class_id=class_id)
            action, log_prob, entropy = self.actor.act(h_comm, deterministic)
            value = self.critic(state)
        return {"action": action, "log_prob": log_prob, "entropy": entropy, "value": value}

    def evaluate(
        self,
        obs: Tensor,
        state: Tensor,
        actions: Tensor,
        mask: Optional[Tensor] = None,
        class_id: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Re-evaluate stored actions under the current policy (PPO update inner loop).

        Called once per minibatch per PPO epoch.
        Old log-probs are NOT passed here — they live in the rollout buffer.

        Actor path  : obs → encode(mask, class_id) → actor.evaluate_actions(actions)
        Critic path : state (RAW) → critic.forward()

        Parameters
        ----------
        obs      : [(T*B), N, obs_dim]   flattened minibatch observations.
        state    : [(T*B), state_dim]    flattened minibatch raw states.
        actions  : [(T*B), N] long       stored actions.
        mask     : optional [(T*B), N, N] comm mask forwarded to the comm slot.
        class_id : optional [N] agent-class ids forwarded to the comm slot.

        Returns dict with keys:
            "log_prob" : [(T*B), N]  new log-probs under current policy.
            "entropy"  : [(T*B), N]  per-agent entropy.
            "value"    : [(T*B), N]  per-agent value estimates.
        """
        h_comm = self.encode(obs, mask=mask, class_id=class_id)
        log_prob, entropy = self.actor.evaluate_actions(h_comm, actions)
        value = self.critic(state)
        return {"log_prob": log_prob, "entropy": entropy, "value": value}

    # ------------------------------------------------------------------
    # State construction fallback
    # ------------------------------------------------------------------

    def build_state_from_obs(self, obs: Tensor) -> Tensor:
        """
        Construct a critic-compatible global state by concatenating all agent obs.

        Used when the environment provides no privileged global state.

        [B, N, obs_dim] → [B, N * obs_dim]  (per Q4 decision)
        """
        assert obs.shape[-2:] == (self.num_agents, self.obs_dim), (
            f"obs shape[-2:] {obs.shape[-2:]} != ({self.num_agents}, {self.obs_dim})"
        )
        return obs.reshape(*obs.shape[:-2], self.num_agents * self.obs_dim)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str, optimizer=None, global_step: int = 0, extra=None) -> None:
        """
        Save a checkpoint to disk.

        Checkpoint contains:
            encoder_state_dict   : SharedMLPEncoder weights
            actor_state_dict     : Actor weights
            critic_state_dict    : CentralCritic weights
            comm_state_dict      : CommModule weights (empty for IdentityComm)
            config               : dict of constructor hyperparameters
            global_step          : int training step at save time
            optimizer_state_dict : AdamW/Adam state (caller passes via kwargs — Step 2)
        """
        ckpt = {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "comm": self.comm.state_dict(),
            "config": {
                "obs_dim": self.obs_dim,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "num_agents": self.num_agents,
                "hidden_dim": self.hidden_dim,
                "share_encoder": self.share_encoder,
                "encoder_layers": self.encoder_layers,
                "actor_layers": self.actor_layers,
                "critic_layers": self.critic_layers,
                "device": str(self.device),
            },
            "global_step": global_step,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "extra": extra,
        }
        torch.save(ckpt, path)

    def load(self, path: str, map_location=None) -> dict:
        """
        Load a checkpoint saved by save().

        Restores encoder, actor, critic, and comm weights.
        Optimizer state is restored by the caller from the same checkpoint dict.

        Returns the full checkpoint dict so caller can restore optimizer + global_step.
        """
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.comm.load_state_dict(ckpt["comm"])
        return ckpt
