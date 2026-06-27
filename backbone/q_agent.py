import itertools
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional

from backbone.encoder import SharedMLPEncoder
from comm.identity import IdentityComm
from comm.base import CommModule


class QAgent(nn.Module):
    """
    Per-agent Q-utility network: obs -> encoder -> comm slot -> Q-head -> Q[B,N,A].
    Exposes the SAME encode()/act()/save()/load() interface as MAPPO so the probe +
    eval-sweep + checkpoint loaders work unchanged. The mixer (separate module) reads
    raw state only.

    Capacity/architecture parity with the MAPPO actor: encoder (shared, tanh) -> comm ->
    `head_layers` x [Linear, Tanh] trunk -> Linear readout. `head_layers` has the SAME
    semantics as MAPPO's `actor_layers` (number of hidden trunk blocks before the readout),
    so `head_layers == actor_layers` gives a capacity- and activation-matched network whose
    ONLY difference from the actor is that the readout produces Q-values instead of logits.
    """

    def __init__(self, obs_dim, action_dim, num_agents, hidden_dim,
                 comm_module: Optional[CommModule] = None,
                 encoder_layers=2, head_layers=1, state_dim=None, device="cpu"):

        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.head_layers = head_layers
        # state_dim is NOT used by the agent net (the mixer consumes raw state). It is
        # stored only so save() can record it in the checkpoint config for inspection /
        # mixer reconstruction on resume.
        self.state_dim = state_dim
        self.device = torch.device(device)

        self.encoder = SharedMLPEncoder(obs_dim, hidden_dim, num_layers=encoder_layers)
        self.comm = comm_module if comm_module is not None else IdentityComm()

        # Q-head: `head_layers` x [Linear, Tanh] trunk + Linear readout. Matches the
        # MAPPO Actor's trunk depth (Actor builds actor_layers x [Linear, Tanh] + readout),
        # so head_layers=1 is capacity-matched to actor_layers=1 (the configured default).
        layers, d = [], hidden_dim
        for _ in range(max(0, head_layers)):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        layers += [nn.Linear(d, action_dim)]
        self.q_head = nn.Sequential(*layers)

        self.to(self.device)

    def encode(self, obs: Tensor, mask=None, class_id=None) -> Tensor:
        """obs -> encoder -> comm(mask, class_id). Identical to MAPPO.encode (probe uses this)."""
        return self.comm(self.encoder(obs), mask=mask, class_id=class_id)

    def q_values(self, obs: Tensor, mask=None, class_id=None) -> Tensor:
        return self.q_head(self.encode(obs, mask=mask, class_id=class_id))  # [B,N,A]

    def act(self, obs, state=None, deterministic=False, mask=None, class_id=None,
            epsilon: float = 0.0) -> Dict[str, Tensor]:
        """Greedy (deterministic / eval) or epsilon-greedy (collection). `state` unused
        for action selection — kept for MAPPO interface parity. At eval mask=None ->
        full comm, exactly like MAPPO eval."""
        with torch.no_grad():
            q = self.q_values(obs, mask=mask, class_id=class_id)  # [B,N,A]
            greedy = q.argmax(dim=-1)
            if deterministic or epsilon <= 0.0:
                action = greedy
            else:
                rand = torch.randint(0, q.shape[-1], greedy.shape, device=q.device)
                explore = torch.rand(greedy.shape, device=q.device) < epsilon
                action = torch.where(explore, rand, greedy)

        return {"action": action}

    def parameters_group(self):
        """Encoder + comm + q_head params (the single optimization group; the mixer's
        params are added separately by the learner)."""
        return itertools.chain(self.encoder.parameters(), self.comm.parameters(),
                               self.q_head.parameters())

    # ------------------------------------------------------------------
    # Checkpoint I/O (MAPPO-compatible surface; algo="qmix" discriminator)
    # ------------------------------------------------------------------

    def save(self, path: str, optimizer=None, global_step: int = 0, mixer=None, extra=None) -> None:
        """Save a QMIX checkpoint. The agent sub-net (encoder+comm+q_head) is what the
        probe / eval model-factory reloads; mixer + optimizer are for resume only."""
        ckpt = {
            "agent": self.state_dict(),  # encoder.* + comm.* + q_head.*
            "config": {
                "algo": "qmix",
                "obs_dim": self.obs_dim,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "num_agents": self.num_agents,
                "hidden_dim": self.hidden_dim,
                "encoder_layers": self.encoder_layers,
                "head_layers": self.head_layers,
                "device": str(self.device),
            },
            "global_step": global_step,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "mixer": mixer.state_dict() if mixer is not None else None,
            "extra": extra,
        }
        torch.save(ckpt, path)

    def load(self, path: str, map_location=None) -> dict:
        """Load a checkpoint saved by save(). Restores the agent sub-net and returns the
        full dict so the caller can restore mixer/optimizer/global_step."""
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.load_state_dict(ckpt["agent"])
        return ckpt
