"""
PursuitAdapter: VectorEnvAdapter subclass wrapping PursuitEnv.

Provides the exact same reset()/step() interface contract as VMASAdapter
so the MAPPO training loop (train_vmas_mappo.py) can use pursuit without
any changes to backbone code.

IMPORTANT: No VMAS import anywhere in this file.  PursuitEnv is pure PyTorch,
so the import chain is completely VMAS-free.

Trainer-facing attributes mirrored from VMASAdapter:
    self.obs_dim    : per-agent observation dimension
    self.action_dim : 5 (discrete, fixed)
    self.state_dim  : privileged critic state dimension
    self.num_agents : N
    self.B          : num_envs
    self.device     : torch.device

Extra attribute (forward-compat, not consumed by Phase 2 trainer/backbone):
    self.class_id   : [N] long — 0=full-sight, 1=sensor-limited.
                      The Phase 3 backbone will thread this into the comm slot.
                      In Phase 2 IdentityComm ignores it completely.
"""

from typing import Tuple, Dict, Any

import torch
from torch import Tensor

from envs.adapters import VectorEnvAdapter
from envs.pursuit import PursuitEnv


class PursuitAdapter(VectorEnvAdapter):
    """
    Thin adapter that wraps PursuitEnv to the MAPPO backbone VectorEnvAdapter interface.

    All reset()/step() calls are delegated verbatim to PursuitEnv, which already
    returns the exact (obs, state, reward, done, info) contract expected by the trainer.

    Constructor accepts the trainer-facing names used by _build_env():
        scenario        : ignored (kept for API parity with VMASAdapter)
        num_envs        : B — parallel environments
        n_agents        : N — number of pursuing agents
        max_steps       : T — fixed episode length
        seed            : random seed
        device          : torch device string
        include_timestep: append normalised timestep to critic state

    Plus pursuit-specific params forwarded to PursuitEnv:
        alpha, num_full_sight, observe_teammates, world_size, capture_radius,
        move_step, target_speed, dist_coef, step_penalty, capture_bonus, capture_k
    """

    def __init__(
        self,
        scenario: str = "pursuit",           # ignored; kept for API parity
        num_envs: int = 32,
        n_agents: int = 3,
        max_steps: int = 100,
        seed: int = 0,
        device: str = "cpu",
        include_timestep: bool = True,
        # Pursuit-specific
        alpha: float = 1.0,
        num_full_sight: int = 2,
        observe_teammates: bool = False,
        world_size: float = 1.0,
        capture_radius: float = 0.1,
        move_step: float = 0.05,
        target_speed: float = 0.03,
        dist_coef: float = 1.0,
        step_penalty: float = 0.01,
        capture_bonus: float = 10.0,
        capture_k: int = 1,
    ):
        self.B = num_envs
        self.num_agents = n_agents
        self.device = torch.device(device)

        # Build the wrapped environment
        self.env = PursuitEnv(
            num_envs=num_envs,
            num_agents=n_agents,
            max_steps=max_steps,
            seed=seed,
            device=device,
            world_size=world_size,
            capture_radius=capture_radius,
            move_step=move_step,
            target_speed=target_speed,
            alpha=alpha,
            num_full_sight=num_full_sight,
            observe_teammates=observe_teammates,
            include_timestep=include_timestep,
            dist_coef=dist_coef,
            step_penalty=step_penalty,
            capture_bonus=capture_bonus,
            capture_k=capture_k,
        )

        # Mirror the attribute surface VMASAdapter exposes to the trainer
        self.obs_dim    = self.env.obs_dim
        self.action_dim = self.env.action_dim   # 5
        self.state_dim  = self.env.state_dim
        self.num_agents = n_agents
        self.N          = n_agents

        # Forward-compat: class_id [N] — Phase 3 comm slot will consume this.
        # Phase 2 IdentityComm ignores it; do NOT thread into model.act/evaluate.
        self.class_id = self.env.class_id

    @property
    def agent_positions(self) -> "Tensor":
        """
        Read-only property returning pursuer positions: [B, N, 2].

        Used by the training loop to build radius-based comm masks when
        comm.comm_radius is set in the config.
        """
        return self.env.pursuer_pos.clone()

    def reset(self) -> Tuple[Tensor, Tensor]:
        """
        Delegate to PursuitEnv.reset().

        Returns
        -------
        obs   : [B, N, obs_dim]
        state : [B, state_dim]
        """
        return self.env.reset()

    def step(
        self, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        """
        Delegate to PursuitEnv.step().

        Parameters
        ----------
        actions : [B, N] long

        Returns
        -------
        obs    : [B, N, obs_dim]
        state  : [B, state_dim]
        reward : [B, N]
        done   : [B, N] float
        info   : dict  (contains all required keys: truncated, terminated,
                        terminal_obs, terminal_state, pursuer_target_distance,
                        capture, captured_episode, target_pos)
        """
        return self.env.step(actions)

    def render(self):
        """Render is not supported for PursuitEnv in Phase 2."""
        raise NotImplementedError(
            "PursuitAdapter.render() is not implemented in Phase 2. "
            "Use --render off when training on pursuit."
        )
