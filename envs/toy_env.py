"""
Toy multi-agent environment: shared-target voting.

Each env draws a random target index g ∈ [0, action_dim-1] at reset.
Per-agent obs [B, N, 4] = [one_hot(g, 3), agent_id_norm].
State [B, 8] = obs.reshape(B, N*obs_dim).
Team reward: +1.0 to ALL agents iff every agent picks g, else 0.0.
Episode ends after episode_length steps; auto-reset on done.

Interface
---------
reset() → (obs [B, N, obs_dim], state [B, state_dim])
step(actions [B, N]) → (obs, state, reward [B, N], done [B, N], info dict)
"""

from typing import Tuple, Dict, Any

import torch
from torch import Tensor


class ToyMultiAgentEnv:
    """
    Minimal toy multi-agent environment for backbone sanity checking.

    Class attributes act as defaults; __init__ may override them.
    """

    obs_dim: int = 4
    state_dim: int = 8      # will be num_agents * obs_dim = 2 * 4
    action_dim: int = 3
    num_agents: int = 2

    def __init__(
        self,
        num_envs: int = 4,
        num_agents: int = 2,
        episode_length: int = 32,
        seed: int = 0,
    ):
        """
        Parameters
        ----------
        num_envs       : B — number of parallel environments.
        num_agents     : N.
        episode_length : maximum steps per episode.
        seed           : random seed for reproducibility.
        """
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.state_dim = num_agents * self.obs_dim
        self.episode_length = episode_length
        self.seed = seed

        # Seeded generator for reproducible target sampling
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

        # Step counter per env
        self._step = torch.zeros(num_envs, dtype=torch.long)
        # Target index per env
        self._target = torch.zeros(num_envs, dtype=torch.long)

    def _build_obs(self) -> Tuple[Tensor, Tensor]:
        """
        Build obs and state from current targets.

        Returns
        -------
        obs   : [B, N, obs_dim]   (one-hot target + agent_id_norm)
        state : [B, state_dim]    (obs.reshape(B, N*obs_dim))
        """
        target_oh = torch.nn.functional.one_hot(
            self._target, num_classes=self.action_dim
        ).float()  # [B, 3]
        obs = torch.zeros(self.num_envs, self.num_agents, self.obs_dim)
        for a in range(self.num_agents):
            obs[:, a, :self.action_dim] = target_oh
            obs[:, a, self.action_dim] = a / max(self.num_agents - 1, 1)
        state = obs.reshape(self.num_envs, self.num_agents * self.obs_dim)
        return obs, state

    def reset(self) -> Tuple[Tensor, Tensor]:
        """
        Reset all environments.

        Returns
        -------
        obs   : [B, N, obs_dim]
        state : [B, state_dim]
        """
        self._step.zero_()
        self._target = torch.randint(
            0, self.action_dim, (self.num_envs,), generator=self._gen
        )
        return self._build_obs()

    def step(
        self, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        """
        Advance all environments by one step.

        Parameters
        ----------
        actions : [B, N] long — one discrete action per agent per env.

        Returns
        -------
        obs    : [B, N, obs_dim]
        state  : [B, state_dim]
        reward : [B, N]          float
        done   : [B, N]          float  (1.0 = terminal)
        info   : dict
        """
        assert actions.shape == (self.num_envs, self.num_agents), (
            f"actions shape {actions.shape} != ({self.num_envs}, {self.num_agents})"
        )
        actions = actions.cpu().long()

        # All agents must pick the target for success
        correct = (actions == self._target.unsqueeze(1)).all(dim=1).float()  # [B]
        reward = correct.unsqueeze(1).expand(self.num_envs, self.num_agents).clone()  # [B, N]

        self._step += 1
        done_env = (self._step >= self.episode_length)  # [B] bool

        # Auto-reset where done
        if done_env.any():
            self._step[done_env] = 0
            n_done = int(done_env.sum().item())
            self._target[done_env] = torch.randint(
                0, self.action_dim, (n_done,), generator=self._gen
            )

        obs, state = self._build_obs()  # uses new targets where reset
        done = done_env.unsqueeze(1).expand(self.num_envs, self.num_agents).float()
        info = {"success": correct}
        return obs, state, reward, done, info
