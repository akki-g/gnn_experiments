"""
Environment adapters for MAPPO backbone.

VectorEnvAdapter provides the standard reset/step interface.
VMASAdapter wraps VMAS vectorized environments to that interface.

Phase 1: VectorEnvAdapter stubs are preserved.
Phase 2+: VMASAdapter is a concrete subclass.

Interface contract (reset/step):
    reset() -> (obs [B, N, obs_dim], state [B, N*obs_dim])
    step(actions [B, N]) -> (obs, state, reward [B, N], done [B, N] float, info dict)
"""

from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch import Tensor


class VectorEnvAdapter:
    """
    Thin adapter that normalises any vectorised multi-agent environment
    to the (obs, state, reward, done, info) interface expected by the
    MAPPO training loop.

    Future subclasses: VMASAdapter, PredCapPreyAdapter.
    """

    def reset(self) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        obs   : [B, N, obs_dim]
        state : [B, state_dim]
        """
        raise NotImplementedError("Step 2")

    def step(
        self, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        """
        Parameters
        ----------
        actions : [B, N] long.

        Returns
        -------
        obs    : [B, N, obs_dim]
        state  : [B, state_dim]
        reward : [B, N]
        done   : [B, N]
        info   : dict
        """
        raise NotImplementedError("Step 2")


class VMASAdapter(VectorEnvAdapter):
    """
    Adapter that wraps a VMAS vectorized environment to the MAPPO backbone interface.

    VMAS facts that shape this implementation:
    - step returns: obs_list, reward_list, dones, info_list
      where obs_list is a list of N tensors [B, obs_dim],
      reward_list is a list of N tensors [B],
      dones is a single [B] bool tensor,
      info_list is a list of N empty dicts.
    - Action input MUST be a Python list of N tensors each [B] long.
    - VMAS does NOT auto-reset, so this adapter resets finished envs before
      returning the next observation.
    - For this smoke-test adapter, VMAS dones are treated as time-limit
      truncations. Terminal observations are preserved in info before reset.
    - render: env.render(mode="rgb_array", env_index=0, agent_index_focus=None)
      returns [1400, 1400, 3] uint8 numpy array.

    Attributes
    ----------
    obs_dim    : per-agent observation dimension (inferred from a probe reset).
    action_dim : number of discrete actions (inferred from env.action_space[0].n).
    state_dim  : num_agents * obs_dim [+ 1 if include_timestep] (concat fallback for critic).
    num_agents : N.
    B          : num_envs.
    """

    def __init__(
        self,
        scenario: str = "simple_spread",
        num_envs: int = 32,
        n_agents: int = 3,
        max_steps: int = 100,
        seed: int = 0,
        device: str = "cpu",
        continuous_actions: bool = False,
        include_timestep: bool = True,
    ):
        """
        Parameters
        ----------
        scenario         : VMAS scenario name (default "simple_spread").
        num_envs         : B — number of parallel environments.
        n_agents         : N — number of agents.
        max_steps        : episode length (VMAS time limit).
        seed             : random seed passed to vmas.make_env.
        device           : torch device string ("cpu" or "cuda").
        continuous_actions: must be False for Phase 2 discrete backbone.
        include_timestep : if True, append a scalar normalised timestep
                           [0,1] as the last column of the critic state,
                           making state_dim = N*obs_dim + 1.
        """
        self.B = num_envs
        self.N = n_agents
        self.num_agents = n_agents
        self.device = torch.device(device)
        self._max_steps = max_steps
        self._t = 0
        self.include_timestep = include_timestep

        # Build VMAS environment
        import vmas

        self.env = vmas.make_env(
            scenario=scenario,
            num_envs=num_envs,
            device=device,
            continuous_actions=False,   # Phase 2: discrete only
            n_agents=n_agents,
            max_steps=max_steps,
            seed=seed,
        )

        # Probe one reset to infer obs_dim and action_dim
        obs_list = self.env.reset()
        assert len(obs_list) == self.N, (
            f"Expected {self.N} obs tensors from reset, got {len(obs_list)}"
        )
        # obs_list[i] is [B, obs_dim]
        self.obs_dim = int(obs_list[0].shape[-1])
        # action_space is a list of N gym.spaces.Discrete
        self.action_dim = int(self.env.action_space[0].n)
        self.state_dim = self.num_agents * self.obs_dim + (1 if self.include_timestep else 0)

        # Reset internal step counter (probe used one reset call)
        self._t = 0

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def _stack_obs(self, obs_list) -> Tensor:
        """Stack VMAS obs list of N [B, obs_dim] tensors into [B, N, obs_dim]."""
        assert len(obs_list) == self.N, (
            f"Expected {self.N} obs tensors, got {len(obs_list)}"
        )
        obs = torch.stack(obs_list, dim=1).to(self.device)
        assert obs.shape == (self.B, self.N, self.obs_dim), (
            f"obs shape {obs.shape} != ({self.B}, {self.N}, {self.obs_dim})"
        )
        return obs

    def _state_from_obs(self, obs: Tensor) -> Tensor:
        """Build concat-observation critic state [B, N * obs_dim] (no timestep column)."""
        base_width = self.num_agents * self.obs_dim
        state = obs.reshape(self.B, base_width)
        assert state.shape == (self.B, base_width), (
            f"state shape {state.shape} != ({self.B}, {base_width})"
        )
        return state

    def _append_timestep(self, state: Tensor, t_norm) -> Tensor:
        """
        Append a normalised timestep scalar as the last column of state.

        If self.include_timestep is False, returns state unchanged.

        Parameters
        ----------
        state  : [B, N*obs_dim] tensor (base width, no timestep column yet).
        t_norm : float or [B] tensor — normalised time in [0, 1].
                 scalar self._t is a pre-existing lockstep assumption.

        Returns
        -------
        state with shape [B, N*obs_dim+1] (or unchanged if not include_timestep).
        """
        if not self.include_timestep:
            return state
        if isinstance(t_norm, torch.Tensor):
            col = t_norm.reshape(self.B, 1).to(dtype=state.dtype, device=state.device)
        else:
            col = torch.full((self.B, 1), float(t_norm), dtype=state.dtype, device=state.device)
        return torch.cat([state, col], dim=1)

    def _normalise_dones(self, dones) -> Tensor:
        """Convert VMAS done output to a bool tensor shaped [B] on self.device."""
        done_env = torch.as_tensor(dones, device=self.device, dtype=torch.bool)
        if done_env.dim() == 0:
            done_env = done_env.repeat(self.B)
        assert done_env.shape == (self.B,), (
            f"VMAS dones shape {done_env.shape} != ({self.B},)"
        )
        return done_env

    def _task_diagnostics(self) -> Dict[str, Tensor]:
        """
        Return simple_spread task diagnostics without changing reward dynamics.

        coverage_distance is the mean over landmarks of the closest agent
        distance. collision_pairs counts overlapping unordered agent pairs.
        """
        world = getattr(self.env, "world", None)
        if world is None or not hasattr(world, "agents") or not hasattr(world, "landmarks"):
            nan_vec = torch.full((self.B,), float("nan"), device=self.device)
            return {
                "coverage_distance": nan_vec,
                "collision_pairs": nan_vec,
            }

        if not world.agents or not world.landmarks:
            nan_vec = torch.full((self.B,), float("nan"), device=self.device)
            return {
                "coverage_distance": nan_vec,
                "collision_pairs": nan_vec,
            }

        agent_pos = torch.stack([agent.state.pos for agent in world.agents], dim=1)
        landmark_pos = torch.stack(
            [landmark.state.pos for landmark in world.landmarks], dim=1
        )
        distances = torch.linalg.vector_norm(
            agent_pos.unsqueeze(2) - landmark_pos.unsqueeze(1),
            dim=-1,
        )
        closest_per_landmark = distances.min(dim=1).values
        coverage_distance = closest_per_landmark.mean(dim=1).to(self.device)

        collision_pairs = torch.zeros(self.B, device=self.device)
        for i, agent_i in enumerate(world.agents):
            if not getattr(agent_i, "collide", False):
                continue
            for agent_j in world.agents[i + 1:]:
                if getattr(agent_j, "collide", False):
                    collision_pairs += world.is_overlapping(agent_i, agent_j).to(self.device).float()

        return {
            "coverage_distance": coverage_distance,
            "collision_pairs": collision_pairs,
        }

    def reset(self) -> Tuple[Tensor, Tensor]:
        """
        Reset all parallel environments.

        Returns
        -------
        obs   : [B, N, obs_dim]  on self.device
        state : [B, state_dim]   on self.device (N*obs_dim [+1 timestep])
        """
        obs_list = self.env.reset()
        self._t = 0

        obs = self._stack_obs(obs_list)
        state = self._state_from_obs(obs)
        state = self._append_timestep(state, 0.0)
        return obs, state

    def step(
        self, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        """
        Step all parallel environments.

        VMAS does NOT auto-reset. This adapter resets finished environments
        before returning so the caller can continue collecting rollouts without
        an explicit reset branch. The terminal observation/state from the VMAS
        step is preserved in info before any reset happens.

        Parameters
        ----------
        actions : [B, N] long — one discrete action per agent per env.
                  Must be on CPU or will be moved to env device internally.

        Returns
        -------
        obs    : [B, N, obs_dim]  — next obs on self.device. Finished envs
                 contain fresh reset observations.
        state  : [B, state_dim]  — next state on self.device (N*obs_dim [+1 timestep]).
        reward : [B, N]          — on self.device.
        done   : [B, N] float    — 1.0 at VMAS done boundary, else 0.0.
        info   : dict with keys "truncated" [B] bool, "terminated" [B] bool,
                 "terminal_obs" [B,N,obs_dim], and "terminal_state" [B,state_dim].
        """
        assert actions.shape == (self.B, self.N), (
            f"actions shape {actions.shape} != ({self.B}, {self.N})"
        )

        # VMAS expects a list of N tensors each [B] long (on env device)
        action_list = [
            actions[:, i].contiguous().long().to(self.env.device)
            for i in range(self.N)
        ]

        obs_list, reward_list, dones, info_list = self.env.step(action_list)

        assert len(obs_list) == self.N, (
            f"Expected {self.N} obs tensors from step, got {len(obs_list)}"
        )

        # reward: stack list of N [B] tensors → [B, N]
        reward = torch.stack(reward_list, dim=1).to(self.device)  # [B, N]

        self._t += 1
        vmas_done = self._normalise_dones(dones)
        local_time_limit = torch.full(
            (self.B,), self._t >= self._max_steps, device=self.device, dtype=torch.bool
        )
        truncated_env = vmas_done | local_time_limit
        terminated_env = torch.zeros_like(truncated_env)

        done = truncated_env.float().unsqueeze(1).expand(self.B, self.N).clone()

        # Terminal obs/state are from the VMAS step before any reset.
        # LOAD-BEARING: terminal_state timestep must be self._t/max_steps, captured
        # AFTER the _t += 1 increment. At truncation this equals max_steps/max_steps = 1.0.
        terminal_obs = self._stack_obs(obs_list)
        terminal_state = self._state_from_obs(terminal_obs)
        terminal_state = self._append_timestep(terminal_state, self._t / self._max_steps)

        obs = terminal_obs
        if truncated_env.any():
            if bool(truncated_env.all()):
                fresh_obs_list = self.env.reset()
                self._t = 0
            else:
                fresh_obs_list = None
                for env_idx in torch.nonzero(truncated_env, as_tuple=False).flatten().tolist():
                    fresh_obs_list = self.env.reset_at(int(env_idx))
                assert fresh_obs_list is not None
            obs = self._stack_obs(fresh_obs_list)

        # Build per-env t_norm: 0.0 for envs that just truncated+reset (new episode
        # start); self._t/max_steps for non-truncated envs. Under enforced
        # rollout_length==max_steps all envs truncate in lockstep, but the per-env
        # vector keeps the mixed/reset_at branch correct.
        t_norm_vec = torch.where(
            truncated_env,
            torch.zeros(self.B, device=self.device),
            torch.full((self.B,), self._t / self._max_steps, device=self.device),
        )
        state = self._append_timestep(self._state_from_obs(obs), t_norm_vec)

        assert obs.shape == (self.B, self.N, self.obs_dim), (
            f"step obs shape {obs.shape} != ({self.B}, {self.N}, {self.obs_dim})"
        )
        assert reward.shape == (self.B, self.N), (
            f"step reward shape {reward.shape} != ({self.B}, {self.N})"
        )
        assert done.shape == (self.B, self.N), (
            f"step done shape {done.shape} != ({self.B}, {self.N})"
        )
        assert state.shape == (self.B, self.state_dim), (
            f"step state shape {state.shape} != ({self.B}, {self.state_dim})"
        )

        info = {
            "truncated": truncated_env,
            "terminated": terminated_env,
            "terminal_obs": terminal_obs,
            "terminal_state": terminal_state,
            "raw_vmas_dones": vmas_done,
        }
        info.update(self._task_diagnostics())
        return obs, state, reward, done, info

    def render(self) -> np.ndarray:
        """
        Render the first environment (env_index=0) to a numpy RGB array.

        Returns
        -------
        frame : [1400, 1400, 3] uint8 numpy array.
        """
        return self.env.render(
            mode="rgb_array",
            env_index=0,
            agent_index_focus=None,
        )
