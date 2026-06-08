"""
Vectorized cooperative pursuit environment in pure PyTorch (no VMAS dependency).

Architecture: fixed-length mode A (lock-step truncation across all envs).
    - N pursuers cooperatively chase 1 randomly-walking target.
    - Episode length = max_steps (scalar step counter, lockstep like VMASAdapter).
    - No true termination in mode A — only truncation at max_steps.
    - Capture events are dense reward, not terminal.

Observation layout (per pursuer, obs_dim fixed across BOTH classes)
--------------------------------------------------------------------
observe_teammates=False (default, obs_dim=5):
    [0,1]  own_pos (x, y)               — self-state, never masked
    [2,3]  target_rel (tx-ox, ty-oy)    — TARGET SLICE, masked by sensor radius
    [4]    target_visible (0.0/1.0)      — TARGET SLICE

observe_teammates=True (obs_dim = 5 + 2*(N-1)):
    [0..4] as above
    [5..] teammate relative positions (NO masking — isolation guarantee)

The target slice is produced by envs.asymmetry.build_obs.
Masking happens ONLY inside build_obs, never in the trainer.

Privileged critic state (CTDE — TRUE unmasked target always present)
---------------------------------------------------------------------
state = concat([pursuer_pos.reshape(B, N*2), target_pos (B,2)], dim=-1)
    → base state_dim = N*2 + 2
If include_timestep=True: append normalized timestep t/max_steps
    → state_dim = N*2 + 2 + 1

CRITICAL: build_obs output is NEVER used to build state.
The critic always sees the true target regardless of alpha. This is the CTDE
guarantee — masking affects only actor inputs, not the critic.

Action space
------------
action_dim = 5 discrete:
    0 = stay, 1 = +x, 2 = -x, 3 = +y, 4 = -y
Each action moves a pursuer by move_step in the given direction (clamped to world).

Reward (mode A — dense, NOT terminating)
-----------------------------------------
    dist_term    = mean of capture_k smallest ||pursuer_i - target||  [B]
    n_within     = count of pursuers with dist < capture_radius        [B] int
    captured_now = (n_within >= capture_k)                             [B] bool
    team_reward  = -dist_coef*dist_term - step_penalty + capture_bonus*captured_now
    reward [B,N] = team_reward.unsqueeze(1).expand(B, N)  (shared team reward)

    min_dist is also computed and logged separately under "pursuer_target_distance"
    for continuity with prior k=1 runs and to feed _pursuit_task_score.

Target motion
-------------
    target_pos += target_speed * unit_vector(randn(generator))
    then reflected/clamped into [-world_size, world_size]^2
    Seeded by self.generator for determinism.

Episode / termination (mode A)
-------------------------------
    terminated = always False
    truncated  = (self._t >= max_steps)  lockstep across ALL envs
    done [B,N] = truncated.float().unsqueeze(1).expand(B, N)
    Auto-reset on truncation exactly like VMASAdapter.
    terminal_obs/terminal_state are built BEFORE reset (pre-reset positions).

Info dict keys (required by trainer + probe)
-------------------------------------------
    "truncated"               [B] bool
    "terminated"              [B] bool  (always False, mode A)
    "terminal_obs"            [B, N, obs_dim]
    "terminal_state"          [B, state_dim]
    "pursuer_target_distance" [B] float  — min_dist (primary diagnostic; lower better)
    "pursuer_k_distance"      [B] float  — mean of capture_k smallest dists (optimized by reward)
    "capture"                 [B] float  — captured_now.float() (per-step capture)
    "captured_episode"        [B] float  — ever_captured this episode
    "target_pos"              [B, 2]    — ground-truth xy for probe (NEVER from obs)
"""

import math
from typing import Tuple, Dict, Any

import torch
from torch import Tensor

from envs.asymmetry import build_obs, class_id_vector


# ---------------------------------------------------------------------------
# Action delta map: 5 discrete actions → (dx, dy)
# ---------------------------------------------------------------------------
_ACTION_DELTAS = torch.tensor([
    [0.0,  0.0],   # 0: stay
    [1.0,  0.0],   # 1: +x
    [-1.0, 0.0],   # 2: -x
    [0.0,  1.0],   # 3: +y
    [0.0, -1.0],   # 4: -y
], dtype=torch.float32)  # [5, 2]


class PursuitEnv:
    """
    Vectorized cooperative pursuit environment.

    All state tensors live on `device`.  Uses a seeded torch.Generator for
    reproducible stochastic resets and target random-walk noise.

    Attributes
    ----------
    obs_dim    : per-agent observation dimension (5 or 5+2*(N-1)).
    state_dim  : privileged critic state dimension (N*2 + 2 [+1 if include_timestep]).
    action_dim : 5 (fixed discrete action space).
    num_agents : N.
    B          : num_envs.
    class_id   : [N] long — 0=full-sight, 1=sensor-limited.
    """

    action_dim: int = 5

    def __init__(
        self,
        num_envs: int = 32,
        num_agents: int = 3,
        max_steps: int = 100,
        seed: int = 0,
        device: str = "cpu",
        world_size: float = 1.0,
        capture_radius: float = 0.1,
        move_step: float = 0.05,
        target_speed: float = 0.03,
        alpha: float = 1.0,
        num_full_sight: int = 2,
        observe_teammates: bool = False,
        include_timestep: bool = True,
        dist_coef: float = 1.0,
        step_penalty: float = 0.01,
        capture_bonus: float = 10.0,
        capture_k: int = 1,
        terminate_on_capture: bool = False,
    ):
        """
        Parameters
        ----------
        num_envs        : B — parallel environments.
        num_agents      : N — number of pursuing agents.
        max_steps       : T — fixed episode length (lockstep mode A).
        seed            : random seed for torch.Generator.
        device          : torch device string.
        world_size      : half-extent of the square world [-w,w]^2.
        capture_radius  : distance threshold for a capture event.
        move_step       : pursuer move step per action.
        target_speed    : target random-walk step size per step.
        alpha           : ∈ [0,1], sensor-limited visibility fraction.
        num_full_sight  : number of class-0 (full-sight) agents; rest are class 1.
        observe_teammates: if True, teammates' positions are included in obs (obs_dim grows).
        include_timestep: if True, append normalized t/max_steps to critic state.
        dist_coef       : weight for the coalition distance term (mean of capture_k smallest dists) in team reward.
        step_penalty    : constant per-step penalty.
        capture_bonus   : bonus reward on capture event.
        capture_k       : number of pursuers required within capture_radius for a capture event.
        terminate_on_capture: if True, raise NotImplementedError (mode B not implemented).
        """
        if terminate_on_capture:
            raise NotImplementedError(
                "mode B (true capture termination + per-env counters) is documented "
                "future work for the ablation phase; Phase 2 uses fixed-length mode A "
                "— see design memo"
            )

        self.B = num_envs
        self.N = num_agents
        self.num_agents = num_agents
        self._max_steps = max_steps
        self.device = torch.device(device)
        self.world_size = float(world_size)
        self.capture_radius = float(capture_radius)
        self.move_step = float(move_step)
        self.target_speed = float(target_speed)
        self.alpha = float(alpha)
        self.num_full_sight = num_full_sight
        self.observe_teammates = observe_teammates
        self.include_timestep = include_timestep
        self.dist_coef = float(dist_coef)
        self.step_penalty = float(step_penalty)
        self.capture_bonus = float(capture_bonus)
        self.capture_k = int(capture_k)

        # R_max = 2 * world_size * sqrt(2) — diameter of the bounding box,
        # guaranteed >= max possible distance within [-world_size, world_size]^2.
        self.R_max = 2.0 * self.world_size * math.sqrt(2.0)

        # Class assignment: first num_full_sight agents are class 0, rest class 1.
        self.class_id = class_id_vector(num_agents, num_full_sight).to(self.device)

        # obs_dim depends on observe_teammates
        self.obs_dim = 5 + (2 * (num_agents - 1) if observe_teammates else 0)

        # Privileged state: pursuer_pos flat + target_pos [+timestep]
        self.state_dim = num_agents * 2 + 2 + (1 if include_timestep else 0)

        # Action deltas on device
        self._action_deltas = _ACTION_DELTAS.to(self.device)  # [5, 2]

        # Seeded generator (mirrors toy_env.py pattern)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        # State tensors (allocated in reset)
        self.pursuer_pos: Tensor = torch.zeros(self.B, self.N, 2, device=self.device)
        self.target_pos: Tensor = torch.zeros(self.B, 2, device=self.device)
        self._ever_captured: Tensor = torch.zeros(self.B, dtype=torch.bool, device=self.device)

        # Scalar lockstep step counter (mirrors VMASAdapter._t)
        self._t: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_positions(self, shape: tuple) -> Tensor:
        """Sample random positions in [-world_size, world_size]^2."""
        return (
            torch.rand(shape, device=self.device, generator=self.generator) * 2.0
            - 1.0
        ) * self.world_size

    def _clamp_to_world(self, pos: Tensor) -> Tensor:
        """Clamp positions to [-world_size, world_size]^2."""
        return torch.clamp(pos, -self.world_size, self.world_size)

    def _build_state(self, t_norm: float | Tensor) -> Tensor:
        """
        Build privileged critic state from ground-truth positions.

        NEVER uses build_obs — the critic always sees the true target.

        Returns [B, state_dim]
        """
        # Flatten pursuer positions: [B, N*2]
        pursuer_flat = self.pursuer_pos.reshape(self.B, self.N * 2)
        # Concatenate with target_pos: [B, N*2 + 2]
        state = torch.cat([pursuer_flat, self.target_pos], dim=-1)
        assert state.shape == (self.B, self.N * 2 + 2), (
            f"state base shape {state.shape} != ({self.B}, {self.N*2+2})"
        )
        if self.include_timestep:
            if isinstance(t_norm, torch.Tensor):
                col = t_norm.reshape(self.B, 1).to(dtype=state.dtype, device=state.device)
            else:
                col = torch.full(
                    (self.B, 1), float(t_norm), dtype=state.dtype, device=state.device
                )
            state = torch.cat([state, col], dim=-1)
        assert state.shape == (self.B, self.state_dim), (
            f"state shape {state.shape} != ({self.B}, {self.state_dim})"
        )
        return state

    def _build_obs(self) -> Tensor:
        """
        Build masked per-agent observations via envs.asymmetry.build_obs.

        Masking happens here ONLY — not in the trainer.
        Returns [B, N, obs_dim].
        """
        obs = build_obs(
            pursuer_pos=self.pursuer_pos,
            target_pos=self.target_pos,
            class_id=self.class_id,
            alpha=self.alpha,
            r_max=self.R_max,
            world_size=self.world_size,
            observe_teammates=self.observe_teammates,
        )
        assert obs.shape == (self.B, self.N, self.obs_dim), (
            f"obs shape {obs.shape} != ({self.B}, {self.N}, {self.obs_dim})"
        )
        return obs

    def _move_target(self) -> None:
        """
        Move the target via a bounded random walk.

        target_pos += target_speed * randn(generator)
        then clamped to [-world_size, world_size]^2.
        """
        noise = torch.randn(
            self.B, 2, device=self.device, generator=self.generator
        )
        self.target_pos = self._clamp_to_world(
            self.target_pos + self.target_speed * noise
        )

    def _compute_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute per-step team reward.

        Returns
        -------
        team_reward  : [B]      -- one scalar reward per env
        min_dist     : [B]      -- min pursuer-target distance (primary diagnostic)
        captured_now : [B] bool -- True if >= capture_k pursuers are within capture_radius

        Stashed on self (for step info dict, NEVER entered into obs/state):
            _last_dist_term               [B]
            _last_dists                   [B, N]
            _last_n_within                [B] int
            _last_dist_full_sight         [B]  min dist from full-sight (class-0) agents
            _last_dist_sensor_limited     [B]  min dist from sensor-limited (class-1) agents
            _last_capture_set_has_c1      [B] bool  capture_now AND any class-1 in within set
            _last_marginal_is_c1          [B] bool  marginal capturer is class-1
            _last_dist_contrib            [B]  -dist_coef * dist_term
            _last_capture_contrib         [B]  capture_bonus * captured_now.float()
        """
        # distances: [B, N]
        diff  = self.pursuer_pos - self.target_pos.unsqueeze(1)   # [B, N, 2]
        dists = torch.linalg.vector_norm(diff, dim=-1)             # [B, N]
        min_dist = dists.min(dim=1).values                         # [B]  (continuity key)

        # Coalition-distance dense term: mean of k smallest distances.
        # Guard k <= N so topk never requests more elements than available.
        k = min(self.capture_k, self.N)
        dist_term = torch.topk(dists, k=k, largest=False).values.mean(dim=1)  # [B]

        # Capture: require >= capture_k agents strictly within capture_radius.
        n_within     = (dists < self.capture_radius).sum(dim=1)    # [B] int
        captured_now = n_within >= self.capture_k                   # [B] bool

        # Stash dist_term and dists (for diagnostics below and step info dict).
        self._last_dist_term = dist_term
        self._last_dists     = dists
        self._last_n_within  = n_within

        # ------------------------------------------------------------------ #
        # Phase 3 diagnostics (stashed only; never enter obs/state)
        # ------------------------------------------------------------------ #

        class0_mask = (self.class_id == 0)  # [N] bool -- full-sight agents
        class1_mask = (self.class_id == 1)  # [N] bool -- sensor-limited agents

        # Minimum distance from each class to the target [B]
        if class0_mask.any():
            self._last_dist_full_sight = dists[:, class0_mask].min(dim=1).values  # [B]
        else:
            self._last_dist_full_sight = torch.full(
                (self.B,), float("nan"), device=self.device
            )

        if class1_mask.any():
            self._last_dist_sensor_limited = dists[:, class1_mask].min(dim=1).values  # [B]
        else:
            self._last_dist_sensor_limited = torch.full(
                (self.B,), float("nan"), device=self.device
            )

        # within[b, n] = True if agent n is strictly within capture_radius in env b
        # REUSE the same strict-< used for n_within (not recomputed differently)
        within = dists < self.capture_radius  # [B, N]

        # has_c1_within[b] = any sensor-limited agent is within capture_radius in env b
        has_c1_within = (within & class1_mask.unsqueeze(0)).any(dim=1)  # [B]

        # capture_set_has_sensor_limited: capture occurred AND a class-1 agent was in the set
        self._last_capture_set_has_c1 = captured_now & has_c1_within  # [B] bool

        # Marginal capturer: among within==True agents, the one with the LARGEST dist
        # still strictly < capture_radius.  Guard envs where n_within==0.
        # Set dists of non-within agents to -inf, then argmax.
        dists_within = dists.clone()
        dists_within[~within] = float("-inf")
        marginal_idx = dists_within.argmax(dim=1)  # [B] -- index of marginal agent

        # marginal_is_sensor_limited: capture occurred AND marginal agent is class-1
        marginal_class = self.class_id[marginal_idx]  # [B] long
        self._last_marginal_is_c1 = captured_now & (marginal_class == 1)  # [B] bool

        # Reward decomposition terms
        self._last_dist_contrib    = -self.dist_coef * dist_term      # [B]
        self._last_capture_contrib = self.capture_bonus * captured_now.float()  # [B]

        team_reward = (
            self._last_dist_contrib
            - self.step_penalty
            + self._last_capture_contrib
        )
        return team_reward, min_dist, captured_now

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Tensor, Tensor]:
        """
        Reset all parallel environments.

        Returns
        -------
        obs   : [B, N, obs_dim]  — masked per-agent observations
        state : [B, state_dim]   — privileged critic state (unmasked true target)
        """
        # Seeded spawn of pursuer and target positions
        self.pursuer_pos = self._random_positions((self.B, self.N, 2))
        self.target_pos  = self._random_positions((self.B, 2))

        # Reset counters
        self._t = 0
        self._ever_captured = torch.zeros(self.B, dtype=torch.bool, device=self.device)

        obs   = self._build_obs()
        state = self._build_state(t_norm=0.0)
        return obs, state

    def step(
        self, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        """
        Step all parallel environments.

        Auto-resets ALL envs in lockstep when _t >= max_steps.
        Terminal obs/state are built from PRE-RESET positions (mirrors VMASAdapter).

        Parameters
        ----------
        actions : [B, N] long — discrete action 0-4 for each pursuer.

        Returns
        -------
        obs    : [B, N, obs_dim]
        state  : [B, state_dim]
        reward : [B, N]          float
        done   : [B, N]          float (1.0 at truncation, else 0.0)
        info   : dict (see module docstring for required keys)
        """
        assert actions.shape == (self.B, self.N), (
            f"actions shape {actions.shape} != ({self.B}, {self.N})"
        )
        actions = actions.long().to(self.device)

        # Apply pursuer movements
        # delta: [B, N, 2] = action_deltas[actions] * move_step
        deltas = self._action_deltas[actions] * self.move_step  # [B, N, 2]
        self.pursuer_pos = self._clamp_to_world(self.pursuer_pos + deltas)

        # Move target (stochastic random walk)
        self._move_target()

        # Compute reward
        team_reward, min_dist, captured_now = self._compute_reward()

        # Update ever-captured flag
        self._ever_captured = self._ever_captured | captured_now
        captured_episode = self._ever_captured.clone()

        # Increment step counter BEFORE building terminal signals
        self._t += 1

        # Build done/truncation signals (lockstep: all envs truncate together)
        truncated_all = self._t >= self._max_steps
        truncated_env = torch.full(
            (self.B,), truncated_all, dtype=torch.bool, device=self.device
        )
        terminated_env = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        done = truncated_env.float().unsqueeze(1).expand(self.B, self.N).clone()

        # Expand team reward to per-agent reward: [B, N]
        reward = team_reward.unsqueeze(1).expand(self.B, self.N).clone()

        # Build terminal obs/state BEFORE reset (pre-reset positions).
        # terminal_state timestep = self._t / max_steps = 1.0 at truncation.
        terminal_obs   = self._build_obs()
        terminal_state = self._build_state(t_norm=self._t / self._max_steps)

        # Auto-reset on truncation (lockstep: all or none)
        if truncated_all:
            self.pursuer_pos = self._random_positions((self.B, self.N, 2))
            self.target_pos  = self._random_positions((self.B, 2))
            self._t = 0
            self._ever_captured = torch.zeros(self.B, dtype=torch.bool, device=self.device)

        # Build next obs/state from (possibly reset) positions
        # t_norm for returned state:
        #   - if just reset: 0.0 (new episode)
        #   - if not reset: self._t / max_steps
        t_norm_next = 0.0 if truncated_all else (self._t / self._max_steps)
        obs   = self._build_obs()
        state = self._build_state(t_norm=t_norm_next)

        # Shape assertions
        assert obs.shape    == (self.B, self.N, self.obs_dim),   f"obs shape {obs.shape}"
        assert state.shape  == (self.B, self.state_dim),          f"state shape {state.shape}"
        assert reward.shape == (self.B, self.N),                  f"reward shape {reward.shape}"
        assert done.shape   == (self.B, self.N),                  f"done shape {done.shape}"

        info = {
            "truncated":               truncated_env,
            "terminated":              terminated_env,
            "terminal_obs":            terminal_obs,
            "terminal_state":          terminal_state,
            "pursuer_target_distance": min_dist.clone(),
            "pursuer_k_distance":      self._last_dist_term.clone(),
            "capture":                 captured_now.float().clone(),
            "captured_episode":        captured_episode.float(),
            "target_pos":              self.target_pos.clone(),  # ground-truth for probe
            # Phase 3 diagnostics (stashed in _compute_reward; NEVER from obs/state)
            "dist_full_sight":              self._last_dist_full_sight.clone(),
            "dist_sensor_limited":          self._last_dist_sensor_limited.clone(),
            "capture_set_has_sensor_limited": self._last_capture_set_has_c1.clone(),
            "n_within_mean":                self._last_n_within.float().clone(),
            "marginal_is_sensor_limited":   self._last_marginal_is_c1.clone(),
            "dist_term_contrib":            self._last_dist_contrib.clone(),
            "capture_bonus_contrib":        self._last_capture_contrib.clone(),
        }
        return obs, state, reward, done, info
