"""
Predator-Capture-Prey (PCP) environment — subclass of PursuitEnv.

PCP semantics
-------------
Two agent classes, determined by class_id (same as PursuitEnv):
  class 0 = detectors  (full-sight, use r_max; observe the prey)
  class 1 = capturers  (observation radius = alpha * r_max; alpha=0.0 default => hard-blind)

alpha = capturer observation-radius fraction (the OBSERVATION-HETEROGENEITY axis):
  alpha=0.0 (default): capturers cannot see the prey at all -> communication is a
    STRUCTURAL precondition for capture (the canonical PCP comm-necessity regime).
  alpha in (0,1]: capturers see the prey within alpha*r_max themselves, so they
    depend on the detector relay less and less. Sweeping alpha 0 -> 1 turns the
    binary PCP-vs-PP contrast into a continuum: comm necessity is expected to decay
    smoothly toward alpha=1. Detectors (class 0) are ALWAYS full-sight regardless
    of alpha; only capturer sight varies.

Prey (target): STATIONARY (target_speed=0.0, enforced).

Capture is gated by a detection window:
  - A detector within detect_radius of the prey fires a detection event.
  - After a detection event, the window is open for the next `detection_window` steps
    (steps_since_detection <= detection_window).
  - A capture event requires:
      * >= capture_k CAPTURERS (class-1 agents, NOT detectors) within capture_radius, AND
      * the window is currently open (a detector saw the prey recently enough).

Reward (Scheme B):
  - Dense distance term uses DETECTOR-ONLY min distance to prey (capturer positions
    never appear in the reward except via sparse capture bonus).
  - dist_contrib    = -dist_coef * detector_min_dist
  - capture_contrib = capture_bonus * captured_now.float()
  - team_reward     = dist_contrib - step_penalty + capture_contrib

Observation isolation:
  - At alpha=0.0 (default): capturers are structurally blind; their obs are
    byte-identical regardless of prey position (the strict-isolation regime that
    makes comm the ONLY route to prey location).
  - At alpha>0: isolation is intentionally relaxed by exactly alpha — capturer obs
    becomes prey-dependent within alpha*r_max. This is the continuum knob, not a leak.
  - observe_teammates MUST be False regardless (no cross-agent position leakage).

CONSTRUCT-VALIDITY caveat:
  PCP capture is position+class+detection-window based, not an explicit capture
  action as in Seraj et al. 2022; the detection window substitutes for the
  explicit-action gating of capture validity.
"""

import torch
from torch import Tensor
from typing import Tuple, Dict, Any

from envs.pursuit import PursuitEnv


class PCPEnv(PursuitEnv):
    """
    Predator-Capture-Prey environment.

    Subclass of PursuitEnv. All structural changes:
      - Prey is stationary (target_speed=0.0 forced).
      - Capturers are hard-blind (alpha=0.0 forced).
      - Capture requires class-1 (capturers) within capture_radius AND a recent
        detection event from class-0 (detectors) within detect_radius.
      - Reward dense term is detector-min-dist only (Scheme B).

    Extra constructor parameters
    ----------------------------
    detect_radius    : float — distance within which a detector triggers a detection event.
    detection_window : int  — number of steps after a detection event the window stays open (T).
    """

    def __init__(
        self,
        detect_radius: float,
        detection_window: int,
        **kwargs,
    ):
        """
        Parameters
        ----------
        detect_radius    : float, detection radius for class-0 agents.
        detection_window : int, T — window open for T steps after detection.
        **kwargs         : forwarded to PursuitEnv.__init__.

        Enforced constraints (ValueError/AssertionError on violation):
          - alpha must be in [0, 1] (capturer observation-radius fraction; default 0.0
            = hard-blind canonical regime; >0 = observation-heterogeneity continuum).
          - target_speed must be 0.0 (prey is stationary in PCP).
          - observe_teammates must be False (no inter-agent position leakage).
          - num_full_sight >= 1 (need at least 1 detector).
          - num_agents - num_full_sight >= 1 (need at least 1 capturer).
        """
        # --- Pre-construction guards ---
        # alpha = capturer (class-1) observation-radius fraction. Default 0.0 keeps
        # the canonical hard-blind comm-necessity regime so existing configs are
        # unchanged; alpha in (0,1] enables the observation-heterogeneity continuum
        # (capturers see the prey within alpha*r_max; detectors stay full-sight).
        alpha = float(kwargs.get("alpha", 0.0))
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"PCPEnv alpha (capturer observation-radius fraction) must be in "
                f"[0, 1], got alpha={alpha}."
            )
        kwargs["alpha"] = alpha  # make the default explicit for super().__init__

        # Force stationary prey before calling super().__init__
        kwargs["target_speed"] = 0.0

        observe_teammates = bool(kwargs.get("observe_teammates", False))
        if observe_teammates:
            raise ValueError(
                "PCPEnv requires observe_teammates=False. "
                "Capturers must not see detector positions — that could leak prey "
                "location and re-confound the ablation."
            )

        num_agents = int(kwargs.get("num_agents", 3))
        num_full_sight = int(kwargs.get("num_full_sight", 2))

        if num_full_sight < 1:
            raise ValueError(
                f"PCPEnv requires num_full_sight >= 1 (at least 1 detector). "
                f"Got num_full_sight={num_full_sight}."
            )
        n_capturers = num_agents - num_full_sight
        if n_capturers < 1:
            raise ValueError(
                f"PCPEnv requires at least 1 capturer (class-1 agent). "
                f"Got num_agents={num_agents}, num_full_sight={num_full_sight} "
                f"=> n_capturers={n_capturers}."
            )

        super().__init__(**kwargs)

        # Post-init assertions
        assert self.target_speed == 0.0, (
            f"PCP requires stationary prey (target_speed=0.0), got {self.target_speed}"
        )
        assert 0.0 <= self.alpha <= 1.0, (
            f"PCP capturer alpha must be in [0,1], got {self.alpha}"
        )
        assert not self.observe_teammates, (
            "PCP requires observe_teammates=False"
        )

        # PCP-specific attributes
        self.detect_radius = float(detect_radius)
        self.detection_window = int(detection_window)

        # steps_since_detection: [B] long — allocated here, set properly in reset()
        self.steps_since_detection: Tensor = torch.full(
            (self.B,), self.detection_window + 1,
            dtype=torch.long, device=self.device
        )

        # Stash PCP-specific last-step values (init to safe defaults)
        self._last_detection_event: Tensor = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self._last_window_open: Tensor = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self._last_cap_without_recent: Tensor = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self._last_steps_since_detection: Tensor = self.steps_since_detection.clone()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Tensor, Tensor]:
        """
        Reset all parallel environments.

        Delegates to PursuitEnv.reset() then initialises the detection timer
        to (detection_window + 1) — window CLOSED at episode start.

        Returns
        -------
        obs   : [B, N, obs_dim]
        state : [B, state_dim]
        """
        obs, state = super().reset()
        # Window CLOSED at episode start: no prior detection event.
        self.steps_since_detection = torch.full(
            (self.B,), self.detection_window + 1,
            dtype=torch.long, device=self.device
        )
        return obs, state

    # ------------------------------------------------------------------
    # _compute_reward (Scheme B + detection timer + class-1-only windowed capture)
    # ------------------------------------------------------------------

    def _compute_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        PCP reward: Scheme B (detector-only dense term) + detection-window gating.

        Overrides PursuitEnv._compute_reward completely.

        Returns
        -------
        team_reward  : [B]      — one scalar reward per env
        min_dist     : [B]      — min pursuer-target distance (all agents, for logging)
        captured_now : [B] bool — valid capture (class-1 only, window open)

        All parent step() stash attrs are set here to avoid AttributeError.
        PCP-specific stash attrs are also set for use in step() info dict.
        """
        # --- Distances: all agents ---
        diff  = self.pursuer_pos - self.target_pos.unsqueeze(1)  # [B, N, 2]
        dists = torch.linalg.vector_norm(diff, dim=-1)            # [B, N]
        min_dist = dists.min(dim=1).values                        # [B]

        # --- Class masks ---
        class0_mask = (self.class_id == 0)   # [N] bool — detectors
        class1_mask = (self.class_id == 1)   # [N] bool — capturers

        detector_dists  = dists[:, class0_mask]   # [B, n_det]
        capturer_dists  = dists[:, class1_mask]   # [B, n_cap]

        detector_min_dist = detector_dists.min(dim=1).values  # [B]

        # --- Detection event at this step ---
        detection_event = (detector_dists < self.detect_radius).any(dim=1)  # [B] bool

        # --- Update detection timer ---
        # Reset to 0 on detection; increment otherwise (clamped at window+1 for safety)
        self.steps_since_detection = torch.where(
            detection_event,
            torch.zeros_like(self.steps_since_detection),
            self.steps_since_detection + 1,
        )

        window_open = self.steps_since_detection <= self.detection_window  # [B] bool

        # --- Class-1-only proximity capture, gated by window ---
        n_within_c1 = (capturer_dists < self.capture_radius).sum(dim=1)   # [B] int
        proximity_capture = n_within_c1 >= self.capture_k                  # [B] bool
        captured_now      = proximity_capture & window_open                # [B] bool — VALID capture

        # Hard self-check: a counted capture must never have a closed window.
        cap_without_recent = captured_now & (~window_open)  # must be all-False by construction
        assert not bool(cap_without_recent.any()), (
            "PCP detection-window gating bug: counted capture with closed window"
        )

        # --- Scheme B reward ---
        dist_contrib    = -self.dist_coef * detector_min_dist        # [B]
        capture_contrib =  self.capture_bonus * captured_now.float()  # [B]
        team_reward     =  dist_contrib - self.step_penalty + capture_contrib  # [B]

        # ------------------------------------------------------------------
        # Stash parent-required attrs (PursuitEnv.step() info-dict reads these)
        # ------------------------------------------------------------------
        self._last_dist_term         = detector_min_dist       # parent uses as "pursuer_k_distance"
        self._last_dists             = dists
        self._last_n_within          = n_within_c1             # class-1 within count
        self._last_dist_full_sight   = detector_min_dist       # detectors
        self._last_dist_sensor_limited = capturer_dists.min(dim=1).values   # capturers (diagnostic)
        self._last_capture_set_has_c1  = captured_now          # capturers are the ONLY capturers
        self._last_marginal_is_c1      = captured_now          # marginal capturer always class-1
        self._last_dist_contrib        = dist_contrib
        self._last_capture_contrib     = capture_contrib

        # ------------------------------------------------------------------
        # PCP-specific stash attrs (used in step() override below)
        # ------------------------------------------------------------------
        self._last_detection_event         = detection_event
        self._last_window_open             = window_open
        self._last_cap_without_recent      = cap_without_recent
        self._last_steps_since_detection   = self.steps_since_detection.clone()

        return team_reward, min_dist, captured_now

    # ------------------------------------------------------------------
    # step — add PCP diagnostics to info
    # ------------------------------------------------------------------

    def step(
        self, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        """
        Step all parallel environments.

        Delegates to PursuitEnv.step() then appends PCP-specific info keys.
        Resets the detection timer after lockstep truncation.

        Returns
        -------
        obs    : [B, N, obs_dim]
        state  : [B, state_dim]
        reward : [B, N] float
        done   : [B, N] float
        info   : dict — all PursuitEnv keys PLUS PCP diagnostics
        """
        obs, state, reward, done, info = super().step(actions)

        # Add PCP diagnostics (stashed by _compute_reward before the parent auto-reset)
        info["detection_event"]                   = self._last_detection_event.float().clone()
        info["window_open"]                       = self._last_window_open.float().clone()
        info["capture_without_recent_detection"]  = self._last_cap_without_recent.float().clone()
        info["steps_since_detection"]             = self._last_steps_since_detection.float().clone()

        # After lockstep truncation, reset the detection timer for the next episode.
        if bool(info["truncated"].any()):
            self.steps_since_detection = torch.full_like(
                self.steps_since_detection, self.detection_window + 1
            )

        return obs, state, reward, done, info
