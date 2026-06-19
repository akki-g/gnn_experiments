"""
Hard-assertion tests for PCPEnv.

Tests:
1. Construction: valid defaults work; invalid args raise.
2. Capturer observation isolation: capturer obs byte-identical across prey positions at alpha=0.
   Complement: detector obs DOES change with prey position.
3. Detection timer: detector within detect_radius -> steps_since_detection==0; all outside -> +1.
4. Window gating: capture_without_recent_detection is always 0; constructed scenario confirms
   capture is blocked when window is closed and unblocked when opened.
5. Class-1-only capture: detector ON prey (within capture_radius), no capturer near, window open
   -> captured_now must be False.
6. Free-rider check: moving a non-capturing capturer does NOT change team_reward when neither
   capturer placement causes a capture.
"""

import math
import torch
import pytest

from envs.pcp import PCPEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(**overrides):
    """Minimal valid PCPEnv constructor."""
    defaults = dict(
        num_envs=4,
        num_agents=5,
        num_full_sight=3,
        alpha=0.0,
        target_speed=0.0,
        world_size=12.0,
        move_step=0.2,
        capture_radius=0.6,
        capture_k=1,
        detect_radius=2.0,
        detection_window=5,
        dist_coef=0.1,
        step_penalty=0.01,
        capture_bonus=10.0,
        observe_teammates=False,
        max_steps=100,
        seed=42,
        device="cpu",
    )
    defaults.update(overrides)
    return PCPEnv(**defaults)


# ---------------------------------------------------------------------------
# 1. Construction guards
# ---------------------------------------------------------------------------

def test_construction_defaults_works():
    env = _make_env()
    obs, state = env.reset()
    assert obs.shape == (4, 5, env.obs_dim)
    assert state.shape == (4, env.state_dim)
    assert env.target_speed == 0.0
    assert env.alpha == 0.0
    assert not env.observe_teammates
    assert env.detect_radius == 2.0
    assert env.detection_window == 5


def test_construction_alpha_nonzero_raises():
    with pytest.raises(ValueError, match="alpha=0.0"):
        _make_env(alpha=0.5)


def test_construction_alpha_one_raises():
    with pytest.raises(ValueError, match="alpha=0.0"):
        _make_env(alpha=1.0)


def test_construction_observe_teammates_true_raises():
    with pytest.raises(ValueError, match="observe_teammates=False"):
        _make_env(observe_teammates=True)


def test_construction_zero_detectors_raises():
    # num_full_sight=0 => no detectors
    with pytest.raises(ValueError, match="num_full_sight >= 1"):
        _make_env(num_full_sight=0)


def test_construction_zero_capturers_raises():
    # all agents are detectors => no capturers
    with pytest.raises(ValueError, match="capturer"):
        _make_env(num_agents=3, num_full_sight=3)


# ---------------------------------------------------------------------------
# 2. Capturer observation isolation
# ---------------------------------------------------------------------------

def test_capturer_obs_independent_of_prey_position():
    """
    At alpha=0, capturer (class-1) obs must be byte-identical regardless of prey position.
    This replicates the pattern in test_observation_isolation.py for PCPEnv.
    """
    B, N = 4, 5
    num_full_sight = 3
    env = _make_env(num_envs=B, num_agents=N, num_full_sight=num_full_sight)
    env.reset()

    # Fix pursuer positions manually
    env.pursuer_pos = torch.zeros(B, N, 2)
    env.pursuer_pos[:, 0, :] = torch.tensor([1.0, 2.0])
    env.pursuer_pos[:, 1, :] = torch.tensor([-3.0, 1.5])
    env.pursuer_pos[:, 2, :] = torch.tensor([0.5, -1.0])
    env.pursuer_pos[:, 3, :] = torch.tensor([-2.0, -2.0])
    env.pursuer_pos[:, 4, :] = torch.tensor([4.0, 0.5])

    # Reference obs: prey at origin
    env.target_pos = torch.zeros(B, 2)
    ref_obs = env._build_obs()

    # Capturer indices (class-1 agents = index num_full_sight..N-1)
    capturer_indices = list(range(num_full_sight, N))

    # Sweep target positions: corners, origin, capturer own positions, random
    target_positions = [
        torch.zeros(B, 2),
        torch.full((B, 2), 11.9),
        torch.full((B, 2), -11.9),
        torch.stack([torch.full((B,), 11.9), torch.full((B,), -11.9)], dim=-1),
    ]
    # Each capturer's own position (dist==0 case)
    for idx in capturer_indices:
        target_positions.append(env.pursuer_pos[:, idx, :].clone())
    # Random positions
    for _ in range(30):
        target_positions.append((torch.rand(B, 2) * 2 - 1) * env.world_size)

    for target_pos in target_positions:
        env.target_pos = target_pos
        obs = env._build_obs()
        for n in capturer_indices:
            assert torch.equal(obs[:, n, :], ref_obs[:, n, :]), (
                f"Capturer agent {n} obs differs at alpha=0 for target={target_pos[0].tolist()}. "
                "Target slice must be zeroed regardless of prey position."
            )


def test_detector_obs_changes_with_prey_position():
    """Complement: a detector's obs DOES change with prey position (test is sensitive)."""
    B, N = 4, 5
    num_full_sight = 3
    env = _make_env(num_envs=B, num_agents=N, num_full_sight=num_full_sight)
    env.reset()
    env.pursuer_pos = torch.zeros(B, N, 2)

    env.target_pos = torch.zeros(B, 2)
    obs_a = env._build_obs()

    env.target_pos = torch.full((B, 2), 5.0)
    obs_b = env._build_obs()

    # Detector (class-0, index 0) obs should differ
    assert not torch.equal(obs_a[:, 0, :], obs_b[:, 0, :]), (
        "Detector (class-0) obs must change when prey moves. "
        "If this fails, the isolation test above is vacuous."
    )


# ---------------------------------------------------------------------------
# 3. Detection timer
# ---------------------------------------------------------------------------

def test_detection_timer_resets_on_detection():
    """
    When a detector is placed within detect_radius of prey, steps_since_detection becomes 0.
    """
    B = 4
    env = _make_env(num_envs=B, num_agents=5, num_full_sight=3,
                    detect_radius=2.0, detection_window=5)
    env.reset()

    # Place prey at origin, detector (agent 0, class-0) at distance 1.0 (< detect_radius=2.0)
    env.pursuer_pos = torch.zeros(B, 5, 2)
    env.pursuer_pos[:, 0, :] = torch.tensor([1.0, 0.0])  # distance = 1.0 from origin
    env.target_pos = torch.zeros(B, 2)

    # Run _compute_reward directly to check timer
    env._compute_reward()

    assert (env.steps_since_detection == 0).all(), (
        f"steps_since_detection should be 0 when a detector is within detect_radius, "
        f"got {env.steps_since_detection}"
    )


def test_detection_timer_increments_when_no_detection():
    """
    When all detectors are outside detect_radius, steps_since_detection increments by 1.
    """
    B = 4
    detect_radius = 2.0
    env = _make_env(num_envs=B, num_agents=5, num_full_sight=3,
                    detect_radius=detect_radius, detection_window=5)
    env.reset()

    # Place all agents far from prey (well outside detect_radius)
    env.pursuer_pos = torch.zeros(B, 5, 2)
    env.pursuer_pos[:, :, 0] = 10.0   # all agents at x=10, prey at origin => dist=10 >> detect_radius
    env.target_pos = torch.zeros(B, 2)

    # Set initial timer to a known value
    initial_val = 3
    env.steps_since_detection = torch.full((B,), initial_val, dtype=torch.long)

    env._compute_reward()

    assert (env.steps_since_detection == initial_val + 1).all(), (
        f"steps_since_detection should increment by 1 when no detection, "
        f"got {env.steps_since_detection}"
    )


# ---------------------------------------------------------------------------
# 4. Window gating / capture_without_recent_detection
# ---------------------------------------------------------------------------

def test_capture_without_recent_detection_always_zero():
    """
    Over random steps, capture_without_recent_detection must always be 0.
    This is guaranteed by construction: captured_now = proximity_capture & window_open.
    """
    env = _make_env(num_envs=8, seed=99)
    env.reset()
    actions = torch.zeros(8, 5, dtype=torch.long)

    for _ in range(30):
        _, _, _, _, info = env.step(actions)
        assert info["capture_without_recent_detection"].sum() == 0, (
            "capture_without_recent_detection must always be 0 by construction."
        )


def test_window_blocks_capture_when_closed():
    """
    Capturer is ON prey (within capture_radius), but window is forced closed
    (steps_since_detection > detection_window) => captured_now must be False.
    """
    B = 4
    env = _make_env(num_envs=B, num_agents=5, num_full_sight=3,
                    detect_radius=2.0, detection_window=5,
                    capture_radius=0.6, capture_k=1)
    env.reset()

    # Force window closed: steps_since_detection = window+1
    env.steps_since_detection = torch.full((B,), env.detection_window + 1, dtype=torch.long)

    # Place capturer (agent 3, class-1) directly ON prey (dist=0 < capture_radius)
    env.pursuer_pos = torch.zeros(B, 5, 2)
    env.target_pos = torch.zeros(B, 2)
    # Detectors far from detect_radius (no detection event this step either)
    env.pursuer_pos[:, 0, :] = torch.tensor([10.0, 0.0])
    env.pursuer_pos[:, 1, :] = torch.tensor([0.0, 10.0])
    env.pursuer_pos[:, 2, :] = torch.tensor([-10.0, 0.0])
    # Capturer at origin (== prey position, dist=0)
    env.pursuer_pos[:, 3, :] = torch.tensor([0.0, 0.0])
    env.pursuer_pos[:, 4, :] = torch.tensor([5.0, 5.0])

    team_reward, min_dist, captured_now = env._compute_reward()
    assert not captured_now.any(), (
        f"With window closed and no detector in range, captured_now must be False. "
        f"Got captured_now={captured_now}"
    )


def test_window_allows_capture_when_open():
    """
    Capturer within capture_radius AND detector within detect_radius => capture is True.
    """
    B = 4
    env = _make_env(num_envs=B, num_agents=5, num_full_sight=3,
                    detect_radius=2.0, detection_window=5,
                    capture_radius=0.6, capture_k=1)
    env.reset()

    # Force window closed initially; detector will fire this step and open window
    env.steps_since_detection = torch.full((B,), env.detection_window + 1, dtype=torch.long)

    # Place detector (agent 0, class-0) within detect_radius (1.0 < 2.0)
    # Place capturer (agent 3, class-1) within capture_radius (0.1 < 0.6)
    env.pursuer_pos = torch.zeros(B, 5, 2)
    env.target_pos = torch.zeros(B, 2)
    env.pursuer_pos[:, 0, :] = torch.tensor([1.0, 0.0])   # detector: dist=1.0 < detect_radius=2.0
    env.pursuer_pos[:, 1, :] = torch.tensor([10.0, 0.0])
    env.pursuer_pos[:, 2, :] = torch.tensor([0.0, 10.0])
    env.pursuer_pos[:, 3, :] = torch.tensor([0.1, 0.0])   # capturer: dist=0.1 < capture_radius=0.6
    env.pursuer_pos[:, 4, :] = torch.tensor([5.0, 5.0])

    team_reward, min_dist, captured_now = env._compute_reward()
    # Detection fires this step: window is now open (steps_since_detection=0 <= window)
    assert captured_now.all(), (
        f"With detector in detect_radius AND capturer in capture_radius, "
        f"captured_now must be True. Got captured_now={captured_now}"
    )


def test_window_expires_after_detection_window_steps():
    """
    After a detection fires, the window should close after detection_window steps
    if no further detection occurs. Confirm captured_now is False when window expires.
    """
    B = 2
    detection_window = 3
    env = _make_env(num_envs=B, num_agents=5, num_full_sight=3,
                    detect_radius=2.0, detection_window=detection_window,
                    capture_radius=0.6, capture_k=1, max_steps=200)
    env.reset()

    # Fire a detection event now: steps_since_detection -> 0
    env.steps_since_detection = torch.zeros(B, dtype=torch.long)

    # Place all detectors far (no detection event in subsequent steps)
    # Place capturer within capture_radius
    env.pursuer_pos = torch.zeros(B, 5, 2)
    env.target_pos = torch.zeros(B, 2)
    env.pursuer_pos[:, 0, :] = torch.tensor([10.0, 0.0])
    env.pursuer_pos[:, 1, :] = torch.tensor([0.0, 10.0])
    env.pursuer_pos[:, 2, :] = torch.tensor([-10.0, 0.0])
    env.pursuer_pos[:, 3, :] = torch.tensor([0.1, 0.0])   # capturer within capture_radius
    env.pursuer_pos[:, 4, :] = torch.tensor([5.0, 5.0])

    # Steps 1..detection_window: window should be open (steps_since_detection <= window)
    for step in range(1, detection_window + 1):
        _, _, captured_now = env._compute_reward()
        assert captured_now.all(), (
            f"Capture should succeed at step {step} (within window={detection_window}), "
            f"steps_since_detection={env.steps_since_detection}"
        )

    # Step detection_window+1: window should be closed
    _, _, captured_now = env._compute_reward()
    assert not captured_now.any(), (
        f"Capture should fail at step {detection_window+1} (window expired), "
        f"steps_since_detection={env.steps_since_detection}"
    )


# ---------------------------------------------------------------------------
# 5. Class-1-only capture: detector ON prey, no capturer near -> no capture
# ---------------------------------------------------------------------------

def test_detector_cannot_trigger_capture():
    """
    Even if a detector (class-0) is within capture_radius and the window is open,
    captured_now must be False when NO class-1 capturer is within capture_radius.
    """
    B = 4
    env = _make_env(num_envs=B, num_agents=5, num_full_sight=3,
                    detect_radius=2.0, detection_window=5,
                    capture_radius=0.6, capture_k=1)
    env.reset()

    # Open window: a prior detection event
    env.steps_since_detection = torch.zeros(B, dtype=torch.long)

    env.pursuer_pos = torch.zeros(B, 5, 2)
    env.target_pos = torch.zeros(B, 2)
    # Detector (class-0) at distance 0.1 < capture_radius (AND triggers detection_event)
    env.pursuer_pos[:, 0, :] = torch.tensor([0.1, 0.0])   # class-0, dist=0.1 < 0.6
    env.pursuer_pos[:, 1, :] = torch.tensor([0.2, 0.0])   # class-0, dist=0.2 < 0.6
    env.pursuer_pos[:, 2, :] = torch.tensor([0.3, 0.0])   # class-0, dist=0.3 < 0.6
    # Capturers (class-1) far from prey
    env.pursuer_pos[:, 3, :] = torch.tensor([8.0, 0.0])
    env.pursuer_pos[:, 4, :] = torch.tensor([0.0, 9.0])

    _, _, captured_now = env._compute_reward()

    assert not captured_now.any(), (
        "Detectors (class-0) within capture_radius must NOT trigger a capture. "
        f"Capture requires class-1 capturers within capture_radius. "
        f"Got captured_now={captured_now}"
    )


# ---------------------------------------------------------------------------
# 6. Free-rider check: non-capturing capturer does not change team_reward
# ---------------------------------------------------------------------------

def test_capturer_position_absent_from_reward_when_no_capture():
    """
    Scheme B: team_reward dense term uses detector-min-dist only.
    Moving a capturer (class-1) that fails to capture must NOT change team_reward.
    """
    B = 4
    env = _make_env(num_envs=B, num_agents=5, num_full_sight=3,
                    detect_radius=2.0, detection_window=5,
                    capture_radius=0.6, capture_k=1,
                    dist_coef=0.1, step_penalty=0.01, capture_bonus=10.0)
    env.reset()

    # Fix detector positions (determine the dense term)
    base_pursuer_pos = torch.zeros(B, 5, 2)
    base_pursuer_pos[:, 0, :] = torch.tensor([3.0, 0.0])  # detector, dist=3.0
    base_pursuer_pos[:, 1, :] = torch.tensor([4.0, 0.0])
    base_pursuer_pos[:, 2, :] = torch.tensor([5.0, 0.0])
    env.target_pos = torch.zeros(B, 2)

    # Force window closed so no capture can occur regardless of capturer pos
    env.steps_since_detection = torch.full((B,), env.detection_window + 1, dtype=torch.long)

    # Placement A: capturers far from prey
    env.pursuer_pos = base_pursuer_pos.clone()
    env.pursuer_pos[:, 3, :] = torch.tensor([7.0, 0.0])
    env.pursuer_pos[:, 4, :] = torch.tensor([8.0, 0.0])
    reward_a, _, captured_a = env._compute_reward()
    # Reset timer (compute_reward updated it)
    env.steps_since_detection = torch.full((B,), env.detection_window + 1, dtype=torch.long)

    # Placement B: capturers at different positions (still far, no capture)
    env.pursuer_pos = base_pursuer_pos.clone()
    env.pursuer_pos[:, 3, :] = torch.tensor([9.0, 1.0])
    env.pursuer_pos[:, 4, :] = torch.tensor([-2.0, 3.0])
    reward_b, _, captured_b = env._compute_reward()

    # No capture in either placement
    assert not captured_a.any(), "Placement A should not trigger capture"
    assert not captured_b.any(), "Placement B should not trigger capture"

    # team_reward should be identical (detector positions fixed, capturers absent from dense term)
    assert torch.allclose(reward_a, reward_b, atol=1e-6), (
        "Moving non-capturing capturers must NOT change team_reward. "
        f"reward_a={reward_a}, reward_b={reward_b}. "
        "Scheme B: dense term is detector-only."
    )


# ---------------------------------------------------------------------------
# 7. Step-level info dict has required PCP keys
# ---------------------------------------------------------------------------

def test_step_info_has_pcp_keys():
    """PCPEnv.step() must return info with PCP diagnostic keys."""
    env = _make_env(num_envs=4, seed=7)
    env.reset()
    actions = torch.zeros(4, 5, dtype=torch.long)
    _, _, _, _, info = env.step(actions)

    required_pcp_keys = [
        "detection_event",
        "window_open",
        "capture_without_recent_detection",
        "steps_since_detection",
    ]
    for key in required_pcp_keys:
        assert key in info, f"PCP info key '{key}' missing from info dict"

    # Also verify parent keys are present
    required_parent_keys = [
        "truncated", "terminated", "terminal_obs", "terminal_state",
        "pursuer_target_distance", "pursuer_k_distance", "capture", "captured_episode",
        "target_pos",
    ]
    for key in required_parent_keys:
        assert key in info, f"Parent info key '{key}' missing from info dict"


# ---------------------------------------------------------------------------
# 8. Stationary prey: target_pos does NOT change during step
# ---------------------------------------------------------------------------

def test_prey_is_stationary():
    """PCP requires stationary prey: target_pos must not change after step."""
    env = _make_env(num_envs=4, seed=5)
    env.reset()
    # Manually set target_pos to known value
    env.target_pos = torch.tensor([[1.0, 2.0]] * 4)
    target_before = env.target_pos.clone()

    actions = torch.zeros(4, 5, dtype=torch.long)
    for _ in range(5):
        obs, state, reward, done, info = env.step(actions)

    # target_pos in info should match what was set (and not change)
    # After lockstep truncation it resets; run fewer steps than max_steps
    # (max_steps=100, we ran 5 steps — no reset)
    assert torch.allclose(env.target_pos, target_before), (
        "Prey must be stationary (target_speed=0.0). target_pos should not change."
    )


# ---------------------------------------------------------------------------
# 9. Detection timer resets after episode boundary (lockstep truncation)
# ---------------------------------------------------------------------------

def test_detection_timer_resets_after_episode():
    """After lockstep truncation, steps_since_detection should reset to window+1."""
    env = _make_env(num_envs=2, max_steps=5, seed=3)
    env.reset()

    # Set timer to 0 (window open)
    env.steps_since_detection = torch.zeros(2, dtype=torch.long)

    actions = torch.zeros(2, 5, dtype=torch.long)
    # Run max_steps to trigger truncation
    for _ in range(5):
        _, _, _, _, info = env.step(actions)

    # After truncation, timer should be reset to window+1 (closed)
    assert (env.steps_since_detection == env.detection_window + 1).all(), (
        f"After episode truncation, steps_since_detection should be {env.detection_window+1}. "
        f"Got {env.steps_since_detection}"
    )
