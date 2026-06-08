"""
Tests for PursuitEnv.

Mirrors tests/test_vmas_adapter.py style.
Uses small B=4, max_steps=5 for speed.
"""

import pytest
import torch

from envs.pursuit import PursuitEnv


def make_env(B=4, N=3, max_steps=5, seed=0, alpha=1.0):
    return PursuitEnv(
        num_envs=B,
        num_agents=N,
        max_steps=max_steps,
        seed=seed,
        device="cpu",
        alpha=alpha,
        num_full_sight=2,
        include_timestep=True,
    )


# ---------------------------------------------------------------------------
# Shape & dtype tests
# ---------------------------------------------------------------------------

def test_reset_shapes_and_dtypes():
    env = make_env()
    obs, state = env.reset()

    assert obs.shape   == (4, 3, env.obs_dim),  f"obs shape {obs.shape}"
    assert state.shape == (4, env.state_dim),    f"state shape {state.shape}"
    assert obs.dtype   == torch.float32
    assert state.dtype == torch.float32


def test_step_shapes_and_dtypes():
    env = make_env()
    env.reset()
    actions = torch.zeros(4, 3, dtype=torch.long)
    obs, state, reward, done, info = env.step(actions)

    assert obs.shape    == (4, 3, env.obs_dim),  f"obs shape {obs.shape}"
    assert state.shape  == (4, env.state_dim),    f"state shape {state.shape}"
    assert reward.shape == (4, 3),                f"reward shape {reward.shape}"
    assert done.shape   == (4, 3),                f"done shape {done.shape}"
    assert obs.dtype    == torch.float32
    assert reward.dtype == torch.float32
    assert done.dtype   == torch.float32


def test_action_input_accepted_as_long():
    env = make_env()
    env.reset()
    actions = torch.randint(0, 5, (4, 3), dtype=torch.long)
    obs, state, reward, done, info = env.step(actions)
    assert obs.shape == (4, 3, env.obs_dim)


def test_state_dim_with_include_timestep():
    """state_dim = N*2 + 2 + 1 with include_timestep=True."""
    N = 3
    env = PursuitEnv(num_envs=4, num_agents=N, include_timestep=True)
    expected = N * 2 + 2 + 1
    assert env.state_dim == expected, f"state_dim={env.state_dim}, expected {expected}"


def test_state_dim_without_include_timestep():
    """state_dim = N*2 + 2 without timestep."""
    N = 3
    env = PursuitEnv(num_envs=4, num_agents=N, include_timestep=False)
    expected = N * 2 + 2
    assert env.state_dim == expected, f"state_dim={env.state_dim}, expected {expected}"


def test_obs_dim_no_teammates():
    env = PursuitEnv(num_envs=4, num_agents=3, observe_teammates=False)
    assert env.obs_dim == 5


def test_obs_dim_with_teammates():
    N = 3
    env = PursuitEnv(num_envs=4, num_agents=N, observe_teammates=True)
    expected = 5 + 2 * (N - 1)
    assert env.obs_dim == expected, f"obs_dim={env.obs_dim}, expected {expected}"


# ---------------------------------------------------------------------------
# Reward is shared across all agents
# ---------------------------------------------------------------------------

def test_reward_is_shared_across_agents():
    """All agents receive the same team reward (reward[:,0] == reward[:,k] for all k)."""
    env = make_env()
    env.reset()
    actions = torch.zeros(4, 3, dtype=torch.long)
    _, _, reward, _, _ = env.step(actions)

    for k in range(1, 3):
        assert torch.allclose(reward[:, 0], reward[:, k]), (
            f"reward[:,0] != reward[:,{k}]: {reward}"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    """Two environments with the same seed produce identical obs/reward over several steps."""
    env1 = make_env(seed=42)
    env2 = make_env(seed=42)

    obs1, state1 = env1.reset()
    obs2, state2 = env2.reset()
    assert torch.equal(obs1, obs2), "reset obs differ"
    assert torch.equal(state1, state2), "reset state differ"

    actions = torch.zeros(4, 3, dtype=torch.long)
    for _ in range(3):
        obs1, s1, r1, d1, _ = env1.step(actions)
        obs2, s2, r2, d2, _ = env2.step(actions)
        assert torch.equal(obs1, obs2), "step obs differ"
        assert torch.equal(r1,   r2),   "step reward differ"


def test_different_seeds_differ():
    """Different seeds should produce different initial observations."""
    env1 = make_env(seed=0)
    env2 = make_env(seed=99)
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    assert not torch.equal(obs1, obs2), "Different seeds gave same obs"


# ---------------------------------------------------------------------------
# Truncation at max_steps
# ---------------------------------------------------------------------------

def test_truncation_at_max_steps():
    """After max_steps steps, all envs truncate, done==1, terminal_state timestep==1."""
    env = make_env(max_steps=5)
    env.reset()
    actions = torch.zeros(4, 3, dtype=torch.long)

    # Take max_steps-1 steps: no truncation
    for i in range(4):
        obs, state, reward, done, info = env.step(actions)
        assert not info["truncated"].any(), f"Unexpected truncation at step {i+1}"
        assert not done.any(), f"Unexpected done at step {i+1}"

    # Final step: truncation
    obs, state, reward, done, info = env.step(actions)
    assert info["truncated"].all(), "All envs should be truncated at max_steps"
    assert done.eq(1.0).all(), "done should be 1.0 at truncation"
    # terminal_state last column (timestep) should be 1.0
    assert info["terminal_state"][:, -1].eq(1.0).all(), (
        f"terminal_state timestep must be 1.0, got {info['terminal_state'][:, -1]}"
    )


def test_auto_reset_after_truncation():
    """After truncation the returned obs/state should come from a fresh episode (t=0)."""
    env = make_env(max_steps=5)
    env.reset()
    actions = torch.zeros(4, 3, dtype=torch.long)
    for _ in range(5):
        obs, state, reward, done, info = env.step(actions)

    # state last column should be 0.0 (new episode, t=0/max_steps=0.0)
    assert (state[:, -1] == 0.0).all(), (
        f"Post-truncation state timestep must be 0.0, got {state[:, -1]}"
    )
    assert env._t == 0, f"env._t should be 0 after auto-reset, got {env._t}"


def test_captured_episode_survives_auto_reset_info():
    """
    At a truncation boundary, info must describe the episode that just ended
    even though the returned obs/state have already been auto-reset.
    """
    env = PursuitEnv(
        num_envs=4,
        num_agents=3,
        max_steps=1,
        capture_radius=10.0,
        target_speed=0.0,
    )
    env.reset()
    actions = torch.zeros(4, 3, dtype=torch.long)

    _, _, _, _, info = env.step(actions)

    assert info["truncated"].all()
    assert info["captured_episode"].eq(1.0).all(), (
        "captured_episode should report captures from the just-finished episode"
    )


# ---------------------------------------------------------------------------
# Info dict keys
# ---------------------------------------------------------------------------

def test_info_contains_required_keys():
    env = make_env()
    env.reset()
    actions = torch.zeros(4, 3, dtype=torch.long)
    _, _, _, _, info = env.step(actions)

    required_keys = [
        "truncated", "terminated",
        "terminal_obs", "terminal_state",
        "pursuer_target_distance", "capture",
        "captured_episode", "target_pos",
    ]
    for key in required_keys:
        assert key in info, f"Missing info key: {key}"


def test_info_shapes():
    env = make_env()
    env.reset()
    actions = torch.zeros(4, 3, dtype=torch.long)
    obs, state, reward, done, info = env.step(actions)

    assert info["truncated"].shape         == (4,)
    assert info["terminated"].shape        == (4,)
    assert info["terminal_obs"].shape      == (4, 3, env.obs_dim)
    assert info["terminal_state"].shape    == (4, env.state_dim)
    assert info["pursuer_target_distance"].shape == (4,)
    assert info["capture"].shape           == (4,)
    assert info["captured_episode"].shape  == (4,)
    assert info["target_pos"].shape        == (4, 2)


# ---------------------------------------------------------------------------
# Mode A: terminated is always False
# ---------------------------------------------------------------------------

def test_terminated_always_false():
    env = make_env(max_steps=5)
    env.reset()
    actions = torch.zeros(4, 3, dtype=torch.long)
    for _ in range(5):
        _, _, _, _, info = env.step(actions)
        assert not info["terminated"].any(), "terminated should always be False (mode A)"


# ---------------------------------------------------------------------------
# Mode B raises NotImplementedError
# ---------------------------------------------------------------------------

def test_terminate_on_capture_raises():
    with pytest.raises(NotImplementedError):
        PursuitEnv(
            num_envs=4, num_agents=3, max_steps=5,
            terminate_on_capture=True,
        )


# ---------------------------------------------------------------------------
# pursuer_target_distance is non-negative
# ---------------------------------------------------------------------------

def test_pursuer_target_distance_non_negative():
    env = make_env()
    env.reset()
    actions = torch.zeros(4, 3, dtype=torch.long)
    for _ in range(4):
        _, _, _, _, info = env.step(actions)
        assert (info["pursuer_target_distance"] >= 0.0).all(), (
            "pursuer_target_distance must be non-negative"
        )


# ---------------------------------------------------------------------------
# class_id attribute
# ---------------------------------------------------------------------------

def test_class_id_shape_and_values():
    env = PursuitEnv(num_envs=4, num_agents=4, num_full_sight=2)
    cid = env.class_id
    assert cid.shape == (4,)
    assert (cid[:2] == 0).all(), "First 2 agents should be class 0 (full-sight)"
    assert (cid[2:] == 1).all(), "Last 2 agents should be class 1 (sensor-limited)"


# ---------------------------------------------------------------------------
# capture_k coalition semantics
# ---------------------------------------------------------------------------

def test_k2_capture_requires_two_agents():
    """
    k=2 semantics: exactly testing the three cases.

    Row 0: 1 agent inside capture_radius -> captured_now False.
    Row 1: 2 agents inside capture_radius -> captured_now True.
    Row 2: 2 agents exactly AT capture_radius (strict <) -> captured_now False.
    """
    env = PursuitEnv(
        num_envs=3,
        num_agents=3,
        max_steps=5,
        capture_k=2,
        capture_radius=0.5,
        world_size=10.0,
        target_speed=0.0,
        move_step=0.0,
        seed=0,
    )
    env.reset()

    # Fix target at origin, pursuers placed as axis-aligned points
    # so vector_norm returns the x-coordinate exactly.
    env.target_pos = torch.zeros(3, 2)
    env.pursuer_pos = torch.tensor([
        # Row 0: dists [0.3, 0.9, 0.9] -> 1 inside  -> not captured
        [[0.3, 0.0], [0.9, 0.0], [0.9, 0.0]],
        # Row 1: dists [0.3, 0.4, 0.9] -> 2 inside  -> captured
        [[0.3, 0.0], [0.4, 0.0], [0.9, 0.0]],
        # Row 2: dists [0.5, 0.5, 0.9] -> 0 inside (strict <) -> not captured
        [[0.5, 0.0], [0.5, 0.0], [0.9, 0.0]],
    ])  # [3, 3, 2]

    actions = torch.zeros(3, 3, dtype=torch.long)
    _, _, _, _, info = env.step(actions)

    assert info["capture"][0].item() == pytest.approx(0.0), "Row 0: 1 agent inside should not capture"
    assert info["capture"][1].item() == pytest.approx(1.0), "Row 1: 2 agents inside should capture"
    assert info["capture"][2].item() == pytest.approx(0.0), "Row 2: boundary (d==r) should not capture"

    # Dense term for Row 1: mean of 2 smallest = (0.3 + 0.4) / 2 = 0.35
    assert info["pursuer_k_distance"][1].item() == pytest.approx(0.35, abs=1e-5), \
        "pursuer_k_distance must equal mean of k smallest distances"

    # For Row 1, min_dist should be 0.3 (continuity key unchanged)
    assert info["pursuer_target_distance"][1].item() == pytest.approx(0.3, abs=1e-5), \
        "pursuer_target_distance must still equal min_dist"


def test_k1_backward_compat():
    """
    k=1 (default) behavior must be bit-identical to old single-agent capture.
    - 1 agent inside capture_radius is sufficient for captured_now=True.
    - pursuer_k_distance == pursuer_target_distance (dist_term == min_dist when k=1).
    """
    env = PursuitEnv(
        num_envs=3,
        num_agents=3,
        max_steps=5,
        capture_k=1,
        capture_radius=0.5,
        world_size=10.0,
        target_speed=0.0,
        move_step=0.0,
        seed=0,
    )
    env.reset()

    env.target_pos = torch.zeros(3, 2)
    env.pursuer_pos = torch.tensor([
        [[0.3, 0.0], [0.9, 0.0], [0.9, 0.0]],
        [[0.3, 0.0], [0.4, 0.0], [0.9, 0.0]],
        [[0.5, 0.0], [0.5, 0.0], [0.9, 0.0]],
    ])

    actions = torch.zeros(3, 3, dtype=torch.long)
    _, _, _, _, info = env.step(actions)

    # k=1: any single agent inside suffices
    assert info["capture"][0].item() == pytest.approx(1.0), "k=1 Row 0: 1 agent inside should capture"
    assert info["capture"][1].item() == pytest.approx(1.0), "k=1 Row 1: 2 agents inside should capture"
    assert info["capture"][2].item() == pytest.approx(0.0), "k=1 Row 2: boundary should not capture"

    # dist_term == min_dist when k=1
    assert torch.allclose(
        info["pursuer_k_distance"], info["pursuer_target_distance"], atol=1e-6
    ), "pursuer_k_distance must equal pursuer_target_distance when capture_k=1"
