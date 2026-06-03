"""
Smoke tests for the VMAS adapter used by the HPC simple_spread deployment.
"""

import pytest
import torch

pytest.importorskip("vmas")

from envs.adapters import VMASAdapter
from experiments.train_vmas_mappo import _bad_mask_from_info


def make_env(max_steps=2):
    return VMASAdapter(
        scenario="simple_spread",
        num_envs=2,
        n_agents=3,
        max_steps=max_steps,
        seed=123,
        device="cpu",
    )


def test_vmas_adapter_reset_and_step_shapes():
    env = make_env(max_steps=3)
    obs, state = env.reset()

    assert obs.shape == (2, 3, env.obs_dim)
    assert state.shape == (2, env.state_dim)
    # include_timestep=True by default: state_dim = N*obs_dim + 1
    assert env.state_dim == env.num_agents * env.obs_dim + 1
    assert (state[:, -1] == 0.0).all()

    actions = torch.zeros(2, 3, dtype=torch.long)
    next_obs, next_state, reward, done, info = env.step(actions)

    assert next_obs.shape == (2, 3, env.obs_dim)
    assert next_state.shape == (2, env.state_dim)
    assert reward.shape == (2, 3)
    assert done.shape == (2, 3)
    assert info["truncated"].shape == (2,)
    assert info["terminated"].shape == (2,)
    assert info["terminal_obs"].shape == (2, 3, env.obs_dim)
    assert info["terminal_state"].shape == (2, env.state_dim)
    assert not info["truncated"].any()


def test_vmas_adapter_truncation_metadata_and_bad_mask():
    env = make_env(max_steps=2)
    obs, _ = env.reset()
    actions = torch.zeros(2, 3, dtype=torch.long)

    next_obs_1, _, _, done_1, info_1 = env.step(actions)
    assert not done_1.any()
    assert not info_1["truncated"].any()

    next_obs_2, _, _, done_2, info_2 = env.step(actions)
    assert done_2.eq(1.0).all()
    assert info_2["truncated"].all()
    assert info_2["terminal_obs"].shape == (2, 3, env.obs_dim)

    # The adapter auto-resets before returning next_obs, but keeps the terminal
    # observation for value bootstrapping.
    assert not torch.allclose(next_obs_2, info_2["terminal_obs"])

    bad_mask = _bad_mask_from_info(info_2, env.num_agents, torch.device("cpu"))
    assert bad_mask.shape == (2, 3)
    assert bad_mask.eq(0.0).all()

    bad_mask_none = _bad_mask_from_info(info_1, env.num_agents, torch.device("cpu"))
    assert bad_mask_none is None

    # Terminal state at truncation must have timestep column == 1.0
    assert info_2["terminal_state"][:, -1].eq(1.0).all()


def test_vmas_adapter_legacy_state_dim_no_timestep():
    """When include_timestep=False, state_dim is N*obs_dim with no extra column."""
    env = VMASAdapter(
        scenario="simple_spread",
        num_envs=2,
        n_agents=3,
        max_steps=3,
        seed=123,
        device="cpu",
        include_timestep=False,
    )
    assert env.state_dim == env.num_agents * env.obs_dim

    obs, state = env.reset()
    assert state.shape == (2, env.state_dim)

    actions = torch.zeros(2, 3, dtype=torch.long)
    _, next_state, _, _, _ = env.step(actions)
    assert next_state.shape == (2, env.state_dim)


def test_vmas_adapter_timestep_alignment():
    """Verify timestep column values are correct at reset, mid-episode, and truncation."""
    env = VMASAdapter(
        scenario="simple_spread",
        num_envs=2,
        n_agents=2,
        max_steps=5,
        seed=123,
        device="cpu",
        include_timestep=True,
    )
    actions = torch.zeros(2, 2, dtype=torch.long)

    # (a) After reset: timestep column == 0.0
    obs, state = env.reset()
    assert (state[:, -1] == 0.0).all()

    # (b) One step mid-episode (t=1, no truncation): timestep == 1/5 = 0.2
    _, state_1, _, _, info_1 = env.step(actions)
    assert not info_1["truncated"].any(), "Expected no truncation on step 1 of 5"
    expected_t = env._t / env._max_steps  # should be 1/5 = 0.2
    assert torch.allclose(
        state_1[:, -1],
        torch.full((2,), expected_t),
        atol=1e-6,
    ), f"Expected timestep {expected_t}, got {state_1[:, -1]}"

    # (c) Step until truncation: take 4 more steps (total max_steps=5)
    for _ in range(3):
        _, _, _, _, _ = env.step(actions)
    _, state_trunc, _, _, info_trunc = env.step(actions)

    assert info_trunc["truncated"].all(), "Expected all envs to truncate at max_steps"
    # terminal_state last column must be 1.0 (the landmine check)
    assert info_trunc["terminal_state"][:, -1].eq(1.0).all(), (
        f"terminal_state timestep must be 1.0 at truncation, "
        f"got {info_trunc['terminal_state'][:, -1]}"
    )
    # Returned state after reset should have timestep == 0.0 (new episode start)
    assert (state_trunc[:, -1] == 0.0).all(), (
        f"Post-truncation state timestep must be 0.0, got {state_trunc[:, -1]}"
    )
