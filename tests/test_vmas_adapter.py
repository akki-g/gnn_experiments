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
