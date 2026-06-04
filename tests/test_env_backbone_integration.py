"""
End-to-end integration test: PursuitAdapter + MAPPO backbone + RolloutBuffer + ppo_update.

Confirms that pursuit satisfies every shape contract the frozen backbone asserts,
and that a full PPO update produces no NaNs and mean_ratio_epoch0 ≈ 1.0.
"""

import math
import torch
import pytest

from envs.pursuit_adapter import PursuitAdapter
from backbone.mappo import MAPPO
from backbone.rollout_buffer import RolloutBuffer
from backbone.ppo_update import ppo_update
from backbone.utils import make_optimizer
from comm.identity import IdentityComm
from experiments.train_vmas_mappo import _bad_mask_from_info, _terminal_state_from_info


def test_pursuit_backbone_integration_no_nan_and_ratio_near_one():
    """
    Full end-to-end: PursuitAdapter → MAPPO.act → buffer.insert → GAE →
    ppo_update → check no NaNs and mean_ratio_epoch0 ≈ 1.0.
    """
    B   = 4
    N   = 3
    T   = 5
    device = torch.device("cpu")

    # Build environment
    env = PursuitAdapter(
        num_envs=B,
        n_agents=N,
        max_steps=T,
        seed=0,
        device="cpu",
        include_timestep=True,
        alpha=1.0,
        num_full_sight=2,
    )

    obs_dim    = env.obs_dim
    state_dim  = env.state_dim
    action_dim = env.action_dim  # 5

    assert obs_dim   == 5,            f"obs_dim={obs_dim}"
    assert action_dim == 5,           f"action_dim={action_dim}"
    assert state_dim == N * 2 + 2 + 1, f"state_dim={state_dim}"

    # Build model
    model = MAPPO(
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=N,
        hidden_dim=16,
        comm_module=IdentityComm(),
        encoder_layers=1,
        actor_layers=1,
        critic_layers=1,
        device="cpu",
    )

    optimizer = make_optimizer(model.parameters(), lr=3e-4)

    # Build rollout buffer
    buffer = RolloutBuffer(
        rollout_length=T,
        num_envs=B,
        num_agents=N,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        device="cpu",
    )

    # Reset environment
    obs, state = env.reset()
    obs   = obs.to(device)
    state = state.to(device)

    last_info = None

    # Collect T steps
    model.eval()
    for t in range(T):
        out = model.act(obs, state)

        # Shape checks on act output
        assert out["action"].shape   == (B, N)
        assert out["log_prob"].shape == (B, N)
        assert out["value"].shape    == (B, N)

        # No NaNs in act output
        assert not torch.isnan(out["log_prob"]).any(),  "NaN in log_prob"
        assert not torch.isnan(out["value"]).any(),     "NaN in value"

        next_obs, next_state, reward, done, info = env.step(out["action"].cpu())
        reward = reward.to(device)
        done   = done.to(device)

        # bad_mask: 0 for time-limit truncation, 1 otherwise
        bad_mask = _bad_mask_from_info(info, N, device)

        buffer.insert(
            obs=obs,
            state=state,
            action=out["action"],
            log_prob=out["log_prob"],
            value=out["value"],
            reward=reward,
            done=done,
            bad_mask=bad_mask,
        )

        obs   = next_obs.to(device)
        state = next_state.to(device)
        last_info = info

    assert buffer.is_full, "Buffer should be full after T steps"

    # Bootstrap value from terminal state
    with torch.no_grad():
        bootstrap_state = _terminal_state_from_info(last_info or {}, state, device)
        last_value = model.critic(bootstrap_state)  # [B, N]

    assert last_value.shape == (B, N)
    assert not torch.isnan(last_value).any(), "NaN in last_value"

    # Compute GAE
    buffer.compute_returns_and_advantages(
        last_value, gamma=0.99, gae_lambda=0.95,
        value_normalizer=model.value_normalizer,
    )

    # No NaNs in computed returns/advantages
    assert not torch.isnan(buffer.returns).any(),    "NaN in returns"
    assert not torch.isnan(buffer.advantages).any(), "NaN in advantages"
    assert not torch.isnan(buffer.values).any(),     "NaN in values"
    assert not torch.isnan(buffer.log_probs).any(),  "NaN in log_probs"

    # PPO update
    metrics = ppo_update(
        model,
        optimizer,
        buffer,
        ppo_epochs=2,
        num_minibatches=1,
        clip_eps=0.2,
        value_clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=10.0,
        normalize_advantages=True,
        value_normalizer=model.value_normalizer,
    )

    # No NaNs in loss metrics
    for key in ["policy_loss", "value_loss", "entropy", "total_loss"]:
        assert not math.isnan(metrics[key]), f"NaN in metric '{key}': {metrics[key]}"

    # mean_ratio_epoch0 should be ≈ 1.0 (within 1e-3)
    r0 = metrics["mean_ratio_epoch0"]
    assert r0 is not None, "mean_ratio_epoch0 should not be None"
    assert abs(r0 - 1.0) < 1e-3, (
        f"mean_ratio_epoch0={r0:.6f} should be ≈ 1.0. "
        "Old log-probs may have been recomputed incorrectly."
    )


def test_pursuit_adapter_obs_state_shapes():
    """Adapter attributes and reset/step shapes are correct."""
    env = PursuitAdapter(
        num_envs=4, n_agents=3, max_steps=5, seed=0,
        include_timestep=True, alpha=1.0, num_full_sight=2,
    )
    assert env.obs_dim    == 5
    assert env.action_dim == 5
    assert env.state_dim  == 3 * 2 + 2 + 1  # N*2 + 2 + 1

    obs, state = env.reset()
    assert obs.shape   == (4, 3, 5)
    assert state.shape == (4, env.state_dim)

    actions = torch.zeros(4, 3, dtype=torch.long)
    obs2, state2, reward, done, info = env.step(actions)
    assert obs2.shape    == (4, 3, 5)
    assert state2.shape  == (4, env.state_dim)
    assert reward.shape  == (4, 3)
    assert done.shape    == (4, 3)
    assert "truncated"   in info
    assert "terminal_obs"   in info
    assert "terminal_state" in info


def test_class_id_exposed_on_adapter():
    """PursuitAdapter exposes class_id [N] for forward-compat."""
    env = PursuitAdapter(num_envs=4, n_agents=3, num_full_sight=2, max_steps=5)
    assert env.class_id.shape == (3,)
    assert (env.class_id[:2] == 0).all()
    assert (env.class_id[2:]  == 1).all()
