"""
Rollout buffer tests per planner §13.

Verifies:
- insert stores tensors with the locked [T,B,N,...] shapes.
- compute_returns_and_advantages runs without NaNs.
- get_minibatches yields minibatches with shape [T*B/M, N, ...] (researcher decision 1).
- No stored tensor has requires_grad=True (catches the PPO ratio bug at source).
"""

import torch
import pytest

from backbone.rollout_buffer import RolloutBuffer

# Small dims for tests
T, B, N, obs_dim, state_dim, action_dim = 8, 4, 2, 6, 12, 3


def make_buffer():
    return RolloutBuffer(
        rollout_length=T,
        num_envs=B,
        num_agents=N,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        device="cpu",
    )


def fill_buffer(buf):
    """Fill all T steps with random data (no gradient)."""
    for _ in range(T):
        obs = torch.randn(B, N, obs_dim)
        state = torch.randn(B, state_dim)
        action = torch.randint(0, action_dim, (B, N))
        log_prob = torch.randn(B, N)          # no grad
        value = torch.randn(B, N)             # no grad
        reward = torch.randn(B, N)
        done = torch.zeros(B, N)
        buf.insert(obs=obs, state=state, action=action,
                   log_prob=log_prob, value=value, reward=reward, done=done)


def test_insert_and_shapes():
    """insert() stores tensors with correct shapes [T,B,N,...] / [T,B,state_dim]."""
    buf = make_buffer()
    fill_buffer(buf)
    assert buf.obs.shape == (T, B, N, obs_dim)
    assert buf.state.shape == (T, B, state_dim)
    assert buf.actions.shape == (T, B, N)
    assert buf.log_probs.shape == (T, B, N)
    assert buf.values.shape == (T, B, N)
    assert buf.rewards.shape == (T, B, N)
    assert buf.dones.shape == (T, B, N)
    assert buf.bad_masks.shape == (T, B, N)
    assert buf.active_masks.shape == (T, B, N)
    assert buf.step == T
    assert buf.is_full


def test_compute_returns_advantages_runs():
    """compute_returns_and_advantages runs without NaN/Inf."""
    buf = make_buffer()
    fill_buffer(buf)
    last_value = torch.randn(B, N)
    buf.compute_returns_and_advantages(last_value, gamma=0.99, gae_lambda=0.95)
    assert not torch.isnan(buf.advantages).any(), "NaN in advantages"
    assert not torch.isnan(buf.returns).any(), "NaN in returns"
    assert not torch.isinf(buf.advantages).any(), "Inf in advantages"
    assert buf.advantages.shape == (T, B, N)
    assert buf.returns.shape == (T, B, N)


def test_minibatch_iteration():
    """get_minibatches yields correct shape [T*B/M, N, ...] minibatches."""
    buf = make_buffer()
    fill_buffer(buf)
    last_value = torch.randn(B, N)
    buf.compute_returns_and_advantages(last_value, gamma=0.99, gae_lambda=0.95)

    num_mb = 4
    mb_size = (T * B) // num_mb
    count = 0
    for mb in buf.get_minibatches(num_mb):
        assert mb["obs"].shape == (mb_size, N, obs_dim), f"obs mb shape: {mb['obs'].shape}"
        assert mb["state"].shape == (mb_size, state_dim)
        assert mb["actions"].shape == (mb_size, N)
        assert mb["log_probs"].shape == (mb_size, N)
        assert mb["values"].shape == (mb_size, N)
        assert mb["advantages"].shape == (mb_size, N)
        assert mb["returns"].shape == (mb_size, N)
        assert mb["bad_masks"].shape == (mb_size, N)
        assert mb["active_masks"].shape == (mb_size, N)
        count += 1
    assert count == num_mb, f"expected {num_mb} minibatches, got {count}"


def test_no_grad_on_stored_tensors():
    """insert() must reject log_prob with requires_grad=True."""
    buf = make_buffer()
    obs = torch.randn(B, N, obs_dim)
    state = torch.randn(B, state_dim)
    action = torch.randint(0, action_dim, (B, N))
    # log_prob with grad — should raise AssertionError
    bad_log_prob = torch.randn(B, N, requires_grad=True)
    value = torch.randn(B, N)
    reward = torch.randn(B, N)
    done = torch.zeros(B, N)
    with pytest.raises(AssertionError):
        buf.insert(obs=obs, state=state, action=action,
                   log_prob=bad_log_prob, value=value, reward=reward, done=done)


def test_buffer_full_guard():
    """insert() must raise AssertionError when the buffer is already full."""
    buf = make_buffer()
    fill_buffer(buf)
    assert buf.is_full
    with pytest.raises(AssertionError, match="Buffer is full"):
        buf.insert(
            obs=torch.zeros(B, N, obs_dim),
            state=torch.zeros(B, state_dim),
            action=torch.zeros(B, N, dtype=torch.long),
            log_prob=torch.zeros(B, N),
            value=torch.zeros(B, N),
            reward=torch.zeros(B, N),
            done=torch.zeros(B, N),
        )
