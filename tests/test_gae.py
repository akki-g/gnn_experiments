"""
Manual GAE verification tests on tiny sequences per planner §14.

Two sub-tests:
(a) all-zero dones: manually compute expected advantages and compare.
(b) mid-rollout terminal: verify that the done mask zeroes out bootstrap.
"""

import torch
import pytest

from backbone.rollout_buffer import RolloutBuffer


def make_buffer(T, B, N):
    return RolloutBuffer(
        rollout_length=T,
        num_envs=B,
        num_agents=N,
        obs_dim=4,
        state_dim=8,
        action_dim=3,
        device="cpu",
    )


def test_gae_manual_tiny_sequence():
    """
    Sub-test (a): all-zero dones.

    T=3, B=1, N=1, gamma=1.0, gae_lambda=1.0
    rewards  = [1, 2, 3]
    values   = [0, 0, 0]
    last_val = 0

    With gamma=1, lambda=1, no dones:
        delta[2] = 3 + 1*0*1 - 0 = 3
        delta[1] = 2 + 1*0*1 - 0 = 2
        delta[0] = 1 + 1*0*1 - 0 = 1

        gae[2] = delta[2] = 3
        gae[1] = delta[1] + 1*1*1*gae[2] = 2 + 3 = 5
        gae[0] = delta[0] + 1*1*1*gae[1] = 1 + 5 = 6

    returns = advantages + values = [6, 5, 3]
    """
    T, B, N = 3, 1, 1
    buf = make_buffer(T, B, N)

    rewards = [1.0, 2.0, 3.0]
    for t in range(T):
        buf.insert(
            obs=torch.zeros(B, N, 4),
            state=torch.zeros(B, 8),
            action=torch.zeros(B, N, dtype=torch.long),
            log_prob=torch.zeros(B, N),
            value=torch.zeros(B, N),
            reward=torch.full((B, N), rewards[t]),
            done=torch.zeros(B, N),
        )

    last_value = torch.zeros(B, N)
    buf.compute_returns_and_advantages(last_value, gamma=1.0, gae_lambda=1.0)

    expected_adv = torch.tensor([[[6.0]], [[5.0]], [[3.0]]])  # [T, B, N]
    expected_ret = torch.tensor([[[6.0]], [[5.0]], [[3.0]]])  # values = 0

    assert torch.allclose(buf.advantages, expected_adv, atol=1e-5), (
        f"advantages mismatch: {buf.advantages} vs {expected_adv}"
    )
    assert torch.allclose(buf.returns, expected_ret, atol=1e-5), (
        f"returns mismatch: {buf.returns} vs {expected_ret}"
    )


def test_gae_mid_rollout_terminal():
    """
    Sub-test (b): mid-rollout terminal.

    T=3, B=1, N=1, gamma=1.0, gae_lambda=1.0
    rewards  = [1, 2, 3]
    values   = [0, 0, 0]
    dones    = [0, 1, 0]   <- terminal at t=1
    last_val = 10 (should NOT be bootstrapped through the t=1 done)

    delta[2] = 3 + 1*10*1 - 0 = 13        (last_val bootstrapped for t=2 since done[2]=0)
    delta[1] = 2 + 1*0*(1-1) - 0  = 2     (next_nonterminal=0 at t=1: done[1]=1)
    delta[0] = 1 + 1*0*(1-0) - 0  = 1     (next_values = values[1] = 0)

    gae[2] = 13
    gae[1] = delta[1] + 1*1*(1-done[1])*gae[2] = 2 + 0 = 2
    gae[0] = delta[0] + 1*1*(1-done[0])*gae[1] = 1 + 2 = 3
    """
    T, B, N = 3, 1, 1
    buf = make_buffer(T, B, N)

    rewards = [1.0, 2.0, 3.0]
    dones = [0.0, 1.0, 0.0]
    for t in range(T):
        buf.insert(
            obs=torch.zeros(B, N, 4),
            state=torch.zeros(B, 8),
            action=torch.zeros(B, N, dtype=torch.long),
            log_prob=torch.zeros(B, N),
            value=torch.zeros(B, N),
            reward=torch.full((B, N), rewards[t]),
            done=torch.full((B, N), dones[t]),
        )

    last_value = torch.full((B, N), 10.0)
    buf.compute_returns_and_advantages(last_value, gamma=1.0, gae_lambda=1.0)

    expected_adv = torch.tensor([[[3.0]], [[2.0]], [[13.0]]])  # [T, B, N]

    assert torch.allclose(buf.advantages, expected_adv, atol=1e-5), (
        f"advantages mismatch:\n{buf.advantages}\nvs expected:\n{expected_adv}"
    )
