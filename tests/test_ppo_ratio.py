"""
PPO ratio correctness tests per planner §15.

Three tests collectively catch the common PPO ratio bug:
- test_ratio_approx_one_epoch0   : before any optimizer step, ratio must be ~1.0.
- test_ratio_diverges_after_step : after one optimizer step ratio moves away from 1.0.
- test_old_log_probs_detached    : log-probs stored in buffer carry no gradient.
"""

import torch
import pytest

from backbone import MAPPO, RolloutBuffer
from backbone.utils import make_optimizer
from comm import IdentityComm

# Dims
B, N, obs_dim, state_dim, action_dim, hidden_dim = 4, 2, 6, 12, 3, 32
T = 8


def make_model_and_buffer():
    model = MAPPO(
        obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim,
        num_agents=N, hidden_dim=hidden_dim, comm_module=IdentityComm(),
        device="cpu",
    )
    buf = RolloutBuffer(
        rollout_length=T, num_envs=B, num_agents=N, obs_dim=obs_dim,
        state_dim=state_dim, action_dim=action_dim, device="cpu",
    )
    return model, buf


def fill_buffer_with_model(model, buf):
    """Collect T steps using model, storing detached log_probs."""
    obs_seq = []
    state_seq = []
    for _ in range(T):
        obs = torch.randn(B, N, obs_dim)
        state = torch.randn(B, state_dim)
        obs_seq.append(obs)
        state_seq.append(state)
        out = model.act(obs, state)
        buf.insert(
            obs=obs,
            state=state,
            action=out["action"],
            log_prob=out["log_prob"],   # already detached (no_grad context)
            value=out["value"],
            reward=torch.zeros(B, N),
            done=torch.zeros(B, N),
        )
    return obs_seq, state_seq


def test_ratio_approx_one_epoch0():
    """
    Before any optimizer step, recomputing log-probs should yield ratio ≈ 1.0.
    """
    model, buf = make_model_and_buffer()
    fill_buffer_with_model(model, buf)
    last_value = torch.zeros(B, N)
    buf.compute_returns_and_advantages(last_value, gamma=0.99, gae_lambda=0.95)

    # Get first minibatch and compute ratio WITHOUT taking optimizer step
    mb = next(iter(buf.get_minibatches(1)))  # 1 minibatch = all data
    out = model.evaluate(mb["obs"], mb["state"], mb["actions"])
    ratio = torch.exp(out["log_prob"] - mb["log_probs"])
    ones = torch.ones_like(ratio)
    assert torch.allclose(ratio, ones, atol=1e-5), (
        f"Epoch-0 ratio not ≈ 1.0. mean={ratio.mean().item():.6f}, "
        f"max_abs_diff={(ratio - 1).abs().max().item():.6f}"
    )


def test_ratio_diverges_after_step():
    """
    After one optimizer step, the ratio should no longer be ~1.0.
    (At least some entries should deviate from 1.0.)
    """
    model, buf = make_model_and_buffer()
    fill_buffer_with_model(model, buf)
    last_value = torch.zeros(B, N)
    buf.compute_returns_and_advantages(last_value, gamma=0.99, gae_lambda=0.95)

    optimizer = make_optimizer(model.parameters(), lr=1e-2)

    # Take one update step with a large entropy coef to force policy change
    mb = next(iter(buf.get_minibatches(1)))
    out = model.evaluate(mb["obs"], mb["state"], mb["actions"])
    loss = -out["entropy"].mean()   # maximise entropy → forces policy to move
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # After the step, ratio from the now-stale stored log-probs should drift
    out2 = model.evaluate(mb["obs"], mb["state"], mb["actions"])
    ratio = torch.exp(out2["log_prob"] - mb["log_probs"])
    # We expect at least some deviation from 1.0
    max_diff = (ratio - 1).abs().max().item()
    assert max_diff > 1e-6, (
        f"Ratio should diverge after optimizer step, but max_diff={max_diff:.8f}"
    )


def test_old_log_probs_detached():
    """
    Log-probs stored in the rollout buffer must carry no gradient.
    """
    model, buf = make_model_and_buffer()
    fill_buffer_with_model(model, buf)

    # log_probs stored in buffer must not require grad
    assert not buf.log_probs.requires_grad, (
        "Stored log_probs should NOT require grad — they must be detached at collection time."
    )
