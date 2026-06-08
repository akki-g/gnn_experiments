"""
PPO ratio correctness tests.

Three original tests (identity comm):
- test_ratio_approx_one_epoch0   : before any optimizer step, ratio must be ~1.0.
- test_ratio_diverges_after_step : after one optimizer step ratio moves away from 1.0.
- test_old_log_probs_detached    : log-probs stored in buffer carry no gradient.

Parametrized extension (all four comm modules):
- test_ratio_approx_one_epoch0_comm_param: epoch-0 ratio ≈ 1.0 with a fixed
  radius-style comm mask threaded through model.act() AND buffer.insert()
  and replayed via model.evaluate() with mb["comm_mask"].
  This is the live proof that act/evaluate comm divergence is caught.
"""

import torch
import pytest

from backbone import MAPPO, RolloutBuffer
from backbone.utils import make_optimizer
from comm import IdentityComm, BroadcastComm, AttentionComm, GraphComm

# Dims
B, N, obs_dim, state_dim, action_dim, hidden_dim = 4, 2, 6, 12, 3, 32
T = 8


# ---------------------------------------------------------------------------
# Comm factory parametrization
# ---------------------------------------------------------------------------

COMM_FACTORIES = {
    "identity":  lambda: IdentityComm(),
    "broadcast": lambda: BroadcastComm(hidden_dim, rounds=1),
    "attention": lambda: AttentionComm(hidden_dim, num_heads=1, rounds=1),
    "graph":     lambda: GraphComm(hidden_dim, num_heads=4, num_layers=2),
}


def _make_fixed_comm_mask():
    """Build a fixed [B, N, N] bool comm mask (all off-diagonal True)."""
    m = torch.ones(B, N, N, dtype=torch.bool)
    for i in range(N):
        m[:, i, i] = False
    return m


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_model_and_buffer(comm_module=None):
    if comm_module is None:
        comm_module = IdentityComm()
    model = MAPPO(
        obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim,
        num_agents=N, hidden_dim=hidden_dim, comm_module=comm_module,
        device="cpu",
    )
    buf = RolloutBuffer(
        rollout_length=T, num_envs=B, num_agents=N, obs_dim=obs_dim,
        state_dim=state_dim, action_dim=action_dim, device="cpu",
    )
    return model, buf


def fill_buffer_with_model(model, buf, comm_mask=None):
    """Collect T steps using model, storing detached log_probs and comm_masks."""
    obs_seq = []
    state_seq = []
    for _ in range(T):
        obs = torch.randn(B, N, obs_dim)
        state = torch.randn(B, state_dim)
        obs_seq.append(obs)
        state_seq.append(state)
        out = model.act(obs, state, mask=comm_mask)
        buf.insert(
            obs=obs,
            state=state,
            action=out["action"],
            log_prob=out["log_prob"],   # already detached (no_grad context)
            value=out["value"],
            reward=torch.zeros(B, N),
            done=torch.zeros(B, N),
            comm_mask=comm_mask,
        )
    return obs_seq, state_seq


# ---------------------------------------------------------------------------
# Original identity-only tests (preserved)
# ---------------------------------------------------------------------------

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
    out = model.evaluate(mb["obs"], mb["state"], mb["actions"], mask=mb.get("comm_mask"))
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
    out = model.evaluate(mb["obs"], mb["state"], mb["actions"], mask=mb.get("comm_mask"))
    loss = -out["entropy"].mean()   # maximise entropy → forces policy to move
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # After the step, ratio from the now-stale stored log-probs should drift
    out2 = model.evaluate(mb["obs"], mb["state"], mb["actions"], mask=mb.get("comm_mask"))
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


# ---------------------------------------------------------------------------
# Parametrized tests for all comm modules with radius-style mask
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("comm_name", ["identity", "broadcast", "attention", "graph"])
def test_ratio_approx_one_epoch0_comm_param(comm_name):
    """
    Epoch-0 PPO ratio ≈ 1.0 for all comm modules when a fixed bool comm_mask
    is threaded through model.act(mask=...) and buffer.insert(comm_mask=...)
    and replayed via model.evaluate(..., mask=mb["comm_mask"]).

    This is the guard that catches act()/evaluate() comm divergence.
    """
    comm_module = COMM_FACTORIES[comm_name]()
    model, buf = make_model_and_buffer(comm_module)

    # Fixed radius-style comm mask (all off-diagonal True = no severing)
    comm_mask = _make_fixed_comm_mask()

    fill_buffer_with_model(model, buf, comm_mask=comm_mask)
    last_value = torch.zeros(B, N)
    buf.compute_returns_and_advantages(last_value, gamma=0.99, gae_lambda=0.95)

    # 1 minibatch = all data; comm_mask replayed from buffer
    mb = next(iter(buf.get_minibatches(1)))
    assert "comm_mask" in mb, "comm_mask missing from minibatch"

    out = model.evaluate(mb["obs"], mb["state"], mb["actions"], mask=mb["comm_mask"])
    ratio = torch.exp(out["log_prob"] - mb["log_probs"])

    ones = torch.ones_like(ratio)
    assert torch.allclose(ratio, ones, atol=1e-5), (
        f"[{comm_name}] Epoch-0 ratio not ≈ 1.0. "
        f"mean={ratio.mean().item():.6f}, "
        f"max_abs_diff={(ratio - 1).abs().max().item():.6f}"
    )
    assert torch.isfinite(ratio).all(), f"[{comm_name}] Non-finite values in ratio"
