"""
Tests for comm_mask plumbing in RolloutBuffer.

Verifies:
1. insert() with explicit comm_mask -> get_minibatches returns it correctly.
2. insert() WITHOUT comm_mask -> all-True off-diagonal / False diagonal (default).
3. clear() resets comm_masks to the default full off-diagonal pattern.
"""

import torch
import pytest

from backbone import RolloutBuffer


B, N, T = 4, 3, 5
obs_dim, state_dim, action_dim = 6, 12, 3


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


def fill_buffer(buf, comm_masks_list=None):
    """Fill buffer with T steps. comm_masks_list: list of T optional [B,N,N] masks."""
    for t in range(T):
        obs = torch.randn(B, N, obs_dim)
        state = torch.randn(B, state_dim)
        action = torch.zeros(B, N, dtype=torch.long)
        log_prob = torch.zeros(B, N)
        value = torch.zeros(B, N)
        reward = torch.zeros(B, N)
        done = torch.zeros(B, N)
        cm = comm_masks_list[t] if comm_masks_list is not None else None
        buf.insert(
            obs=obs,
            state=state,
            action=action,
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=done,
            comm_mask=cm,
        )


def test_explicit_comm_mask_roundtrip():
    """Insert explicit comm_mask -> get_minibatches returns it correctly."""
    buf = make_buffer()

    # Build T distinct masks — unique pattern per step
    masks = []
    for t in range(T):
        m = torch.ones(B, N, N, dtype=torch.bool)
        # Diagonal False
        for i in range(N):
            m[:, i, i] = False
        # Extra masking: for step t, agent 0 cannot receive from agent 1
        m[:, 0, 1] = (t % 2 == 0)  # alternating
        masks.append(m)

    fill_buffer(buf, comm_masks_list=masks)
    buf.compute_returns_and_advantages(torch.zeros(B, N), gamma=0.99, gae_lambda=0.95)

    # Use 1 minibatch so we get all T*B rows
    mb = next(iter(buf.get_minibatches(1)))
    assert "comm_mask" in mb, "comm_mask key missing from minibatch"
    assert mb["comm_mask"].shape == (T * B, N, N), (
        f"comm_mask minibatch shape {mb['comm_mask'].shape} != ({T*B}, {N}, {N})"
    )

    # Reconstruct expected from the stored buffer and compare with minibatch
    # (can't compare directly because of random shuffle, but we can check
    #  the multiset of values matches by checking total counts)
    stored_flat = buf.comm_masks.reshape(T * B, N, N)
    mb_cm = mb["comm_mask"]
    # Diagonals should all be False
    for i in range(N):
        assert not mb_cm[:, i, i].any(), f"Diagonal agent {i} should be False in comm_mask"
    # Values are subset of stored (same multiset up to perm, but since 1 minibatch = all rows)
    assert mb_cm.sum() == stored_flat.sum(), (
        "Total True entries in minibatch comm_mask doesn't match stored buffer"
    )


def test_no_comm_mask_default_full_offdiag():
    """insert() without comm_mask -> stored masks should be full off-diagonal."""
    buf = make_buffer()
    fill_buffer(buf, comm_masks_list=None)

    cm = buf.comm_masks  # [T, B, N, N]
    # Off-diagonal should be True
    for i in range(N):
        for j in range(N):
            if i != j:
                assert cm[:, :, i, j].all(), f"Off-diag [{i},{j}] should be True by default"
            else:
                assert not cm[:, :, i, j].any(), f"Diagonal [{i},{i}] should be False"


def test_get_minibatches_no_mask_default():
    """get_minibatches without comm_mask -> full off-diagonal in minibatch."""
    buf = make_buffer()
    fill_buffer(buf, comm_masks_list=None)
    buf.compute_returns_and_advantages(torch.zeros(B, N), gamma=0.99, gae_lambda=0.95)

    mb = next(iter(buf.get_minibatches(1)))
    assert "comm_mask" in mb
    cm = mb["comm_mask"]  # [T*B, N, N]
    for i in range(N):
        assert not cm[:, i, i].any(), f"Diagonal should be False"
        for j in range(N):
            if i != j:
                assert cm[:, i, j].all(), f"Off-diagonal should be True"


def test_clear_resets_comm_masks():
    """clear() resets comm_masks to full off-diagonal."""
    buf = make_buffer()

    # Fill with all-False masks (completely severed)
    masks = [torch.zeros(B, N, N, dtype=torch.bool) for _ in range(T)]
    fill_buffer(buf, comm_masks_list=masks)

    # Verify they were stored as False
    assert not buf.comm_masks.any(), "Expected all-False comm_masks after fill"

    buf.clear()

    # After clear: should be full off-diagonal
    cm = buf.comm_masks
    for i in range(N):
        for j in range(N):
            if i != j:
                assert cm[:, :, i, j].all(), f"Off-diag [{i},{j}] should be True after clear"
            else:
                assert not cm[:, :, i, j].any(), f"Diagonal [{i},{i}] should be False after clear"


def test_no_grad_in_comm_masks():
    """Stored comm_masks must not require gradients."""
    buf = make_buffer()
    fill_buffer(buf, comm_masks_list=None)
    # bool tensors can't require_grad; just confirm dtype
    assert buf.comm_masks.dtype == torch.bool
    assert not buf.comm_masks.requires_grad
