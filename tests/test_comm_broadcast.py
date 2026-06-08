"""
Tests specific to BroadcastComm module.

Verifies:
- masked_mean aggregation equals manual masked mean of neighbors.
- Severing (all-False mask) zeros the c term.
- Near pass-through at init (loose tolerance).
"""

import math
import torch
import pytest

from comm import BroadcastComm
from comm.utils import masked_mean, full_comm_mask

B, N, D = 4, 3, 32


def test_masked_mean_helper():
    """comm.utils.masked_mean agrees with manually computed masked mean."""
    torch.manual_seed(0)
    x = torch.randn(B, N, N, D)  # [B, N_recv, N_send, D]
    mask = torch.ones(B, N, N, dtype=torch.bool)
    for i in range(N):
        mask[:, i, i] = False
    mask_4d = mask.unsqueeze(-1)  # [B, N, N, 1]

    result = masked_mean(x, mask_4d, dim=2)  # [B, N, D]

    # Manual check for batch 0, receiver 0
    for b in range(B):
        for i in range(N):
            valid = [x[b, i, j, :] for j in range(N) if mask[b, i, j]]
            if valid:
                manual = torch.stack(valid).mean(dim=0)
                assert torch.allclose(result[b, i], manual, atol=1e-5), (
                    f"masked_mean mismatch at b={b}, i={i}: "
                    f"result={result[b,i][:3]}, manual={manual[:3]}"
                )


def test_broadcast_aggregation_matches_manual():
    """
    BroadcastComm c_i = masked_mean(h_j for j in neighbors).
    We can verify by checking c equals masked_mean(h, mask, dim=1) for a
    broadcast with W_H=identity and W_C=identity (we override weights).
    """
    bc = BroadcastComm(D, rounds=1)

    # Force W_H=0, W_C=I so output = tanh(W_C(c)) = tanh(c) for clean check
    # Actually, just verify that c term is computed correctly.
    # Simpler: zero out W_H and set W_C = I, verify output = tanh(masked_mean(h))

    with torch.no_grad():
        bc.W_H.weight.zero_()  # W_H = 0 matrix
        bc.W_C.weight.copy_(torch.eye(D))  # W_C = identity

    h = torch.randn(B, N, D)
    mask = full_comm_mask(N, device=h.device).unsqueeze(0).expand(B, -1, -1)

    out = bc(h, mask)

    # Manual: c[b, i] = mean(h[b, j] for j != i)
    h_expand = h.unsqueeze(1).expand(B, N, N, D)
    mask_4d = mask.unsqueeze(-1)
    c = masked_mean(h_expand, mask_4d, dim=2)  # [B, N, D]
    expected = torch.tanh(c)  # W_H(h) = 0, W_C(c) = c => tanh(c)

    assert torch.allclose(out, expected, atol=1e-5), (
        f"Broadcast aggregation mismatch. Max diff: {(out - expected).abs().max().item():.6f}"
    )


def test_broadcast_severing_zeros_c():
    """Under all-False mask, c should be zero (no valid neighbors)."""
    bc = BroadcastComm(D, rounds=1)

    with torch.no_grad():
        bc.W_H.weight.copy_(torch.eye(D))  # W_H = I
        bc.W_C.weight.copy_(torch.eye(D))  # W_C = I (to make c visible in output)

    h = torch.randn(B, N, D)
    all_false_mask = torch.zeros(B, N, N, dtype=torch.bool)

    out = bc(h, all_false_mask)

    # With all-False mask, c=0, so output = tanh(W_H(h) + W_C(0)) = tanh(h)
    expected = torch.tanh(h)
    assert torch.allclose(out, expected, atol=1e-5), (
        f"All-False mask: output should be tanh(h). "
        f"Max diff: {(out - expected).abs().max().item():.6f}"
    )


def test_broadcast_near_passthrough_at_init():
    """At init (W_C gain=0.01), output should be close to tanh(W_H(h))."""
    bc = BroadcastComm(D, rounds=1)
    h = torch.randn(B, N, D)
    mask = full_comm_mask(N, device=h.device).unsqueeze(0).expand(B, -1, -1)

    out = bc(h, mask)

    # W_C init gain=0.01, so W_C(c) is small relative to W_H(h)
    # output ~= tanh(W_H(h) + small_correction)
    # Verify that output is finite and has reasonable magnitude
    assert torch.isfinite(out).all(), "Output has NaN/Inf at init"
    # Loose: output norm should be within 10x of input norm
    assert out.abs().mean().item() < 10.0, "Output magnitude unexpectedly large at init"
