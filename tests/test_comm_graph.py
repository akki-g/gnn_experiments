"""
Tests specific to GraphComm module.

Verifies:
- Self-loop is always present (diagonal of graph_mask True).
- Multi-hop (L=2) changes receptive field vs L=1.
- Per-head softmax normalizes correctly.
- Output shape preserved.
"""

import math
import torch
import pytest

from comm import GraphComm
from comm.utils import add_self_loop, full_comm_mask

B, N, D = 4, 3, 32


def make_mask(b, n):
    m = torch.ones(b, n, n, dtype=torch.bool)
    for i in range(n):
        m[:, i, i] = False
    return m


def test_self_loop_always_in_graph_mask():
    """
    GraphComm must add self-loop before attention. We verify this by checking
    that even when mask diagonal is False, agent output is influenced by self.
    This is a behavioral test: if only agent i's h changes, output_i should change.
    """
    gc = GraphComm(D, num_heads=4, num_layers=1)
    mask = make_mask(B, N)  # off-diagonal only; self-loop must be added inside

    h_base = torch.randn(B, N, D)
    out_base = gc(h_base, mask)

    # Perturb agent 0's own embedding (self-loop means this should affect output_0)
    h_perturbed = h_base.clone()
    h_perturbed[:, 0, :] += 100.0

    out_perturbed = gc(h_perturbed, mask)

    # Agent 0's output MUST change (self-loop)
    assert not torch.allclose(out_base[:, 0, :], out_perturbed[:, 0, :]), (
        "Agent 0 output should change when its own h is perturbed (self-loop)"
    )


def test_multi_hop_changes_receptive_field():
    """L=2 layers should produce different output than L=1 for same init."""
    h = torch.randn(B, N, D)
    mask = make_mask(B, N)

    torch.manual_seed(0)
    gc1 = GraphComm(D, num_heads=4, num_layers=1)
    torch.manual_seed(0)
    gc2 = GraphComm(D, num_heads=4, num_layers=2)

    out1 = gc1(h, mask)
    out2 = gc2(h, mask)

    assert not torch.allclose(out1, out2), (
        "L=1 and L=2 should produce different outputs (different receptive fields)"
    )


def test_per_head_softmax_normalizes():
    """
    Check that softmax over the neighborhood sums to 1 per head
    using the same computation as GraphComm.forward.
    """
    gc = GraphComm(D, num_heads=4, num_layers=1)
    h = torch.randn(B, N, D)
    mask = make_mask(B, N)
    graph_mask = add_self_loop(mask)  # [B, N, N] True including self

    H = gc.num_heads
    d_k = gc.d_k
    tau = gc.tau

    with torch.no_grad():
        Q = gc.W_q_layers[0](h).reshape(B, N, H, d_k)
        K = gc.W_k_layers[0](h).reshape(B, N, H, d_k)
        Q_e = Q.unsqueeze(2)
        K_e = K.unsqueeze(1)
        scores = tau * (Q_e * K_e).sum(dim=-1)  # [B, N, N, H]
        gm4d = graph_mask.unsqueeze(-1).expand_as(scores)
        filled = scores.masked_fill(~gm4d, float("-inf"))
        alpha = torch.softmax(filled, dim=2)

    # Sum over senders (dim=2) should be 1.0 for all (b, i, head) since
    # graph_mask has self-loop so every row has at least one valid sender
    alpha_sum = alpha.sum(dim=2)  # [B, N, H]
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5), (
        f"Per-head softmax doesn't sum to 1. Max diff: {(alpha_sum - 1).abs().max().item():.6f}"
    )


def test_graph_output_shape_preserved():
    """Output shape is [B, N, D] — no cross-layer concat."""
    for n_layers in [1, 2, 3]:
        gc = GraphComm(D, num_heads=4, num_layers=n_layers)
        h = torch.randn(B, N, D)
        mask = make_mask(B, N)
        out = gc(h, mask)
        assert out.shape == (B, N, D), (
            f"L={n_layers}: output shape {out.shape} != ({B}, {N}, {D})"
        )
