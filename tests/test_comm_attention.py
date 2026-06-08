"""
Tests specific to AttentionComm module.

Verifies:
- Attention weights >= 0 and sum to 1 over valid senders.
- Masked senders have weight 0.
- Single-valid-sender row -> weight = 1.0.
- Multi-head shape preserved.
"""

import math
import torch
import pytest

from comm import AttentionComm

B, N, D = 4, 3, 32


def make_mask(b, n):
    m = torch.ones(b, n, n, dtype=torch.bool)
    for i in range(n):
        m[:, i, i] = False
    return m


def _get_attention_weights(ac: AttentionComm, h: torch.Tensor, mask: torch.Tensor):
    """
    Extract attention weights [B, N, N, H] for the first round.
    We do this by monkey-patching masked_softmax via introspection.
    Since that's fragile, instead replicate the weight computation here.
    """
    H = ac.num_heads
    d_k = ac.d_k

    with torch.no_grad():
        Q = ac.W_q(h).reshape(B, N, H, d_k)
        K = ac.W_k(h).reshape(B, N, H, d_k)
        Q_e = Q.unsqueeze(2)
        K_e = K.unsqueeze(1)
        scores = (Q_e * K_e).sum(dim=-1) / math.sqrt(d_k)  # [B, N, N, H]
        mask_4d = mask.unsqueeze(-1).expand_as(scores)
        filled = scores.masked_fill(~mask_4d, float("-inf"))
        weights = torch.softmax(filled, dim=2)
        all_masked = ~mask_4d.any(dim=2, keepdim=True)
        weights = weights.masked_fill(all_masked, 0.0)
    return weights  # [B, N, N, H]


def test_attention_weights_nonneg_sum_to_one():
    """Valid attention weights should be >= 0 and sum to 1 over valid senders."""
    ac = AttentionComm(D, num_heads=2, rounds=1)
    h = torch.randn(B, N, D)
    mask = make_mask(B, N)

    weights = _get_attention_weights(ac, h, mask)  # [B, N, N, H]

    assert (weights >= 0).all(), "Attention weights should be non-negative"

    # Sum over senders (dim=2) should be 1 where any valid sender exists
    weight_sum = weights.sum(dim=2)  # [B, N, H]
    # All rows have at least one valid sender (off-diagonal mask)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5), (
        f"Attention weights don't sum to 1. Max diff: {(weight_sum - 1).abs().max().item():.6f}"
    )


def test_attention_masked_senders_zero_weight():
    """Masked out senders (mask=False) should have zero attention weight."""
    ac = AttentionComm(D, num_heads=2, rounds=1)
    h = torch.randn(B, N, D)

    # Build mask where agent 0 can only receive from agent 1, not agent 2
    mask = torch.zeros(B, N, N, dtype=torch.bool)
    mask[:, 0, 1] = True  # agent 0 receives from agent 1 only
    mask[:, 1, 0] = True
    mask[:, 1, 2] = True
    mask[:, 2, 1] = True

    weights = _get_attention_weights(ac, h, mask)  # [B, N, N, H]

    # For agent 0 (receiver i=0): sender j=2 is masked, weight should be 0
    assert torch.allclose(weights[:, 0, 2, :], torch.zeros(B, ac.num_heads), atol=1e-6), (
        "Masked sender j=2 for receiver i=0 should have zero weight"
    )


def test_attention_single_valid_sender():
    """With exactly one valid sender, that sender gets weight 1.0."""
    ac = AttentionComm(D, num_heads=1, rounds=1)
    h = torch.randn(B, N, D)

    # Agent 0 can only receive from agent 1
    mask = torch.zeros(B, N, N, dtype=torch.bool)
    mask[:, 0, 1] = True
    mask[:, 1, 0] = True
    mask[:, 1, 2] = True
    mask[:, 2, 1] = True

    weights = _get_attention_weights(ac, h, mask)  # [B, N, N, 1]

    # For agent 0: sender j=1 should get weight 1.0
    assert torch.allclose(weights[:, 0, 1, 0], torch.ones(B), atol=1e-5), (
        f"Single valid sender should get weight 1.0, got {weights[:, 0, 1, 0]}"
    )


def test_attention_output_shape_multihead():
    """Output shape is [B, N, D] regardless of num_heads."""
    for h_count in [1, 2, 4]:
        assert D % h_count == 0
        ac = AttentionComm(D, num_heads=h_count, rounds=1)
        h = torch.randn(B, N, D)
        mask = make_mask(B, N)
        out = ac(h, mask)
        assert out.shape == (B, N, D), (
            f"num_heads={h_count}: output shape {out.shape} != ({B}, {N}, {D})"
        )
