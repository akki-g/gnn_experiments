"""
Tests that rounds/layers affect output and message_bits scales correctly.
"""

import torch
import pytest

from comm import BroadcastComm, AttentionComm, GraphComm

B, N, D = 4, 3, 32


def make_mask(b, n):
    m = torch.ones(b, n, n, dtype=torch.bool)
    for i in range(n):
        m[:, i, i] = False
    return m


def test_broadcast_rounds_differ():
    """BroadcastComm with rounds=2 produces different output than rounds=1."""
    mask = make_mask(B, N)
    h = torch.randn(B, N, D)
    torch.manual_seed(0)
    bc1 = BroadcastComm(D, rounds=1)
    torch.manual_seed(0)
    bc2 = BroadcastComm(D, rounds=2)  # different architecture -> different output
    out1 = bc1(h, mask)
    out2 = bc2(h, mask)
    # rounds=2 applies the update twice, result should differ
    assert not torch.allclose(out1, out2), (
        "rounds=1 and rounds=2 should produce different outputs"
    )


def test_attention_rounds_differ():
    """AttentionComm with rounds=2 produces different output than rounds=1."""
    mask = make_mask(B, N)
    h = torch.randn(B, N, D)
    torch.manual_seed(0)
    ac1 = AttentionComm(D, num_heads=1, rounds=1)
    torch.manual_seed(0)
    ac2 = AttentionComm(D, num_heads=1, rounds=2)
    out1 = ac1(h, mask)
    out2 = ac2(h, mask)
    assert not torch.allclose(out1, out2), (
        "rounds=1 and rounds=2 should produce different outputs"
    )


def test_graph_layers_differ():
    """GraphComm with num_layers=2 produces different output than num_layers=1."""
    mask = make_mask(B, N)
    h = torch.randn(B, N, D)
    torch.manual_seed(0)
    gc1 = GraphComm(D, num_heads=4, num_layers=1)
    torch.manual_seed(0)
    gc2 = GraphComm(D, num_heads=4, num_layers=2)
    out1 = gc1(h, mask)
    out2 = gc2(h, mask)
    assert not torch.allclose(out1, out2), (
        "num_layers=1 and num_layers=2 should produce different outputs"
    )


def test_broadcast_message_bits_scales_with_rounds():
    """BroadcastComm message_bits = rounds * D * 32."""
    bc1 = BroadcastComm(D, rounds=1)
    bc2 = BroadcastComm(D, rounds=2)
    assert bc2.message_bits() == 2 * bc1.message_bits(), (
        f"rounds=2 bits={bc2.message_bits()} != 2*rounds=1 bits={2*bc1.message_bits()}"
    )


def test_attention_message_bits_scales_with_rounds():
    """AttentionComm message_bits = rounds * (d_k + d_v) * 32."""
    ac1 = AttentionComm(D, num_heads=1, rounds=1)
    ac2 = AttentionComm(D, num_heads=1, rounds=2)
    assert ac2.message_bits() == 2 * ac1.message_bits(), (
        f"rounds=2 bits={ac2.message_bits()} != 2*rounds=1 bits={2*ac1.message_bits()}"
    )


def test_graph_message_bits_scales_with_layers():
    """GraphComm message_bits = num_layers * 2 * D * 32."""
    gc1 = GraphComm(D, num_heads=4, num_layers=1)
    gc2 = GraphComm(D, num_heads=4, num_layers=2)
    assert gc2.message_bits() == 2 * gc1.message_bits(), (
        f"layers=2 bits={gc2.message_bits()} != 2*layers=1 bits={2*gc1.message_bits()}"
    )
