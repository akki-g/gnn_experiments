"""
Tests for the IdentityComm no-op communication module.

These are the ONLY real tests in Step 1 — the IdentityComm is the only
real implementation in this scaffolding step.
"""

import torch

from comm import IdentityComm


def test_identity_returns_same_tensor():
    """IdentityComm must return the exact same tensor object (no copy)."""
    comm = IdentityComm()
    x = torch.randn(2, 3, 8)
    out = comm(x)
    assert out is x, "IdentityComm should return the input tensor object unchanged."


def test_identity_message_bits_zero():
    """IdentityComm sends no messages; message_bits must be 0."""
    comm = IdentityComm()
    assert comm.message_bits() == 0
