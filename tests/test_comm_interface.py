"""
Interface tests for all three comm modules.

Verifies per module:
- Accepts (h, mask, class_id) and returns same shape for [B,N,D] and [T,B,N,D].
- message_bits() returns the locked formula value.
- class_id doesn't change output shape.
- Gradients flow (output.sum().backward() populates param grads).
- No NaN in output.
"""

import math
import torch
import pytest

from comm import BroadcastComm, AttentionComm, GraphComm

B, N, D = 4, 3, 32
T = 5

MODULES = [
    ("broadcast", BroadcastComm(D, rounds=1)),
    ("attention", AttentionComm(D, num_heads=1, rounds=1)),
    ("graph", GraphComm(D, num_heads=4, num_layers=2)),
]


def make_mask3():
    m = torch.ones(B, N, N, dtype=torch.bool)
    for i in range(N):
        m[:, i, i] = False
    return m


def make_mask4():
    m = torch.ones(T, B, N, N, dtype=torch.bool)
    for i in range(N):
        m[:, :, i, i] = False
    return m


@pytest.mark.parametrize("name,module", MODULES)
def test_shape_3d(name, module):
    h = torch.randn(B, N, D)
    mask = make_mask3()
    out = module(h, mask)
    assert out.shape == (B, N, D), f"{name} [B,N,D] shape: {out.shape}"


@pytest.mark.parametrize("name,module", MODULES)
def test_shape_4d(name, module):
    h = torch.randn(T, B, N, D)
    mask = make_mask4()
    out = module(h, mask)
    assert out.shape == (T, B, N, D), f"{name} [T,B,N,D] shape: {out.shape}"


@pytest.mark.parametrize("name,module", MODULES)
def test_none_mask_3d(name, module):
    h = torch.randn(B, N, D)
    out = module(h, mask=None)
    assert out.shape == (B, N, D), f"{name} None mask shape: {out.shape}"


@pytest.mark.parametrize("name,module", MODULES)
def test_none_mask_4d(name, module):
    h = torch.randn(T, B, N, D)
    out = module(h, mask=None)
    assert out.shape == (T, B, N, D), f"{name} None mask [T,B,N,D] shape: {out.shape}"


@pytest.mark.parametrize("name,module", MODULES)
def test_class_id_doesnt_change_shape(name, module):
    h = torch.randn(B, N, D)
    mask = make_mask3()
    class_id = torch.zeros(N, dtype=torch.long)
    out = module(h, mask, class_id=class_id)
    assert out.shape == (B, N, D), f"{name} class_id shape: {out.shape}"


@pytest.mark.parametrize("name,module", MODULES)
def test_message_bits_formula(name, module):
    bits = module.message_bits()
    assert isinstance(bits, int), f"{name} message_bits must be int"
    assert bits > 0, f"{name} message_bits must be > 0"
    if name == "broadcast":
        # rounds * D * 32
        assert bits == 1 * D * 32, f"{name} message_bits={bits}, expected {1*D*32}"
    elif name == "attention":
        # rounds * (d_k + d_v) * 32  where d_k = d_v = D // H = D // 1 = D
        assert bits == 1 * (D + D) * 32, f"{name} message_bits={bits}, expected {1*(D+D)*32}"
    elif name == "graph":
        # num_layers * 2 * D * 32
        assert bits == 2 * 2 * D * 32, f"{name} message_bits={bits}, expected {2*2*D*32}"


@pytest.mark.parametrize("name,module", MODULES)
def test_gradients_flow(name, module):
    h = torch.randn(B, N, D, requires_grad=False)
    mask = make_mask3()
    out = module(h, mask)
    out.sum().backward()
    # At least one parameter should have a gradient
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert len(grads) > 0, f"{name}: no param gradients after backward"


@pytest.mark.parametrize("name,module", MODULES)
def test_no_nan(name, module):
    h = torch.randn(B, N, D)
    mask = make_mask3()
    out = module(h, mask)
    assert torch.isfinite(out).all(), f"{name}: NaN or Inf in output"
