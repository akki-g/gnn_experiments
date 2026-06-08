"""
Permutation-equivariance tests.

Permuting agent axis + mask rows&cols + class_id consistently should
produce consistently permuted outputs (allclose) for all 3 modules.
"""

import torch
import pytest

from comm import BroadcastComm, AttentionComm, GraphComm

B, N, D = 4, 3, 32

MODULES = [
    ("broadcast", BroadcastComm(D, rounds=1)),
    ("attention", AttentionComm(D, num_heads=2, rounds=1)),
    ("graph",     GraphComm(D, num_heads=4, num_layers=2)),
]


def make_mask(b, n):
    m = torch.ones(b, n, n, dtype=torch.bool)
    for i in range(n):
        m[:, i, i] = False
    return m


@pytest.mark.parametrize("name,module", MODULES)
def test_permutation_equivariance(name, module):
    """
    π(module(h, mask)) == module(π(h), π(mask))
    where π permutes the agent axis.
    """
    module.eval()
    torch.manual_seed(7)
    h = torch.randn(B, N, D)
    mask = make_mask(B, N)
    class_id = torch.tensor([0, 1, 0], dtype=torch.long)

    # Apply module to original input
    out_orig = module(h, mask, class_id=class_id)

    # Build a fixed permutation of agents
    perm = torch.tensor([2, 0, 1], dtype=torch.long)  # N=3 permutation
    inv_perm = torch.argsort(perm)

    # Permute h: [B, N, D] -> [B, N_perm, D]
    h_perm = h[:, perm, :]

    # Permute mask: [B, N, N] -> [B, N_perm, N_perm] — both rows and cols
    mask_perm = mask[:, perm, :][:, :, perm]

    # Permute class_id: [N] -> [N_perm]
    class_id_perm = class_id[perm]

    # Apply module to permuted input
    out_perm = module(h_perm, mask_perm, class_id=class_id_perm)

    # Unpermute output: [B, N_perm, D] -> [B, N, D]
    out_perm_inv = out_perm[:, inv_perm, :]

    assert torch.allclose(out_orig, out_perm_inv, atol=1e-5), (
        f"{name}: permutation equivariance failed. "
        f"Max diff: {(out_orig - out_perm_inv).abs().max().item():.6f}"
    )
