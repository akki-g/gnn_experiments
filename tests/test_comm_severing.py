"""
Communication severing tests.

With an all-False mask (except self-loops in Graph), each agent's output should
depend ONLY on its own embedding.  Perturbing agent j != i should leave agent i's
output unchanged.
"""

import torch
import pytest

from comm import BroadcastComm, AttentionComm, GraphComm

B, N, D = 4, 3, 32


def all_false_mask(b, n):
    """All-False [b, n, n] mask: no agent can receive from any other."""
    return torch.zeros(b, n, n, dtype=torch.bool)


MODULES = [
    ("broadcast", BroadcastComm(D, rounds=1)),
    ("attention", AttentionComm(D, num_heads=1, rounds=1)),
    ("graph",     GraphComm(D, num_heads=4, num_layers=2)),
]


@pytest.mark.parametrize("name,module", MODULES)
def test_severing_isolates_agents(name, module):
    """
    All-False mask -> perturbing agent j should NOT change agent i's output (i != j).
    For GraphComm the self-loop means h_i feeds back to agent i, so the test
    checks the same invariant: output_i unchanged when only h_j (j!=i) changes.
    """
    module.eval()  # deterministic (no dropout; LayerNorm behaves same)
    torch.manual_seed(42)
    h = torch.randn(B, N, D)
    mask = all_false_mask(B, N)

    out_orig = module(h, mask)

    # Perturb agent j=1, check agent i=0 is unchanged
    for j in range(N):
        h_perturbed = h.clone()
        h_perturbed[:, j, :] += 10.0  # large perturbation to agent j

        out_perturbed = module(h_perturbed, mask)

        for i in range(N):
            if i == j:
                continue  # own embedding changed — output IS expected to change
            assert torch.allclose(out_orig[:, i, :], out_perturbed[:, i, :], atol=1e-5), (
                f"{name}: output of agent {i} changed when agent {j} was perturbed "
                f"under all-False mask. Max diff: "
                f"{(out_orig[:, i, :] - out_perturbed[:, i, :]).abs().max().item():.6f}"
            )


@pytest.mark.parametrize("name,module", MODULES)
def test_severing_no_nan(name, module):
    """All-False mask must not produce NaN or Inf."""
    h = torch.randn(B, N, D)
    mask = all_false_mask(B, N)
    out = module(h, mask)
    assert torch.isfinite(out).all(), f"{name}: NaN/Inf under all-False mask"
