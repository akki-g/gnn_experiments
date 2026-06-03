"""
Unit tests for backbone.value_norm.ValueNorm.

CPU-only. Tests cover:
  - normalize/denormalize roundtrip after warmup
  - update shifts stats toward batch mean
  - normalized batch is approximately zero-mean unit-variance
  - no trainable parameters; buffers appear in state_dict
"""

import torch
import pytest

from backbone.value_norm import ValueNorm


def _warmup(vn: ValueNorm, n_batches: int = 200, batch_size: int = 4096,
            mean: float = 5.0, std: float = 3.0) -> None:
    """Run many update calls to push running stats toward (mean, std)."""
    torch.manual_seed(42)
    for _ in range(n_batches):
        x = torch.randn(batch_size, 1) * std + mean
        vn.update(x)


def test_normalize_denormalize_identity_after_warmup():
    """denormalize(normalize(y)) ≈ y after sufficient warmup."""
    vn = ValueNorm(input_dim=1)
    _warmup(vn)

    torch.manual_seed(0)
    y = torch.randn(256, 1) * 3.0 + 5.0
    reconstructed = vn.denormalize(vn.normalize(y))
    assert torch.allclose(reconstructed, y, atol=1e-3), (
        f"max abs error: {(reconstructed - y).abs().max().item():.6f}"
    )


def test_update_shifts_stats():
    """After one update on a non-zero-mean batch, stats must have moved."""
    vn = ValueNorm(input_dim=1)
    # Initial state: all zeros
    assert vn.debiasing_term.item() == 0.0

    x = torch.ones(128, 1) * 10.0
    vn.update(x)

    # debiasing_term must now be > 0
    assert vn.debiasing_term.item() > 0.0, "debiasing_term did not increase after update"

    # running_mean must have moved toward 10.0
    debias = vn.debiasing_term.item()
    debiased_mean = (vn.running_mean / debias).item()
    assert debiased_mean > 0.0, (
        f"debiased mean {debiased_mean} did not move toward batch mean 10.0"
    )


def test_normalized_batch_zero_mean_unit_var():
    """After warmup, normalize(x) should be approximately N(0,1)."""
    vn = ValueNorm(input_dim=1)
    _warmup(vn, n_batches=500)

    torch.manual_seed(7)
    x = torch.randn(8192, 1) * 3.0 + 5.0
    z = vn.normalize(x)

    mean_abs = z.mean().abs().item()
    std_err = (z.std() - 1.0).abs().item()
    assert mean_abs < 0.1, f"normalized mean {mean_abs:.4f} too large (should be < 0.1)"
    assert std_err < 0.2, f"normalized std deviation from 1: {std_err:.4f} (should be < 0.2)"


def test_no_trainable_params():
    """ValueNorm must have zero trainable parameters but three buffer entries."""
    vn = ValueNorm(input_dim=1)

    trainable = list(vn.parameters())
    assert len(trainable) == 0, (
        f"ValueNorm should have no trainable params, found {len(trainable)}"
    )

    sd = vn.state_dict()
    assert "running_mean" in sd, "running_mean missing from state_dict"
    assert "running_mean_sq" in sd, "running_mean_sq missing from state_dict"
    assert "debiasing_term" in sd, "debiasing_term missing from state_dict"
