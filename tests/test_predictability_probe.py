"""
Tests for probes/predictability.py.

Covers:
  - Synthetic recoverable obs → high R² (>0.8)
  - Synthetic unrecoverable obs (target slice zeroed) → near-chance R² (<0.1)
  - probe_gap(high, low) > 0
  - At alpha=1 surrogate (both recoverable) gap ≈ 0 (<0.1)
  - Score bounded in [0, 1]

Uses small M and few epochs for speed.
"""

import torch
import pytest

from probes.predictability import train_probe, probe_score, probe_gap, TargetPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recoverable_obs(M: int, obs_dim: int = 5, noise: float = 0.05, seed: int = 0):
    """
    Build observations where the target xy is directly recoverable:
    obs[:, 2:4] = target_xy + small noise.
    Returns (obs [M, obs_dim], target_xy [M, 2]).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    target_xy = torch.rand(M, 2, generator=gen) * 2 - 1
    obs = torch.zeros(M, obs_dim)
    obs[:, 0:2] = torch.rand(M, 2, generator=gen)  # own_pos (irrelevant)
    obs[:, 2:4] = target_xy + noise * torch.randn(M, 2, generator=gen)
    obs[:, 4]   = 1.0  # visible flag
    return obs, target_xy


def _unrecoverable_obs(M: int, obs_dim: int = 5, seed: int = 0):
    """
    Build observations where the target slice is zeroed (alpha=0 surrogate).
    obs[:, 2:4] = 0, obs[:, 4] = 0.  obs is independent of target_xy.
    Returns (obs [M, obs_dim], target_xy [M, 2]).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    target_xy = torch.rand(M, 2, generator=gen) * 2 - 1
    obs = torch.zeros(M, obs_dim)
    obs[:, 0:2] = torch.rand(M, 2, generator=gen)   # own_pos only
    # target slice stays 0 (zeroed)
    return obs, target_xy


# ---------------------------------------------------------------------------
# test: recoverable obs → high R²
# ---------------------------------------------------------------------------

def test_recoverable_obs_high_score():
    M = 2000
    obs, target_xy = _recoverable_obs(M, noise=0.02, seed=0)
    result = train_probe(obs, target_xy, seed=0, epochs=300, hidden=64,
                         val_frac=0.2, lr=5e-3)
    score = result["score"]
    assert 0.0 <= score <= 1.0, f"Score out of [0,1]: {score}"
    assert score > 0.8, (
        f"Expected high R² for recoverable obs, got score={score:.4f}. "
        "The probe should easily recover target from obs[2:4]."
    )


# ---------------------------------------------------------------------------
# test: unrecoverable obs (zeroed target slice) → near-chance R²
# ---------------------------------------------------------------------------

def test_unrecoverable_obs_chance_score():
    M = 2000
    obs, target_xy = _unrecoverable_obs(M, seed=0)
    result = train_probe(obs, target_xy, seed=0, epochs=300, hidden=64,
                         val_frac=0.2, lr=5e-3)
    score = result["score"]
    assert 0.0 <= score <= 1.0, f"Score out of [0,1]: {score}"
    assert score < 0.1, (
        f"Expected near-chance R² for unrecoverable obs, got score={score:.4f}. "
        "Zeroed target slice should prevent prediction."
    )


# ---------------------------------------------------------------------------
# test: probe_gap(high, low) > 0
# ---------------------------------------------------------------------------

def test_probe_gap_positive():
    M = 2000
    obs_rec, tgt = _recoverable_obs(M, noise=0.02, seed=1)
    obs_unrec, _  = _unrecoverable_obs(M, seed=1)

    r_rec   = train_probe(obs_rec,   tgt, seed=1, epochs=300, hidden=64, lr=5e-3)["score"]
    r_unrec = train_probe(obs_unrec, tgt, seed=1, epochs=300, hidden=64, lr=5e-3)["score"]

    gap = probe_gap(r_rec, r_unrec)
    assert gap > 0.0, (
        f"probe_gap should be positive (full_sight > sensor_limited), got gap={gap:.4f} "
        f"(r_rec={r_rec:.4f}, r_unrec={r_unrec:.4f})"
    )


# ---------------------------------------------------------------------------
# test: alpha=1 surrogate (both recoverable) gap ≈ 0
# ---------------------------------------------------------------------------

def test_alpha1_surrogate_gap_near_zero():
    """
    When both 'classes' receive the same recoverable observations, gap should be < 0.1.
    """
    M = 2000
    obs, tgt = _recoverable_obs(M, noise=0.02, seed=2)

    r_full    = train_probe(obs, tgt, seed=2, epochs=300, hidden=64, lr=5e-3)["score"]
    r_limited = train_probe(obs, tgt, seed=3, epochs=300, hidden=64, lr=5e-3)["score"]

    gap = probe_gap(r_full, r_limited)
    assert abs(gap) < 0.1, (
        f"When both classes have recoverable obs, gap should be ≈ 0, got gap={gap:.4f} "
        f"(r_full={r_full:.4f}, r_limited={r_limited:.4f})"
    )


# ---------------------------------------------------------------------------
# test: score always bounded in [0, 1]
# ---------------------------------------------------------------------------

def test_score_bounded():
    M = 500
    for seed in [0, 1, 2]:
        obs, tgt = _recoverable_obs(M, noise=0.5, seed=seed)
        result = train_probe(obs, tgt, seed=seed, epochs=100, hidden=32, lr=1e-3)
        assert 0.0 <= result["score"] <= 1.0, f"Score={result['score']} out of [0,1]"

    for seed in [0, 1]:
        obs, tgt = _unrecoverable_obs(M, seed=seed)
        result = train_probe(obs, tgt, seed=seed, epochs=100, hidden=32, lr=1e-3)
        assert 0.0 <= result["score"] <= 1.0, f"Score={result['score']} out of [0,1]"


# ---------------------------------------------------------------------------
# test: probe_score API
# ---------------------------------------------------------------------------

def test_probe_score_api():
    M = 1000
    obs, tgt = _recoverable_obs(M, noise=0.02, seed=4)
    result = train_probe(obs, tgt, seed=4, epochs=200, hidden=64, lr=5e-3)
    model = result["model"]

    # Evaluate on new data
    obs_val, tgt_val = _recoverable_obs(M // 2, noise=0.02, seed=5)
    score = probe_score(model, obs_val, tgt_val)
    assert 0.0 <= score <= 1.0
    assert score > 0.5, f"probe_score on new recoverable data should be >0.5, got {score}"


# ---------------------------------------------------------------------------
# test: TargetPredictor shape
# ---------------------------------------------------------------------------

def test_target_predictor_output_shape():
    M = 16
    obs_dim = 5
    model = TargetPredictor(obs_dim=obs_dim, hidden=32, seed=0)
    obs = torch.randn(M, obs_dim)
    pred = model(obs)
    assert pred.shape == (M, 2), f"Expected [M, 2], got {pred.shape}"


# ---------------------------------------------------------------------------
# test: probe_gap sign
# ---------------------------------------------------------------------------

def test_probe_gap_formula():
    assert probe_gap(0.9, 0.1) == pytest.approx(0.8)
    assert probe_gap(0.5, 0.5) == pytest.approx(0.0)
    assert probe_gap(0.3, 0.7) == pytest.approx(-0.4)
