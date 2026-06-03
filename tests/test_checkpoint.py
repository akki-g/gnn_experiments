"""
Checkpoint save/load round-trip tests per planner §16.

Uses pytest's tmp_path fixture for the save path.
"""

import torch
import pytest

from backbone import MAPPO
from backbone.utils import make_optimizer
from comm import IdentityComm

# Dims
obs_dim, state_dim, action_dim, num_agents, hidden_dim = 6, 12, 3, 2, 32


def make_model():
    return MAPPO(
        obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim,
        num_agents=num_agents, hidden_dim=hidden_dim,
        comm_module=IdentityComm(), device="cpu",
    )


def test_save_and_load_roundtrip(tmp_path):
    """Save checkpoint and reload — no crash, ckpt dict has expected keys."""
    model = make_model()
    optimizer = make_optimizer(model.parameters(), lr=3e-4)
    ckpt_path = str(tmp_path / "test_ckpt.pt")

    model.save(ckpt_path, optimizer=optimizer, global_step=42, extra={"test": True})

    # Load via model.load — must return ckpt dict
    model2 = make_model()
    ckpt = model2.load(ckpt_path)

    assert isinstance(ckpt, dict), "load() must return a dict"
    assert ckpt["global_step"] == 42, f"global_step mismatch: {ckpt['global_step']}"
    assert ckpt["optimizer"] is not None, "optimizer state not saved"
    assert ckpt["extra"] == {"test": True}, "extra not saved correctly"
    assert "config" in ckpt, "config not in checkpoint"


def test_loaded_params_match(tmp_path):
    """Save model, load into fresh model, confirm all parameter tensors match."""
    model = make_model()

    # Scramble weights so they are non-default
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.123)

    ckpt_path = str(tmp_path / "params_match.pt")
    model.save(ckpt_path, global_step=0)

    model2 = make_model()
    model2.load(ckpt_path)

    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters()
    ):
        assert n1 == n2, f"parameter name mismatch: {n1} vs {n2}"
        assert torch.allclose(p1, p2, atol=1e-7), (
            f"Parameter {n1} mismatch after load:\n"
            f"  original: {p1.flatten()[:4]}\n"
            f"  loaded:   {p2.flatten()[:4]}"
        )


def test_value_normalizer_roundtrip(tmp_path):
    """ValueNorm buffers survive a save/load round-trip without loss of stats."""
    model = make_model()

    # Warm up value_normalizer so its buffers are non-trivial
    for _ in range(5):
        x = torch.randn(1000, 1) * 2.0 + 3.0
        model.value_normalizer.update(x)

    ckpt_path = str(tmp_path / "vn_roundtrip.pt")
    model.save(ckpt_path, global_step=100)

    model2 = make_model()
    model2.load(ckpt_path)

    assert torch.allclose(
        model.value_normalizer.running_mean,
        model2.value_normalizer.running_mean,
        atol=1e-6,
    ), "running_mean mismatch after load"

    assert torch.allclose(
        model.value_normalizer.running_mean_sq,
        model2.value_normalizer.running_mean_sq,
        atol=1e-6,
    ), "running_mean_sq mismatch after load"

    assert torch.allclose(
        model.value_normalizer.debiasing_term,
        model2.value_normalizer.debiasing_term,
        atol=1e-6,
    ), "debiasing_term mismatch after load"


def test_value_normalizer_old_checkpoint_tolerance(tmp_path):
    """Loading a checkpoint that lacks 'value_normalizer' must not raise."""
    model = make_model()
    ckpt_path = str(tmp_path / "old_ckpt.pt")
    model.save(ckpt_path, global_step=0)

    # Simulate an old checkpoint without value_normalizer key
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    del ckpt["value_normalizer"]
    stripped_path = str(tmp_path / "old_ckpt_stripped.pt")
    torch.save(ckpt, stripped_path)

    # Loading must succeed without raising
    model2 = make_model()
    model2.load(stripped_path)  # Should print a warning but not raise
