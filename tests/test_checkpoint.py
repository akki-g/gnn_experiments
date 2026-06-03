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
