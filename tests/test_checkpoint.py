"""
Checkpoint save/load round-trip tests per planner §16.

Uses pytest's tmp_path fixture for the save path.
"""

import os
import yaml
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


# ---------------------------------------------------------------------------
# Regression test: short training run writes at least one .pt to disk.
#
# This catches the bug where checkpoint_interval > total_iters caused NO .pt
# files to be written (periodic checkpoint never fired, final.pt outside the
# loop was the only safety net).  The fix adds an "also fire on last iter"
# condition so ckpt_{last_iter}.pt always exists even for tiny smoke runs.
# ---------------------------------------------------------------------------

def test_training_run_writes_checkpoint_to_disk(tmp_path):
    """
    A short training run with checkpoint_interval much larger than total_iters
    must still produce at least one .pt file on disk.

    Uses experiments.train_mappo (toy env, no VMAS needed) with an in-memory
    config override to write into tmp_path.
    """
    import sys
    # Ensure project root is importable from within tests
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ckpt_dir = str(tmp_path / "ckpt")
    log_dir  = str(tmp_path / "logs")

    # Load the toy_sanity base config and override for a tiny 4-iteration run
    base_cfg_path = os.path.join(project_root, "configs", "toy_sanity.yaml")
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg["runtime"]["total_env_steps"] = 512   # 4 iters with T=32, B=4
    cfg["runtime"]["checkpoint_interval"] = 9999  # deliberately > total_iters
    cfg["logging"]["checkpoint_dir"] = ckpt_dir
    cfg["logging"]["log_dir"] = log_dir
    cfg["logging"]["console_log_interval"] = 9999  # suppress stdout noise

    tmp_cfg = str(tmp_path / "test_cfg.yaml")
    with open(tmp_cfg, "w") as f:
        yaml.safe_dump(cfg, f)

    # Run training in the project root directory (modules import from there)
    orig_dir = os.getcwd()
    try:
        os.chdir(project_root)
        from experiments.train_mappo import main
        main(tmp_cfg)
    finally:
        os.chdir(orig_dir)

    # At least one .pt file must exist after the run
    pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    assert len(pt_files) >= 1, (
        f"Expected at least one .pt checkpoint in {ckpt_dir}, found: {pt_files}. "
        "This is the checkpoint-interval regression: ckpt_{last_iter}.pt must "
        "always be written even when checkpoint_interval > total_iters."
    )

    # Specifically: the last-iter checkpoint should exist (ckpt_3.pt with 4 iters)
    assert any("ckpt_" in f for f in pt_files), (
        f"Expected ckpt_<it>.pt among files {pt_files}. "
        "The last-iter periodic checkpoint should always be written."
    )
