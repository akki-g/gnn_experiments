"""
Target-position predictability probe.

A linear/MLP probe trained to predict the ground-truth target (x, y) from
a single agent's local observation.  The held-out R² score measures how much
target information is recoverable from that observation.

Corner cases (documented):
    alpha=1 : both full-sight and sensor-limited classes receive full target info
              → both R² scores high, gap ≈ 0.
    alpha=0 : sensor-limited class receives zeroed target slice (obs independent of target)
              → sensor-limited R² ≈ 0 (chance), full-sight R² high → gap large.
    "Chance" = clamped R² ≈ 0: random-walk target is independent of own position,
              so a masked observation has no target signal; the probe cannot beat
              constant-mean baseline.

No environment imports here — purely a PyTorch module + sklearn-free stats.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
from torch import Tensor


class TargetPredictor(nn.Module):
    """
    Small MLP probe: obs_dim → hidden → 2 (predict target x, y).

    Architecture: Linear(obs_dim, hidden) → ReLU → Linear(hidden, 2).
    Seeded initialization via torch.manual_seed at construction time.
    """

    def __init__(self, obs_dim: int, hidden: int = 64, seed: int = 0):
        super().__init__()
        # Seed before weight initialization for reproducibility
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """
        Parameters
        ----------
        obs : [M, obs_dim]

        Returns
        -------
        pred : [M, 2]  predicted (x, y) of target
        """
        return self.net(obs)


def _r2_score(y_pred: Tensor, y_true: Tensor) -> float:
    """
    Compute R² (coefficient of determination) for [M, 2] outputs.

    R² = 1 - SS_res / SS_tot
    Clamped to [0, 1] (negative R² is reported as 0: worse than constant mean).
    """
    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean(dim=0)) ** 2).sum().item()
    if ss_tot < 1e-12:
        # Degenerate: all targets identical → R²=1 if predictions also identical
        return 1.0 if ss_res < 1e-12 else 0.0
    r2 = 1.0 - ss_res / ss_tot
    return float(max(0.0, r2))


def train_probe(
    obs: Tensor,
    target_xy: Tensor,
    seed: int = 0,
    epochs: int = 200,
    hidden: int = 64,
    val_frac: float = 0.2,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Train a TargetPredictor probe and evaluate on a held-out validation set.

    Parameters
    ----------
    obs       : [M, obs_dim]  per-agent observations (one class only).
    target_xy : [M, 2]        ground-truth target positions.
    seed      : int  for train/val split permutation and model init.
    epochs    : int  training epochs.
    hidden    : int  probe hidden width.
    val_frac  : float  fraction of data held out for validation.
    lr        : float  Adam learning rate.
    device    : str  torch device.

    Returns
    -------
    dict with keys:
        "score" : float  clamped R² in [0, 1] on held-out val set.
        "model" : TargetPredictor  (trained probe, on device).
        "mse"   : float  validation MSE.
    """
    M, obs_dim = obs.shape
    assert target_xy.shape == (M, 2), (
        f"target_xy shape {target_xy.shape} != ({M}, 2)"
    )
    assert 0.0 < val_frac < 1.0, f"val_frac must be in (0,1), got {val_frac}"

    dev = torch.device(device)
    obs = obs.float().to(dev)
    target_xy = target_xy.float().to(dev)

    # Deterministic train/val split via seeded permutation
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(M, generator=gen)
    n_val = max(1, int(M * val_frac))
    n_train = M - n_val

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    obs_train = obs[train_idx]
    tgt_train = target_xy[train_idx]
    obs_val   = obs[val_idx]
    tgt_val   = target_xy[val_idx]

    model = TargetPredictor(obs_dim=obs_dim, hidden=hidden, seed=seed).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(obs_train)
        loss = loss_fn(pred, tgt_train)
        loss.backward()
        optimizer.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        pred_val = model(obs_val)
        val_mse  = loss_fn(pred_val, tgt_val).item()
        r2 = _r2_score(pred_val, tgt_val)

    return {
        "score": r2,
        "model": model,
        "mse":   val_mse,
    }


def probe_score(model: TargetPredictor, obs_val: Tensor, target_val: Tensor) -> float:
    """
    Evaluate a trained TargetPredictor on new validation data.

    Parameters
    ----------
    model      : trained TargetPredictor.
    obs_val    : [M, obs_dim]
    target_val : [M, 2]

    Returns
    -------
    float  clamped R² in [0, 1].
    """
    dev = next(model.parameters()).device
    obs_val    = obs_val.float().to(dev)
    target_val = target_val.float().to(dev)
    model.eval()
    with torch.no_grad():
        pred = model(obs_val)
    return _r2_score(pred, target_val)


def probe_gap(score_full_sight: float, score_sensor_limited: float) -> float:
    """
    Compute the probe gap: full_sight_score − sensor_limited_score.

    Interpretation:
        gap ≈ 0  → alpha=1 (symmetric), both classes equally informative.
        gap large → alpha=0 or low, sensor-limited class has no target signal.

    Parameters
    ----------
    score_full_sight    : float  R² for full-sight (class 0) agents.
    score_sensor_limited: float  R² for sensor-limited (class 1) agents.

    Returns
    -------
    float  (non-negative in the expected regime; can be negative if probe training noise)
    """
    return float(score_full_sight) - float(score_sensor_limited)
