"""
Plotting script for the stiffened pursuit sweep.

Reads:
    runs/pursuit_stiffened_20260617_224707/alpha/{module}/{alpha_label}/seed{S}/metrics.csv
    runs/pursuit_stiffened_20260617_224707/analysis/{summary_per_cell,bootstrap_deltas,sanity_gates}.csv

Writes (under runs/.../analysis/plots/):
    curves_capture_rate.png        train capture_rate, 3 modules x 2 alphas, with seed bands
    curves_eval_capture.png        eval_capture (subset of iters), same layout
    curves_eval_dist.png           eval_pursuer_target_distance
    curves_explained_variance.png  EV across training
    curves_entropy.png             entropy schedule
    curves_loss_health.png         policy_loss, value_loss, approx_kl, clipfrac
    final_comparison.png           bar chart of final-window means with seed scatter
    dissociation_panel.png         attention vs attention_zm Delta with 95% CI
    big_sweep_overlay.png          stiffened identity@a=0.5 vs the big-sweep ceiling
    sanity_grid.png                ratio epoch0 and NaN flags as a green/red grid

All plots use plain matplotlib (no seaborn).  Seed bands = mean +/- 1 sd across 5 seeds.
"""

from __future__ import annotations

import csv
import math
import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ALPHA_DIR = ROOT / "alpha"
ANALYSIS = ROOT / "analysis"
PLOTS = ANALYSIS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

MODULES = ["identity", "attention", "attention_zm"]
ALPHAS = [("alpha0p25", 0.25), ("alpha0p5", 0.5)]
SEEDS = list(range(5))

COLORS = {
    "identity":    "#1f77b4",
    "attention":   "#d62728",
    "attention_zm":"#2ca02c",
}
LINESTYLES = {
    "identity":    "-",
    "attention":   "-",
    "attention_zm":"--",
}


def _load(path: pathlib.Path) -> Dict[str, List[float]]:
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        cols: Dict[str, List[float]] = {h: [] for h in header}
        for row in reader:
            for h, v in zip(header, row):
                try:
                    cols[h].append(float(v))
                except (ValueError, TypeError):
                    cols[h].append(float("nan"))
    return cols


def load_cell(module: str, alabel: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Returns (iters, dict of metric -> array shape [n_seeds, n_iters])."""
    arrays: Dict[str, List[List[float]]] = {}
    iters_ref = None
    for s in SEEDS:
        path = ALPHA_DIR / module / alabel / f"seed{s}" / "metrics.csv"
        d = _load(path)
        iters_ref = np.asarray(d["iter"])
        for k, v in d.items():
            arrays.setdefault(k, []).append(v)
    out = {k: np.asarray(v) for k, v in arrays.items()}
    return iters_ref, out


def smooth(y: np.ndarray, w: int = 11) -> np.ndarray:
    if w <= 1 or y.size < w:
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def _band(ax, iters: np.ndarray, ys: np.ndarray, label: str, color: str, linestyle: str = "-", smooth_w: int = 11):
    finite_mask = np.isfinite(ys)
    # If a column has any NaN (e.g. eval columns), drop those rows pointwise.
    means = np.full(iters.size, np.nan)
    stds = np.full(iters.size, np.nan)
    for i in range(iters.size):
        col = ys[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        means[i] = col.mean()
        if col.size >= 2:
            stds[i] = col.std(ddof=1)
        else:
            stds[i] = 0.0
    valid = np.isfinite(means)
    if not valid.any():
        return
    xs = iters[valid]
    ms = means[valid]
    sd = stds[valid]
    if smooth_w > 1 and ms.size >= smooth_w:
        ms_s = smooth(ms, smooth_w)
        sd_s = smooth(np.nan_to_num(sd, nan=0.0), smooth_w)
    else:
        ms_s, sd_s = ms, np.nan_to_num(sd, nan=0.0)
    ax.plot(xs, ms_s, color=color, linestyle=linestyle, label=label, linewidth=1.8)
    ax.fill_between(xs, ms_s - sd_s, ms_s + sd_s, color=color, alpha=0.15, linewidth=0)


def _curves(metric: str, ylabel: str, fname: str, smooth_w: int = 11, ylim=None, train_only: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (alabel, aval) in zip(axes, ALPHAS):
        for mod in MODULES:
            iters, arrs = load_cell(mod, alabel)
            if metric not in arrs:
                continue
            _band(ax, iters, arrs[metric], mod, COLORS[mod], LINESTYLES[mod], smooth_w=smooth_w)
        ax.set_title(f"alpha = {aval}")
        ax.set_xlabel("PPO iter")
        ax.grid(alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="best", fontsize=9)
    fig.suptitle(f"{metric} -- stiffened pursuit sweep", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _curves("capture_rate", "train capture_rate", "curves_capture_rate.png", ylim=(0, 1))
    _curves("eval_capture", "eval_capture", "curves_eval_capture.png", smooth_w=1, ylim=(0, 1))
    _curves("eval_pursuer_target_distance", "eval pursuer-target distance", "curves_eval_dist.png", smooth_w=1)
    _curves("explained_variance", "explained_variance", "curves_explained_variance.png", ylim=(-0.1, 1.05))
    _curves("entropy", "policy entropy", "curves_entropy.png")
    _curves("episode_return", "train episode_return", "curves_episode_return.png")

    # 2x2 loss health overlay (policy, value, kl, clipfrac) at alpha=0.5 only.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, m in zip(axes.flat, ["policy_loss", "value_loss", "approx_kl", "clipfrac"]):
        for mod in MODULES:
            iters, arrs = load_cell(mod, "alpha0p5")
            _band(ax, iters, arrs[m], mod, COLORS[mod], LINESTYLES[mod], smooth_w=11)
        ax.set_title(m)
        ax.set_xlabel("PPO iter")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle("Optimization health -- alpha=0.5", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "curves_loss_health.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Final-window bar chart with seed scatter ---
    # eval_capture and capture_rate at each alpha.
    with open(ANALYSIS / "summary_per_seed.csv") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, metric in zip(axes, ["eval_capture", "capture_rate"]):
        x_positions = []
        labels = []
        pos = 0
        for alabel, aval in ALPHAS:
            for mod in MODULES:
                ys = [float(r[metric]) for r in rows
                      if r["module"] == mod and float(r["alpha"]) == aval]
                ys = [y for y in ys if math.isfinite(y)]
                if not ys:
                    pos += 1
                    continue
                ax.bar(pos, np.mean(ys), color=COLORS[mod], alpha=0.7, edgecolor="black", linewidth=0.5)
                ax.scatter([pos] * len(ys), ys, color="black", s=14, zorder=3)
                x_positions.append(pos)
                labels.append(f"{mod}\nα={aval}")
                pos += 1
            pos += 0.5  # gap between alpha groups
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Final-window finals (mean over last 10% of iters; black dots = per-seed)", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "final_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Dissociation panel: Delta = attention - attention_zm with bootstrap CI ---
    with open(ANALYSIS / "bootstrap_deltas.csv") as f:
        boot = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, ylabel in zip(
        axes,
        ["eval_capture", "capture_rate"],
        ["Δ eval_capture (attention − attention_zm)", "Δ train capture_rate (attention − attention_zm)"],
    ):
        xs, means, lo, hi = [], [], [], []
        for r in boot:
            if r["metric"] != metric:
                continue
            xs.append(float(r["alpha"]))
            means.append(float(r["delta_att_zm_mean"]))
            lo.append(float(r["delta_att_zm_ci_lo"]))
            hi.append(float(r["delta_att_zm_ci_hi"]))
        xs = np.asarray(xs)
        means = np.asarray(means)
        lo = np.asarray(lo)
        hi = np.asarray(hi)
        order = np.argsort(xs)
        xs, means, lo, hi = xs[order], means[order], lo[order], hi[order]
        yerr = np.vstack([means - lo, hi - means])
        ax.errorbar(xs, means, yerr=yerr, fmt="o", color="black", markersize=8,
                    capsize=6, elinewidth=1.5)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("alpha")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
        for x, m_, l, h in zip(xs, means, lo, hi):
            ax.annotate(f"Δ={m_:+.3f}\n[{l:+.3f}, {h:+.3f}]",
                        (x, m_), textcoords="offset points",
                        xytext=(10, 0), fontsize=8, va="center")
    fig.suptitle("Dissociation probe — attention vs attention_zm (95% paired bootstrap CI over 5 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "dissociation_panel.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Sanity-gate grid (ratio_min/max == 1.0; NaN core rows == 0) ---
    with open(ANALYSIS / "sanity_gates.csv") as f:
        srows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    cells = []
    for r in srows:
        ratio_ok = float(r["ratio_min"]) == 1.0 and float(r["ratio_max"]) == 1.0
        nan_ok = int(r["nan_core_rows"]) == 0
        ev_ok = float(r["explained_variance"]) >= 0.85
        cells.append((r["module"], float(r["alpha"]), int(r["seed"]),
                     ratio_ok, nan_ok, ev_ok))
    grid_rows = [(mod, a) for mod in MODULES for _, a in ALPHAS]
    z = np.zeros((len(grid_rows), len(SEEDS)))
    for c in cells:
        i = grid_rows.index((c[0], c[1]))
        j = c[2]
        z[i, j] = (1 if c[3] else 0) + (1 if c[4] else 0) + (1 if c[5] else 0)  # 0..3
    ax.imshow(z, cmap="RdYlGn", vmin=0, vmax=3, aspect="auto")
    ax.set_yticks(range(len(grid_rows)))
    ax.set_yticklabels([f"{m}|α={a}" for m, a in grid_rows], fontsize=9)
    ax.set_xticks(SEEDS)
    ax.set_xticklabels([f"seed{s}" for s in SEEDS])
    for i in range(len(grid_rows)):
        for j in range(len(SEEDS)):
            ax.text(j, i, int(z[i, j]), ha="center", va="center", color="white", fontsize=10)
    ax.set_title("Sanity gates (3 = ratio + NaN + EV all OK)")
    fig.tight_layout()
    fig.savefig(PLOTS / "sanity_grid.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Stiffening confirmation: identity@α=0.5 stiffened vs big-sweep ceiling ---
    iters, arrs = load_cell("identity", "alpha0p5")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _band(ax, iters, arrs["eval_capture"], "identity α=0.5 (stiffened)",
          COLORS["identity"], "-", smooth_w=1)
    _band(ax, iters, arrs["capture_rate"], "identity α=0.5 train cap (stiffened)",
          "#9467bd", "-", smooth_w=11)
    ax.axhline(0.426, color="red", linestyle="--", linewidth=1.5,
               label="big sweep eval_capture ceiling = 0.426 (pursuit_base.yaml)")
    ax.set_xlabel("PPO iter")
    ax.set_ylabel("capture")
    ax.set_title("Did stiffening take?  identity @ α=0.5: stiffened curves vs comm-trivial ceiling")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.0)
    fig.tight_layout()
    fig.savefig(PLOTS / "big_sweep_overlay.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("Plots written to", PLOTS)


if __name__ == "__main__":
    main()
