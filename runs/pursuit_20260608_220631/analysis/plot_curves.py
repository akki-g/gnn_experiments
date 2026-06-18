"""Generate learning-curve and cross-module comparison plots."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ALPHA_ROOT = ROOT / "alpha"
OUT = Path(__file__).resolve().parent
PLOTS = OUT / "plots"
PLOTS.mkdir(exist_ok=True)

ALPHA_MAP = {
    "alpha0p0": 0.0, "alpha0p25": 0.25, "alpha0p5": 0.5,
    "alpha0p75": 0.75, "alpha1p0": 1.0,
}
SEED_RE = re.compile(r"seed(\d+)")

MODULES = ["identity", "broadcast", "attention", "graph",
           "broadcast_zm", "attention_zm", "graph_zm"]

COLORS = {
    "identity": "#444444",
    "broadcast": "#1f77b4",
    "attention": "#2ca02c",
    "graph":     "#d62728",
    "broadcast_zm": "#1f77b4",
    "attention_zm": "#2ca02c",
    "graph_zm":     "#d62728",
}
LS = {True: "--", False: "-"}  # zm = dashed


def load_runs() -> dict:
    """Return {(module, alpha, seed): df} for every run."""
    out = {}
    for mdir in sorted(ALPHA_ROOT.iterdir()):
        if not mdir.is_dir():
            continue
        for adir in sorted(mdir.iterdir()):
            if not adir.is_dir() or adir.name not in ALPHA_MAP:
                continue
            for sdir in sorted(adir.iterdir()):
                m = SEED_RE.search(sdir.name)
                if not m:
                    continue
                df = pd.read_csv(sdir / "metrics.csv")
                out[(mdir.name, ALPHA_MAP[adir.name], int(m.group(1)))] = df
    return out


def smooth(arr: np.ndarray, w: int = 10) -> np.ndarray:
    if len(arr) < w:
        return arr
    k = np.ones(w) / w
    return np.convolve(arr, k, mode="same")


def stat_curve(runs: dict, module: str, alpha: float, col: str):
    """Stack across seeds and return (steps, mean, std)."""
    stack = []
    steps = None
    for seed in range(5):
        df = runs.get((module, alpha, seed))
        if df is None:
            continue
        v = df[col].to_numpy()
        if steps is None:
            steps = df["global_step"].to_numpy()
        stack.append(v)
    if not stack:
        return None, None, None
    stack = np.vstack(stack)
    return steps, stack.mean(0), stack.std(0)


def plot_learning_curves_by_alpha(runs):
    """5-panel grid: one per alpha, all modules overlaid."""
    alphas = sorted(ALPHA_MAP.values())
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=True)
    for ax, alpha in zip(axes, alphas):
        for module in MODULES:
            steps, mean, std = stat_curve(runs, module, alpha, "episode_return")
            if mean is None:
                continue
            mean = smooth(mean, 15)
            std = smooth(std, 15)
            is_zm = module.endswith("_zm")
            base = module.replace("_zm", "")
            label = f"{module}"
            ax.plot(steps, mean, color=COLORS[module], linestyle=LS[is_zm],
                    linewidth=1.2, label=label, alpha=0.9)
            ax.fill_between(steps, mean - std, mean + std,
                            color=COLORS[module], alpha=0.10)
        ax.set_title(f"alpha = {alpha}")
        ax.set_xlabel("env steps")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("episode return")
    axes[-1].legend(fontsize=7, loc="lower right", ncol=1)
    fig.suptitle("Pursuit episode return — 5 seeds (mean ± std), smoothed (window=15)")
    fig.tight_layout()
    fig.savefig(PLOTS / "learning_curves_return.png", dpi=140)
    plt.close(fig)


def plot_capture_curves(runs):
    alphas = sorted(ALPHA_MAP.values())
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=True)
    for ax, alpha in zip(axes, alphas):
        for module in MODULES:
            steps, mean, std = stat_curve(runs, module, alpha, "capture_rate")
            if mean is None:
                continue
            mean = smooth(mean, 15)
            std = smooth(std, 15)
            is_zm = module.endswith("_zm")
            ax.plot(steps, mean, color=COLORS[module], linestyle=LS[is_zm],
                    linewidth=1.2, label=module, alpha=0.9)
            ax.fill_between(steps, np.clip(mean - std, 0, 1),
                            np.clip(mean + std, 0, 1),
                            color=COLORS[module], alpha=0.10)
        ax.set_title(f"alpha = {alpha}")
        ax.set_xlabel("env steps")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("train capture_rate")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Capture rate during training (5 seeds, mean ± std, smoothed)")
    fig.tight_layout()
    fig.savefig(PLOTS / "learning_curves_capture.png", dpi=140)
    plt.close(fig)


def plot_loss_health(runs):
    """4-panel: explained_variance, entropy, approx_kl, value_loss
    averaged across alphas + seeds, one line per module."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    cols = ["explained_variance", "entropy", "approx_kl", "value_loss"]
    for ax, col in zip(axes.flatten(), cols):
        for module in MODULES:
            stack = []
            steps = None
            for alpha in ALPHA_MAP.values():
                for seed in range(5):
                    df = runs.get((module, alpha, seed))
                    if df is None:
                        continue
                    v = df[col].to_numpy()
                    if steps is None:
                        steps = df["global_step"].to_numpy()
                    stack.append(v)
            if not stack:
                continue
            stack = np.vstack(stack)
            mean = smooth(stack.mean(0), 15)
            is_zm = module.endswith("_zm")
            ax.plot(steps, mean, color=COLORS[module], linestyle=LS[is_zm],
                    linewidth=1.2, label=module)
        ax.set_title(col)
        ax.set_xlabel("env steps")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.5, 1.05)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Optimization health (averaged across alpha and seeds)")
    fig.tight_layout()
    fig.savefig(PLOTS / "loss_health.png", dpi=140)
    plt.close(fig)


def plot_final_comparison():
    """Bar charts of final eval capture & eval return per module x alpha."""
    df = pd.read_csv(OUT / "summary_per_seed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # eval_capture bars
    for ax, metric, title in [
        (axes[0], "eval_capture", "Eval capture rate (final 10% mean)"),
        (axes[1], "eval_pursuer_dist", "Eval pursuer→target distance"),
    ]:
        pivot_mean = df.pivot_table(index="alpha", columns="module",
                                     values=metric, aggfunc="mean")
        pivot_std = df.pivot_table(index="alpha", columns="module",
                                    values=metric, aggfunc="std")
        pivot_mean = pivot_mean[MODULES]
        pivot_std = pivot_std[MODULES]
        x = np.arange(len(pivot_mean.index))
        w = 0.11
        for i, module in enumerate(MODULES):
            ax.bar(x + i * w, pivot_mean[module], width=w,
                   yerr=pivot_std[module], capsize=2,
                   label=module, color=COLORS[module],
                   hatch="//" if module.endswith("_zm") else "")
        ax.set_xticks(x + w * (len(MODULES) - 1) / 2)
        ax.set_xticklabels([f"a={a}" for a in pivot_mean.index])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("capture rate")
    axes[1].set_ylabel("distance (lower = better)")
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "final_comparison.png", dpi=140)
    plt.close(fig)


def plot_zm_dissociation():
    """For each comm module, plot delta = real - zm (eval capture) across alpha.
    Positive delta = real comm helps. Negative = comm hurts."""
    df = pd.read_csv(OUT / "summary_per_seed.csv")
    pairs = [("broadcast", "broadcast_zm"),
             ("attention", "attention_zm"),
             ("graph", "graph_zm")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    metrics = [("eval_capture", "Δ eval capture (real − zm)"),
               ("eval_pursuer_dist", "Δ eval distance (real − zm) — lower is better")]
    for ax, (metric, title) in zip(axes, metrics):
        pivot = df.pivot_table(index="alpha", columns="module",
                                values=metric, aggfunc="mean")
        x = np.arange(len(pivot.index))
        w = 0.25
        for i, (real, zm) in enumerate(pairs):
            delta = pivot[real] - pivot[zm]
            ax.bar(x + i * w, delta, width=w, label=real,
                   color=COLORS[real])
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x + w)
        ax.set_xticklabels([f"a={a}" for a in pivot.index])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Δ capture")
    axes[1].set_ylabel("Δ distance")
    axes[0].legend()
    fig.suptitle("Zero-message dissociation probe: positive bar = real messages help")
    fig.tight_layout()
    fig.savefig(PLOTS / "zm_dissociation.png", dpi=140)
    plt.close(fig)


def main():
    runs = load_runs()
    print(f"loaded {len(runs)} runs")
    plot_learning_curves_by_alpha(runs)
    plot_capture_curves(runs)
    plot_loss_health(runs)
    plot_final_comparison()
    plot_zm_dissociation()
    print("plots written to", PLOTS)


if __name__ == "__main__":
    main()
