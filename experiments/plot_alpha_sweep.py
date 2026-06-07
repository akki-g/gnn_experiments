"""
Aggregate MAPPO sweep runs by a config key (default: env.alpha) and plot
mean +/- across-seed spread learning curves.

Unlike plot_run_metrics.py (which overlays one line per run/seed), this groups
runs that share a config value and collapses the seeds into a single mean line
with a shaded band, so each alpha shows up as one curve. Built for the pursuit
alpha sweep but works for any config key whose runs share an iteration grid.

Examples:
    python -m experiments.plot_alpha_sweep runs/pursuit_sweep/pursuit_base \
        --output runs/pursuit_sweep/pursuit_base/alpha_curves.png

    python -m experiments.plot_alpha_sweep runs/pursuit_sweep/pursuit_base \
        --group-key env.alpha --rolling 10 --band std
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# Default metric panels. Each tuple is (column, human title, lower_is_better).
DEFAULT_METRICS = (
    ("episode_return", "Episode return (train)", False),
    ("eval_mean_return", "Episode return (eval)", False),
    ("capture", "Capture rate (train)", False),
    ("eval_capture", "Capture rate (eval)", False),
    ("pursuer_target_distance", "Pursuer-target distance (train)", True),
    ("task_score", "Task score (train)", False),
    ("entropy", "Policy entropy", False),
    ("explained_variance", "Explained variance", False),
)


def _cfg_get(cfg: dict, dotted_key: str):
    """Look up a dotted key like 'env.alpha' in a nested config dict."""
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _csv_group_value(df: pd.DataFrame, group_key: str):
    """Find a constant group value stored directly in metrics.csv."""
    candidates = [
        group_key,
        group_key.replace(".", "_"),
        group_key.split(".")[-1],
    ]
    for column in candidates:
        if column not in df.columns:
            continue
        values = df[column].dropna().unique()
        if len(values) > 0:
            return values[0]
    return None


def _discover_runs(paths: Sequence[str | Path]) -> list[Path]:
    """Return run directories (those containing metrics.csv) under the inputs."""
    found: list[Path] = []
    seen: set[Path] = set()

    def add(run_dir: Path) -> None:
        resolved = run_dir.resolve()
        if resolved not in seen:
            found.append(run_dir)
            seen.add(resolved)

    for raw in paths:
        path = Path(raw).expanduser()
        if (path / "metrics.csv").exists():
            add(path)
            continue
        if path.is_dir():
            for metrics_csv in sorted(path.rglob("metrics.csv")):
                add(metrics_csv.parent)
    return found


def load_grouped_runs(
    paths: Sequence[str | Path],
    group_key: str = "env.alpha",
) -> dict[object, list[pd.DataFrame]]:
    """Map each group_key value to the list of per-seed metric DataFrames."""
    groups: dict[object, list[pd.DataFrame]] = {}
    for run_dir in _discover_runs(paths):
        df = pd.read_csv(run_dir / "metrics.csv")
        if df.empty:
            continue

        key = None
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            with config_path.open() as f:
                cfg = yaml.safe_load(f) or {}
            key = _cfg_get(cfg, group_key)
        if key is None:
            key = _csv_group_value(df, group_key)
        if key is None:
            continue
        groups.setdefault(key, []).append(df)
    return groups


def _aligned_matrix(
    seed_frames: list[pd.DataFrame],
    metric: str,
    x_column: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Stack a metric across seeds onto a shared x grid.

    Eval metrics are logged sparsely (NaN on non-eval iters), so we drop NaNs
    per seed and intersect the x values that every seed actually logged. Returns
    (x, matrix[n_seeds, n_x]) or None if the metric is absent/unalignable.
    """
    per_seed_x = []
    per_seed_y = []
    for df in seed_frames:
        if metric not in df.columns or x_column not in df.columns:
            return None
        x = pd.to_numeric(df[x_column], errors="coerce")
        y = pd.to_numeric(df[metric], errors="coerce")
        valid = x.notna() & y.notna()
        if not valid.any():
            return None
        per_seed_x.append(x[valid].to_numpy())
        per_seed_y.append(y[valid].to_numpy())

    shared = per_seed_x[0]
    for xs in per_seed_x[1:]:
        shared = np.intersect1d(shared, xs)
    if shared.size == 0:
        return None

    rows = []
    for xs, ys in zip(per_seed_x, per_seed_y):
        lookup = dict(zip(xs, ys))
        rows.append([lookup[v] for v in shared])
    return shared, np.asarray(rows, dtype="float64")


def plot_alpha_sweep(
    paths: Sequence[str | Path],
    *,
    output: str | Path | None = None,
    group_key: str = "env.alpha",
    metrics: Sequence[tuple[str, str, bool]] = DEFAULT_METRICS,
    x_column: str = "iter",
    rolling: int = 10,
    band: str = "std",
    columns: int = 2,
    dpi: int = 220,
    show: bool = False,
):
    """Plot per-group mean curves with a shaded across-seed band per metric."""
    groups = load_grouped_runs(paths, group_key=group_key)
    if not groups:
        raise FileNotFoundError(
            f"No runs with metrics.csv carrying {group_key} found."
        )

    group_values = sorted(groups.keys())
    cmap = plt.get_cmap("viridis")
    colors = {
        val: cmap(i / max(1, len(group_values) - 1))
        for i, val in enumerate(group_values)
    }

    n_panels = len(metrics)
    rows = (n_panels + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(7.0 * columns, 4.0 * rows), squeeze=False
    )
    axes_flat = axes.ravel()
    rolling = max(1, int(rolling))
    plotted_group_values: set[object] = set()

    for ax, (metric, title, lower_better) in zip(axes_flat, metrics):
        plotted = False
        for val in group_values:
            aligned = _aligned_matrix(groups[val], metric, x_column)
            if aligned is None:
                continue
            x, mat = aligned
            if rolling > 1:
                smoothed = pd.DataFrame(mat.T).rolling(
                    rolling, min_periods=1
                ).mean().to_numpy().T
            else:
                smoothed = mat
            mean = smoothed.mean(axis=0)
            if band == "minmax":
                low, high = smoothed.min(axis=0), smoothed.max(axis=0)
            else:  # std
                sd = smoothed.std(axis=0)
                low, high = mean - sd, mean + sd
            color = colors[val]
            n_seeds = mat.shape[0]
            ax.plot(x, mean, color=color, linewidth=1.8,
                    label=f"{group_key.split('.')[-1]}={val} (n={n_seeds})")
            ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
            plotted_group_values.add(val)
            plotted = True
        suffix = " (lower is better)" if lower_better else ""
        ax.set_title(title + suffix)
        ax.set_xlabel(x_column)
        ax.grid(True, alpha=0.25)
        if not plotted:
            ax.text(0.5, 0.5, f"no data: {metric}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    band_label = "min/max" if band == "minmax" else "+/-1 std"
    fig.suptitle(
        f"Per-{group_key} learning curves (mean over seeds, shaded {band_label}, "
        f"rolling={rolling})",
        fontsize=13,
    )
    legend_values = [val for val in group_values if val in plotted_group_values]
    group_label = group_key.split(".")[-1]
    handles = [
        Line2D([0], [0], color=colors[val], linewidth=2.2)
        for val in legend_values
    ]
    labels = [
        f"{group_label}={val} (n={len(groups[val])})"
        for val in legend_values
    ]
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(labels), 5), fontsize=9, frameon=False,
                   title=group_key)
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if output is not None:
        out = Path(output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot per-group (default env.alpha) mean +/- seed-spread curves."
    )
    p.add_argument("paths", nargs="+",
                   help="Sweep parent dir(s) or run dir(s) containing metrics.csv.")
    p.add_argument("--output", default=None, help="Output image path (PNG).")
    p.add_argument("--group-key", default="env.alpha",
                   help="Dotted config key to group runs by (default env.alpha).")
    p.add_argument("--x", default="iter", help="X-axis column (default iter).")
    p.add_argument("--rolling", type=int, default=10,
                   help="Rolling-mean window applied per seed before aggregating.")
    p.add_argument("--band", choices=("std", "minmax"), default="std",
                   help="Shaded band: +/-1 std (default) or min/max envelope.")
    p.add_argument("--columns", type=int, default=2, help="Subplot columns.")
    p.add_argument("--dpi", type=int, default=220, help="Output resolution.")
    p.add_argument("--show", action="store_true",
                   help="Open an interactive window after plotting.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plot_alpha_sweep(
        args.paths,
        output=args.output,
        group_key=args.group_key,
        x_column=args.x,
        rolling=args.rolling,
        band=args.band,
        columns=args.columns,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
