"""
Aggregate MAPPO sweep runs and plot mean +/- across-seed spread learning curves.

Two modes:
  --mode alpha  (default)
      Groups runs by a single config key (default env.alpha) and plots one
      mean±band learning curve per group value. Built for the identity alpha sweep.

  --mode comm
      Groups runs by both comm module and alpha level. Produces a subplot grid:
      rows = metrics, cols = alpha values.  Within each cell, one colored line per
      comm module (identity / broadcast / attention / graph).  Use this to compare
      communication architectures across asymmetry levels.

Examples:
    # Classic alpha sweep (identity baseline):
    python -m experiments.plot_alpha_sweep runs/pursuit_sweep/pursuit_base \\
        --output runs/pursuit_sweep/pursuit_base/alpha_curves.png

    # Comm-module vs alpha grid:
    python -m experiments.plot_alpha_sweep runs/pursuit/alpha \\
        --mode comm \\
        --output runs/pursuit/alpha/comm_vs_alpha.png
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

# Metrics shown in comm-sweep mode (fewer panels for legibility in the grid).
COMM_METRICS = (
    ("capture", "Capture rate (train)", False),
    ("pursuer_target_distance", "Pursuer-target distance", True),
    ("episode_return", "Episode return (train)", False),
    ("explained_variance", "Explained variance", False),
)

# Fixed colors and display order for comm modules so plots are consistent.
COMM_MODULE_ORDER = ["identity", "broadcast", "attention", "graph"]
COMM_MODULE_COLORS = {
    "identity":  "#888888",
    "broadcast": "#1f77b4",
    "attention": "#ff7f0e",
    "graph":     "#2ca02c",
}


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


def _load_run_metadata(run_dir: Path) -> tuple[object, str] | None:
    """
    Return (alpha, comm_module) for a run directory or None if unresolvable.

    Reads config.yaml for comm.module; falls back to parent directory name if
    config is absent.  Alpha is read from config then from metrics.csv.
    """
    config_path = run_dir / "config.yaml"
    alpha = None
    comm_module = None

    if config_path.exists():
        with config_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        alpha = _cfg_get(cfg, "env.alpha")
        comm_module = _cfg_get(cfg, "comm.module")

    if alpha is None or comm_module is None:
        csv_path = run_dir / "metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, nrows=5)
            if alpha is None:
                val = _csv_group_value(df, "env.alpha")
                if val is not None:
                    alpha = val
        # fallback: infer comm module from grandparent directory name
        if comm_module is None:
            for part in run_dir.parts:
                if part.lower() in COMM_MODULE_ORDER:
                    comm_module = part.lower()
                    break

    if alpha is None or comm_module is None:
        return None
    return alpha, comm_module


def load_comm_runs(
    paths: Sequence[str | Path],
) -> dict[str, dict[object, list[pd.DataFrame]]]:
    """
    Return nested dict: comm_module -> alpha -> [seed DataFrames].

    Discovers all runs with metrics.csv under *paths*, then reads each run's
    config.yaml to obtain comm.module and env.alpha.
    """
    result: dict[str, dict[object, list[pd.DataFrame]]] = {}
    for run_dir in _discover_runs(paths):
        meta = _load_run_metadata(run_dir)
        if meta is None:
            continue
        alpha, comm_module = meta
        df = pd.read_csv(run_dir / "metrics.csv")
        if df.empty:
            continue
        result.setdefault(comm_module, {}).setdefault(alpha, []).append(df)
    return result


def plot_comm_sweep(
    paths: Sequence[str | Path],
    *,
    output: str | Path | None = None,
    metrics: Sequence[tuple[str, str, bool]] = COMM_METRICS,
    x_column: str = "iter",
    rolling: int = 10,
    band: str = "std",
    dpi: int = 220,
    show: bool = False,
):
    """
    Plot a grid: rows = metrics, cols = alpha values.

    Within each cell, one learning curve per comm module (identity/broadcast/
    attention/graph) with a shaded across-seed band.  Colors are consistent
    across all cells so the legend only appears once.
    """
    comm_runs = load_comm_runs(paths)
    if not comm_runs:
        raise FileNotFoundError(
            "No runs with metrics.csv and resolvable (alpha, comm.module) found."
        )

    all_alphas: list[object] = sorted(
        {a for module_data in comm_runs.values() for a in module_data}
    )
    present_modules = [m for m in COMM_MODULE_ORDER if m in comm_runs]
    if not present_modules:
        present_modules = sorted(comm_runs.keys())

    n_rows = len(metrics)
    n_cols = len(all_alphas)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.8 * n_rows),
        squeeze=False,
        sharey="row",
    )
    rolling = max(1, int(rolling))

    for row_idx, (metric, title, lower_better) in enumerate(metrics):
        for col_idx, alpha in enumerate(all_alphas):
            ax = axes[row_idx][col_idx]
            any_plotted = False

            for module in present_modules:
                seed_frames = comm_runs.get(module, {}).get(alpha, [])
                if not seed_frames:
                    continue
                aligned = _aligned_matrix(seed_frames, metric, x_column)
                if aligned is None:
                    continue
                x, mat = aligned
                if rolling > 1:
                    smoothed = (
                        pd.DataFrame(mat.T)
                        .rolling(rolling, min_periods=1)
                        .mean()
                        .to_numpy()
                        .T
                    )
                else:
                    smoothed = mat
                mean = smoothed.mean(axis=0)
                if band == "minmax":
                    low, high = smoothed.min(axis=0), smoothed.max(axis=0)
                else:
                    sd = smoothed.std(axis=0)
                    low, high = mean - sd, mean + sd
                color = COMM_MODULE_COLORS.get(module, None)
                n_seeds = mat.shape[0]
                ax.plot(x, mean, color=color, linewidth=1.8,
                        label=f"{module} (n={n_seeds})")
                ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
                any_plotted = True

            suffix = " ↓" if lower_better else ""
            if row_idx == 0:
                ax.set_title(f"α = {alpha}", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(title + suffix, fontsize=9)
            ax.set_xlabel(x_column if row_idx == n_rows - 1 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)
            if not any_plotted:
                ax.text(0.5, 0.5, f"no data\n{metric}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")

    band_label = "min/max" if band == "minmax" else "±1 std"
    fig.suptitle(
        f"Comm module comparison across α levels  "
        f"(mean over seeds, shaded {band_label}, rolling={rolling})",
        fontsize=13,
    )

    handles = [
        Line2D([0], [0], color=COMM_MODULE_COLORS.get(m, "black"), linewidth=2.2)
        for m in present_modules
    ]
    labels = [m for m in present_modules]
    fig.legend(handles, labels, loc="lower center",
               ncol=len(labels), fontsize=10, frameon=False,
               title="comm module")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))

    if output is not None:
        out = Path(output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[plot_comm_sweep] saved → {out}")
    if show:
        plt.show()
    return fig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot sweep learning curves.  "
            "--mode alpha (default): one curve per alpha value.  "
            "--mode comm: grid of alpha cols x metric rows, one line per comm module."
        )
    )
    p.add_argument("paths", nargs="+",
                   help="Sweep parent dir(s) or run dir(s) containing metrics.csv.")
    p.add_argument("--mode", choices=("alpha", "comm"), default="alpha",
                   help="alpha: group by a single config key (default).  "
                        "comm: comm-module vs alpha grid.")
    p.add_argument("--output", default=None, help="Output image path (PNG).")
    p.add_argument("--group-key", default="env.alpha",
                   help="[--mode alpha] Dotted config key to group runs by.")
    p.add_argument("--x", default="iter", help="X-axis column (default iter).")
    p.add_argument("--rolling", type=int, default=10,
                   help="Rolling-mean window applied per seed before aggregating.")
    p.add_argument("--band", choices=("std", "minmax"), default="std",
                   help="Shaded band: +/-1 std (default) or min/max envelope.")
    p.add_argument("--columns", type=int, default=2,
                   help="[--mode alpha] Subplot columns.")
    p.add_argument("--dpi", type=int, default=220, help="Output resolution.")
    p.add_argument("--show", action="store_true",
                   help="Open an interactive window after plotting.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "comm":
        plot_comm_sweep(
            args.paths,
            output=args.output,
            x_column=args.x,
            rolling=args.rolling,
            band=args.band,
            dpi=args.dpi,
            show=args.show,
        )
    else:
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
