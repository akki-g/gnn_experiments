"""
Plot training metrics from one or more MAPPO run directories.

Examples:
    python -m experiments.plot_run_metrics \
        runs/vmas_simple_spread/20260603_192930_* \
        --output runs/vmas_simple_spread/metrics_633898.png

    python -m experiments.plot_run_metrics runs/vmas_simple_spread \
        --recursive --rolling 5 --show
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_EXCLUDE_COLUMNS = {
    "iter",
    "global_step",
    "seed",
    "run",
    "run_label",
    "source_path",
}


def _discover_metrics_files(paths: Iterable[str | Path], recursive: bool = False) -> list[Path]:
    """Return metrics.csv/json files from explicit files, run dirs, or parent dirs."""
    discovered: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen:
            discovered.append(path)
            seen.add(resolved)

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if path.name in {"metrics.csv", "metrics.json"}:
                add(path)
            continue

        if not path.is_dir():
            continue

        csv_path = path / "metrics.csv"
        json_path = path / "metrics.json"
        if csv_path.exists():
            add(csv_path)
        elif json_path.exists():
            add(json_path)

        if recursive:
            for metrics_csv in sorted(path.rglob("metrics.csv")):
                add(metrics_csv)
            for metrics_json in sorted(path.rglob("metrics.json")):
                if not (metrics_json.parent / "metrics.csv").exists():
                    add(metrics_json)

    return discovered


def _read_metrics_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported metrics file type: {path}")


def _base_label_for_run(df: pd.DataFrame) -> str:
    if "seed" in df.columns and df["seed"].nunique(dropna=True) == 1:
        return f"seed={df['seed'].iloc[0]}"
    return "run"


def load_run_metrics(paths: Sequence[str | Path], recursive: bool = False) -> pd.DataFrame:
    """
    Load and concatenate metrics from run directories or metrics files.

    Parameters
    ----------
    paths:
        Run directories, parent directories, or explicit metrics.csv/metrics.json
        files.
    recursive:
        If true, search directories recursively for metrics files.

    Returns
    -------
    DataFrame containing all metric rows plus run/run_label/source_path columns.
    """
    metrics_files = _discover_metrics_files(paths, recursive=recursive)
    if not metrics_files:
        raise FileNotFoundError("No metrics.csv or metrics.json files found.")

    frame_records = []
    for metrics_file in metrics_files:
        df = _read_metrics_file(metrics_file)
        if df.empty:
            continue

        df = df.copy()
        df["run"] = metrics_file.parent.name
        df["source_path"] = str(metrics_file)
        frame_records.append((df, _base_label_for_run(df), metrics_file.parent.name))

    if not frame_records:
        raise ValueError("All discovered metrics files were empty.")

    label_counts = {}
    for _, base_label, _ in frame_records:
        label_counts[base_label] = label_counts.get(base_label, 0) + 1

    frames = []
    for df, base_label, run_name in frame_records:
        if label_counts[base_label] > 1:
            df["run_label"] = f"{base_label} {run_name}"
        else:
            df["run_label"] = base_label
        frames.append(df)

    return pd.concat(frames, ignore_index=True, sort=False)


def metric_columns(
    metrics_df: pd.DataFrame,
    x_column: str = "global_step",
    exclude_columns: Iterable[str] = DEFAULT_EXCLUDE_COLUMNS,
) -> list[str]:
    """Return numeric columns that should be plotted as y-axis metrics."""
    exclude = set(exclude_columns)
    exclude.add(x_column)

    numeric_cols = [
        col
        for col in metrics_df.columns
        if pd.api.types.is_numeric_dtype(metrics_df[col])
    ]
    return [col for col in numeric_cols if col not in exclude]


def plot_run_metrics(
    paths: Sequence[str | Path],
    *,
    output: str | Path | None = None,
    recursive: bool = False,
    metrics: Sequence[str] | None = None,
    x_column: str = "global_step",
    rolling: int = 1,
    columns: int = 3,
    dpi: int = 160,
    show: bool = False,
):
    """
    Plot every numeric metric across one or more runs.

    The function overlays one line per run/seed on each metric subplot. Columns
    used for indexing/grouping (`global_step`, `iter`, `seed`) are not plotted as
    y-axis metrics by default.
    """
    metrics_df = load_run_metrics(paths, recursive=recursive)

    if x_column not in metrics_df.columns:
        if "iter" in metrics_df.columns:
            x_column = "iter"
        else:
            x_column = "__row_index__"
            metrics_df[x_column] = metrics_df.groupby("run_label").cumcount()

    if metrics is None:
        metrics_to_plot = metric_columns(metrics_df, x_column=x_column)
    else:
        missing = [metric for metric in metrics if metric not in metrics_df.columns]
        if missing:
            raise ValueError(f"Requested metrics are missing: {missing}")
        metrics_to_plot = list(metrics)

    if not metrics_to_plot:
        raise ValueError("No numeric metric columns found to plot.")

    columns = max(1, int(columns))
    rows = math.ceil(len(metrics_to_plot) / columns)
    fig_width = min(18, 5.4 * columns)
    fig_height = max(3.2, 3.2 * rows)
    fig, axes = plt.subplots(rows, columns, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.ravel()

    rolling = max(1, int(rolling))
    grouped = list(metrics_df.groupby("run_label", sort=False))

    for ax, metric in zip(axes_flat, metrics_to_plot):
        for run_label, run_df in grouped:
            run_df = run_df.sort_values(x_column)
            x_values = pd.to_numeric(run_df[x_column], errors="coerce")
            y_values = pd.to_numeric(run_df[metric], errors="coerce")
            valid = x_values.notna() & y_values.notna()
            if rolling > 1:
                y_values = y_values.rolling(rolling, min_periods=1).mean()
            ax.plot(
                x_values[valid],
                y_values[valid],
                linewidth=1.4,
                alpha=0.9,
                label=run_label,
            )
        ax.set_title(metric)
        ax.set_xlabel(x_column)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(metrics_to_plot):]:
        ax.set_visible(False)

    title = "Training metrics"
    if rolling > 1:
        title += f" (rolling={rolling})"
    fig.suptitle(title)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(3, len(labels)),
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.97))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    if output is not None:
        output_path = Path(output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot numeric metrics from MAPPO metrics.csv/json files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run directories, parent directories, or metrics.csv/json files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path, e.g. runs/vmas_simple_spread/metrics.png.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover metrics files under directory inputs.",
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=None,
        help="Metric column to plot. Repeat to select multiple metrics.",
    )
    parser.add_argument(
        "--x",
        default="global_step",
        help="X-axis column. Falls back to iter or row index if missing.",
    )
    parser.add_argument(
        "--rolling",
        type=int,
        default=1,
        help="Optional rolling mean window for smoothing plotted curves.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="Number of subplot columns.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plot_run_metrics(
        args.paths,
        output=args.output,
        recursive=args.recursive,
        metrics=args.metrics,
        x_column=args.x,
        rolling=args.rolling,
        columns=args.columns,
        show=args.show,
    )


if __name__ == "__main__":
    main()
