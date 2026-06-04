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
    "config_label",
    "source_path",
}

MAXIMIZE_SUMMARY_METRICS = (
    "episode_return",
    "mean_step_reward",
    "optimized_episode_return",
    "optimized_mean_step_reward",
    "episode_return_delta_from_start",
    "mean_step_reward_delta_from_start",
    "task_score",
    "eval_mean_return",
    "eval_mean_step_reward",
    "eval_optimized_mean_return",
    "eval_optimized_mean_step_reward",
    "eval_return_delta_from_start",
    "eval_task_score",
    "capture_rate",
    "eval_capture",
)

MINIMIZE_SUMMARY_METRICS = (
    "episode_return_gap_to_zero",
    "mean_step_reward_gap_to_zero",
    "coverage_distance",
    "collision_pairs",
    "eval_return_gap_to_zero",
    "eval_coverage_distance",
    "eval_collision_pairs",
    "pursuer_target_distance",
    "eval_pursuer_target_distance",
)

FINAL_ONLY_SUMMARY_METRICS = (
    "episodes_completed",
    "policy_loss",
    "value_loss",
    "entropy",
    "entropy_coef",
    "total_loss",
    "grad_norm",
    "actor_grad_norm",
    "critic_grad_norm",
    "approx_kl",
    "clipfrac",
    "explained_variance",
    "mean_ratio_epoch0",
    "lr",
    "reward_shift",
)

DEFAULT_SUMMARY_SORT = "best_eval_mean_return"
SUMMARY_SORT_FALLBACKS = (
    "best_eval_mean_return",
    "best_episode_return",
    "best_eval_task_score",
    "best_task_score",
    "best_eval_optimized_mean_return",
    "best_optimized_episode_return",
)


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


def _python_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _single_value(series: pd.Series):
    values = series.dropna().unique()
    if len(values) != 1:
        return None
    return _python_scalar(values[0])


def _single_string_value(series: pd.Series) -> str | None:
    value = _single_value(series)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _config_label_for_metrics_file(metrics_file: Path) -> str | None:
    config_path = metrics_file.parent / "config.yaml"
    if not config_path.exists():
        return None

    try:
        import yaml
    except ImportError:
        return None

    try:
        with config_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None

    label = cfg.get("logging", {}).get("run_label")
    if label is None:
        return None
    label = str(label).strip()
    return label or None


def _base_label_for_run(df: pd.DataFrame) -> str:
    config_label = (
        _single_string_value(df["config_label"])
        if "config_label" in df.columns
        else None
    )
    if "seed" in df.columns and df["seed"].nunique(dropna=True) == 1:
        seed_label = f"seed={df['seed'].iloc[0]}"
        if config_label is not None:
            return f"{config_label} {seed_label}"
        return seed_label
    if config_label is not None:
        return config_label
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
        existing_run_label = (
            _single_string_value(df["run_label"])
            if "run_label" in df.columns
            else None
        )
        if "config_label" not in df.columns:
            config_label = _config_label_for_metrics_file(metrics_file) or existing_run_label
            if config_label is not None:
                df["config_label"] = config_label
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


def _numeric_values(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _last_numeric_value(df: pd.DataFrame, column: str) -> float:
    values = _numeric_values(df, column)
    if values.empty:
        return float("nan")
    return float(values.iloc[-1])


def _first_numeric_value(df: pd.DataFrame, column: str) -> float:
    values = _numeric_values(df, column)
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def _max_numeric_value(df: pd.DataFrame, column: str) -> float:
    values = _numeric_values(df, column)
    if values.empty:
        return float("nan")
    return float(values.max())


def _min_numeric_value(df: pd.DataFrame, column: str) -> float:
    values = _numeric_values(df, column)
    if values.empty:
        return float("nan")
    return float(values.min())


def _sort_summary(
    summary_df: pd.DataFrame,
    sort_by: str | None,
    ascending: bool | None,
) -> pd.DataFrame:
    candidates = []
    if sort_by:
        candidates.append(sort_by)
    candidates.extend(col for col in SUMMARY_SORT_FALLBACKS if col not in candidates)

    active_sort = None
    for column in candidates:
        if column in summary_df.columns and summary_df[column].notna().any():
            active_sort = column
            break

    if active_sort is None:
        return summary_df

    sort_ascending = active_sort.startswith("min_") if ascending is None else ascending
    return summary_df.sort_values(
        active_sort,
        ascending=sort_ascending,
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_run_metrics(
    paths: Sequence[str | Path],
    *,
    recursive: bool = False,
    sort_by: str | None = DEFAULT_SUMMARY_SORT,
    ascending: bool | None = None,
) -> pd.DataFrame:
    """
    Produce one ranked summary row per metrics file.

    The summary is intentionally tolerant of old run schemas. Missing metrics are
    included as NaN so older CSVs can be compared with newer VMAS/pursuit logs.
    """
    metrics_df = load_run_metrics(paths, recursive=recursive)
    sort_column = "global_step" if "global_step" in metrics_df.columns else "iter"

    summary_rows = []
    for source_path, run_df in metrics_df.groupby("source_path", sort=False):
        if sort_column in run_df.columns:
            run_df = run_df.sort_values(sort_column, kind="mergesort")

        row = {
            "run_label": _single_string_value(run_df["run_label"]),
            "config_label": (
                _single_string_value(run_df["config_label"])
                if "config_label" in run_df.columns
                else None
            ),
            "run": _single_string_value(run_df["run"]),
            "seed": _single_value(run_df["seed"]) if "seed" in run_df.columns else None,
            "source_path": source_path,
            "rows": int(len(run_df)),
            "first_global_step": _first_numeric_value(run_df, "global_step"),
            "final_global_step": _last_numeric_value(run_df, "global_step"),
        }

        for metric in MAXIMIZE_SUMMARY_METRICS:
            row[f"best_{metric}"] = _max_numeric_value(run_df, metric)
            row[f"final_{metric}"] = _last_numeric_value(run_df, metric)

        for metric in MINIMIZE_SUMMARY_METRICS:
            row[f"min_{metric}"] = _min_numeric_value(run_df, metric)
            row[f"final_{metric}"] = _last_numeric_value(run_df, metric)

        for metric in FINAL_ONLY_SUMMARY_METRICS:
            row[f"final_{metric}"] = _last_numeric_value(run_df, metric)

        summary_rows.append(row)

    if not summary_rows:
        raise ValueError("All discovered metrics files were empty.")

    summary_df = pd.DataFrame(summary_rows)
    return _sort_summary(summary_df, sort_by=sort_by, ascending=ascending)


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
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print one ranked summary row per run.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Optional output CSV path for --summary.",
    )
    parser.add_argument(
        "--summary-sort",
        default=DEFAULT_SUMMARY_SORT,
        help=(
            "Summary column to sort by. Falls back to episode return/task score "
            "columns when the requested metric is missing."
        ),
    )
    parser.add_argument(
        "--summary-ascending",
        action="store_true",
        help="Sort summary in ascending order.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.summary:
        summary_df = summarize_run_metrics(
            args.paths,
            recursive=args.recursive,
            sort_by=args.summary_sort,
            ascending=True if args.summary_ascending else None,
        )
        print(summary_df.to_string(index=False))
        if args.summary_output is not None:
            output_path = Path(args.summary_output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(output_path, index=False)

    if not args.summary or args.output is not None or args.show:
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
