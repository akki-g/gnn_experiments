import pytest

pytest.importorskip("pandas")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from experiments.plot_run_metrics import (  # noqa: E402
    load_run_metrics,
    metric_columns,
    plot_run_metrics,
    summarize_run_metrics,
)


def _write_metrics(run_dir, seed):
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text(
        "\n".join(
            [
                "iter,global_step,episode_return,value_loss,entropy,seed",
                f"0,100,{10 + seed},0.5,2.0,{seed}",
                f"1,200,{20 + seed},0.25,1.8,{seed}",
            ]
        )
        + "\n"
    )


def _write_sweep_metrics(run_dir, seed, label, eval_returns):
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "logging:",
                f"  run_label: {label}",
            ]
        )
        + "\n"
    )
    (run_dir / "metrics.csv").write_text(
        "\n".join(
            [
                (
                    "iter,global_step,episode_return,optimized_episode_return,"
                    "eval_mean_return,eval_optimized_mean_return,coverage_distance,"
                    "eval_coverage_distance,collision_pairs,eval_collision_pairs,"
                    "task_score,eval_task_score,seed"
                ),
                (
                    f"0,100,-10,20,{eval_returns[0]},30,1.5,1.4,"
                    f"2,2,0.35,0.40,{seed}"
                ),
                (
                    f"1,200,-5,25,{eval_returns[1]},35,0.8,0.7,"
                    f"1,1,0.65,0.70,{seed}"
                ),
            ]
        )
        + "\n"
    )


def test_load_run_metrics_and_metric_columns(tmp_path):
    _write_metrics(tmp_path / "run_seed0", seed=0)
    _write_metrics(tmp_path / "run_seed1", seed=1)

    df = load_run_metrics([tmp_path], recursive=True)

    assert len(df) == 4
    assert set(df["seed"]) == {0, 1}
    assert "episode_return" in metric_columns(df)
    assert "value_loss" in metric_columns(df)
    assert "global_step" not in metric_columns(df)
    assert "seed" not in metric_columns(df)


def test_plot_run_metrics_writes_output(tmp_path):
    _write_metrics(tmp_path / "run_seed0", seed=0)
    output_path = tmp_path / "plots" / "metrics.png"

    fig = plot_run_metrics(
        [tmp_path / "run_seed0"],
        output=output_path,
        metrics=["episode_return", "value_loss"],
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    fig.clf()


def test_plot_run_metrics_defaults_to_iteration_axis(tmp_path):
    _write_metrics(tmp_path / "run_seed0", seed=0)

    fig = plot_run_metrics(
        [tmp_path / "run_seed0"],
        metrics=["episode_return"],
    )

    ax = fig.axes[0]
    assert ax.get_xlabel() == "iter"
    assert list(ax.lines[0].get_xdata()) == [0, 1]
    fig.clf()


def test_summarize_run_metrics_ranks_runs_and_preserves_config_labels(tmp_path):
    _write_sweep_metrics(
        tmp_path / "weak_seed0",
        seed=0,
        label="entropy_anneal",
        eval_returns=(-12, -8),
    )
    _write_sweep_metrics(
        tmp_path / "strong_seed1",
        seed=1,
        label="wide128_lr5e-4",
        eval_returns=(-9, -4),
    )

    summary = summarize_run_metrics([tmp_path], recursive=True)

    assert list(summary["config_label"]) == ["wide128_lr5e-4", "entropy_anneal"]
    assert list(summary["seed"]) == [1, 0]
    assert summary.loc[0, "best_eval_mean_return"] == -4
    assert summary.loc[0, "final_episode_return"] == -5
    assert summary.loc[0, "min_eval_coverage_distance"] == 0.7
    assert summary.loc[0, "final_eval_task_score"] == 0.70


def test_summarize_run_metrics_handles_old_metrics_without_eval_columns(tmp_path):
    _write_metrics(tmp_path / "old_seed0", seed=0)
    _write_metrics(tmp_path / "old_seed1", seed=1)

    summary = summarize_run_metrics([tmp_path], recursive=True)

    assert list(summary["seed"]) == [1, 0]
    assert list(summary["best_episode_return"]) == [21, 20]
    assert summary["best_eval_mean_return"].isna().all()
