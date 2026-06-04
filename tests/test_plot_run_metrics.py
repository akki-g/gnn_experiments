import pytest

pytest.importorskip("pandas")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from experiments.plot_run_metrics import (  # noqa: E402
    load_run_metrics,
    metric_columns,
    plot_run_metrics,
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
