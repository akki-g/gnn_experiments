import pytest

import math
import torch

from experiments.train_vmas_mappo import (
    _delta_from_start,
    _gap_to_zero,
    _linear_schedule,
    _safe_run_label,
    _set_optimizer_lr,
    _task_score,
)


def test_linear_schedule_interpolates_start_middle_end():
    assert _linear_schedule(0.01, 0.001, step=0, total_steps=11) == pytest.approx(0.01)
    assert _linear_schedule(0.01, 0.001, step=5, total_steps=11) == pytest.approx(0.0055)
    assert _linear_schedule(0.01, 0.001, step=10, total_steps=11) == pytest.approx(0.001)


def test_linear_schedule_handles_single_step_run():
    assert _linear_schedule(0.01, 0.001, step=0, total_steps=1) == pytest.approx(0.001)


def test_gap_to_zero_reports_remaining_non_positive_reward_gap():
    assert _gap_to_zero(-2.5) == pytest.approx(2.5)
    assert _gap_to_zero(0.0) == pytest.approx(0.0)
    assert _gap_to_zero(1.0) == pytest.approx(0.0)
    assert math.isnan(_gap_to_zero(float("nan")))


def test_delta_from_start_is_positive_when_reward_improves():
    assert _delta_from_start(-2.0, -5.0) == pytest.approx(3.0)
    assert _delta_from_start(-6.0, -5.0) == pytest.approx(-1.0)
    assert math.isnan(_delta_from_start(-2.0, None))
    assert math.isnan(_delta_from_start(float("nan"), -5.0))


def test_safe_run_label_keeps_run_directory_names_simple():
    assert _safe_run_label(None) == ""
    assert _safe_run_label("wide 128 / lr=5e-4") == "wide_128_lr5e-4"
    assert _safe_run_label("entropy_anneal") == "entropy_anneal"


def test_set_optimizer_lr_updates_all_param_groups():
    p0 = torch.nn.Parameter(torch.tensor(1.0))
    p1 = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.Adam(
        [
            {"params": [p0], "lr": 1.0e-3},
            {"params": [p1], "lr": 5.0e-4},
        ]
    )

    _set_optimizer_lr(optimizer, 1.0e-5)

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1.0e-5, 1.0e-5]
    )


def test_task_score_is_positive_bounded_and_monotonic():
    perfect = _task_score(0.0, 0.0)
    farther = _task_score(1.0, 0.0)
    collided = _task_score(0.0, 1.0)

    assert perfect == pytest.approx(1.0)
    assert 0.0 < farther < perfect
    assert 0.0 < collided < perfect
    assert math.isnan(_task_score(float("nan"), 0.0))
