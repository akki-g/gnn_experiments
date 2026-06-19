"""
Stiffened sweep aggregator.

Reads every metrics.csv in:
    runs/pursuit_stiffened_20260617_224707/alpha/{module}/{alpha_label}/seed{S}/metrics.csv

Produces:
    summary_per_seed.csv      one row per (module, alpha, seed) on final-window means
    summary_per_cell.csv      one row per (module, alpha): mean/std over 5 seeds
    summary_per_module.csv    one row per module averaged over alphas
    bootstrap_deltas.csv      paired bootstrap CI for attention - attention_zm at each alpha
    sanity_gates.csv          per-run gate checks (NaN, ratio epoch0, EV final)
    nan_report.csv            empty if no NaNs in final-window metrics

Numeric conventions:
- "Final window" = last 10% of iters (iter 563-624, inclusive) = 63 rows.
- Train metrics aggregated as plain means over the final-window rows.
- Eval columns are NaN on non-eval rows; aggregated by dropping NaN.
- Paired bootstrap: 10_000 resamples, seeded for reproducibility.

Run:
    python runs/pursuit_stiffened_20260617_224707/analysis/aggregate.py
"""

from __future__ import annotations

import csv
import math
import pathlib
import statistics
from typing import Dict, List, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ALPHA_DIR = ROOT / "alpha"
OUT = ROOT / "analysis"

MODULES = ["identity", "attention", "attention_zm"]
ALPHAS = [("alpha0p25", 0.25), ("alpha0p5", 0.5)]
SEEDS = list(range(5))

# Metric subsets we summarize.
TRAIN_METRICS = [
    "episode_return",
    "mean_step_reward",
    "pursuer_target_distance",
    "capture",
    "capture_rate",
    "policy_loss",
    "value_loss",
    "entropy",
    "total_loss",
    "grad_norm",
    "approx_kl",
    "clipfrac",
    "explained_variance",
    "mean_ratio_epoch0",
    "capture_c1_participation",
    "dist_sensor_limited",
    "marginal_c1_rate",
    "knn_c1_sighted_frac",
]
EVAL_METRICS = [
    "eval_mean_return",
    "eval_mean_episode_length",
    "eval_pursuer_target_distance",
    "eval_capture",
]


def _load_csv(path: pathlib.Path) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def _to_float(s: str) -> float:
    if s == "" or s is None:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _final_window(rows: List[List[str]], header: List[str], frac: float = 0.10):
    n = len(rows)
    cutoff = int(round((1.0 - frac) * n))
    return rows[cutoff:], header


def _mean_finite(values: List[float]) -> float:
    fin = [v for v in values if math.isfinite(v)]
    if not fin:
        return float("nan")
    return sum(fin) / len(fin)


def _std_finite(values: List[float]) -> float:
    fin = [v for v in values if math.isfinite(v)]
    if len(fin) < 2:
        return float("nan")
    return statistics.stdev(fin)


def _nan_count(values: List[float]) -> int:
    return sum(1 for v in values if not math.isfinite(v))


def summarize_run(path: pathlib.Path) -> Dict[str, float]:
    header, rows = _load_csv(path)
    final_rows, _ = _final_window(rows, header)
    col = {name: i for i, name in enumerate(header)}

    out: Dict[str, float] = {}
    for m in TRAIN_METRICS + EVAL_METRICS:
        if m not in col:
            out[m] = float("nan")
            continue
        idx = col[m]
        vals = [_to_float(r[idx]) for r in final_rows]
        out[m] = _mean_finite(vals)
        out[f"{m}__nan_count"] = float(_nan_count(vals))

    # Whole-run NaN sweep (any row, any metric)
    all_nan = 0
    for r in rows:
        for m in TRAIN_METRICS:
            v = _to_float(r[col[m]]) if m in col else float("nan")
            if not math.isfinite(v) and m not in {"capture_c1_participation",
                                                  "dist_full_sight",
                                                  "dist_sensor_limited",
                                                  "marginal_c1_rate",
                                                  "knn_c1_sighted_frac"}:
                # only count NaN on core (non-probe) metrics for the gate
                all_nan += 1
    out["nan_core_rows"] = float(all_nan)

    # Ratio sanity gate: should be exactly 1.0 every iter (old logprobs detached).
    if "mean_ratio_epoch0" in col:
        all_ratios = [_to_float(r[col["mean_ratio_epoch0"]]) for r in rows]
        out["ratio_min"] = min(v for v in all_ratios if math.isfinite(v))
        out["ratio_max"] = max(v for v in all_ratios if math.isfinite(v))
    else:
        out["ratio_min"] = float("nan")
        out["ratio_max"] = float("nan")

    out["num_iters"] = float(len(rows))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed: List[dict] = []
    for module in MODULES:
        for alabel, aval in ALPHAS:
            for s in SEEDS:
                path = ALPHA_DIR / module / alabel / f"seed{s}" / "metrics.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                stats = summarize_run(path)
                stats.update({
                    "module": module,
                    "alpha": aval,
                    "alpha_label": alabel,
                    "seed": s,
                })
                per_seed.append(stats)

    # Write summary_per_seed.csv
    fields = (
        ["module", "alpha", "alpha_label", "seed"]
        + TRAIN_METRICS
        + EVAL_METRICS
        + ["ratio_min", "ratio_max", "nan_core_rows", "num_iters"]
    )
    with open(OUT / "summary_per_seed.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_seed:
            w.writerow({k: r.get(k, "") for k in fields})

    # Per-cell aggregates
    per_cell: List[dict] = []
    by_cell: Dict[Tuple[str, float], List[dict]] = {}
    for r in per_seed:
        by_cell.setdefault((r["module"], r["alpha"]), []).append(r)
    for (mod, a), rows in sorted(by_cell.items()):
        cell = {"module": mod, "alpha": a, "n_seeds": len(rows)}
        for m in TRAIN_METRICS + EVAL_METRICS:
            vals = [r[m] for r in rows]
            cell[f"{m}__mean"] = _mean_finite(vals)
            cell[f"{m}__std"] = _std_finite(vals)
        per_cell.append(cell)

    pc_fields = ["module", "alpha", "n_seeds"] + [
        f"{m}__{s}" for m in TRAIN_METRICS + EVAL_METRICS for s in ("mean", "std")
    ]
    with open(OUT / "summary_per_cell.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=pc_fields)
        w.writeheader()
        for r in per_cell:
            w.writerow({k: r.get(k, "") for k in pc_fields})

    # Per-module averages across alphas (rough overview)
    per_module: List[dict] = []
    by_mod: Dict[str, List[dict]] = {}
    for r in per_seed:
        by_mod.setdefault(r["module"], []).append(r)
    for mod, rows in sorted(by_mod.items()):
        row = {"module": mod, "n_seeds": len(rows)}
        for m in TRAIN_METRICS + EVAL_METRICS:
            vals = [r[m] for r in rows]
            row[f"{m}__mean"] = _mean_finite(vals)
            row[f"{m}__std"] = _std_finite(vals)
        per_module.append(row)
    pm_fields = ["module", "n_seeds"] + [f"{m}__{s}" for m in TRAIN_METRICS + EVAL_METRICS for s in ("mean", "std")]
    with open(OUT / "summary_per_module.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=pm_fields)
        w.writeheader()
        for r in per_module:
            w.writerow({k: r.get(k, "") for k in pm_fields})

    # Paired bootstrap: attention - attention_zm at each alpha, on eval_capture and eval_pursuer_target_distance.
    rng = np.random.default_rng(20260617)

    def cell_seeds(mod: str, a: float, metric: str) -> np.ndarray:
        return np.array(
            sorted(
                [(r["seed"], r[metric]) for r in per_seed if r["module"] == mod and r["alpha"] == a],
                key=lambda x: x[0],
            ),
            dtype=object,
        )

    boot_rows = []
    for a_label, a in ALPHAS:
        for metric in ["eval_capture", "eval_pursuer_target_distance", "capture_rate", "pursuer_target_distance"]:
            att = cell_seeds("attention", a, metric)[:, 1].astype(float)
            zm = cell_seeds("attention_zm", a, metric)[:, 1].astype(float)
            iden = cell_seeds("identity", a, metric)[:, 1].astype(float)
            # paired delta att - zm, seed-aligned
            delta = att - zm
            # iid bootstrap over seed index
            B = 10_000
            n = len(delta)
            idx = rng.integers(0, n, size=(B, n))
            boot = delta[idx].mean(axis=1)
            ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
            # Also bootstrap att vs identity
            delta_id = att - iden
            boot_id = delta_id[idx].mean(axis=1)
            ci_lo_id, ci_hi_id = np.quantile(boot_id, [0.025, 0.975])
            boot_rows.append({
                "alpha": a,
                "metric": metric,
                "att_mean": float(att.mean()),
                "zm_mean": float(zm.mean()),
                "iden_mean": float(iden.mean()),
                "delta_att_zm_mean": float(delta.mean()),
                "delta_att_zm_ci_lo": float(ci_lo),
                "delta_att_zm_ci_hi": float(ci_hi),
                "delta_att_iden_mean": float(delta_id.mean()),
                "delta_att_iden_ci_lo": float(ci_lo_id),
                "delta_att_iden_ci_hi": float(ci_hi_id),
            })
    with open(OUT / "bootstrap_deltas.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
        w.writeheader()
        for r in boot_rows:
            w.writerow(r)

    # Sanity gates report
    with open(OUT / "sanity_gates.csv", "w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "module",
                "alpha",
                "seed",
                "num_iters",
                "ratio_min",
                "ratio_max",
                "nan_core_rows",
                "explained_variance",
                "eval_capture",
                "capture_rate",
            ],
        )
        w.writeheader()
        for r in per_seed:
            w.writerow({
                "module": r["module"],
                "alpha": r["alpha"],
                "seed": r["seed"],
                "num_iters": int(r["num_iters"]),
                "ratio_min": f"{r['ratio_min']:.6f}",
                "ratio_max": f"{r['ratio_max']:.6f}",
                "nan_core_rows": int(r["nan_core_rows"]),
                "explained_variance": f"{r['explained_variance']:.4f}",
                "eval_capture": f"{r['eval_capture']:.4f}",
                "capture_rate": f"{r['capture_rate']:.4f}",
            })

    # NaN report
    with open(OUT / "nan_report.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=["module", "alpha", "seed", "nan_core_rows"])
        w.writeheader()
        for r in per_seed:
            if int(r["nan_core_rows"]) > 0:
                w.writerow({k: r[k] for k in ["module", "alpha", "seed", "nan_core_rows"]})

    print("Aggregation complete.  Wrote:")
    for name in [
        "summary_per_seed.csv",
        "summary_per_cell.csv",
        "summary_per_module.csv",
        "bootstrap_deltas.csv",
        "sanity_gates.csv",
        "nan_report.csv",
    ]:
        p = OUT / name
        print(f"  {p}  ({p.stat().st_size} B)")


if __name__ == "__main__":
    main()
