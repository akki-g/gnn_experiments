"""Aggregate metrics across all 175 pursuit runs in this sweep.

Layout: alpha/<module>/<alphaXpY>/seed<k>/metrics.csv  (k in 0..4, alpha in {0.0,0.25,0.5,0.75,1.0})
Modules: identity, broadcast, attention, graph, broadcast_zm, attention_zm, graph_zm

Outputs:
- summary_per_seed.csv   (one row per run)
- summary_per_cell.csv   (one row per module x alpha)
- summary_per_module.csv (averaged across alpha)
- nan_report.csv         (rows with NaN in train metrics)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # .../pursuit_20260608_220631
ALPHA_ROOT = ROOT / "alpha"
OUT = Path(__file__).resolve().parent

ALPHA_MAP = {
    "alpha0p0": 0.0,
    "alpha0p25": 0.25,
    "alpha0p5": 0.5,
    "alpha0p75": 0.75,
    "alpha1p0": 1.0,
}
SEED_RE = re.compile(r"seed(\d+)")

# Late-window = last 10% of training (~iter 562..624)
LATE_FRAC = 0.10


def collect() -> pd.DataFrame:
    rows = []
    for module_dir in sorted(ALPHA_ROOT.iterdir()):
        if not module_dir.is_dir():
            continue
        module = module_dir.name
        for alpha_dir in sorted(module_dir.iterdir()):
            if not alpha_dir.is_dir():
                continue
            alpha = ALPHA_MAP[alpha_dir.name]
            for seed_dir in sorted(alpha_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                m = SEED_RE.search(seed_dir.name)
                if not m:
                    continue
                seed = int(m.group(1))
                csv = seed_dir / "metrics.csv"
                if not csv.exists():
                    continue
                df = pd.read_csv(csv)
                n = len(df)
                last_n = max(1, int(n * LATE_FRAC))
                tail = df.iloc[-last_n:]

                # Train metrics
                final_ret = tail["episode_return"].mean()
                final_score = tail["task_score"].mean()
                final_cap = tail["capture_rate"].mean()
                final_dist = tail["pursuer_target_distance"].mean()

                # Eval metrics — drop NaN before averaging (NaN at non-eval iters)
                eval_tail = tail.dropna(subset=["eval_mean_return"])
                eval_ret = eval_tail["eval_mean_return"].mean() if len(eval_tail) else np.nan
                eval_score = eval_tail["eval_task_score"].mean() if len(eval_tail) else np.nan
                eval_cap = eval_tail["eval_capture"].mean() if len(eval_tail) else np.nan
                eval_dist = eval_tail["eval_pursuer_target_distance"].mean() if len(eval_tail) else np.nan

                # Optimization health
                ev = tail["explained_variance"].mean()
                ent_final = tail["entropy"].mean()
                pl_final = tail["policy_loss"].mean()
                vl_final = tail["value_loss"].mean()
                kl_final = tail["approx_kl"].mean()
                clipfrac_final = tail["clipfrac"].mean()
                grad_final = tail["grad_norm"].mean()
                ratio0_max = df["mean_ratio_epoch0"].max()
                ratio0_min = df["mean_ratio_epoch0"].min()

                # NaN sweep across full run
                train_cols = ["episode_return", "policy_loss", "value_loss", "entropy",
                              "total_loss", "grad_norm", "approx_kl", "explained_variance"]
                nans = df[train_cols].isna().sum().sum()

                # Improvement: late - early(window of first 10%)
                head = df.iloc[:last_n]
                early_ret = head["episode_return"].mean()
                delta_ret = final_ret - early_ret

                # Phase-3 / Phase-4 dissociation metrics (post-comm probe)
                cap_c1 = tail["capture_c1_participation"].mean()
                marg_c1 = tail["marginal_c1_rate"].mean()
                knn_sighted = tail["knn_c1_sighted_frac"].mean()
                dist_full = tail["dist_full_sight"].mean()
                dist_sensor = tail["dist_sensor_limited"].mean()

                rows.append(dict(
                    module=module, alpha=alpha, seed=seed,
                    n_iters=n, total_steps=int(df["global_step"].iloc[-1]),
                    train_return_late=final_ret, train_return_early=early_ret,
                    train_return_delta=delta_ret,
                    train_task_score=final_score, train_capture_rate=final_cap,
                    train_pursuer_dist=final_dist,
                    eval_return=eval_ret, eval_task_score=eval_score,
                    eval_capture=eval_cap, eval_pursuer_dist=eval_dist,
                    explained_variance=ev, entropy=ent_final,
                    policy_loss=pl_final, value_loss=vl_final,
                    approx_kl=kl_final, clipfrac=clipfrac_final,
                    grad_norm=grad_final,
                    ratio0_max=ratio0_max, ratio0_min=ratio0_min,
                    nan_count=int(nans),
                    cap_c1_participation=cap_c1, marginal_c1_rate=marg_c1,
                    knn_c1_sighted_frac=knn_sighted,
                    dist_full_sight=dist_full, dist_sensor_limited=dist_sensor,
                ))
    return pd.DataFrame(rows)


def summarize(per_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_cols = [
        "train_return_late", "train_return_delta", "train_task_score",
        "train_capture_rate", "train_pursuer_dist",
        "eval_return", "eval_task_score", "eval_capture", "eval_pursuer_dist",
        "explained_variance", "entropy", "policy_loss", "value_loss",
        "approx_kl", "clipfrac", "grad_norm",
        "cap_c1_participation", "marginal_c1_rate", "knn_c1_sighted_frac",
        "dist_full_sight", "dist_sensor_limited",
    ]
    # Per (module, alpha) cell
    cell = per_seed.groupby(["module", "alpha"])[agg_cols].agg(["mean", "std", "count"])
    # Flatten columns
    cell.columns = [f"{c}_{stat}" for c, stat in cell.columns]
    cell = cell.reset_index()

    # Per module (avg across alpha — note this averages across an asymmetry sweep)
    mod = per_seed.groupby("module")[agg_cols].agg(["mean", "std"])
    mod.columns = [f"{c}_{stat}" for c, stat in mod.columns]
    mod = mod.reset_index()
    return cell, mod


def main():
    per_seed = collect()
    per_seed.to_csv(OUT / "summary_per_seed.csv", index=False)
    print(f"per_seed rows: {len(per_seed)}")

    cell, mod = summarize(per_seed)
    cell.to_csv(OUT / "summary_per_cell.csv", index=False)
    mod.to_csv(OUT / "summary_per_module.csv", index=False)

    # NaN report
    nan_rows = per_seed[per_seed["nan_count"] > 0]
    nan_rows.to_csv(OUT / "nan_report.csv", index=False)
    print(f"runs with NaN in train metrics: {len(nan_rows)}")

    # Console snapshot
    print("\n=== Per-module averages (mean across alpha & seed) ===")
    pretty = per_seed.groupby("module")[
        ["train_return_late", "train_return_delta",
         "train_capture_rate", "train_pursuer_dist",
         "eval_return", "eval_capture", "eval_pursuer_dist",
         "explained_variance", "entropy"]
    ].mean().round(3)
    print(pretty.to_string())

    print("\n=== Capture rate by module x alpha (mean of 5 seeds) ===")
    cap = per_seed.pivot_table(index="module", columns="alpha",
                                values="train_capture_rate", aggfunc="mean").round(4)
    print(cap.to_string())

    print("\n=== Eval pursuer distance by module x alpha ===")
    pdist = per_seed.pivot_table(index="module", columns="alpha",
                                  values="eval_pursuer_dist", aggfunc="mean").round(3)
    print(pdist.to_string())


if __name__ == "__main__":
    main()
