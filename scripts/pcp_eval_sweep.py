"""
PCP large-sample deterministic eval sweep.

The training-time eval uses render.episodes (=2) single-env episodes, so eval_capture
is a tiny, high-variance estimate of a rare event -> erratic across cells. This
re-evaluates each cell/seed checkpoint with MANY deterministic episodes (default 100)
by reusing the trainer's own _evaluate_deterministic_policy, then reports per-cell
eval_capture (seed-avg) + paired-bootstrap deltas (live - _zm, live - identity).

Reuses checkpoints in place (run on the cluster where final.pt live). Forces CPU and
maps the load to CPU so GPU-trained ckpts work on a CPU node.

Usage:
    python scripts/pcp_eval_sweep.py runs/pcp_smoke_v2_XXXX \
        --episodes 100 --out-csv runs/pcp_smoke_v2_XXXX/eval_results.csv
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/pcp_eval_sweep.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from backbone.utils import load_yaml_config
from backbone import MAPPO
from experiments.train_vmas_mappo import _build_comm_module, _evaluate_deterministic_policy

CELLS = ["identity", "broadcast", "broadcast_zm", "attention", "attention_zm"]
SEEDS = [0, 1, 2]


def load_model(ckpt: str, device_str: str):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    # Force probe/eval device: GPU-trained ckpts save device="cuda".
    mcfg = dict(ck["config"])
    mcfg["device"] = device_str
    run_cfg = load_yaml_config(str(Path(ckpt).parent.parent / "config.yaml"))
    comm = _build_comm_module(run_cfg.get("comm", {}), run_cfg.get("model", {}))
    model = MAPPO(**mcfg, comm_module=comm)
    model.load(ckpt, map_location=torch.device(device_str))
    model.eval()
    return model, run_cfg


def eval_one(ckpt: str, episodes: int, device_str: str) -> dict:
    model, run_cfg = load_model(ckpt, device_str)
    cfg = dict(run_cfg)
    cfg["render"] = dict(cfg.get("render", {}))
    cfg["render"]["episodes"] = episodes   # large-sample override
    cfg["render"]["mode"] = "off"
    return _evaluate_deterministic_policy(model, cfg, torch.device(device_str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt-name", default="final.pt")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    per = {c: [] for c in CELLS}
    rows = []

    for c in CELLS:
        for s in SEEDS:
            ckpt = run_dir / c / f"seed{s}" / "ckpt" / args.ckpt_name
            if not ckpt.exists():
                rows.append(dict(cell=c, seed=s, eval_capture="", eval_dist="", eval_return="", note="MISSING_CKPT"))
                continue
            try:
                m = eval_one(str(ckpt), args.episodes, args.device)
            except Exception as e:
                sys.stderr.write(f"[err] {ckpt}: {type(e).__name__}: {e}\n")
                rows.append(dict(cell=c, seed=s, eval_capture="", eval_dist="", eval_return="", note=f"ERR:{type(e).__name__}"))
                continue
            ec = float(m.get("eval_capture", float("nan")))
            per[c].append(ec)
            rows.append(dict(cell=c, seed=s, eval_capture=ec,
                             eval_dist=m.get("eval_pursuer_target_distance", ""),
                             eval_return=m.get("eval_mean_return", ""), note="ok"))
            print(f"  {c}/seed{s}: eval_capture={ec:.4f} (episodes={args.episodes})")

    def mean(xs):
        xs = [x for x in xs if isinstance(x, float) and not math.isnan(x)]
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\n=== per-cell eval_capture (deterministic, {args.episodes} episodes) ===")
    for c in CELLS:
        print(f"  {c:<14} {mean(per[c]):.4f}  n={len(per[c])}  seeds={[round(x,4) for x in per[c]]}")

    random.seed(0)
    def boot(live, ctrl, n=20000):
        a, b = per[live], per[ctrl]
        if not a or not b:
            return float("nan"), float("nan"), float("nan")
        pairs = list(zip(a, b))
        ds = []
        for _ in range(n):
            samp = [random.choice(pairs) for _ in pairs]
            ds.append(mean([x - y for x, y in samp]))
        ds.sort()
        return mean([x - y for x, y in pairs]), ds[int(0.025 * n)], ds[int(0.975 * n)]

    print("\n=== paired bootstrap deltas (eval_capture) ===")
    for live, ctrl in [("broadcast", "broadcast_zm"), ("attention", "attention_zm"),
                       ("broadcast", "identity"), ("attention", "identity")]:
        dd, lo, hi = boot(live, ctrl)
        tag = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
        print(f"  {live:<10} - {ctrl:<13} delta={dd:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] {tag}")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cell", "seed", "eval_capture", "eval_dist", "eval_return", "note"])
            w.writeheader()
            w.writerows(rows)
        print(f"\n[eval-sweep] wrote {out}")


if __name__ == "__main__":
    main()
