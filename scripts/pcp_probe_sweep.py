"""
PCP post-comm probe sweep driver.

Runs the post-comm predictability probe (experiments/run_probe.py --probe-mode post)
on every cell/seed checkpoint of a PCP smoke-sweep run dir, then tabulates the
decisive contrast:

    R2_c1_post   capturer POST-COMM hidden -> true prey position   (channel signal)
    R2_c1_raw    capturer RAW obs          -> true prey position   (should be ~chance)
    R2_c0_local  detector RAW obs          -> true prey position   (ceiling)

Interpretation:
    - Live comm (attention/broadcast): R2_c1_post should be HIGH.
    - _zm twin: R2_c1_post should collapse to ~chance (messages zeroed).
      If a _zm twin's R2_c1_post stays HIGH, prey info reaches capturers through
      something other than the message channel -> a leak/bug, not a result.
    - identity: R2_c1_post ~ R2_c1_raw (no neighbor pooling) -> implicit no-channel control.

NOTE on architectural baseline: attention/broadcast pool neighbor embeddings, and
detector embeddings carry prey info, so R2_post can be high even at near-init. The
load-bearing number is therefore the DELTA R2_post(live) - R2_post(_zm), not the
absolute live value.

Usage:
    python scripts/pcp_probe_sweep.py runs/pcp_smoke_20260619_113708 \
        --config configs/pcp_smoke.yaml --probe-seed 0
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean

LINE_RE = re.compile(
    r"R2_c1_post=([-\d.]+)\s+R2_c1_raw=([-\d.]+)\s+R2_c0_local\(ceil\)=([-\d.]+)\s+branch=(\w+)"
)


def discover_cells(run_dir: Path):
    """Cells = subdirs containing seed* dirs. Auto-handles 5 or 7 cells, any seed count."""
    return sorted(
        p.name for p in run_dir.iterdir()
        if p.is_dir() and any(d.is_dir() and d.name.startswith("seed") for d in p.iterdir())
    )


def discover_seeds(run_dir: Path, cell: str):
    return sorted(
        int(d.name[4:]) for d in (run_dir / cell).iterdir()
        if d.is_dir() and d.name.startswith("seed") and d.name[4:].isdigit()
    )


def build_contrasts(cells):
    """(live, control) pairs: each comm module vs its _zm twin and vs identity."""
    comm_mods = [c for c in cells if c != "identity" and not c.endswith("_zm")]
    out = []
    for m in comm_mods:
        if f"{m}_zm" in cells:
            out.append((m, f"{m}_zm"))
        if "identity" in cells:
            out.append((m, "identity"))
    return out


def run_one(config: str, ckpt: Path, probe_seed: int):
    cmd = [
        sys.executable, "-m", "experiments.run_probe",
        "--config", config, "--probe-mode", "post",
        "--ckpt", str(ckpt), "--seed", str(probe_seed),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    m = LINE_RE.search(out.stdout)
    if not m:
        sys.stderr.write(f"[warn] no probe line for {ckpt}\n{out.stdout[-500:]}\n{out.stderr[-500:]}\n")
        return None
    post, raw, ceil, branch = m.groups()
    return dict(post=float(post), raw=float(raw), ceil=float(ceil), branch=branch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="PCP smoke-sweep run dir (contains <cell>/seed<s>/ckpt/final.pt)")
    ap.add_argument("--config", default="configs/pcp_smoke.yaml")
    ap.add_argument("--probe-seed", type=int, default=0,
                    help="Fixed probe rollout seed for cross-cell comparability.")
    ap.add_argument("--ckpt-name", default="final.pt")
    ap.add_argument("--out-csv", default=None,
                    help="Optional path to write per-(cell,seed) rows for sync-back.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cells = discover_cells(run_dir)
    if not cells:
        sys.stderr.write(f"[err] no cell/seed dirs found under {run_dir}\n")
        return
    per_cell = {c: {"post": [], "raw": [], "ceil": [], "branch": [], "missing": []} for c in cells}
    rows = []  # per-(cell,seed) for CSV

    for cell in cells:
        for s in discover_seeds(run_dir, cell):
            ckpt = run_dir / cell / f"seed{s}" / "ckpt" / args.ckpt_name
            if not ckpt.exists():
                per_cell[cell]["missing"].append(s)
                rows.append(dict(cell=cell, seed=s, r2_post="", r2_raw="", r2_ceil="", branch="MISSING_CKPT"))
                continue
            r = run_one(args.config, ckpt, args.probe_seed)
            if r is None:
                per_cell[cell]["missing"].append(s)
                rows.append(dict(cell=cell, seed=s, r2_post="", r2_raw="", r2_ceil="", branch="PARSE_FAIL"))
                continue
            per_cell[cell]["post"].append(r["post"])
            per_cell[cell]["raw"].append(r["raw"])
            per_cell[cell]["ceil"].append(r["ceil"])
            per_cell[cell]["branch"].append(r["branch"])
            rows.append(dict(cell=cell, seed=s, r2_post=r["post"], r2_raw=r["raw"],
                             r2_ceil=r["ceil"], branch=r["branch"]))
            print(f"  {cell}/seed{s}: R2_post={r['post']:.4f} R2_raw={r['raw']:.4f} "
                  f"ceil={r['ceil']:.4f} branch={r['branch']}")

    def avg(xs):
        return mean(xs) if xs else float("nan")

    print("\n=== per-cell (seed-averaged) ===")
    print(f"{'cell':<14} {'R2_post':>9} {'R2_raw':>9} {'ceil':>9} {'n':>3} {'missing':>10}")
    for cell in cells:
        d = per_cell[cell]
        print(f"{cell:<14} {avg(d['post']):>9.4f} {avg(d['raw']):>9.4f} {avg(d['ceil']):>9.4f} "
              f"{len(d['post']):>3} {str(d['missing']):>10}")

    # Decisive deltas: live - zm (and a note vs identity)
    print("\n=== decisive contrast: R2_post(live) - R2_post(_zm) ===")
    for live, ctrl in build_contrasts(cells):
        if not ctrl.endswith("_zm"):
            continue
        lv, zv = avg(per_cell[live]["post"]), avg(per_cell[ctrl]["post"])
        print(f"  {live:<10} {lv:.4f}  -  {ctrl:<13} {zv:.4f}  =  delta {lv - zv:+.4f}")
    if "identity" in per_cell:
        print(f"  identity (no-channel control) R2_post = {avg(per_cell['identity']['post']):.4f}")
    print("\nVerdict guide: channel CARRIES the signal if live R2_post is high AND")
    print("delta(live - _zm) is large AND _zm/identity R2_post collapse to ~R2_raw.")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cell", "seed", "r2_post", "r2_raw", "r2_ceil", "branch"])
            w.writeheader()
            w.writerows(rows)
        print(f"\n[probe-sweep] wrote per-run rows to {out}")


if __name__ == "__main__":
    main()
