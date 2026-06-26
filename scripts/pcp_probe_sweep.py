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


def discover_alphas(run_dir: Path):
    """
    Detect an alpha-continuum layout: subdirs named alpha<label> (e.g. alpha0p5) that
    contain cell/seed dirs. Returns sorted [(label, value, path)] by value, or [] for
    a flat layout.
    """
    out = []
    for p in run_dir.iterdir():
        if not (p.is_dir() and p.name.startswith("alpha")):
            continue
        if not discover_cells(p):
            continue
        try:
            value = float(p.name[len("alpha"):].replace("p", "."))
        except ValueError:
            continue
        out.append((p.name, value, p))
    return sorted(out, key=lambda t: t[1])


def run_one(config: str, ckpt: Path, probe_seed: int, alpha=None):
    cmd = [
        sys.executable, "-m", "experiments.run_probe",
        "--config", config, "--probe-mode", "post",
        "--ckpt", str(ckpt), "--seed", str(probe_seed),
    ]
    # Build the probe env at the ckpt's TRAINED capturer alpha (so obs masking matches).
    if alpha is not None:
        cmd += ["--alpha", str(alpha)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    m = LINE_RE.search(out.stdout)
    if not m:
        sys.stderr.write(f"[warn] no probe line for {ckpt}\n{out.stdout[-500:]}\n{out.stderr[-500:]}\n")
        return None
    post, raw, ceil, branch = m.groups()
    return dict(post=float(post), raw=float(raw), ceil=float(ceil), branch=branch)


def _avg(xs):
    return mean(xs) if xs else float("nan")


def probe_root(root: Path, config: str, probe_seed: int, ckpt_name: str, alpha=None):
    """Probe every cell/seed under one cell-root. Returns (cells, per_cell, rows)."""
    cells = discover_cells(root)
    per_cell = {c: {"post": [], "raw": [], "ceil": [], "branch": [], "missing": []} for c in cells}
    rows = []
    for cell in cells:
        for s in discover_seeds(root, cell):
            ckpt = root / cell / f"seed{s}" / "ckpt" / ckpt_name
            if not ckpt.exists():
                per_cell[cell]["missing"].append(s)
                rows.append(dict(cell=cell, seed=s, r2_post="", r2_raw="", r2_ceil="", branch="MISSING_CKPT"))
                continue
            r = run_one(config, ckpt, probe_seed, alpha=alpha)
            if r is None:
                per_cell[cell]["missing"].append(s)
                rows.append(dict(cell=cell, seed=s, r2_post="", r2_raw="", r2_ceil="", branch="PARSE_FAIL"))
                continue
            for k in ("post", "raw", "ceil", "branch"):
                per_cell[cell][k].append(r[k])
            rows.append(dict(cell=cell, seed=s, r2_post=r["post"], r2_raw=r["raw"],
                             r2_ceil=r["ceil"], branch=r["branch"]))
            print(f"  {cell}/seed{s}: R2_post={r['post']:.4f} R2_raw={r['raw']:.4f} "
                  f"ceil={r['ceil']:.4f} branch={r['branch']}")
    return cells, per_cell, rows


def summarize_probe(cells, per_cell, header=""):
    """Print per-cell R2 + live-zm deltas. Returns {(live,zm): delta_post}."""
    print(f"\n=== per-cell R2 (seed-averaged){header} ===")
    print(f"{'cell':<14} {'R2_post':>9} {'R2_raw':>9} {'ceil':>9} {'n':>3}")
    for cell in cells:
        d = per_cell[cell]
        print(f"{cell:<14} {_avg(d['post']):>9.4f} {_avg(d['raw']):>9.4f} {_avg(d['ceil']):>9.4f} {len(d['post']):>3}")
    res = {}
    for live, ctrl in build_contrasts(cells):
        if not ctrl.endswith("_zm"):
            continue
        lv, zv = _avg(per_cell[live]["post"]), _avg(per_cell[ctrl]["post"])
        res[(live, ctrl)] = lv - zv
        print(f"  delta R2_post  {live:<10} - {ctrl:<13} = {lv - zv:+.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="run dir: flat <cell>/seed<s>/ or alpha<label>/<cell>/seed<s>/")
    ap.add_argument("--config", default="configs/pcp_smoke_v2.yaml")
    ap.add_argument("--probe-seed", type=int, default=0,
                    help="Fixed probe rollout seed for cross-cell comparability.")
    ap.add_argument("--ckpt-name", default="final.pt")
    ap.add_argument("--out-csv", default=None,
                    help="Optional path to write per-(alpha,cell,seed) rows for sync-back.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    alphas = discover_alphas(run_dir)
    all_rows = []

    if alphas:
        # alpha-continuum layout: each cell probed at its TRAINED alpha; curve at the end.
        curve = {}  # (live, ctrl) -> [(alpha, delta_post)]
        for label, val, path in alphas:
            print(f"\n########## alpha={val} ({label}) ##########")
            cells, per_cell, rows = probe_root(path, args.config, args.probe_seed, args.ckpt_name, alpha=val)
            for r in rows:
                r["alpha"] = val
            all_rows += rows
            res = summarize_probe(cells, per_cell, header=f" — alpha={val}")
            for k, v in res.items():
                curve.setdefault(k, []).append((val, v))
        print("\n=== delta R2_post(live - _zm) vs alpha (comm's marginal info; should decay toward alpha=1) ===")
        for (live, ctrl), pts in curve.items():
            s = "  ".join(f"a{a:g}:{d:+.4f}" for a, d in pts)
            print(f"  {live}-{ctrl:<13} {s}")
    else:
        cells, per_cell, rows = probe_root(run_dir, args.config, args.probe_seed, args.ckpt_name)
        if not cells:
            sys.stderr.write(f"[err] no cell/seed dirs found under {run_dir}\n")
            return
        for r in rows:
            r["alpha"] = ""
        all_rows = rows
        summarize_probe(cells, per_cell)
        print("\nVerdict guide: channel CARRIES the signal if live R2_post is high AND")
        print("delta(live - _zm) is large AND _zm/identity R2_post collapse to ~R2_raw.")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["alpha", "cell", "seed", "r2_post", "r2_raw", "r2_ceil", "branch"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[probe-sweep] wrote per-run rows to {out}")


if __name__ == "__main__":
    main()
