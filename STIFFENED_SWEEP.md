# Stiffened pursuit mini-sweep (comm-necessity sanity check)

A 30-run mini-sweep designed to test ONE question:

> Under the homogeneous pursuit env, can we make any communication module
> show a positive Δ vs its zero-message twin by removing the three knobs
> that made messages redundant in `runs/pursuit_20260608_220631`?

If the answer is no, homogeneous-pursuit is structurally a dead end for
the dissociation thesis question and the next move is
predator-capture-prey (two non-overlapping roles → comm becomes a
precondition for non-trivial reward).

## What changes vs the big sweep

| Knob              | `pursuit_base.yaml` (175-run sweep) | `pursuit_stiffened.yaml` (this sweep) |
| ----------------- | ----------------------------------- | -------------------------------------- |
| `num_full_sight`  | 3                                   | **0**                                  |
| `capture_k`       | 4                                   | **6**                                  |
| `dist_coef`       | 1.0                                 | **0.1**                                |
| everything else   | —                                   | identical                              |

Why these three:

- `num_full_sight=3` always let three pursuers see all targets, so a
  working policy = sighted three converge + any blind one stumbles in
  → comm never has to fire.
- `capture_k=4` is loose enough that the sighted three + 1 random
  blind agent solves the task.
- `dist_coef=1.0` made the dense distance reward an implicit broadcast
  ("the swarm is closer/farther from a target") so even blind agents
  got a centralized gradient for free.

## Grid

3 configs × 2 alphas × 5 seeds = **30 runs**.

| Module          | zero_messages | alpha    | seeds |
| --------------- | ------------- | -------- | ----- |
| `identity`      | n/a (control) | 0.25, 0.5 | 0..4 |
| `attention`     | false         | 0.25, 0.5 | 0..4 |
| `attention`     | true (`_zm`)  | 0.25, 0.5 | 0..4 |

We pick `attention` (not broadcast or graph) because:
- `broadcast` was net-neutral in the big sweep; nothing about
  stiffening should change that ranking enough to be informative.
- `graph` collapsed in the big sweep (Δ ≈ −0.23 vs `graph_zm`,
  oversmoothing signature) and stiffening the task does not address
  the architectural problem.
- `attention` is the cheapest module that could plausibly route
  information selectively if there were information to route.

## Submit

```bash
cd ~/gnn_experiments
sbatch scripts/slurm/train_pursuit_stiffened_sweep.sbatch
```

The script forces `--array=0-29`. Do NOT override `ALPHAS`,
`COMM_MODULES`, or `NUM_SEEDS` from the environment — those are fixed
in this sbatch (unlike `train_pursuit_comm_sweep.sbatch`, which is
overridable). The generation block asserts `capture_k=6`,
`num_full_sight=0`, `dist_coef=0.1` after merging — if any of those
get clobbered, the job fails loud.

## Output

```
runs/pursuit_stiffened_<YYYYMMDD_HHMMSS>/alpha/
  identity/alpha0p25/seed{0..4}/metrics.csv
  identity/alpha0p5/seed{0..4}/metrics.csv
  attention/alpha0p25/seed{0..4}/metrics.csv
  attention/alpha0p5/seed{0..4}/metrics.csv
  attention_zm/alpha0p25/seed{0..4}/metrics.csv
  attention_zm/alpha0p5/seed{0..4}/metrics.csv
```

Each `metrics.csv` has the same column set as the big sweep, including
the dissociation probe columns:

- `capture_c1_participation`, `marginal_c1_rate`, `knn_c1_sighted_frac`
- `dist_full_sight` — **NaN by design** in this sweep
  (`num_full_sight=0` means there are zero class-0 agents)
- `dist_sensor_limited` — populated normally

The c0/c1 split-based probes degenerate when `num_full_sight=0`
(every agent is class 1). That's expected — the meaningful probe in
this regime is the dissociation Δ between `attention` and
`attention_zm`, not the c0/c1 split.

## Decision criterion

After the sweep finishes (~12 hours wall on the cluster, 30 × ~2M steps),
compute per-cell mean `eval_capture` over the final 10 % of iterations:

```
delta_alpha025 = mean(attention | alpha=0.25) - mean(attention_zm | alpha=0.25)
delta_alpha050 = mean(attention | alpha=0.50) - mean(attention_zm | alpha=0.50)
```

with paired-bootstrap CIs over the 5 seeds.

**Kill criterion (pivot to PCP):** both deltas' 95 % CIs cross zero.
That replicates the big-sweep result under stiffer task knobs and
demonstrates that homogeneous-pursuit cannot be made
comm-necessary by knob tuning alone.

**Continue criterion (keep homogeneous-pursuit):** at least one
`Δ_alpha` is strictly positive with a CI excluding zero, AND
`attention > identity` on the same alpha. Then we can sweep wider on
the stiffened config.

**Ambiguous (the boring case):** noisy deltas with overlapping CIs but
no clear sign reversal. Re-run with 10 seeds per cell at the more
informative alpha before pivoting.

## Sanity gates (must pass before reading deltas)

Same gates the big-sweep used:

- `mean_ratio_epoch0 == 1.0` in 30/30 runs (PPO old-logprob bookkeeping).
- 0 NaNs in `train_*` and `eval_*` columns.
- `explained_variance` ≥ 0.85 by the final 10 % of training.
- `eval_capture` at `alpha=0.5` for `identity` should be **strictly
  less than** the 0.426 ceiling we saw in the big sweep. If it's not,
  the stiffening didn't take and we have a config bug, not a result.
