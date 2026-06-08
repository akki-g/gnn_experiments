# Phase E / F Runbook — Communication-Necessity Dissociation

Code for Phases A-D is implemented, reviewed (APPROVE), and integration-tested
(end-to-end train + post-comm probe; ratio r0=1.0 in the live loop; 170 tests green).
This runbook covers the compute-heavy run steps on the SLURM cluster.

Env knobs now in effect: `capture_k=4` (= num_full_sight+1, minimal strict necessity gap),
unified OR-symmetrized k-NN comm topology (`knn_k=4`), `zero_messages` ablation control,
and a `--probe-mode post` post-comm target predictor. All gated runs save checkpoints
and a sibling `config.yaml` (so the probe can rebuild the comm module).

The Phase E gated runs reuse the tested sweep sbatch with environment overrides, so they
are config-identical to the full sweep. SLURM exports the submitting shell environment by
default, so the inline `VAR=val ... sbatch` prefix form works.

---

## Phase E — gated single runs (decide before any sweep)

Single seed (seed 0) is intentional here: these are cheap gates, not the statistical claim.

### E1. Learnability anchor — identity @ alpha=1.0 (must clear the easy end)
```bash
cd ~/gnn_experiments
COMM_MODULES=identity ALPHAS=1.0 NUM_SEEDS=1 \
  sbatch --array=0-0 scripts/slurm/train_pursuit_comm_sweep.sbatch
```
GATE: final `capture` and `eval_capture` > 0.3. If this fails, the capture_k=4 gap broke the
easy end and nothing downstream is interpretable. Also note `capture_c1_participation`.

### E2. Dissociation candidates — identity + 3 comm modules @ alpha=0.0
```bash
COMM_MODULES="identity broadcast attention graph" ALPHAS=0.0 NUM_SEEDS=1 \
  sbatch --array=0-3 scripts/slurm/train_pursuit_comm_sweep.sbatch
```
Task map: 0=identity, 1=broadcast, 2=attention, 3=graph (all alpha=0, seed0).

GATE (dissociation): at least one comm module `capture` > 0.15 while identity stays near floor.
Confirm the mechanism, not just the number:
- `capture_set_has_sensor_limited` > 0 for the winning comm module (blind agents are in the ball).
- `dist_sensor_limited` falls over training for comm but not identity.
- `knn_c1_sighted_frac` ~ 0.9+ (blind agents are actually wired to sighted ones).

CHANCE-CAPTURE GUARD: read identity's `capture_c1_participation`. If it is > ~0.05, capture_k=4
is too chancy (a lucky blind agent floors the contrast). Fix by REDUCING `capture_radius`
(e.g. 0.5 -> 0.4), NOT by restoring capture_k=5.

### E3. Post-comm probe (folds in the Phase A diagnostic; run on the winning comm checkpoint)
Find the sweep-stamped run dir under `runs/pursuit_<DT>/alpha/<module>/alpha0p0/seed0/`, then:
```bash
python -m experiments.run_probe \
  --config configs/pursuit_base.yaml \
  --ckpt runs/pursuit_<DT>/alpha/broadcast/alpha0p0/seed0/ckpt/final.pt \
  --probe-mode post --alpha 0.0 --seed 0
```
Prints `R2_c1_post / R2_c1_raw / R2_c0_local(ceil)` and a branch label:
- HIGH (R2_c1_post >= 0.5*ceiling AND gain >= 0.3): channel carries navigable target info ->
  the floor was a rendezvous problem -> capture_k=4 is the right fix -> proceed to Phase F.
- LOW (R2_c1_post < 0.15 AND gain < 0.1): channel never learned to pass target info ->
  protocol-learning problem -> STOP and escalate (auxiliary target-reconstruction loss or
  larger message dim; this is deliberately NOT built yet).
- INTERMEDIATE: capture_k=4 already applied; only add aux-loss if still intermediate AND
  capture < 0.15 after.

DO NOT launch the sweep if no alpha=0 comm module crosses 0.15. Iterate on these single
runs (capture_radius / knn_k), never on the sweep.

---

## Phase F — sweep + analysis (only after the E gate passes)

### F1. Main sweep: 4 modules x 5 alphas x 5 seeds = 100 tasks
```bash
sbatch --array=0-99 scripts/slurm/train_pursuit_comm_sweep.sbatch
```

### F1b. With the zero_messages ablation arm: 175 tasks (load-bearing control)
```bash
sbatch --array=0-174 --export=ALL,ZERO_MESSAGES_SWEEP=1 \
  scripts/slurm/train_pursuit_comm_sweep.sbatch
```
The zero_messages arm is the control that separates genuine communication from the privileged
CTDE critic's gradient pathway (the single most dangerous confound). If zeroed-broadcast matches
real-broadcast at low alpha, the gain was capacity, not communication.

### F2. Regenerate the comparison figure
```bash
python -m experiments.plot_alpha_sweep runs/pursuit_<DT>/alpha \
  --mode comm --output runs/pursuit_<DT>/alpha/comm_vs_alpha.png
```
Read separation specifically at alpha 0.0 and 0.25 (where capture_k > num_full_sight forces
blind agents in). Report per-seed alpha=0 capture-rate and return gaps (comm vs identity) with
across-seed std, so the edge is reported as statistically real, not merely visual.

### F3. Cross-alpha N_test eval (separates task necessity from policy fragility)
Evaluate every trained policy at alpha=0 regardless of its training alpha. A comm-trained policy
should degrade gracefully when agents go blind at eval; an identity-trained policy should collapse.
There is no turnkey cross-alpha evaluator yet (out of scope for the A-D implementation). Two options
when you reach this step:
- Resume each checkpoint into a short eval-only config with `env.alpha: 0.0` and a small step
  budget (`--resume <ckpt> --config <alpha0-eval-cfg>`), reading `eval_capture`.
- Or have a small `evaluate_at_alpha` helper added (ask and I will build it against the existing
  deterministic-eval path) to produce a clean N_train (training-alpha) vs N_test (eval-at-alpha=0)
  table per module.

---

## Reporting checklist (Section 9 of the work order)
- Class-1 post-comm R2 from E3 and which branch was taken.
- The two gate runs and whether the dissociation criterion was met, plus any capture_k /
  capture_radius change made and why.
- Regenerated figure; low-alpha comm-vs-identity gaps with std; the zero_messages comparison;
  the N_train/N_test table.
- Confirmation nothing out of scope moved and the isolation invariant holds (adjacency positions
  never leaked into local obs).
