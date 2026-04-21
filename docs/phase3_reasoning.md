---
name: Phase 3 Reasoning
description: Decision on value clipping removal, exp_var computation, and advantage normalization floor
type: project
---

# Phase 3 Reasoning: Critic Correctness

## 1. Value Clipping Decision — Remove Entirely

**Recommendation:** Remove value clipping from `baseline/trainer.py` and `gnn/trainer_multienv.py`. Replace with simple MSE. (`HetGAT/mahac.py` already uses simple MSE.)

**Mechanistic justification.** Value clipping was introduced as a heuristic analog to the policy-ratio clip to prevent large value-function updates. However, Engstrom et al. (2020) demonstrate that value clipping provides no consistent benefit and actively harms learning when returns are not normalized. This codebase explicitly does NOT normalize returns (confirmed: no `RunningMeanStd` exists anywhere). With unnormalized returns spanning several units (once Phase 1 dones are correct and episodes run full 80 steps), the `clip_eps=0.2` threshold severely throttles critic learning speed.

**Alignment across algorithms.** `HetGAT/mahac.py` already uses simple MSE. Bringing baseline and GNN-MAPPO into alignment eliminates an implementation asymmetry that could confound the ablation comparison. For thesis reviewers, uniformity of critic implementation is essential to isolating the architectural variable.

**CleanRL precedent.** CleanRL's canonical `ppo_continuous_action.py` uses simple MSE for value loss. This codebase is benchmarked against CleanRL conventions elsewhere (per-rollout advantage normalization, GAE structure).

## 2. exp_var Computation — Once Per Iteration, Pre-Update

**Recommendation:** Compute `exp_var = 1 - var(returns - values_before_update) / (var(returns) + 1e-8)` once per iteration on the full rollout, using buffer-stored values (pre-update), outside the critic training loop.

**Why per-minibatch is wrong.** Explained variance is a ratio of variances, and the mean of ratios is not equal to the ratio of means. A single minibatch with low return variance can produce `exp_var = -99`, dragging the average to a deeply negative number even when the critic fits the overall distribution well. Additionally, computing inside the update loop mixes pre-update and mid-update evaluations — the metric becomes uninterpretable.

**Where `values_before_update` comes from.** After `compute_advantages()` but before the critic loop, `self.buffer.values` contains the critic's pre-update predictions. For `baseline` and `gnn`, these are stored as a list of `(E, N)` tensors per timestep; flatten and concatenate to get the full `(T*E*N,)` tensor. Compare against `self.buffer.returns` (same shape, computed by GAE).

`mahac.py` already implements this correctly. No change needed there.

## 3. Advantage Normalization Floor — clamp(min=1e-4)

**Recommendation:** Replace `adv_std + 1e-8` with `adv_std.clamp(min=1e-4)` at all three normalization sites.

When `advantages.std() = 1e-6`, the current `+ 1e-8` provides essentially no protection: `1e-6 + 1e-8 ≈ 1e-6`, and normalized advantages blow up to order `1e6`. With `.clamp(min=1e-4)`, maximum amplification is `1/1e-4 = 10000` — large but bounded enough to prevent gradient explosions.

**This is defense-in-depth, not a root-cause fix.** The root cause was Bug #1 (stale dones), fixed in Phase 1. After Phase 1, advantage stds should be in `[0.1, 1.0]` for healthy training and the floor should never activate.

Three sites to fix:
- `baseline/rollout_buffer.py::get_batches`
- `gnn/batch_rollout.py::get_batches` (verify same pattern)
- `HetGAT/mahac.py` lines 110-111 (inline normalization)

## 4. exp_var Invariants Post-Phase-3

- `[0, 1]` → critic explains that fraction of return variance; above 0.8 is well-fit
- `[-1, 0]` in first 5-10 iters → acceptable; critic is untrained
- `< -5` after iter 10 → signals a real problem (near-constant returns, or critic is anti-correlated)
- Oscillating `[0.3, 0.7]` at steady state → normal (non-stationary returns as policy improves)

The key improvement: `exp_var` is now a single, interpretable number per iteration, directly comparable across IPPO, GNN-MAPPO, and HetNet.

## Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Value clipping | Remove entirely | CleanRL standard; no return normalization; aligns mahac.py |
| `exp_var` location | Once per iteration, pre-update | Statistically meaningful; matches mahac.py and SpinUp |
| Advantage std floor | `.clamp(min=1e-4)` | Defense-in-depth; bounds amplification |
| Return normalization | Do not add | Already decided |
