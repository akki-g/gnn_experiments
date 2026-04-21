---
name: Phase 2 Reasoning
description: Root cause analysis of r_s=1.000 PPO ratio and log_std labeling fix
type: project
---

# Phase 2 Reasoning: PPO Ratio and log_std Labeling

## 1. Root Cause of r_s = r_i = 1.000

The Phase 2 Researcher confirmed the ratio computation in `mahac.py` is **structurally correct**:
- `old_log_probs_s = buffer.scout_log_probs.detach()` — reads from buffer (correct, no-grad)
- `new_lp_s = scout_dist.log_prob(flat_actions_s).sum(dim=-1)` — evaluates stored actions under current policy with grad (correct)
- `ratio_s = torch.exp(new_lp_s - flat_old_lp_s)` — standard importance sampling ratio

The ratio reads 1.000 because **the policy gradient signal is near-zero**, not because the ratio is structurally forced to 1.0. The causal chain: Phase 1 done-signal bug → 69% timesteps `done=True` → GAE bootstrap zeroed → advantages near-zero variance → normalized advantages are noise → PPO surrogate loss ≈ 0 → optimizer barely moves params → ratio stays ≈ 1.000.

**Key evidence:** `log_std` saturates at clamp ceiling 0.5, proving the entropy gradient IS reaching the parameters. If ratio=1.000 were caused by a computational-graph detachment bug, entropy would also be zero. The pattern of nonzero entropy gradient but near-zero policy gradient is exactly the signature of zero policy gradient with functional entropy gradient.

**Conclusion:** No code change needed to the ratio computation. The r=1.000 symptom will resolve when Phase 1 delivers non-degenerate advantages.

## 2. Log_std Labeling (Bug #3)

The metrics dictionary in `mahac.py` correctly uses keys `log_std_scout` and `log_std_interc`. The plot in `HetGAT/train.py::plot_metrics` correctly labels the y-axis as `log_std`.

**However, print statements mislabel the value:**
- `HetGAT/train.py` (train loop print): `f"std_s={metrics['log_std_scout']:.3f}"` → should be `log_std_s=`
- `guided_coverage_train.py` (print): `f"std_s={update_metrics['log_std_scout']:.2f}"` → should be `log_std_s=`

Fix: change `std_s=` to `log_std_s=` and `std_i=` to `log_std_i=` at these two print sites. Cosmetic only, but important for thesis reproducibility — a reader should not guess whether the value is std (>0) or log_std (can be negative).

## 3. lp_diff Diagnostic

Add a `lp_diff` metric to `mahac.py::update`. After computing `new_lp_s` and `flat_old_lp_s` for each PPO epoch:

```python
lp_diff_s = (new_lp_s - flat_old_lp_s).abs().mean().item()
```

Log this per epoch. **Invariant:** At epoch 0 (before any optimizer step), `lp_diff_s` must be exactly 0.0 (within ~1e-6 float32 tolerance). At epoch 1+, it must be nonzero. If epoch-0 lp_diff ever exceeds 1e-4, there is a hidden numerical issue in hidden-state reconstruction or buffer integrity.

Include `lp_diff_epoch0_s` and `lp_diff_final_s` in the returned metrics dict for monitoring.

## 4. Dual Clamp on log_std (documentation only, no code change)

Two clamp operations exist:
1. **Inner (non-in-place)** in `hetnetPolicy.py::forward`: `log_std.clamp(min=-5.0, max=2.0)` — affects only the computed distribution, not the parameter. Gradient flows through.
2. **Outer (in-place)** in `mahac.py` post-optimizer: `log_std_scout.clamp_(min=-1.5, max=0.5)` — hard-resets the parameter.

The outer clamp at `[-1.5, 0.5]` is strictly tighter, making the inner clamp redundant. `hetnetPolicy.py` is **out of scope for Phase 2** — document this duality for Phase 4, which can decide whether to remove the inner clamp or widen the outer bounds.

## Summary of Phase 2 Actions

| Item | Action | File |
|------|--------|------|
| PPO ratio code | No change — structurally correct | `mahac.py` |
| log_std labeling | Change `std_s=` → `log_std_s=` at 2 print sites | `HetGAT/train.py`, `guided_coverage_train.py` |
| lp_diff diagnostic | Add epoch-0 and final-epoch lp_diff metrics | `mahac.py::update` |
| Dual clamp | Document only; no code change (out of scope) | N/A |
