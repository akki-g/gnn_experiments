---
name: Phase 3 Research Memo
description: Analysis of value clipping and exp_var computation in all three trainers
type: project
---

# Phase 3 Research Memo

**Date:** 2026-04-19
**Agent:** Researcher (Phase 3)
**Scope:** Value clipping, exp_var computation, advantage normalization, return normalization — all three trainers.

---

## Q1: `value_loss` computation in `baseline/trainer.py::update`

The value loss uses the clipped form (not simple MSE). The exact code is at lines 217-221:

```python
# baseline/trainer.py lines 217–221
# value loss clipping
values_clipped = flat_old_values + torch.clamp(pred_values - flat_old_values, -self.clip_eps, self.clip_eps)
value_loss_unclipped = F.mse_loss(pred_values, flat_returns, reduction='none')
value_loss_clipped = F.mse_loss(values_clipped, flat_returns, reduction='none')
value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
```

`exp_var` is computed immediately after the loss call, inside the same inner critic loop (lines 223–228):

```python
# baseline/trainer.py lines 223–228
td_error    = flat_returns - pred_values
mean_bellman_error = td_error.abs().mean()

# Explained variance diagnostic
var_returns = flat_returns.var() + 1e-8
exp_var = 1.0 - td_error.var() / var_returns
```

Both `value_loss` and `exp_var` are computed inside the same `for obs, actions, old_log_probs, advantages, returns, old_values in self.buffer.get_batches(B):` loop on the per-minibatch `pred_values`. They are computed on the *same* `pred_values` tensor. `exp_var` is appended to `explained_vars` every minibatch and later averaged via `_safe_mean`.

**Note on `exp_var` placement relative to optimizer step:** `exp_var` is computed BEFORE `self.critic_optim.step()` (line 232). So it measures the critic's fit *before* the parameter update, which is not quite pre-update (pre-epoch) nor post-update — it is per-minibatch-pre-step.

---

## Q2: Math check on exp_var with near-constant returns

If `returns.var() = 0.01` and `(pred_values - returns).var() = 1.0`, then:

```
exp_var = 1 - 1.0 / 0.01 = 1 - 100 = -99
```

**Interpretation:** When returns have very low variance (near-constant across the rollout), the denominator `var(returns)` is tiny. Any non-zero prediction error — even a small absolute error — produces a massive negative `exp_var`. This is not a pathological critic; it is a near-constant target whose explained variance is ill-defined because a trivially constant predictor would score `exp_var = 1.0`. With stale dones causing degenerate GAE (Bug #1), returns collapse to `value[t]` plus per-step TD residuals, producing exactly this low-variance regime. The -99 or similar deeply negative `exp_var` observed in job 585255 is therefore an artifact of Bug #1, not evidence of a broken critic.

**Implication for Phase 3:** Even after Phase 1 fixes dones, `exp_var` computed per-minibatch inside the update loop is still unreliable because:
- Each minibatch may have different return variance, producing inconsistent numbers
- The average over minibatches of a quantity that can be in (-100, 1) is not meaningful

---

## Q3: Advantage normalization in `baseline/rollout_buffer.py::get_batches`

The normalization is per-rollout (full buffer), not per-minibatch. Code at lines 94–97:

```python
# baseline/rollout_buffer.py lines 94–97
# Normalize advantages across the whole buffer
adv_mean = advantages.mean()
adv_std  = advantages.std() + 1e-8
advantages = (advantages - adv_mean) / adv_std
```

This normalization is applied once before the `perm.split(B)` loop, so every minibatch sees the same (globally normalized) advantages. This matches the CleanRL convention.

**There is no floor (clamp min) on `adv_std`.** The only stabilizer is the additive `1e-8`. If `advantages.std()` is extremely small (e.g., `1e-6` from degenerate GAE), the normalized advantages become `(adv - mean) / (1e-6 + 1e-8) ≈ (adv - mean) / 1e-6`. For a typical per-step advantage deviation of `1e-5`, this produces normalized values on the order of `10.0` — large but not catastrophic. However if `adv.std() = 1e-8` or below, normalized values could reach `1e8 / 1e-8 = 1e16`. This is the amplification risk noted in the Phase 3 plan.

**No clamp min is currently present.** The fix is `adv_std = advantages.std().clamp(min=1e-4)`.

---

## Q4: GNN-MAPPO value clipping in `gnn/trainer_multienv.py::update`

The pattern is **identical** to `baseline/trainer.py`. Code at lines 223–227:

```python
# gnn/trainer_multienv.py lines 223–227
# value loss clipping
values_clipped = flat_old_values + torch.clamp(pred_values - flat_old_values, -self.clip_eps, self.clip_eps)
value_loss_unclipped = F.mse_loss(pred_values, flat_returns, reduction='none')
value_loss_clipped = F.mse_loss(values_clipped, flat_returns, reduction='none')
value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
```

`exp_var` in `gnn/trainer_multienv.py` is computed inside the same critic inner loop, also per-minibatch, but with one structural difference: it is wrapped in a `with torch.no_grad():` block (lines 232–234):

```python
# gnn/trainer_multienv.py lines 232–234
with torch.no_grad():
    ev = 1 - (flat_returns - pred_values).var() / (flat_returns.var() + 1e-8)
    explained_vars.append(ev.item())
```

This `with torch.no_grad()` is placed BEFORE `self.critic_optim.step()`, same as baseline. The `exp_var` formula is identical. The averaging behavior (stored per-minibatch, then averaged) is identical.

---

## Q5: Return normalization

**Result: No return normalization is applied anywhere in the active training paths.**

Grep for `ret_rms`, `return_normalizer`, `RunningMeanStd` across all `.py` files returns only two comment lines in `HetGAT/mahac.py`:

```
HetGAT/mahac.py:113:  # FIX-2: do NOT normalize returns — train critic on raw returns (CleanRL/SpinUp standard)
HetGAT/mahac.py:144:  # FIX-2: use raw (unnormalized) returns as critic targets
```

These are comments documenting the decision to *not* normalize returns, not active normalization code. No `RunningMeanStd`, `ret_rms`, or `return_normalizer` object exists anywhere in the codebase.

---

## Summary Table

| Trainer | Value Loss Type | exp_var Location | exp_var Computed | adv_std Floor |
|---------|-----------------|------------------|------------------|---------------|
| `baseline/trainer.py` | Clipped (`torch.max(unclip, clip)`) | Per-minibatch, inside critic loop | Before optimizer step | No (only `+1e-8`) |
| `gnn/trainer_multienv.py` | Clipped (identical pattern) | Per-minibatch, inside critic loop | Before optimizer step | No (only `+1e-8`) |
| `HetGAT/mahac.py` | Simple MSE (FIX-2 already applied) | Per-iteration, after critic epochs | Post-update, `torch.no_grad` block | N/A (HetNet uses its own path) |

**Key finding:** `HetGAT/mahac.py` already has the Phase 3 fix applied (simple MSE, no value clipping, per-iteration `exp_var`). The two trainers that still need Phase 3 fixes are `baseline/trainer.py` and `gnn/trainer_multienv.py`.

---

## Inconsistencies Found

1. **HetNet already fixed, baseline/GNN not.** `mahac.py` already has the simple MSE critic (comment says `# FIX-2+6: simple MSE against raw returns — no value clipping`). `baseline/trainer.py` and `gnn/trainer_multienv.py` still use the clipped form. Phase 3 must bring the two lagging trainers into alignment with mahac.

2. **exp_var averaging semantics differ.** In `baseline` and `gnn`, `exp_var` is the mean over all `(num_critic_epochs × num_minibatches)` values, each computed pre-step. In `mahac`, it is a single computation after all critic epochs complete. The mahac approach is more interpretable. Phase 3 should harmonize the baseline/gnn trainers to match mahac's post-update single-computation pattern.

3. **`exp_var` in baseline is NOT guarded by `torch.no_grad()`.** In `baseline/trainer.py` lines 227–228, `td_error.var()` and `flat_returns.var()` are computed in the active computation graph (no `no_grad` wrapper), whereas `gnn/trainer_multienv.py` wraps this in `torch.no_grad()`. In baseline, this builds a graph for the exp_var scalars that is then discarded — wasteful but not incorrect (since `.item()` is called before backward). However it is an inconsistency between trainers.

---

## Open Questions for Planner/Thinker

1. **Rollout buffer floor vs. trainer floor.** The advantage `std` floor should go in `baseline/rollout_buffer.py::get_batches` (line 96, change `+ 1e-8` to `.clamp(min=1e-4)`). For GNN-MAPPO, it goes in `gnn/batch_rollout.py::get_batches`. Confirm both buffer files use the same pattern before editing.

2. **HetNet's advantage normalization** is done inside `mahac.py::update` directly (lines 110–111), not in a buffer `get_batches`. It already uses `(adv - adv.mean()) / (adv.std() + 1e-8)`. The same floor should be added there: `.clamp(min=1e-4)` in place of `+ 1e-8`.

3. **How to compute exp_var "once per iteration" in baseline/gnn trainers.** The cleanest approach is to compute it in `update()` before the critic training loop, using the buffer's stored `values` (pre-update) against `returns`. This requires reading both from the buffer before `get_batches` shuffles them. Check whether `self.buffer.returns` and `self.buffer.values` are accessible as flat tensors after `compute_advantages()` but before any minibatch loop.

