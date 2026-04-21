---
name: Phase 3 Plan
description: Remove value clipping, move exp_var out of inner loop, add adv_std floor
type: project
---

# Phase 3 — Concrete Diff Plan

Bugs: #6 (value clipping + deceptive exp_var), #10 (advantage normalization amplification)

## Files Changed

- `baseline/trainer.py` — `update()`: remove clipped value loss, move exp_var out of loop
- `baseline/rollout_buffer.py` — `get_batches()`: add adv_std floor
- `gnn/trainer_multienv.py` — `update()`: same as baseline
- `gnn/batch_rollout.py` — `get_batches()`: same floor
- `HetGAT/mahac.py` — advantage normalization lines only (adv_std floor)

## Files NOT Touched

- Any environment, policy, or critic file
- `HetGAT/hetgatLayer.py`, `HetGAT/utils.py`, `HetGAT/rollout.py`
- Actor portions of any trainer

---

## Edit 1: `baseline/trainer.py::update` — remove clipped value loss

**Find:** The `torch.max(value_loss_unclipped, value_loss_clipped).mean()` block.

**Replace with:**
```python
# FIX-P3-B6: simple MSE — value clipping removed (no return normalization; aligns with CleanRL)
value_loss = F.mse_loss(pred_values, flat_returns)
```

## Edit 2: `baseline/trainer.py::update` — move exp_var out of inner loop

**Remove** the per-minibatch exp_var lines from inside the critic loop.
**Add before the minibatch loop** (after compute_advantages, before the for loop):
```python
# FIX-P3-B6: capture pre-update values for post-loop exp_var computation
with torch.no_grad():
    _all_v = self.buffer.get_values_tensor()   # (T*E*N,) pre-update
    _all_r = self.buffer.get_returns_tensor()  # (T*E*N,) GAE returns
    exp_var = (1.0 - (_all_r - _all_v).var() / (_all_r.var() + 1e-8)).item()
```

Then replace `"explained_var": self._safe_mean(explained_vars)` with `"explained_var": exp_var`.

## Edit 3: `baseline/rollout_buffer.py::get_batches` — adv_std floor

**Find:** `adv_std  = advantages.std() + 1e-8`

**Replace with:**
```python
adv_std = advantages.std().clamp(min=1e-4)  # FIX-P3-B10: floor prevents amplification
```

## Edit 4: `gnn/trainer_multienv.py::update` — same as edits 1+2

Remove clipped value loss block, replace with `F.mse_loss(pred_values, flat_returns)`.
Move exp_var computation outside the critic loop.

## Edit 5: `gnn/batch_rollout.py::get_batches` — adv_std floor

**Find:** `adv_std = advantages.std() + 1e-8`

**Replace with:**
```python
adv_std = advantages.std().clamp(min=1e-4)  # FIX-P3-B10: floor prevents amplification
```

## Edit 6: `HetGAT/mahac.py` — advantage normalization floor

**Find (lines ~110-111):**
```python
adv_s = (adv_s - adv_s.mean()) / (adv_s.std() + 1e-8)
adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)
```

**Replace with:**
```python
adv_s = (adv_s - adv_s.mean()) / adv_s.std().clamp(min=1e-4)  # FIX-P3-B10
adv_i = (adv_i - adv_i.mean()) / adv_i.std().clamp(min=1e-4)  # FIX-P3-B10
```

## Note on HetGAT/mahac.py

mahac.py already uses simple MSE and per-iteration exp_var (previously fixed). Only edit 6 (adv_std floor) applies.
