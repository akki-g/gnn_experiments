---
name: Phase 2 Plan
description: Concrete diff plan for log_std labeling fix and lp_diff diagnostic
type: project
---

# Phase 2 Plan: log_std Labeling Fix and lp_diff Diagnostic

## Summary

The PPO ratio computation in `mahac.py` is **structurally correct** (confirmed by Researcher). The `r_s = r_i = 1.000` symptom is caused by degenerate near-zero advantages from the Phase 1 done-signal bug, not by a ratio code bug. Phase 2 makes **no changes** to the ratio arithmetic. It addresses:

1. **Bug #3 (log_std labeling):** Print statements display `std_s=` / `std_i=` when the value is `log_std`. Fix labels at all sites.
2. **lp_diff diagnostic:** Add per-epoch log-prob difference tracking to `mahac.py::update`.

## Files Changed

| File | What changes |
|------|-------------|
| `HetGAT/train.py` | Relabel `std_s=` → `log_std_s=`, `std_i=` → `log_std_i=` in train loop print |
| `guided_coverage_train.py` | Same relabeling in run_hetnet print |
| `train_hetnet_envs.py` | Same relabeling (this is the script that produced job_585255 logs) |
| `HetGAT/mahac.py` | Add lp_diff tracking per epoch; add 4 keys to returned metrics dict |

## Files NOT Touched

- `HetGAT/hetnetPolicy.py`, `HetGAT/hetgatLayer.py`, `HetGAT/rollout.py`
- Any environment file, any critic file
- `baseline/trainer.py`, `gnn/trainer_multienv.py`

---

## Edit 1: `HetGAT/train.py` — fix std label in train loop print

**Find:** `f"std_s={metrics['log_std_scout']:.3f}` and `f"std_i={metrics['log_std_interc']:.3f}`

**Replacement:**
```python
                f"log_std_s={metrics['log_std_scout']:.3f}  "  # FIX-P2-B3: label matches value
                f"log_std_i={metrics['log_std_interc']:.3f}  "  # FIX-P2-B3
```

---

## Edit 2: `guided_coverage_train.py` — fix std label in run_hetnet print

**Find:** `f"std_s={update_metrics['log_std_scout']:.2f}` and `f"std_i={update_metrics['log_std_interc']:.2f}`

**Replacement:**
```python
                f"log_std_s={update_metrics['log_std_scout']:.2f} "  # FIX-P2-B3: label matches value
                f"log_std_i={update_metrics['log_std_interc']:.2f}",  # FIX-P2-B3
```

---

## Edit 3: `train_hetnet_envs.py` — fix std label

**Find:** same pattern `f"std_s={update_metrics['log_std_scout']:.2f}` and `f"std_i=...`

**Replacement:** same as Edit 2.

---

## Edit 4: `HetGAT/mahac.py::update` — add lp_diff diagnostic

### 4a. Add tracking variables near the top of `update()` (after other metric accumulators)

```python
        lp_diff_epoch0_s = 0.0  # FIX-P2-DIAG: log-prob diff at epoch 0 (should be ~0)
        lp_diff_epoch0_i = 0.0
        lp_diff_final_s = 0.0
        lp_diff_final_i = 0.0
```

### 4b. After computing `ratio_s` and `ratio_i`, insert lp_diff recording

```python
            # FIX-P2-DIAG: track log-prob difference per epoch
            with torch.no_grad():
                _lp_diff_s = (new_lp_s - flat_old_lp_s).abs().mean().item()
                _lp_diff_i = (new_lp_i - flat_old_lp_i).abs().mean().item()
                if epoch == 0:
                    lp_diff_epoch0_s = _lp_diff_s
                    lp_diff_epoch0_i = _lp_diff_i
                lp_diff_final_s = _lp_diff_s
                lp_diff_final_i = _lp_diff_i
```

At epoch 0 (before any optimizer step), `lp_diff_epoch0_s` should be exactly 0.0 (float32 tolerance ~1e-6). After final epoch, `lp_diff_final_s` should be nonzero if PPO gradient is functional.

### 4c. Add keys to the returned metrics dict

```python
            'lp_diff_epoch0_scout': lp_diff_epoch0_s,
            'lp_diff_epoch0_interc': lp_diff_epoch0_i,
            'lp_diff_final_scout': lp_diff_final_s,
            'lp_diff_final_interc': lp_diff_final_i,
```

---

## Smoke Tests

```bash
# Label audit
grep -rn 'std_s=' --include='*.py' /Users/akshatguduru/Desktop/Thesis/gnn_experiments | grep -v 'log_std_s='
# Should return zero print-context matches

# After Phase 1 also landed: run 3-iter HetNet PP and check lp_diff_epoch0_scout < 1e-5
```
