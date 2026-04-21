---
name: Phase 1 Plan
description: Concrete diff plan for PP/PCP done-signal fixes
type: project
---

# Phase 1 — Concrete Diff Plan

Bugs addressed: #1 (stale VMAS dones in PP/PCP), #5 (stale `expected_if_100step_eps` diagnostic), #7 (truncation vs. termination conflation), #8 (PP `prey_pos` scaling).

## Files Changed

1. `environments/predator_prey.py` — Edits 1, 2, 3, 5
2. `environments/predator_capture_prey.py` — Edit 4
3. `HetGAT/utils.py` — Edit 6
4. `gnn/trainer_multienv.py` — Edit 7
5. `baseline/trainer.py` — Edit 8

## Files NOT Touched

- `HetGAT/mahac.py`, `HetGAT/hetnetPolicy.py`, `HetGAT/hetgatLayer.py`, `HetGAT/rollout.py`, `HetGAT/critic.py`, `HetGAT/train.py`
- `gnn/run.py`, `gnn/comm.py`, `gnn/critic.py`, `gnn/batch_rollout.py`
- `baseline/ippo_policy.py`
- `environments/guarded_territory.py`, `environments/guided_coverage.py`

---

## Edit 1: Add `_step_counters` to PP adapter `__init__`

**File:** `environments/predator_prey.py`
**Class:** `PredatorPreyAdapter.__init__`
**After the line:** `self._max_steps = max_steps` (verify present)

**Insert:**
```python
        # FIX-P1-B1: port step-counter FIX-1B from guided_coverage.py
        self._step_counters = torch.zeros(num_envs, device=self.device, dtype=torch.long)
```

**Why sufficient:** `self._max_steps` already exists. Only the counter tensor is missing. Matches `guided_coverage.py` pattern.

---

## Edit 2: Zero counters on PP `reset` and `hetnet_reset`

**File:** `environments/predator_prey.py`
**Functions:** `PredatorPreyAdapter.reset` and `PredatorPreyAdapter.hetnet_reset`

In each, add after `all_obs = self.env.reset()`:
```python
        self._step_counters.zero_()  # FIX-P1-B1: reset step counters on full reset
```

---

## Edit 3: Add step-counter logic to PP `step` and `hetnet_step`

**File:** `environments/predator_prey.py`

In both `step` and `hetnet_step`, after `all_obs, all_rew, dones, all_infos = self.env.step(all_actions)`, insert:

```python
        # FIX-P1-B1: port step-counter FIX-1B from guided_coverage.py
        self._step_counters += 1
        forced_dones = self._step_counters >= self._max_steps
        is_truncation = forced_dones & ~dones
        dones = dones | forced_dones
        self._step_counters[dones] = 0
```

And before the `return` statement, add to `info`:
```python
        # FIX-P1-B7: emit truncation flag for downstream GAE
        info["is_truncation"] = is_truncation
```

**Why sufficient:** Counter increments every step, fires `forced_dones` at `_max_steps`, ORs into `dones`, resets for all done envs. `is_truncation` lets GAE bootstrap correctly.

**Invariant:** LSTM hidden state reset in `HetGAT/utils.py` fires on `done.any()`, which now includes both scenario dones and forced truncation dones — correct behavior.

---

## Edit 4: Identical edits for PCP adapter

**File:** `environments/predator_capture_prey.py`
**Class:** `PredatorCapturePreyAdapter`

Apply identical edits 1–3 to the PCP adapter:
- Edit 4a: Insert `self._step_counters = torch.zeros(...)` after `self._max_steps = max_steps` in `__init__`
- Edit 4b: Add `self._step_counters.zero_()` in both `reset` and `hetnet_reset`
- Edit 4c: Add step-counter block in both `step` and `hetnet_step`, with `info["is_truncation"] = is_truncation`

---

## Edit 5: Fix PP prey position scaling (Bug #8)

**File:** `environments/predator_prey.py`
**Function:** `Scenario.reset_world_at`

**Current code (find the prey_pos = torch.rand(...) line):**
```python
        prey_pos = (torch.rand(
            (1,2) if env_index is not None else (batch, 2),
            device=device
        ))
```

**Replacement:**
```python
        # FIX-P1-B8: scale prey_pos to [-inner, inner] (was [0, 1]; PCP already correct)
        prey_pos = (torch.rand(
            (1,2) if env_index is not None else (batch, 2),
            device=device
        ) * 2 - 1) * inner
```

**Why sufficient:** For `world_size=5.0`, `inner=3.0`, prey now spawns in `[-3.0, 3.0]^2` instead of `[0.0, 1.0]^2`. Observation isolation preserved: observation uses `prey_rel = prey.pos - agent.pos` (FOV-masked), per-agent only.

---

## Edit 6: Fix diagnostic print in HetGAT `collect_rollout`

**File:** `HetGAT/utils.py`
**Function:** `collect_rollout`

**Find:** `expected_if_100step_eps≈{B * (T // 100):.0f}`

**Replacement:**
```python
    # FIX-P1-B5: use adapter's actual max_steps instead of hardcoded 100
    _ms = getattr(env_adapter, '_max_steps', 100)
    # ... in the print f-string:
    f"expected_{_ms}step_eps≈{B * (T // _ms):.0f} "
```

**Why sufficient:** Uses `getattr` with fallback 100 for backward compatibility. For PP/PCP, prints `expected_80step_eps≈768`.

---

## Edit 7: Rewrite `dones_for_gae` in GNN-MAPPO trainer

**File:** `gnn/trainer_multienv.py`
**Function:** `GNNTrainerMultiEnv.collect_rollouts`

**Find and replace the `true_done_mask` block (the n_tagged/n_breached logic):**

```python
                # FIX-P1-B7: use adapter-emitted is_truncation to distinguish
                # truncation (bootstrap V) from true termination (bootstrap 0)
                is_trunc = info.get("is_truncation", torch.zeros_like(dones))
                true_done_mask = dones & ~is_trunc
                dones_for_gae = true_done_mask.float().unsqueeze(-1).expand(E, N)
```

**Why sufficient:** Old `n_tagged/n_breached` logic was equivalent to raw dones for PP/PCP (research memo). Fallback `torch.zeros_like(dones)` is conservative (treats all dones as terminations) for envs without `is_truncation`.

Note: `completed_mask = dones` (raw dones) for episode-return tracking is already correct in GNN-MAPPO trainer — no change needed there.

---

## Edit 8: Same `dones_for_gae` fix in IPPO baseline trainer

**File:** `baseline/trainer.py`
**Function:** `IPPOTrainer.collect_rollouts` (NOT `update` — `collect_rollouts` is in scope)

**Find and replace the `true_done_mask` block:**

```python
                # FIX-P1-B7: use adapter-emitted is_truncation to distinguish
                # truncation (bootstrap V) from true termination (bootstrap 0)
                is_trunc = info.get("is_truncation", torch.zeros_like(dones))
                true_done_mask = dones & ~is_trunc
                dones_for_gae = true_done_mask.float().unsqueeze(-1).expand(E, N)
```

**Also fix episode-return tracking** — change the reset from `true_done_mask` to raw `dones`:
```python
                if dones.any():
                    for e in dones.nonzero(as_tuple=True)[0].tolist():
                        completed_episode_returns.append(
                            self._running_episode_returns[e].item()
                        )
                    self._running_episode_returns[dones] = 0.0
```

**Why:** After the `dones_for_gae` change, `true_done_mask` excludes truncations, so truncation-only episode returns would accumulate across boundaries without this fix.

---

## Smoke Test Commands (after Coder applies edits)

```bash
cd /Users/akshatguduru/Desktop/Thesis/gnn_experiments
python scripts/smoke_p1_dones.py
python scripts/smoke_p1_prey_pos.py
```
