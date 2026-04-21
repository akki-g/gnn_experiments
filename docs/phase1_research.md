---
name: Phase 1 Research Memo
description: Analysis of done signal bugs in PP/PCP environments
type: project
---

# Phase 1 Research Memo

## Q1: FIX-1B Step-Counter Logic in `guided_coverage.py`

The relevant block in `environments/guided_coverage.py` (both `step` and `hetnet_step`, lines ~401–405 and ~492–496):

```python
# FIX-1B: force dones at max_steps regardless of VMAS internal behavior
self._step_counters += 1
forced_dones = self._step_counters >= self._max_steps
dones = dones | forced_dones
self._step_counters[dones] = 0  # reset counters for done envs (VMAS auto-resets them)
```

- **When is `_step_counters` incremented?** After every call to `self.env.step(...)`, unconditionally, before `forced_dones` is computed. It is initialized to `torch.zeros(num_envs, ...)` in `__init__` and zeroed on `reset()` / `hetnet_reset()`.
- **When is `forced_dones` computed?** Immediately after the increment, on the same step — any env whose counter has reached `_max_steps` fires `forced_dones=True`, which is OR-ed into the VMAS-returned `dones`.
- **Why is `self._step_counters[dones] = 0` needed?** VMAS auto-resets individual envs internally when `done=True` (via `env_reset_world_at(index)` in `_done`), but it does NOT call the adapter's reset logic. Without this line the adapter's counter would keep incrementing past `_max_steps` for an env that was truly done by the scenario (not by truncation), causing `forced_dones` to fire on the very next step for that env — generating spurious dones on every subsequent step. Resetting the counter for all done envs (both scenario-terminated and truncated) keeps the adapter counter in sync with VMAS's internal per-env state.

## Q2: PP/PCP Adapters — Step-Counter Absence and VMAS Auto-Reset

**PP (`environments/predator_prey.py`) and PCP (`environments/predator_capture_prey.py`) both lack any `_step_counters` logic.** The `step` and `hetnet_step` methods call `self.env.step(...)` and return the raw VMAS `dones` with no truncation guard:

```python
# predator_prey.py hetnet_step (line 478)
all_obs, all_rew, dones, all_info = self.env.step(all_actions)
...
return reward, dones, info
```

Both adapters do pass `max_steps=max_steps` to `vmas.make_env(...)`, so VMAS internally uses `truncated = self.steps >= self.max_steps` in `_done()`. **However**, VMAS `_done()` returns `terminated + truncated` (boolean OR as integer sum) when `terminated_truncated=False` (the default). VMAS increments `self.steps` each call and sets `self.steps[index] = 0` in `_reset_at(index)`. Critically, VMAS does NOT auto-call `reset_at` after returning `done=True` — the caller is responsible. After `done=True`, VMAS's `self.steps` for that env is NOT reset until `reset_at(index)` is called externally. This means on the next call to `step`, VMAS's `self.steps` for that env is still at `max_steps`, so `truncated` fires again immediately — yielding `done=True` on every subsequent step for that env until an explicit reset. This is the root cause of 69% done-rate: every env stuck at step 80 keeps returning `done=True` forever.

**Observations after done** come from the post-step world state (the terminal or auto-advanced state), not a fresh episode, because VMAS never calls `reset_at` automatically.

## Q3: `dones_for_gae` Logic and `info["n_tagged"]` in PP

**Active path is `gnn/trainer_multienv.py`** (not the notebook). Lines 129–142:

```python
if self.adapter.n_intruders == 0:
    true_done_mask = dones.bool() if dones.dtype != torch.bool else dones
else:
    n_tagged = info.get("n_tagged", torch.zeros(E, device=self.device))
    n_breached = info.get("n_breached", torch.zeros(E, device=self.device))
    true_done_mask = (
        dones
        & ((n_tagged >= self.adapter.n_intruders)
           | (n_breached >= self.adapter.n_zones))
    )
dones_for_gae = true_done_mask.float().unsqueeze(-1).expand(E, N)
```

**In `environments/predator_prey.py::step` (line 418):**

```python
all_found = info.get("all_found", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
info["n_tagged"] = all_found.float()
```

`all_found` is `True` only when all agents are within `capture_radius` of the prey. `n_intruders=1` for PP, so the condition `n_tagged >= 1` is equivalent to `all_found=True`. VMAS also sets `done=True` (via `scenario.done()`) when `all_found=True`. Therefore `done=True AND n_tagged>=1` is **always identical** to `done=True` alone — there is no observable difference. The `dones_for_gae` filter adds no discrimination over raw dones for PP. Because the step-counter bug causes `done=True` on every step past step 80, `n_tagged` will be `0` on those spurious dones, so `true_done_mask` will suppress them — but only partially masks the bug; the counter is never reset, so subsequent steps continue firing.

## Q4: Expected vs. Observed Done Count

For `T=256`, `B=256`, `max_steps=80`:

- Expected episodes per env in 256 steps: `256 / 80 = 3.2`
- Expected done events (episode boundaries): `256 * 3.2 = 819.2 ≈ 768` (at the 3 clean boundaries t=79, 159, 239, all 256 envs fire: `3 * 256 = 768`)
- **Observed: 45,312 dones**
- Ratio: `45312 / 768 ≈ 59×` more dones than expected
- As fraction of total timesteps: `45312 / (256 * 256) = 45312 / 65536 ≈ 69%` of all timesteps have `done=True`

This confirms the VMAS step-counter is not being reset: after t=80 each env fires `done=True` on every single subsequent step (176 remaining steps × 256 envs = 45,056 spurious dones, plus the ~256 legitimate ones at t=79).
