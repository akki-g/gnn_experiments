---
name: Phase 1 Reasoning
description: Mathematical analysis of done-signal failure mode and fix invariants
type: project
---

# Phase 1 Reasoning: Done-Signal Failure Mode

## 1. Mathematical Characterization of the Failure

The `compute_gae` function in `HetGAT/utils.py` implements the standard GAE recursion:

```
mask[t] = 1 - dones[t]
delta[t] = r[t] + gamma * V(s_{t+1}) * mask[t] - V(s_t)
A[t] = delta[t] + gamma * lambda * mask[t] * A[t+1]
```

**Observed done pattern.** VMAS's internal step counter reaches `max_steps=80` at t=79 and never resets (no external `reset_at` is called by the adapter). From t=79 onward, every call to `env.step()` returns `done=True` for every environment. Thus within a rollout of T=256 steps:

- t in [0, 78]: `dones[t] = 0`, `mask[t] = 1` (normal)
- t in [79, 255]: `dones[t] = 1`, `mask[t] = 0` (pathological)

**Degenerate advantages for t >= 79.** When `mask[t] = 0`:

```
delta[t] = r[t] + 0 - V(s_t) = r[t] - V(s_t)
A[t] = delta[t] + 0 = r[t] - V(s_t)
```

The GAE recursion is completely severed. No temporal credit assignment occurs. The return degenerates to:

```
returns[t] = A[t] + V(s_t) = r[t]
```

**The critic effectively learns to predict the per-step reward rather than any discounted future sum.** With `step_penalty = -0.05` dominating, returns cluster tightly near -0.05 to -0.55. Return variance is near-zero, which artificially inflates `exp_var` metrics and makes the critic appear well-fit when it is not.

Of 65,536 total buffer entries (256 steps × 256 envs), approximately 45,312 (69%) have `dones=1` and degenerate GAE. Only steps 0–78 of the first episode segment per env contribute meaningful multi-step advantages.

## 2. VMAS Bug or Adapter Bug?

This is an **adapter-level bug**. VMAS auto-resets world state (positions, velocities) for done environments via `reset_world_at`, but its internal `self.steps` counter remains at `max_steps` after the done fires. Since no external code calls `reset_at` on VMAS, `_done()` returns `truncated = self.steps >= max_steps = True` for every subsequent step indefinitely.

The `guided_coverage.py` adapter solves this with FIX-1B: an adapter-level counter that increments after each `env.step()`, fires `forced_dones` at `max_steps`, and resets for any done env. PP and PCP adapters lack this entirely.

The fix is adapter-level: port FIX-1B to both PP and PCP adapters. This complements VMAS's internal mechanism; it does not replace it.

## 3. Contract for the Truncation Signal

In PP, `done=True` iff `all_found=True` (prey captured). The `dones_for_gae` filter in `trainer_multienv.py` computes `dones & (n_tagged >= 1)`, which equals raw scenario dones — providing zero additional discrimination. For PCP, the same equivalence holds.

GAE needs to distinguish:
- **True termination** (prey captured): bootstrap = 0 (no future reward)
- **Truncation** (max_steps reached): bootstrap = V(s_T) (agent would continue earning)

The adapter must emit `info["is_truncation"]` as a bool tensor of shape `(B,)`:

```python
forced_dones = step_counter >= max_steps
is_truncation = forced_dones & ~scenario_dones
dones = scenario_dones | forced_dones
info["is_truncation"] = is_truncation
```

Downstream `dones_for_gae` should be:

```python
is_trunc = info.get("is_truncation", torch.zeros_like(dones))
true_termination = dones & ~is_trunc
dones_for_gae = true_termination.float().unsqueeze(1).expand(-1, N)
```

## 4. Invariants Phase 1 Changes Must Preserve

**(a) All three algorithms must see fixed dones.** The step-counter logic must appear in both `step()` (used by GNN-MAPPO and IPPO) and `hetnet_step()` (used by HetNet). Both methods in both PP and PCP adapters must include counter increment, forced-done OR, counter reset, and `is_truncation` emission. Both `reset()` and `hetnet_reset()` must zero the counter.

**(b) Observations at t=0 of a new episode must be fresh.** VMAS auto-resets world state before returning observations from the next `step()` call. The adapter's observation-stacking reads the post-step world state, so new-episode observations are naturally correct. The adapter counter fix must NOT call `env.reset()` globally or disable per-env auto-reset — only layer a counter on top.

**(c) LSTM hidden state reset must still fire on true dones.** In `HetGAT/utils.py`, LSTM hidden states are zeroed for any env where `done=True`. After Phase 1, `done = scenario_dones | forced_dones` ensures the LSTM reset fires at both termination types (success AND truncation). This is correct — the LSTM should reset at any episode boundary. The `is_truncation` flag is used only by GAE (bootstrap decisions), not by the LSTM reset.

**(d) Prey position scaling (Bug #8) preserves observation isolation.** The fix changes `prey_pos = torch.rand(...)` → `prey_pos = (torch.rand(...) * 2 - 1) * inner`, centering prey in [-3.0, 3.0] for world_size=5.0. The observation function computes `prey_rel = prey.pos - agent.pos` with FOV masking — a function only of the querying agent's own position. No inter-agent information leaks.
