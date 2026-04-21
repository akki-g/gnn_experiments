---
name: Phase 4 Reasoning
description: Decision on entropy_coef, tanh eval removal, and grad-norm logging
type: project
---

# Phase 4 Reasoning: Entropy Coefficient and Eval Consistency

## 1. Entropy Coefficient Decision — Keep 0.01, Add Logging

**Decision:** Keep `entropy_coef = 0.01` unchanged. Add grad-norm logging so the next run can inform whether to tune.

**Rationale.** With `entropy_coef=0.01` for a 2-D Gaussian action, the entropy bonus pushes `log_std` upward with force `0.01 * 1.0 = 0.01` per dimension per step. Now that Phases 1-2 have made the policy gradient non-zero, the PPO surrogate will compete with this. We cannot measure the empirical ratio without running 50+ post-Phase-3 iterations first.

Changing `entropy_coef` simultaneously with Phase 2's ratio fix would confound attribution: if HetNet starts learning, we would not know which fix caused it. Keeping it constant isolates Phase 2's effect.

`entropy_coef = 0.01` is standard for continuous control PPO (CleanRL uses 0.0-0.01 range; Andrychowicz et al. 2021 cover this). It is not obviously wrong.

**Contingency rule:** if the logged `entropy_grad / total_actor_grad > 0.5` at iteration 50 of the post-Phase-3 HetNet PP run, reduce `entropy_coef` to 0.003 and document as a hyperparameter sensitivity finding, not a bug fix.

## 2. Tanh Eval Decision — Remove (Option A)

**Decision:** Remove `torch.tanh(...)` from eval. Use `scout_dist.mean` / `interc_dist.mean` directly. VMAS `clamp_actions=True` handles bounds.

**Why not option (b) (add tanh to training).** Option (b) is SAC-style squashed Gaussian, which requires the log-prob Jacobian correction `log(1 - tanh(a)^2)`. This introduces:
1. Non-trivial code complexity
2. Numerical instability near `tanh(a) = ±1`
3. A change to the effective policy class (squashed-Gaussian vs. clipped-Gaussian) that requires retuning hyperparameters and would create an asymmetry between HetNet and the IPPO/GNN-MAPPO baselines (which use no tanh). This would invalidate the ablation comparison.

**Why option (a) is correct.** During training, actions are raw Gaussian samples clipped by VMAS. The deterministic analog is the raw Gaussian mean clipped by VMAS. This is the standard "deterministic eval" pattern in continuous-control RL. Using `tanh(mean)` instead introduces a nonlinear transformation the policy was never optimized for — even though both land in `[-1, 1]`, the mapping from latent policy parameters to physical action is different, meaning eval behavior does not faithfully represent what the policy learned.

**Edit:** In `HetGAT/train.py::evaluate` (and any `render_evaluation` function), replace:
```python
scout_actions = torch.tanh(scout_dist.mean)
interc_actions = torch.tanh(interc_dist.mean)
```
with:
```python
scout_actions = scout_dist.mean   # FIX-P4-B9: no tanh; VMAS clamp_actions handles bounds
interc_actions = interc_dist.mean  # FIX-P4-B9
```

## 3. log_std Clamp — Keep [-1.5, 0.5] Unchanged

Keep the post-update clamp in `mahac.py` at `[-1.5, 0.5]`. No evidence from logs that `log_std` presses against the lower bound. The inner clamp in `hetnetPolicy.py` at `[-5.0, 2.0]` is redundant but harmless and **out of scope** — do not touch `hetnetPolicy.py` in Phase 4.

## 4. Grad-Norm Logging — Add Pre-Clip Actor Grad Norm

Add `actor_grad_norm_preclip` metric: compute after `actor_loss.backward()`, before `clip_grad_norm_()`:

```python
actor_grad_norm_preclip = sum(
    p.grad.norm() ** 2 for p in self.policy.parameters() if p.grad is not None
).sqrt().item()
```

Also log `ent_loss_contribution = self.entropy_coef * (entropy_s + entropy_i).item()` as a scalar approximation of entropy's share of the actor gradient.

Include both in the returned metrics dict.

## Summary

| Item | Decision | Justification |
|------|----------|---------------|
| `entropy_coef` | Keep 0.01, add grad-norm logging | Isolate Phase 2 effect; tune empirically if ratio > 0.5 |
| Eval tanh | Remove (option a) | Consistency with clipped-Gaussian training; deterministic eval standard |
| `log_std` clamp | Keep `[-1.5, 0.5]` | No evidence of binding at lower bound |
| Grad-norm logging | Add pre-clip `actor_grad_norm` + `ent_loss_contribution` | Diagnostic for future entropy_coef tuning |
| `hetnetPolicy.py` | Do not touch | Out of scope |
