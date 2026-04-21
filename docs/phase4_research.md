---
name: Phase 4 Research Memo
description: Analysis of entropy coef dominance and tanh eval/train mismatch
type: project
---

# Phase 4 Research Memo

## 1. Current entropy_coef value

**`HetGAT/mahac.py`:**
- Default value: `entropy_coef: float = 0.01` (line 30)
- Storage: `self.entropy_coef = entropy_coef` (line 42)
- Usage in actor loss (line 210-213):
  ```python
  actor_loss = (
      policy_loss_s + policy_loss_i
      - self.entropy_coef * (entropy_s + entropy_i)
  )
  ```
- The value is a constructor parameter with default 0.01. It is NOT hardcoded at the call sites; callers pass it through.

**`guided_coverage_train.py`:**
- Module-level constant: `entropy_coef = 0.01` (line 41), comment: `# entropy regularization — prevents policy collapse in continuous MARL`
- Passed to `MAHAC(entropy_coef=entropy_coef, ...)` at line 318 inside `run_hetnet()`
- NOT an argparse argument in this file — it is a hardcoded module-level constant.

**`train_hetnet_envs.py`:**
- Module-level constant: `entropy_coef = 0.01` (line 39)
- Passed to `MAHAC(entropy_coef=entropy_coef, ...)` at line 296 inside `run_hetnet()`
- NOT an argparse argument — also hardcoded.

**Summary:** `entropy_coef = 0.01` everywhere. It is a default in `mahac.py::MAHAC.__init__` and a hardcoded module-level constant in both training scripts. To change it requires editing the constant in each training script (or promoting it to an argparse argument).

---

## 2. tanh eval path

**During evaluation — `HetGAT/train.py::evaluate` (lines 97-98):**
```python
scout_actions = torch.tanh(scout_dist.mean)
interc_actions = torch.tanh(interc_dist.mean)
```
`torch.tanh(...)` is applied to the **distribution mean** (the deterministic mode), not a sample. The same pattern appears in `render_evaluation()` at line 271:
```python
all_actions = torch.cat([torch.tanh(scout_dist.mean), torch.tanh(interc_dist.mean)], dim=1)
```

**During training — `HetGAT/utils.py::collect_rollout` (lines 138-139):**
```python
scout_actions, scout_log_probs = policy.sample_action(scout_dist)
interc_actions, interc_log_probs = policy.sample_action(interc_dist)
```

`policy.sample_action` is defined in `HetGAT/hetnetPolicy.py` (lines 150-158):
```python
@staticmethod
def sample_action(dist):
    """Sample action from Normal distribution. No tanh — VMAS clamps via max_speed."""
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)
    return action, log_prob
```

The docstring is explicit: **no tanh**. The raw Gaussian sample is returned. VMAS then clips the action internally.

**The mismatch is confirmed:**
- Training: `action = dist.sample()` — raw Gaussian sample, unbounded, VMAS clips to `u_range=1.0`
- Eval: `action = torch.tanh(dist.mean)` — deterministic tanh-squashed mean, lives in `(-1, 1)`

These are different transformations of different moments of the same distribution. Training actions can exceed `[-1, 1]` before VMAS clips them; eval actions are guaranteed in `(-1, 1)` before VMAS sees them. VMAS clips with `u_range=1.0` and `clamp_actions=True` (confirmed in `environments/predator_prey.py` lines 76, 93, 329), so both paths land in `[-1, 1]` after VMAS processing, but the pre-clip distributions differ meaningfully.

**Magnitude of mismatch:**
- If `log_std = 0.5`, then `std = exp(0.5) ≈ 1.65`. A raw sample has probability mass significantly outside `[-1, 1]`; those actions get hard-clipped by VMAS. The effective action distribution during training is a truncated Gaussian (or piecewise-constant at boundaries). During eval, `tanh(mean)` applies a smooth squash that is different from VMAS's hard clip. The behavioral difference is largest when the mean is near 0 (tanh ≈ 0, clip passes through the same) but can diverge when the mean is large.
- If `log_std = -1.5` (minimum after clamp), `std = exp(-1.5) ≈ 0.22`. Actions rarely exceed `[-1, 1]`, so the clip rarely triggers and the mismatch is negligible.

---

## 3. log_std clamp values

**`HetGAT/mahac.py::update` (lines 221-223):**
```python
with torch.no_grad():
    self.policy.log_std_scout.clamp_(min=self.log_std_min, max=0.5)
    self.policy.log_std_interc.clamp_(min=self.log_std_min, max=0.5)
```
- Applied **after** `actor_optim.step()`, each PPO epoch
- `min = self.log_std_min` (default `-1.5`, passed as constructor arg)
- `max = 0.5` — hardcoded in `mahac.py`, not a constructor arg

**`HetGAT/hetnetPolicy.py::forward` (lines 141-142):**
```python
scout_std = self.log_std_scout.clamp(min=-5.0, max=2.0).exp().expand_as(scout_mean)
interc_std = self.log_std_interc.clamp(min=-5.0, max=2.0).exp().expand_as(interc_mean)
```
- This is a **functional clamp** (not in-place), applied during every forward pass to compute std from log_std
- Range: `[-5.0, 2.0]` — much wider than the post-update clamp in `mahac.py`

**Two-clamp architecture:**
1. Forward-pass clamp (`[-5.0, 2.0]`): protects against numerical explosion of `exp(log_std)` during forward
2. Post-update clamp (`[log_std_min, 0.5]` = `[-1.5, 0.5]`): the effective policy constraint

The post-update clamp in `mahac.py` is the binding one for training behavior. The forward-pass clamp in `hetnetPolicy.py` would only activate if `log_std` escaped to `|log_std| > 2.0`, which the post-update clamp prevents from happening. In practice, `hetnetPolicy.py`'s clamp is a safety net that never triggers under normal training.

**Entropy at ceiling:** at `log_std = 0.5`, entropy of 2D diagonal Gaussian = `sum_d [0.5 * (1 + ln(2pi)) + log_std]` = `2 * [0.5 * (1 + ln(2pi)) + 0.5]` = `2 * [0.9189 + 0.5]` = `2 * 1.4189 = 2.838` per dimension, total `= 2 * 1.4189 * 2 = ... `. More precisely:

Entropy of N(mu, sigma^2) in d=2 = `d/2 * (1 + ln(2*pi)) + sum_k log_std_k` = `2 * 0.5 * ln(2*pi*e) + 2 * log_std` = `ln(2*pi*e) + 2*log_std` = `1.8379 + 2*0.5 = 2.8379`.

The observed `ent_s = 3.838` in the pre-Phase-2 logs indicates `log_std` was at ceiling (as predicted by the Phase 2 Thinker's analysis). After Phase 2 fixes, the clamp ceiling at 0.5 will still cap entropy at ≈2.84 if `log_std` continues to climb.

---

## 4. Grad-norm logging feasibility

**Backward call location in `mahac.py::update` (line 216):**
```python
self.actor_optim.zero_grad()
actor_loss.backward()
nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
self.actor_optim.step()
```

The sequence is: `zero_grad()` → `backward()` → `clip_grad_norm_()` → `step()`.

After `actor_loss.backward()` and before `actor_optim.step()`, all gradients are populated. The correct insertion point for grad-norm logging is between lines 217 (`clip_grad_norm_`) and 218 (`step()`):

```python
self.actor_optim.zero_grad()
actor_loss.backward()
nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
# INSERT HERE: policy_grad_norm computation
policy_grad_norm = sum(
    p.grad.norm() ** 2 for p in self.policy.parameters() if p.grad is not None
).sqrt()
self.actor_optim.step()
```

Note: `clip_grad_norm_` modifies the gradients in-place before the logging. Therefore the logged norm will reflect the **post-clip** policy gradient, which is already capped at `max_grad_norm = 0.5`. To measure the **pre-clip** entropy-vs-policy gradient ratio, the computation would need to happen before `clip_grad_norm_` on a separate backward pass, or computed as a fraction: `entropy_grad_frac = entropy_coef * entropy_sum / actor_loss.detach().abs()`.

A simpler feasible approach: log `policy_grad_norm` (post-clip, capped at 0.5) alongside `entropy_coef * (entropy_s + entropy_i).item()` to estimate the relative entropy contribution to the loss, without a second backward pass.

**Optimizer identity:** `self.actor_optim = torch.optim.Adam(policy.parameters(), lr=lr_actor)` (line 49-51). There is no `self.optimizer` attribute — `train.py` line 447 references `trainer.optimizer.state_dict()` which is a latent bug (AttributeError at checkpoint save time), but this is out of scope for Phase 4.

---

## Summary of findings for Thinker

| Question | Finding |
|---|---|
| entropy_coef in mahac.py | 0.01 default, passed as arg |
| entropy_coef in guided_coverage_train.py | hardcoded module constant 0.01 |
| entropy_coef in train_hetnet_envs.py | hardcoded module constant 0.01 |
| eval tanh applied to | `scout_dist.mean` and `interc_dist.mean` |
| training actions | raw `dist.sample()`, no tanh |
| VMAS u_range | 1.0 |
| VMAS clamp_actions | True |
| log_std post-update clamp min | `self.log_std_min` = -1.5 |
| log_std post-update clamp max | 0.5 (hardcoded in mahac.py) |
| log_std forward-pass clamp | [-5.0, 2.0] (hetnetPolicy.py, non-binding) |
| grad-norm insertion point | after `backward()`, before or after `clip_grad_norm_()` |
| actor optimizer attribute name | `self.actor_optim` (not `self.optimizer`) |

**Side-channel bug found (out of scope for Phase 4):** `HetGAT/train.py` line 447 saves `trainer.optimizer.state_dict()` but `MAHAC` has no `.optimizer` attribute — only `.actor_optim` and `.critic_optim`. This will raise `AttributeError` at every checkpoint save interval during a `train.py`-based run. Not relevant to PP/PCP runs (which use `train_hetnet_envs.py`), but should be escalated to Planner for a follow-up fix.
