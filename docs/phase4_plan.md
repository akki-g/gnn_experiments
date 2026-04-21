---
name: Phase 4 Plan
description: Remove tanh from eval, add grad-norm logging
type: project
---

# Phase 4 Plan: Tanh Eval Removal and Grad-Norm Logging

Bugs: #4 (entropy coef dominance — add logging), #9 (tanh in eval but not training)

## Files Changed

- `HetGAT/train.py` — `evaluate()` and `render_evaluation()`: remove torch.tanh
- `HetGAT/mahac.py` — `update()`: add grad-norm metric trackers + metrics dict entries

## Files NOT Touched

- `HetGAT/hetnetPolicy.py` — out of scope
- Any environment, baseline, or gnn file
- `entropy_coef` stays 0.01 (unchanged)

---

## Edit 1: `HetGAT/train.py::evaluate` — remove tanh

**Find:**
```python
scout_actions = torch.tanh(scout_dist.mean)
interc_actions = torch.tanh(interc_dist.mean)
```

**Replace:**
```python
scout_actions = scout_dist.mean    # FIX-P4-B9: no tanh; VMAS clamp_actions handles bounds
interc_actions = interc_dist.mean  # FIX-P4-B9
```

## Edit 2: `HetGAT/train.py::render_evaluation` — remove tanh

**Find:** `torch.cat([torch.tanh(scout_dist.mean), torch.tanh(interc_dist.mean)], dim=1)`

**Replace:** `torch.cat([scout_dist.mean, interc_dist.mean], dim=1)  # FIX-P4-B9`

## Edit 3: `HetGAT/mahac.py::update` — add grad-norm accumulator init

After `n_critic_updates = 0`, add:
```python
total_actor_grad_norm = 0.0   # FIX-P4-B4: pre-clip grad norm for entropy_coef diagnostics
total_ent_loss_contrib = 0.0  # FIX-P4-B4: entropy loss contribution tracking
```

## Edit 4: `HetGAT/mahac.py::update` — record grad-norm after backward()

Between `actor_loss.backward()` and `clip_grad_norm_()`:
```python
# FIX-P4-B4: log pre-clip actor grad norm
actor_grad_norm_preclip = sum(
    p.grad.norm() ** 2 for p in self.policy.parameters() if p.grad is not None
).sqrt().item()
total_actor_grad_norm += actor_grad_norm_preclip
total_ent_loss_contrib += abs(self.entropy_coef * (entropy_s.item() + entropy_i.item()))
```

## Edit 5: `HetGAT/mahac.py::update` — add to metrics dict

```python
'actor_grad_norm_preclip': total_actor_grad_norm / max(n_actor_updates, 1),  # FIX-P4-B4
'ent_loss_contribution': total_ent_loss_contrib / max(n_actor_updates, 1),   # FIX-P4-B4
```

## Contingency rule (post-integration test)

If `ent_loss_contribution / actor_grad_norm_preclip > 0.5` at iter 50 of HetNet PP, reduce `entropy_coef` from 0.01 to 0.003 in `guided_coverage_train.py` and `train_hetnet_envs.py`.
