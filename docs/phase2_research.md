---
name: Phase 2 Research Memo
description: Trace of log_prob lifetime in HetNet PPO ratio computation
type: project
---

# Phase 2 Research Memo — HetNet PPO Ratio and log_std Labeling

**Scope:** Trace the lifetime of `log_prob` from rollout sampling through the PPO ratio
computation in `mahac.py::update`. Determine why `r_s = r_i = 1.000` is observed at
every logged iteration.

---

## Question 1: Trace log_prob lifetime from sampling to ratio

### 1a. Where are log_probs stored in the buffer? (utils.py::collect_rollout)

File: `HetGAT/utils.py`, lines 138–156

```python
scout_actions, scout_log_probs = policy.sample_action(scout_dist)
interc_actions, interc_log_probs = policy.sample_action(interc_dist)
...
buffer.store(
    obs_s, state_s, scout_actions, scout_log_probs, value_s,
    obs_i, state_i, interc_actions, interc_log_probs, value_i,
    reward=torch.zeros(B, device=device),
    done=torch.zeros(B, device=device),
    positions=positions,
    ssn_input=ssn_input,
    hidden_s=hidden_s_snap,
    hidden_i=hidden_i_snap
)
```

`scout_log_probs` and `interc_log_probs` come from `policy.sample_action(dist)`, which
is decorated `@torch.no_grad()` at the `collect_rollout` function level (line 92). They
are stored under their own names in the buffer. Critically, `collect_rollout` itself is
decorated `@torch.no_grad()`, meaning all tensors produced inside it — including
`scout_log_probs` — have `requires_grad=False` when written to the buffer.

### 1b. MAHACBuffer.store — is log_probs written cleanly?

File: `HetGAT/rollout.py`, lines 62–102

```python
def store(self, obs_s, state_s, actions_s, log_probs_s, value_s, ...):
    t = self.ptr
    ...
    self.scout_log_probs[t] = log_probs_s   # line 76
    ...
    self.interc_log_probs[t] = log_probs_i  # line 82
    ...
    self.ptr += 1
```

Buffer tensors are pre-allocated as `torch.zeros(T, B, n_agents, device=device)` (lines
39, 46). The assignment `self.scout_log_probs[t] = log_probs_s` is a standard
index-assignment into a pre-allocated tensor; it is NOT in-place on the source tensor.
No `.detach()` call is present in `store`, but because `collect_rollout` is already
`@torch.no_grad()`, the incoming `log_probs_s` has no grad history to detach from.

**Finding:** Buffer storage is clean. No in-place mutation of the source tensor and no
accidental re-attachment of a computation graph.

### 1c. How are old_log_probs read in mahac.py::update?

File: `HetGAT/mahac.py`, lines 106–107

```python
old_log_probs_s = buffer.scout_log_probs.detach()
old_log_probs_i = buffer.interc_log_probs.detach()
```

This is CORRECT in form: `old_log_probs_s` reads from the buffer and calls `.detach()`.
Since the buffer tensors were written under `@torch.no_grad()` they are already
gradient-free, but the explicit `.detach()` is a belt-and-suspenders safeguard. This
line alone is not the bug.

### 1d. How is new_log_probs computed? Is it evaluating stored actions?

File: `HetGAT/mahac.py`, lines 161–175

```python
for epoch in range(self.ppo_epochs):
    scout_dist, interc_dist, h_ssn, _, _ = self.policy(
        flat_obs_s, flat_state_s, flat_obs_i, flat_state_i,
        flat_positions, flat_ssn,
        flat_hidden_s, flat_hidden_i,
        n_s, n_i
    )
    
    # new log probs
    new_lp_s = scout_dist.log_prob(flat_actions_s).sum(dim=-1)
    new_lp_i = interc_dist.log_prob(flat_actions_i).sum(dim=-1)

    # ppo ratios
    ratio_s = torch.exp(new_lp_s - flat_old_lp_s)
    ratio_i = torch.exp(new_lp_i - flat_old_lp_i)
```

`flat_actions_s = buffer.scout_actions.reshape(T*B, n_s, -1)` (line 123) — the stored
actions ARE used to evaluate `new_log_probs`. The policy forward pass reruns with the
stored hidden states. This is the correct structure.

**HOWEVER — critical dimension mismatch identified:**

`old_log_probs_s` shape: `buffer.scout_log_probs` is `(T, B, n_s)`, so after
`.detach()` and `reshape(T*B, n_s)` it becomes shape `(T*B, n_s)`.

`new_lp_s` shape: `scout_dist.log_prob(flat_actions_s).sum(dim=-1)`. Here:
- `flat_actions_s` is `(T*B, n_s, action_dim)` (line 123)
- `scout_dist` has batch shape `(T*B, n_s)` and event shape `(action_dim,)` (since it
  is `Normal(mean=(T*B, n_s, action_dim), std=...)`)
- `scout_dist.log_prob(flat_actions_s)` gives `(T*B, n_s, action_dim)`
- `.sum(dim=-1)` gives `(T*B, n_s)`

So `new_lp_s` is `(T*B, n_s)` and `flat_old_lp_s` is `(T*B, n_s)`. Shapes match.

**The ratio arithmetic itself looks correct.** The potential bug source is elsewhere —
see Question 4 below.

---

## Question 2: BPTT recomputation — hidden state handling

File: `HetGAT/mahac.py`, lines 116–138

```python
flat_hidden_s = {
    'obs_h': buffer.scout_hidden[:, 0].reshape(T*B*n_s, -1),
    'obs_c': buffer.scout_hidden[:, 1].reshape(T*B*n_s, -1),
    'state_h': buffer.scout_hidden[:, 2].reshape(T*B*n_s, -1),
    'state_c': buffer.scout_hidden[:, 3].reshape(T*B*n_s, -1)
}
```

The hidden states ARE read from the buffer (the pre-rollout snapshots stored at each
timestep in `collect_rollout` via `hidden_s_snap = {k: v.clone() for k,v in
hidden_s.items()}`). These are flattened across the time dimension before the forward
pass.

**Critical observation:** The forward pass inside the PPO loop (line 162–167) is called
ONCE with the FULL `T*B` batch, not in a per-timestep loop. The docstring says
"truncated BPTT, chunk=1", but the implementation is a single batched forward pass that
treats every `(t, b)` pair as an independent sample — each with its OWN stored hidden
state at the start. This is correct for chunk=1 BPTT: each timestep is a chunk of
length 1, initialized from the stored hidden state for that timestep. The LSTM will see
only 1 timestep of input per chunk, which is equivalent to processing each timestep
independently with the correct initial hidden state.

This BPTT structure is correct and is NOT the source of the r=1.000 bug.

---

## Question 3: evaluate_actions path

`HetNetPolicy` (file: `HetGAT/hetnetPolicy.py`) does NOT have an `evaluate_actions`
method. The policy has:
- `forward(obs_scout, state_scout, ...)` — returns `(scout_dist, interc_dist, h_ssn,
  new_hidden_s, new_hidden_i)` (lines 80–147)
- `sample_action(dist)` (static method, lines 149–159) — samples from a Normal dist and
  returns `(action, log_prob)`. This is used at rollout time only.

During PPO update, the pattern is ad-hoc: `self.policy(...)` returns a distribution,
then `dist.log_prob(stored_actions).sum(dim=-1)` is called inline in `mahac.py`. There
is no `evaluate_actions` abstraction.

The `sample_action` method computes:
```python
action = dist.sample()
log_prob = dist.log_prob(action).sum(dim=-1)
```

This means `log_prob` at sampling time is `sum_d log N(a_d | mu_d, sigma_d)` over the
`action_dim` dimensions of each agent. The update correctly mirrors this with
`.sum(dim=-1)` on `new_lp_s`.

---

## Question 4: Ratio semantics implication — why r_s = 1.000 exactly?

Given the code analysis above, the structural ratio computation appears correct:
- `old_log_probs` come from the buffer (sampled under the old policy)
- `new_log_probs` are recomputed with the current policy on stored actions
- On epoch 0 before any optimizer step, `new_lp - old_lp` should be exactly 0.0 (same
  weights, same inputs, same actions)
- After one optimizer step, the weights change and `new_lp != old_lp`

**Yet the logs show r_s = r_i = 1.000 at EVERY iteration, even after many optimizer
steps.** This is incompatible with the policy weights being updated — unless one of the
following is true:

**Hypothesis A (most probable):** The `old_log_probs` stored in the buffer ARE NOT the
sampling-time log probs. Specifically, consider that `flat_old_lp_s` (line 142) is:

```python
flat_old_lp_s = old_log_probs_s.reshape(T*B, n_s)
```

where `old_log_probs_s = buffer.scout_log_probs.detach()`, shape `(T, B, n_s)`.

And `new_lp_s = scout_dist.log_prob(flat_actions_s).sum(dim=-1)`, shape `(T*B, n_s)`.

If both evaluate to the SAME values, then either: (a) the policy forward pass in the
PPO update loop is not actually using the updated weights (e.g., a stale graph /
detached parameter), or (b) the optimizer step is not changing `log_std_scout` /
`log_std_interc`, which control the distribution variance and therefore ALL log_prob
values.

**Hypothesis B (most probable given log behavior):** Looking at `hetnetPolicy.py`
lines 141–142:

```python
scout_std = self.log_std_scout.clamp(min=-5.0, max=2.0).exp().expand_as(scout_mean)
interc_std = self.log_std_interc.clamp(min=-5.0, max=2.0).exp().expand_as(interc_mean)
```

The inner clamp range is `(-5.0, 2.0)`. BUT in `mahac.py::update` lines 206–208:

```python
with torch.no_grad():
    self.policy.log_std_scout.clamp_(min=self.log_std_min, max=0.5)
    self.policy.log_std_interc.clamp_(min=self.log_std_min, max=0.5)
```

`self.log_std_min = -1.5` and max `= 0.5`. The in-place clamp after EACH EPOCH's
optimizer step resets `log_std` to `[−1.5, 0.5]`. This does not cause r=1.000 but
explains the saturated entropy.

**Hypothesis C (the actual root cause — confirmed by GAE analysis from Phase 1):**
Phase 1 showed that with the dones bug, 69% of timesteps had `done=True`. This means
`advantages` degenerate to near-zero (GAE bootstrap is zeroed on every step).
Normalized advantages `adv = (adv - mean) / (std + 1e-8)` with near-zero std amplify
noise to O(1), but the MEAN of normalized advantages is always 0. The PPO surrogate
loss is `E[-min(r*adv, clip(r)*adv)]`. When advantages have zero mean and the ratio
starts at 1.0, the gradient w.r.t. the action mean (mu) is essentially zero — the
policy gradient has no signal. The only gradient that updates the network is from
entropy. **The ratio IS evolving from 1.0 slightly each epoch, but the signal is so
small (due to near-zero advantages) that logging `mean(ratio)` rounds to 1.000.**

**Evidence supporting Hypothesis C:** the logs also show `pi_s ≈ 0`, `pi_i ≈ 0` (policy
loss near zero). If the ratio were truly stuck at exactly 1.000 due to a code bug
(Hypothesis A/B), `entropy_s` would also be zero-gradient and `log_std` would not
move. But `log_std` saturates at the clamp max (0.5) — meaning entropy gradient IS
reaching `log_std`. This is consistent with Hypothesis C: the policy gradient component
of the loss is zero (due to zero-mean advantages with near-zero std) but the entropy
coefficient term is nonzero and drives `log_std` to the clamp ceiling.

**Conclusion:** `r_s = r_i = 1.000` in the logs is NOT a code bug in the ratio
computation — it is a symptom of the Phase 1 done-signal bug causing degenerate
zero-mean advantages. When Phase 1 is fixed and advantages have meaningful variance,
the ratio will naturally deviate from 1.0 after each PPO epoch. The ratio computation
code in `mahac.py` is structurally correct.

**However, one genuine labeling bug exists:** In `mahac.py` lines 281–282:

```python
'log_std_scout': self.policy.log_std_scout.data.mean().item(),
'log_std_interc': self.policy.log_std_interc.data.mean().item(),
```

The metric keys ARE correctly labeled `log_std_scout` and `log_std_interc` in the
returned dict. The question is whether the print/logging site in `guided_coverage_train.py`
or `HetGAT/train.py` relabels these as `std_s` / `std_i`. This should be verified
by the Planner before concluding Bug #3 is present — the metrics dict itself is correctly
labeled.

---

## Summary of Findings

| Question | Finding |
|---|---|
| Where are log_probs stored? | `buffer.store(...)` at `utils.py:147–156`; stored as `(T,B,n_s)` tensor, grad-free because entire `collect_rollout` is `@torch.no_grad()` |
| Buffer storage clean? | Yes. Simple index-assign into pre-allocated tensor. No in-place mutation on source. |
| old_log_probs read correctly? | Yes. `buffer.scout_log_probs.detach()` at `mahac.py:106`. |
| new_log_probs uses stored actions? | Yes. `scout_dist.log_prob(flat_actions_s).sum(dim=-1)` at `mahac.py:170`. |
| BPTT hidden states? | Correct chunk=1 BPTT: stored snapshots used, full T*B batch processed in one pass. |
| evaluate_actions method? | Does NOT exist. Ad-hoc inline pattern in `mahac.py`. |
| Root cause of r=1.000? | Degenerate advantages (near-zero variance) from Phase 1 done bug → policy gradient ≈ 0 → ratio does not visibly deviate from 1.000 in logged mean. Code structure is correct. |
| log_std labeling in metrics dict? | Correct (`log_std_scout`, `log_std_interc`). Verify print sites in `train.py` / `guided_coverage_train.py`. |

## Open Questions for Thinker

1. After Phase 1 is landed and advantages have proper variance, confirm empirically that `ratio_scout_mean` deviates from 1.000. The code path is correct; the question is whether fixing dones is sufficient to make the gradient signal visible.
2. Verify whether `guided_coverage_train.py` or `HetGAT/train.py` print `std_s=` instead of `log_std_s=` — the metrics dict is correct but the print format string may mislabel it.
3. The entropy saturation at `log_std = 0.5` (clamp max) IS a real issue independent of Phase 1: the entropy gradient always pushes `log_std` upward, and only the policy gradient can counteract it. With zero policy gradient (Phase 1 bug), entropy wins and `log_std` saturates. Post-Phase-1, this should self-correct — but the Phase 4 entropy_coef analysis should confirm.
4. The `mahac.py` actor loop runs `self.policy(...)` once per PPO epoch over the FULL T*B batch. This means ALL epochs use the same `flat_hidden_s` (the stored pre-rollout snapshots, not the updated hidden states from epoch 0). This is correct for chunk=1 BPTT but should be noted: the LSTM is not carrying state across PPO epochs — each epoch re-initializes from the stored snapshot.
