# Phase 1 — MAPPO Backbone

This phase implements the complete shared **MAPPO backbone** for the thesis MARL
research project.  The backbone is the stable foundation on which future
communication modules (Broadcast, Attention, Graph, …) and the pursuit
(predator-cap-prey) environment will be layered.

## What Phase 1 implements

- `backbone/` — full MAPPO algorithm:
  - `encoder.py` — `SharedMLPEncoder`: shared MLP that maps per-agent observations
    to hidden embeddings, used by all agents.
  - `actor.py` — decentralized `Actor` head (discrete Categorical actions).
  - `critic.py` — centralized `CentralCritic` that reads raw global state (never
    comm-slot output — this is the Q5 isolation invariant).
  - `rollout_buffer.py` — pre-allocated on-policy buffer with correct GAE
    (`bad_masks` for truncated-episode bootstrapping, `active_masks` for dead agents).
  - `ppo_update.py` — PPO clipped-surrogate update with `model.train()` discipline.
  - `losses.py` — `ppo_policy_loss`, `value_loss`, `entropy_loss`,
    `explained_variance`, all supporting optional `active_masks`.
  - `mappo.py` — top-level `MAPPO` module wiring encoder → comm → actor and
    raw-state → critic.  Accepts configurable layer counts.  Comm slot receives
    `mask` and `class_id` at every call so future modules can use them without
    changing any trainer code.
  - `utils.py` — config loading, seeding, device selection, optimizer factory.
- `comm/` — communication slot interface:
  - `base.py` — `CommModule` abstract interface (Phase 2+ subclasses go here).
  - `identity.py` — `IdentityComm` no-op default.
- `envs/` — toy multi-agent environment for backbone validation:
  - `toy_env.py` — `ToyMultiAgentEnv`: vectorized, discrete, N=2 agents, known
    learnable reward signal, episode-length cap.
  - `adapters.py` — environment adapter utilities.
- `configs/` — YAML configs (`mappo_base.yaml`, `toy_sanity.yaml`).
- `experiments/` — training and evaluation entry points.
- `tests/` — six test modules; all pass.

## What Phase 1 does NOT implement

- VMAS predator-cap-prey / pursuit environment.
- Alpha observation-radius knob or observation-isolation probe.
- Real communication modules: Broadcast, Attention, Graph, Gated, CommFormer, MAGIC.
- HAPPO / HASAC.
- Necessity coefficients or severed-communication evaluation.
- SLURM sweep scripts.
- rliable analysis or thesis figures.

## Tensor-shape conventions

| Symbol | Meaning |
|---|---|
| `T` | rollout horizon |
| `B` | parallel environments (`num_envs`) |
| `N` | number of agents |
| `obs_dim` | per-agent observation dim |
| `state_dim` | global state dim (or `N * obs_dim` concat fallback) |
| `D` | hidden embedding width |

Stored rollout tensors:

| Field | Shape | Notes |
|---|---|---|
| `obs` | `[T, B, N, obs_dim]` | |
| `state` | `[T, B, state_dim]` | raw global state, or concat fallback |
| `actions` | `[T, B, N]` long | discrete action index |
| `log_probs` | `[T, B, N]` | detached at collection |
| `values` | `[T, B, N]` | detached at collection (per-agent) |
| `rewards` | `[T, B, N]` | per-agent |
| `dones` | `[T, B, N]` | 1.0 = episode ended (truncated or truly terminal) |
| `bad_masks` | `[T, B, N]` | 0.0 = truncated time-limit end; 1.0 = normal |
| `active_masks` | `[T, B, N]` | 0.0 = dead agent; 1.0 = alive (all-ones for toy env) |
| `advantages` | `[T, B, N]` | GAE output |
| `returns` | `[T, B, N]` | GAE output |

Minibatches flatten `T*B` into one batch dimension and keep `N` as an inner
axis (researcher decision 1):

```
obs_mb          : [T*B, N, obs_dim]
state_mb        : [T*B, state_dim]
actions_mb      : [T*B, N]
log_probs_mb    : [T*B, N]
values_mb       : [T*B, N]
advantages_mb   : [T*B, N]
returns_mb      : [T*B, N]
bad_masks_mb    : [T*B, N]
active_masks_mb : [T*B, N]
```

## GAE and truncation handling

`bad_masks` separates truncated episodes (time-limit) from truly terminal ones:

```
bootstrap_coef[t] = 1 - dones[t] * bad_masks[t]
```

- `done=0, bad=1` → `coef=1` (not done — always bootstrap).
- `done=1, bad=0` → `coef=1` (truncated — still bootstrap with next state value).
- `done=1, bad=1` → `coef=0` (truly terminal — no bootstrap).

The GAE lambda-continuation trace uses `1 - dones[t]` (stops at any episode
boundary, truncated or truly terminal).

For Phase 1 (toy env, no time-limit), `bad_masks` is always 1.0.

## Action space

Discrete (`torch.distributions.Categorical`).  Each agent selects one of
`action_dim` actions per step.

## Centralized critic

`[B, N]` per-agent values from `CentralCritic`.  Input: raw global state
`[B, state_dim]` or `build_state_from_obs()` concat fallback `[B, N * obs_dim]`.
The critic never sees comm-slot output — swapping comm modules does not change
what the critic sees.

## How future communication modules plug in

The backbone architecture:
```
raw_obs → SharedMLPEncoder → CommModule → Actor → action / log_prob / entropy
                               ↑
                    (identity by default; future: Broadcast, Attention, Graph)

raw_state ─────────────────────────────────────────→ CentralCritic → value
```

`MAPPO.__init__` accepts `comm_module: Optional[CommModule]`, defaulting to
`IdentityComm()`.  The comm slot receives `mask` and `class_id` at every
forward call so future radius-limited or class-differentiated modules work
without changing the trainer, rollout collection, PPO update, or critic code.

Future swap example — zero trainer changes required:

```python
from backbone import MAPPO
from comm.attention import AttentionComm
model = MAPPO(..., comm_module=AttentionComm(hidden_dim=64, num_heads=4))
```

The `configs/toy_sanity.yaml` `comm.module` key controls which module is used.

## How to run tests

```bash
cd /path/to/gnn_experiments
pytest tests/ -v
```

All six test modules should pass.

## How to run toy training

```bash
python -m experiments.train_mappo --config configs/toy_sanity.yaml
```

Training runs for 5 000 env steps on the toy environment and saves a checkpoint
to `runs/toy_sanity/ckpt/final.pt`.

Resume from checkpoint:

```bash
python -m experiments.train_mappo --config configs/toy_sanity.yaml \
    --resume runs/toy_sanity/ckpt/final.pt
```

## Research references

- **Official MAPPO** (Yu et al., 2022) — PPO update structure, rollout buffer
  semantics, GAE, value loss, entropy loss, old-log-prob storage, `bad_masks`
  for time-limit bootstrapping.
- **EPyMARL** — multi-agent experiment organization, IPPO/MAPPO distinction,
  parameter-sharing patterns, `active_masks` / death masking.
- **BenchMARL** — reproducible config patterns, clean algorithm/task/model
  separation, checkpointing, evaluation discipline.
- **VMAS / TorchRL** — vectorized multi-agent environment conventions (researched
  for forward compatibility; not yet integrated).

## Known limitations

- Truncation (`bad_masks=0`) is wired but not exercised — the toy env ends
  naturally.  Truly correct truncation for time-limited envs also requires the
  environment to provide the terminal observation value `V(s_terminal)`; this
  is beyond Phase 1 scope.
- Continuous action spaces are not yet supported (deferred to a later phase).
- No W&B integration yet (config flag `logging.wandb` is parsed but ignored).
- The evaluation script (`experiments/evaluate_mappo.py`) is a stub.
