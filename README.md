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

---

## VMAS Simple Spread Validation (MAPPO backbone smoke test)

This section validates that the frozen MAPPO backbone trains, checkpoints,
reloads, and renders correctly on a real vectorized VMAS task using IdentityComm.

### Local sanity run (~1 minute)

```bash
python -m experiments.train_vmas_mappo \
    --config configs/vmas_simple_spread_mappo.yaml \
    --render off
```

To shrink to a ~1-minute run, override three keys in the YAML (or copy to `/tmp/`):
- `runtime.num_envs: 4`
- `runtime.total_env_steps: 20000`
- `model.hidden_dim: 32`

### Resume from checkpoint

```bash
python -m experiments.train_vmas_mappo \
    --config configs/vmas_simple_spread_mappo.yaml \
    --resume runs/vmas_simple_spread/<run_id>/ckpt/final.pt \
    --render off
```

### Render a trained policy

```bash
python -m experiments.render_trained_policy \
    --config configs/vmas_simple_spread_mappo.yaml \
    --checkpoint runs/vmas_simple_spread/<run_id>/ckpt/best.pt
```

On a headless node, write deterministic evaluation metrics without video:

```bash
python -m experiments.render_trained_policy \
    --config configs/vmas_simple_spread_mappo.yaml \
    --checkpoint runs/vmas_simple_spread/<run_id>/ckpt/best.pt \
    --no-video
```

### Plot run metrics

Plot every numeric metric from one or more run directories:

```bash
python -m experiments.plot_run_metrics \
    runs/vmas_simple_spread/20260603_192930_* \
    --rolling 5 \
    --output runs/vmas_simple_spread/metrics_633898.png
```

For a parent directory, use recursive discovery:

```bash
python -m experiments.plot_run_metrics runs/vmas_simple_spread \
    --recursive \
    --rolling 5 \
    --output runs/vmas_simple_spread/all_metrics.png
```

### HPC dependencies

For the VMAS training job on a cluster, use the minimal pinned batch
dependencies:

```bash
pip install -r requirements-hpc.txt
```

The SLURM script also activates `~/expenv`, creates it if it does not exist,
checks for `yaml`, `numpy`, `torch`, and `vmas`, and installs any missing pinned
packages under a simple lock so array tasks do not write to the venv at the same
time. `req.txt` remains the broader local/dev dependency list, including
rendering and analysis packages. The HPC list intentionally omits rendering-only
packages because the batch script runs with `--render off`.

### SLURM (seeds 0, 1, 2; account cenyioha, partition normal)

One-time setup before first submission from the repo root. SLURM needs the log
directory to exist before it opens `--output` / `--error`:

```bash
cd $HOME/gnn_experiments
mkdir -p $HOME/gnn_experiments/slurm_logs
```

Submit:

```bash
cd $HOME/gnn_experiments
sbatch scripts/slurm/train_vmas_simple_spread.sbatch
squeue -u $USER            # monitor
```

The script writes scheduler logs directly to
`slurm_logs/vmas_ss_mappo_<jobid>_<arrayid>.out` and `.err`. It resolves the
runtime repo root from `SLURM_SUBMIT_DIR` and falls back to
`$HOME/gnn_experiments`, so submitting from the repo root remains the safest
default. Array tasks 0/1/2 → seeds 0/1/2. The trainer writes its own
`console.log` inside a collision-resistant run dir containing timestamp,
seed, pid, and SLURM job/task ids when present.

`CUDA_VISIBLE_DEVICES` is set by SLURM automatically — the trainer uses `cuda:0`
which maps to the assigned GPU. No manual device override needed.

The batch script disables rendering with `--render off` and sets conservative
thread environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`) to
avoid accidental CPU oversubscription. To get a video from a cluster checkpoint,
run `experiments.render_trained_policy` locally, or submit the render under
`xvfb-run`.

### Artifacts produced in `runs/vmas_simple_spread/<run_id>/`

| File | Description |
|---|---|
| `ckpt/final.pt` | Checkpoint after final iteration |
| `ckpt/best.pt` | Checkpoint at best mean episode return |
| `config.yaml` | Copy of the config used for this run |
| `metrics.json` | Per-iteration metrics (JSON list of dicts) |
| `metrics.csv` | Per-iteration metrics (CSV) |
| `console.log` | Copy of all stdout printed during training |
| `policy.mp4` | Optional rendered evaluation video (system ffmpeg) |
| `policy.gif` | Optional fallback video if ffmpeg is unavailable (imageio) |

### **Limitations (read before using results)**

**simple_spread observations include other-agent relative positions
(obs_dim=14). Observations are NOT isolated: each agent can observe every
other agent's position directly. This is a backbone smoke test ONLY and
must never be used as the IPPO-vs-GNN ablation environment, because the
observation-isolation invariant (all inter-agent information must flow only
through GNN graph edges) is violated.**

The VMAS adapter auto-resets finished environments so rollout collection can
continue without a trainer-side reset branch. It also preserves
`terminal_obs`/`terminal_state` in `info`; the trainer uses that terminal state
for the final bootstrap when a rollout ends on a time-limit boundary.
