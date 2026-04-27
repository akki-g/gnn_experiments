# GNN / GAT Architectures for Heterogeneous MARL

Honors research under Dr. Enyioha. Current focus: exploring GNN- and GAT-based communication architectures in cooperative MARL settings where agents have heterogeneous observation capabilities. The codebase is a working testbed — three swappable algorithms on a shared environment interface — for running controlled comparisons.

---

## Algorithms

Three PPO-family variants, all sharing the same environment adapters so we can A/B them cleanly.

**IPPO** — fully independent PPO, no communication. Control / lower bound.

**GNN-MAPPO** (`gnn/`) — homogeneous shared-GCN communication. Follows Li et al. 2020 (CNN→GNN→MLP decomposition, adapted for feature-vector observations) and Khan et al. 2019 (GCN-parameterized policy, permutation-equivariant). The `GraphConv` module implements K-hop localized convolution **z = Σₖ hₖ Sᵏ x** using a symmetric-normalized adjacency D⁻¹ᐟ²AD⁻¹ᐟ² with self-loops stripped (handled by the k=0 term). Components: `obs_encode.py` (MLP encoder), `graph_conv.py` (K-hop GCN), `action.py` (policy head with learned `log_std`), `comm.py` (wires them together), `critic.py` (independent MLP baseline), `batch_rollout.py` (on-policy buffer with GAE), `trainer_multienv.py` (PPO outer loop with LR annealing and counterfactual communication-saliency logging).

**HetNet / MAHAC** (`HetGAT/`) — heterogeneous GAT, adapted from Seraj et al. 2021. Per-class preprocessors, multi-head HetGAT stack with edge-type-specific message transforms and attention vectors, centralized State Summary Node (SSN) feeding a CTDE critic.
- `preProcessing.py` — per-class `Linear → LayerNorm → LSTMCell` for ego state and observation (LSTM for partial observability). Deliberately not a GCN here — that would add a second aggregation step outside the main HetGAT layer.
- `graph.py` — per-edge-type binary adjacency `{s→s, s→i, i→s, i→i, s→ssn, i→ssn}` from positions and `r_comm`.
- `hetgatLayer.py` — multi-head HetGAT with **GATv2 attention** (Brody et al. 2022): LeakyReLU before the attention dot product, which restores expressivity vs. vanilla GAT.
- `hetnetPolicy.py` — stacks L HetGAT layers with LayerNorm + residuals (residual skips layer 0 because dims change), per-class action heads, `log_std` clamped at construction and post-step.
- `critic.py` — `HetNetCritic` with centralized / per-class / per-agent modes, reads from SSN residually enriched with mean node features.
- `rollout.py` — `MAHACBuffer`: (T, B, …) pre-allocated storage plus per-timestep LSTM hidden-state snapshots so PPO can recompute log-probs on the same hidden-state trajectory.
- `mahac.py` — PPO outer loop, GAE on per-class values, plain MSE critic on raw returns (no value clipping, no return normalization — CleanRL/SpinUp style).
- `utils.py` — `build_ssn_input`, `compute_gae`, `collect_rollout`.

---

## Environments

Four VMAS scenarios, each exposing three parallel adapter interfaces (`reset`/`step` for IPPO + GNN-MAPPO, `hetnet_*` for HetNet) so trainer code doesn't change across algorithms.

**Guarded Territory** (`guarded_territory.py`) — mixed cooperative–competitive. Scouts (wide FOV, can't tag) and interceptors (narrow FOV, can tag) defend zones from scripted intruders whose policy blends goal-seeking + evasion + noise, parameterized by `intruder_skill` for curriculum. The richest scenario, but the most confounded.

**Guided Coverage** (`guided_coverage.py`) — no intruders. Coverage task where only interceptors count toward `−mean(min_interceptor_distance_to_landmark)`, but scouts have the wider FOV. Scouts are useful only if they relay landmark locations over the GNN — clean sanity check.

**Predator-Prey (PP)** — homogeneous mode of `predator_capture_prey.py`. Agents split into "scout"/"interceptor" labels with identical FOV and capabilities. Homogeneous control for the PCP comparison.

**Predator-Capture-Prey (PCP)** — heterogeneous mode of `predator_capture_prey.py`, adapted from Section 7 of Seraj et al. 2021. Continuous-action 2D version of their grid-world. `interceptor_prey_fov = 0.0` is hard-coded — interceptors never observe the prey directly, so they can only close on it through scout-relayed information over the GNN. The cleanest observation-asymmetry setup we have.

---

## Notable Implementation Choices

A few things worth being ready to explain:

- **Other-agent positions are FOV-masked in the observation.** So the communication graph is the primary inter-agent information channel (not strictly exclusive — agents still see peers inside their FOV — but the asymmetry is concentrated there).
- **Reward caching** uses `if agent is self.all_agents[0]:` rather than `_elapsed_steps` (VMAS doesn't expose that field).
- **True-done vs time-limit bootstrap** — for envs without intrinsic termination, every episode end is a time-limit, so GAE bootstraps V(sₜ₊₁) instead of treating it as terminal zero. True completions tracked separately via `info`.
- **GATv2 over GATv1** — LeakyReLU before the attention dot product; vanilla GAT attention is only monotonic (Brody et al. 2022).
- **Plain MSE critic, raw returns, no value clipping** — aligns with CleanRL and SpinUp. Earlier attempts with clipped values or normalized returns caused explained-variance saturation and killed actor gradient signal.
- **`log_std` clamped** to prevent entropy collapse / explosion. No tanh squashing — VMAS already clamps actions via `max_speed`, avoids the Jacobian-correction bug surface.

---

## Infrastructure

VMAS (PyTorch-native vectorized simulator) + UCF ARCC Newton HPC via SLURM (account `cenyioha`, partition `normal`). Experiments sweep algorithm × environment × team-size. References consulted for correctness: CleanRL, SpinningUp, MAPPO/EPyMARL, BenchMARL, VMAS internals.

---

## Primary References

| Reference | Role |
|---|---|
| Li, Gama, Ribeiro, Prorok (2020) — *GNNs for Decentralized Multi-Robot Path Planning* | Architectural template for GNN-MAPPO (encoder + GNN + action head). |
| Khan, Tolstaya, Ribeiro, Kumar (2019) — *Graph Policy Gradients for Large-Scale Robot Control* | GCN-parameterized policy, permutation-equivariance, zero-shot transfer. |
| Kipf & Welling (2017) — *Semi-Supervised Classification with GCNs* | Symmetric-normalized adjacency GCN layer. |
| Seraj et al. (2021) — *HetNet / MAHAC* | HetNet architecture and the PCP scenario. |
| Veličković et al. (2018) — *GAT* | Base graph attention formulation. |
| Brody, Alon, Yahav (2022) — *How Attentive Are GATs?* | Justification for GATv2 in `hetgatLayer.py`. |
| Schulman et al. (2017) — *PPO* / Schulman et al. (2016) — *GAE* | Learning algorithm and advantage estimator. |

---

## Current Status

Full algorithm × environment matrix is running end-to-end on SLURM. Earlier training-dynamics bugs are resolved (GAE bootstrap under `n_intruders=0`, reward-caching gating, LSTM hidden-state reset on done, connectivity-bias sampling over symmetric-normalized adjacency, value-clipping/return-normalization removal).

Two things worth discussing at the meeting:
1. **IPPO is competitive with — sometimes beating — the GNN variants early in training** on several configurations. Could be a genuine finding (observations informative enough that communication doesn't help), could be undertraining, could be a residual issue in the gradient path through the GNN. Need longer runs to disambiguate.
2. **HetNet throughput bottleneck** is the sequential per-timestep LSTM unroll in the PPO update. Batching all T×B samples into a single forward pass is the next engineering task — estimated 30–50× speedup, needed before scaling team sizes.

---

## Layout

```
environments/
  guarded_territory.py     # mixed coop-competitive, scripted intruders
  guided_coverage.py       # no intruders, GNN relay required
  predator_capture_prey.py # PP (homogeneous) + PCP (heterogeneous) modes
gnn/                       # GNN-MAPPO (homogeneous shared GCN)
HetGAT/                    # HetNet / MAHAC (heterogeneous GAT)
baseline/                  # IPPO control
```