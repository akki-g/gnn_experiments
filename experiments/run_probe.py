"""
Observation predictability probe runner.

Rolls a random policy in a PursuitAdapter (no training), collects per-step
per-agent local observations and ground-truth target positions, then trains
one TargetPredictor probe per agent class and reports:

    probe_score_full_sight      R² for class-0 (full-sight) agents
    probe_score_sensor_limited  R² for class-1 (sensor-limited) agents
    probe_gap                   full_sight - sensor_limited

Corner cases:
    alpha=1 → both scores high, gap ≈ 0
    alpha=0 → sensor-limited score ≈ 0 (chance), gap large

Ground-truth target xy comes ONLY from info["target_pos"] — NEVER from obs
slices (which would be circular, especially at alpha=1 where the target slice
is non-zero).

Usage:
    python -m experiments.run_probe --config configs/pursuit_base.yaml
    python -m experiments.run_probe --config configs/pursuit_base.yaml --alpha 0.0
    python -m experiments.run_probe --config configs/pursuit_base.yaml --alpha 1.0 --seed 42
"""

import argparse
import copy

import torch

from backbone.utils import load_yaml_config
from envs.pursuit_adapter import PursuitAdapter
from probes.predictability import train_probe, probe_gap


def main(config_path: str, alpha: float = None, seed: int = None) -> None:
    """
    Run the predictability probe.

    Parameters
    ----------
    config_path : path to YAML config (e.g. configs/pursuit_base.yaml).
    alpha       : optional float override for env.alpha.
    seed        : optional int override for config seed.
    """
    cfg = load_yaml_config(config_path)
    env_cfg = copy.deepcopy(cfg.get("env", {}))

    # Override alpha / seed if provided
    if alpha is not None:
        env_cfg["alpha"] = float(alpha)
    if seed is not None:
        cfg["seed"] = int(seed)

    actual_alpha = float(env_cfg.get("alpha", 1.0))
    run_seed     = int(cfg.get("seed", 0))

    # Build a small/medium env (override num_envs to a moderate value for speed)
    probe_num_envs = 16
    probe_steps    = 200   # steps × envs = total samples per run

    env = PursuitAdapter(
        scenario=env_cfg.get("name", "pursuit"),
        num_envs=probe_num_envs,
        n_agents=int(env_cfg.get("num_agents", 3)),
        max_steps=int(env_cfg.get("max_steps", 100)),
        seed=run_seed,
        device=cfg.get("device", "cpu") if cfg.get("device", "cpu") != "auto" else "cpu",
        include_timestep=bool(env_cfg.get("include_timestep_in_state", True)),
        alpha=actual_alpha,
        num_full_sight=int(env_cfg.get("num_full_sight", 2)),
        observe_teammates=bool(env_cfg.get("observe_teammates", False)),
        world_size=float(env_cfg.get("world_size", 1.0)),
        capture_radius=float(env_cfg.get("capture_radius", 0.1)),
        move_step=float(env_cfg.get("move_step", 0.05)),
        target_speed=float(env_cfg.get("target_speed", 0.03)),
        dist_coef=float(env_cfg.get("dist_coef", 1.0)),
        step_penalty=float(env_cfg.get("step_penalty", 0.01)),
        capture_bonus=float(env_cfg.get("capture_bonus", 10.0)),
    )

    N          = env.num_agents
    B          = env.B
    obs_dim    = env.obs_dim
    class_id   = env.class_id  # [N] long, on env device

    print(
        f"[probe] alpha={actual_alpha} seed={run_seed} "
        f"N={N} B={B} obs_dim={obs_dim} steps={probe_steps} "
        f"total_samples={probe_steps * B * N}"
    )

    # Collect rollout with RANDOM policy (no training)
    obs_list    = []   # will collect [M_total, obs_dim] per class
    target_list = []   # [M_total, 2] per class

    obs, _ = env.reset()  # [B, N, obs_dim]
    actions = torch.zeros(B, N, dtype=torch.long)

    for _ in range(probe_steps):
        # Random actions: uniform over action_dim=5
        actions = torch.randint(0, env.action_dim, (B, N))
        obs, _, _, _, info = env.step(actions)

        # Ground-truth target: info["target_pos"] [B, 2]  — NEVER from obs slice
        target_xy_step = info["target_pos"].cpu()  # [B, 2]

        # Collect per-agent obs + target (repeat target_xy for each agent)
        obs_cpu = obs.cpu()   # [B, N, obs_dim]
        for n in range(N):
            obs_list.append(obs_cpu[:, n, :])      # [B, obs_dim]
            target_list.append(target_xy_step)      # [B, 2]

    # Concatenate all collected samples per (step, agent) — shape [probe_steps*B, obs_dim]
    # We have probe_steps entries per agent → reshape accordingly.
    # obs_list has probe_steps * N entries each [B, obs_dim]
    # Interleaved by agent index within each step.
    # Group by agent index: agent n's data is at indices n, n+N, n+2N, ...
    all_obs    = torch.cat(obs_list, dim=0)     # [probe_steps*B*N, obs_dim]  (interleaved)
    all_target = torch.cat(target_list, dim=0)  # [probe_steps*B*N, 2]

    # Re-split by agent: step i, agent n → index i*N + n  ... but we stacked per step
    # each step appends N chunks of [B, obs_dim], so flat index for step t, agent n:
    # index = t * N + n  in the obs_list list (before cat).
    # After cat with dim=0: flat rows are [step0_agent0, step0_agent1, ..., stepT_agentN-1]
    # each of size B. So for agent n: rows n*B, (n+N)*B ... no, let's just re-split cleanly.

    # Reshape: [probe_steps, N, B, obs_dim] → transpose for clean split
    obs_3d    = torch.zeros(probe_steps, N, B, obs_dim)
    target_3d = torch.zeros(probe_steps, N, B, 2)
    for t in range(probe_steps):
        for n in range(N):
            idx = t * N + n
            obs_3d[t, n]    = obs_list[idx]
            target_3d[t, n] = target_list[idx]

    # Per-agent flat: [probe_steps*B, obs_dim]
    obs_by_agent    = obs_3d.permute(1, 0, 2, 3).reshape(N, probe_steps * B, obs_dim)
    target_by_agent = target_3d.permute(1, 0, 2, 3).reshape(N, probe_steps * B, 2)

    # Split by class using class_id [N]
    class_id_cpu = class_id.cpu()
    full_sight_indices    = (class_id_cpu == 0).nonzero(as_tuple=True)[0].tolist()
    sensor_limited_indices = (class_id_cpu == 1).nonzero(as_tuple=True)[0].tolist()

    def _gather_class(indices):
        if not indices:
            return None, None
        obs_parts    = [obs_by_agent[i] for i in indices]
        target_parts = [target_by_agent[i] for i in indices]
        return torch.cat(obs_parts, dim=0), torch.cat(target_parts, dim=0)

    obs_full, tgt_full     = _gather_class(full_sight_indices)
    obs_limited, tgt_limited = _gather_class(sensor_limited_indices)

    # Train probes
    score_full = float("nan")
    score_limited = float("nan")

    if obs_full is not None:
        result_full = train_probe(
            obs_full, tgt_full,
            seed=run_seed, epochs=200, hidden=64, val_frac=0.2, lr=1e-3,
        )
        score_full = result_full["score"]
        print(
            f"[probe] full_sight    R²={score_full:.4f}  MSE={result_full['mse']:.6f} "
            f"(n={len(obs_full)})"
        )
    else:
        print("[probe] full_sight    — no class-0 agents")

    if obs_limited is not None:
        result_limited = train_probe(
            obs_limited, tgt_limited,
            seed=run_seed, epochs=200, hidden=64, val_frac=0.2, lr=1e-3,
        )
        score_limited = result_limited["score"]
        print(
            f"[probe] sensor_limited R²={score_limited:.4f}  MSE={result_limited['mse']:.6f} "
            f"(n={len(obs_limited)})"
        )
    else:
        print("[probe] sensor_limited — no class-1 agents")

    gap = probe_gap(score_full, score_limited)
    print(f"[probe] probe_gap (full - limited) = {gap:.4f}")
    print(
        f"[probe] Summary: alpha={actual_alpha} | "
        f"score_full_sight={score_full:.4f} | "
        f"score_sensor_limited={score_limited:.4f} | "
        f"gap={gap:.4f}"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the target-position predictability probe on a PursuitAdapter."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/pursuit_base.yaml).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override env.alpha (observation radius fraction). Default: use config value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed. Default: use config seed.",
    )
    args = parser.parse_args()
    main(args.config, alpha=args.alpha, seed=args.seed)
