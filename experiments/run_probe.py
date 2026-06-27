"""
Observation predictability probe runner.

Rolls a random policy in a PursuitAdapter (no training), collects per-step
per-agent local observations and ground-truth target positions, then trains
one TargetPredictor probe per agent class and reports:

    probe_score_full_sight      R2 for class-0 (full-sight) agents
    probe_score_sensor_limited  R2 for class-1 (sensor-limited) agents
    probe_gap                   full_sight - sensor_limited

Corner cases:
    alpha=1 -> both scores high, gap ~= 0
    alpha=0 -> sensor-limited score ~= 0 (chance), gap large

Ground-truth target xy comes ONLY from info["target_pos"] -- NEVER from obs
slices (which would be circular, especially at alpha=1 where the target slice
is non-zero).

Usage:
    python -m experiments.run_probe --config configs/pursuit_base.yaml
    python -m experiments.run_probe --config configs/pursuit_base.yaml --alpha 0.0
    python -m experiments.run_probe --config configs/pursuit_base.yaml --alpha 1.0 --seed 42
    python -m experiments.run_probe --config configs/pursuit_base.yaml --probe-mode post \\
        --ckpt runs/pursuit/alpha/identity/alpha1.0/seed0/ckpt/final.pt
"""

import argparse
import copy
from pathlib import Path
from typing import Optional

import torch

from backbone.utils import load_yaml_config
from envs.pursuit_adapter import PursuitAdapter
from probes.predictability import train_probe, probe_gap, fit_probe_on_features


def _build_probe_env(env_cfg, num_envs: int, seed: int, device_str: str):
    """
    Build the probe env from env_cfg, dispatching on env_cfg['kind'].

    Returns a PursuitAdapter (kind='pursuit', default) or PCPAdapter (kind='pcp').
    Both expose the same reset/step/agent_positions/class_id surface, so the rest
    of the probe is env-agnostic. PCP forwards detect_radius/detection_window.
    """
    kind = env_cfg.get("kind", "pursuit")
    common = dict(
        scenario=env_cfg.get("name", kind),
        num_envs=num_envs,
        n_agents=int(env_cfg.get("num_agents", 3)),
        max_steps=int(env_cfg.get("max_steps", 100)),
        seed=seed,
        device=device_str,
        include_timestep=bool(env_cfg.get("include_timestep_in_state", True)),
        alpha=float(env_cfg.get("alpha", 1.0)),
        num_full_sight=int(env_cfg.get("num_full_sight", 2)),
        observe_teammates=bool(env_cfg.get("observe_teammates", False)),
        world_size=float(env_cfg.get("world_size", 1.0)),
        capture_radius=float(env_cfg.get("capture_radius", 0.1)),
        move_step=float(env_cfg.get("move_step", 0.05)),
        target_speed=float(env_cfg.get("target_speed", 0.03)),
        dist_coef=float(env_cfg.get("dist_coef", 1.0)),
        step_penalty=float(env_cfg.get("step_penalty", 0.01)),
        capture_bonus=float(env_cfg.get("capture_bonus", 10.0)),
        capture_k=int(env_cfg.get("capture_k", 1)),
    )
    if kind == "pcp":
        from envs.pcp_adapter import PCPAdapter
        return PCPAdapter(
            detect_radius=float(env_cfg["detect_radius"]),
            detection_window=int(env_cfg["detection_window"]),
            **common,
        )
    return PursuitAdapter(**common)


def main(
    config_path: str,
    alpha: Optional[float] = None,
    seed: Optional[int] = None,
    ckpt: Optional[str] = None,
    probe_mode: str = "raw",
) -> None:
    """
    Run the predictability probe.

    Parameters
    ----------
    config_path : path to YAML config (e.g. configs/pursuit_base.yaml).
    alpha       : optional float override for env.alpha.
    seed        : optional int override for config seed.
    ckpt        : path to a model checkpoint (required when probe_mode=='post').
    probe_mode  : 'raw' (default) or 'post'. 'raw' trains directly on obs;
                  'post' trains on post-comm encoder embeddings.
    """
    if probe_mode not in ("raw", "post"):
        raise ValueError(f"probe_mode must be 'raw' or 'post', got {probe_mode!r}")
    if probe_mode == "post" and ckpt is None:
        raise ValueError("probe_mode='post' requires --ckpt to be provided.")

    if probe_mode == "raw":
        _run_raw_probe(config_path, alpha=alpha, seed=seed)
    else:
        _run_post_probe(config_path, alpha=alpha, seed=seed, ckpt=ckpt)


# ---------------------------------------------------------------------------
# Raw probe branch (A3: keeps existing logic verbatim)
# ---------------------------------------------------------------------------

def _run_raw_probe(config_path: str, alpha=None, seed=None) -> None:
    """
    Original raw-observation probe.  Computes score_limited used as R2_c1_raw floor.
    """
    cfg = load_yaml_config(config_path)
    env_cfg = copy.deepcopy(cfg.get("env", {}))

    if alpha is not None:
        env_cfg["alpha"] = float(alpha)
    if seed is not None:
        cfg["seed"] = int(seed)

    actual_alpha = float(env_cfg.get("alpha", 1.0))
    run_seed     = int(cfg.get("seed", 0))

    probe_num_envs = 16
    probe_steps    = 200

    raw_device = cfg.get("device", "cpu")
    if raw_device == "auto":
        raw_device = "cpu"
    env = _build_probe_env(env_cfg, probe_num_envs, run_seed, raw_device)

    N        = env.num_agents
    B        = env.B
    obs_dim  = env.obs_dim
    class_id = env.class_id

    print(
        f"[probe] alpha={actual_alpha} seed={run_seed} "
        f"N={N} B={B} obs_dim={obs_dim} steps={probe_steps} "
        f"total_samples={probe_steps * B * N}"
    )

    obs_list    = []
    target_list = []

    obs, _ = env.reset()
    for _ in range(probe_steps):
        actions = torch.randint(0, env.action_dim, (B, N))
        obs, _, _, _, info = env.step(actions)
        target_xy_step = info["target_pos"].cpu()
        obs_cpu = obs.cpu()
        for n in range(N):
            obs_list.append(obs_cpu[:, n, :])
            target_list.append(target_xy_step)

    obs_3d    = torch.zeros(probe_steps, N, B, obs_dim)
    target_3d = torch.zeros(probe_steps, N, B, 2)
    for t in range(probe_steps):
        for n in range(N):
            idx = t * N + n
            obs_3d[t, n]    = obs_list[idx]
            target_3d[t, n] = target_list[idx]

    obs_by_agent    = obs_3d.permute(1, 0, 2, 3).reshape(N, probe_steps * B, obs_dim)
    target_by_agent = target_3d.permute(1, 0, 2, 3).reshape(N, probe_steps * B, 2)

    class_id_cpu = class_id.cpu()
    full_sight_indices     = (class_id_cpu == 0).nonzero(as_tuple=True)[0].tolist()
    sensor_limited_indices = (class_id_cpu == 1).nonzero(as_tuple=True)[0].tolist()

    def _gather_class(indices):
        if not indices:
            return None, None
        obs_parts    = [obs_by_agent[i] for i in indices]
        target_parts = [target_by_agent[i] for i in indices]
        return torch.cat(obs_parts, dim=0), torch.cat(target_parts, dim=0)

    obs_full, tgt_full         = _gather_class(full_sight_indices)
    obs_limited, tgt_limited   = _gather_class(sensor_limited_indices)

    score_full    = float("nan")
    score_limited = float("nan")

    if obs_full is not None:
        result_full = train_probe(
            obs_full, tgt_full,
            seed=run_seed, epochs=200, hidden=64, val_frac=0.2, lr=1e-3,
        )
        score_full = result_full["score"]
        print(
            f"[probe] full_sight    R2={score_full:.4f}  MSE={result_full['mse']:.6f} "
            f"(n={len(obs_full)})"
        )
    else:
        print("[probe] full_sight    -- no class-0 agents")

    if obs_limited is not None:
        result_limited = train_probe(
            obs_limited, tgt_limited,
            seed=run_seed, epochs=200, hidden=64, val_frac=0.2, lr=1e-3,
        )
        score_limited = result_limited["score"]
        print(
            f"[probe] sensor_limited R2={score_limited:.4f}  MSE={result_limited['mse']:.6f} "
            f"(n={len(obs_limited)})"
        )
    else:
        print("[probe] sensor_limited -- no class-1 agents")

    gap = probe_gap(score_full, score_limited)
    print(f"[probe] probe_gap (full - limited) = {gap:.4f}")
    print(
        f"[probe] Summary: alpha={actual_alpha} | "
        f"score_full_sight={score_full:.4f} | "
        f"score_sensor_limited={score_limited:.4f} | "
        f"gap={gap:.4f}"
    )


# ---------------------------------------------------------------------------
# Post-comm probe branch (A4-A7)
# ---------------------------------------------------------------------------

def _run_post_probe(config_path: str, alpha=None, seed=None, ckpt=None) -> None:
    """
    Post-comm embedding probe.

    Loads a trained checkpoint, collects encoder embeddings under a random policy,
    then trains probes on those embeddings to measure R2_c1_post vs R2_c1_raw.
    """
    from backbone import build_agent_from_ckpt
    from experiments.train_vmas_mappo import _build_comm_module, build_knn_mask, build_radius_mask

    cfg = load_yaml_config(config_path)
    env_cfg = copy.deepcopy(cfg.get("env", {}))

    if alpha is not None:
        env_cfg["alpha"] = float(alpha)
    if seed is not None:
        cfg["seed"] = int(seed)

    actual_alpha = float(env_cfg.get("alpha", 1.0))
    run_seed     = int(cfg.get("seed", 0))
    probe_num_envs = 16
    probe_steps    = 200

    # Build probe device from config (cpu only for safety)
    device_str = cfg.get("device", "cpu")
    if device_str == "auto":
        device_str = "cpu"
    dev = torch.device(device_str)

    # A4.1: Build env identically to raw path (kind-dispatched: pursuit or pcp)
    env = _build_probe_env(env_cfg, probe_num_envs, run_seed, device_str)

    N        = env.num_agents
    B        = env.B
    obs_dim  = env.obs_dim
    class_id = env.class_id  # [N] long

    # A4.2: Load checkpoint
    # A4.3: Read sibling config.yaml for comm/model sub-keys
    run_cfg_path = Path(ckpt).parent.parent / "config.yaml"
    if run_cfg_path.exists():
        run_cfg = load_yaml_config(str(run_cfg_path))
    else:
        # Fall back to the provided config if sibling doesn't exist
        run_cfg = cfg

    comm_cfg_run  = run_cfg.get("comm", {})
    model_cfg_run = run_cfg.get("model", {})

    # A4.4: Build model with the checkpoint's comm module via the algo-aware factory.
    # Dispatches on the saved config["algo"] (MAPPO or QMIX QAgent); device_str forces
    # the probe device so GPU-trained ckpts load on a CPU probe node (the factory does
    # the device override internally). The probe only calls encode(), which both
    # backbones provide, so the QMIX mixer is not needed here.
    comm_module = _build_comm_module(comm_cfg_run, model_cfg_run)
    model = build_agent_from_ckpt(ckpt, comm_module, device_str)
    model.to(dev)

    # Topology settings for mask construction
    topology    = comm_cfg_run.get("topology", "full")
    knn_k       = int(comm_cfg_run.get("knn_k", 4))
    symmetrize  = comm_cfg_run.get("symmetrize", "or")
    comm_radius = comm_cfg_run.get("comm_radius", None)

    env_class_id = class_id.to(dev)

    print(
        f"[probe-post] alpha={actual_alpha} seed={run_seed} "
        f"N={N} B={B} obs_dim={obs_dim} D={model.hidden_dim} "
        f"steps={probe_steps} ckpt={ckpt}"
    )

    # A4.5: Roll probe_steps with RANDOM policy
    h_list          = []   # post-comm embeddings per step
    obs_list        = []   # raw obs per step
    target_list     = []   # target_pos per step
    truncated_list  = []   # truncated flag per step

    obs, _ = env.reset()
    for _ in range(probe_steps):
        actions = torch.randint(0, env.action_dim, (B, N))
        obs, _, _, _, info = env.step(actions)

        target_xy_step = info["target_pos"].cpu()
        truncated_step = info["truncated"].cpu()  # [B] bool

        # Build comm mask using same topology branch as trainer
        positions = getattr(env, "agent_positions", None)
        comm_mask = None
        if positions is not None:
            if topology == "knn":
                comm_mask = build_knn_mask(positions.to(dev), knn_k, symmetrize)
            elif topology == "radius" or comm_radius is not None:
                comm_mask = build_radius_mask(positions.to(dev), comm_radius)

        obs_dev = obs.to(dev)
        with torch.no_grad():
            # encode: obs -> encoder -> comm -> h [B, N, D]
            h_step = model.encode(obs_dev, mask=comm_mask, class_id=env_class_id)

        h_list.append(h_step.cpu())
        obs_list.append(obs.cpu())
        target_list.append(target_xy_step)
        truncated_list.append(truncated_step)

    D = model.hidden_dim

    # A5: Truncation filter -- keep only steps where truncated is all-False
    # Shape per step: h [B,N,D], obs [B,N,obs_dim], target [B,2], truncated [B]
    valid_h         = []
    valid_obs       = []
    valid_target    = []
    for t in range(probe_steps):
        trunc = truncated_list[t]  # [B] bool
        if not trunc.any().item():
            # All envs are non-terminal at this step
            valid_h.append(h_list[t])          # [B, N, D]
            valid_obs.append(obs_list[t])      # [B, N, obs_dim]
            valid_target.append(target_list[t])  # [B, 2]

    if not valid_h:
        print("[probe-post] No non-truncation steps found; cannot compute probe.")
        return

    # Stack: [S, B, N, D], [S, B, N, obs_dim], [S, B, 2]
    h_arr     = torch.stack(valid_h,      dim=0)   # [S, B, N, D]
    obs_arr   = torch.stack(valid_obs,    dim=0)   # [S, B, N, obs_dim]
    tgt_arr   = torch.stack(valid_target, dim=0)   # [S, B, 2]
    S = h_arr.shape[0]

    # A6: Compute R2 scores with class split
    # Use the same agent-major interleave/reshape pattern as the raw probe.
    # h [S, B, N, D]  ->  per-agent: [S*B, D]
    # target [S, B, 2] repeats per class member identically.

    class_id_cpu = class_id.cpu()
    class1_mask  = (class_id_cpu == 1)  # sensor-limited
    class0_mask  = (class_id_cpu == 0)  # full-sight

    class1_idx = class1_mask.nonzero(as_tuple=True)[0]
    class0_idx = class0_mask.nonzero(as_tuple=True)[0]

    n_class1 = class1_idx.numel()
    n_class0 = class0_idx.numel()

    # R2_c1_post: probe on post-comm embeddings for class-1 agents
    score_c1_post = float("nan")
    score_c1_raw  = float("nan")
    score_c0_local = float("nan")

    if n_class1 > 0:
        # h for class-1: [S, B, n_class1, D] -> reshape [S*B*n_class1, D]
        h_c1 = h_arr[:, :, class1_idx, :].reshape(-1, D)   # [S*B*n_class1, D]
        # target repeats per class-1 agent: [S, B, 2] -> tile n_class1 times -> [S*B*n_class1, 2]
        tgt_c1 = tgt_arr.unsqueeze(2).expand(S, B, n_class1, 2).reshape(-1, 2)

        # R2_c1_post
        res_c1_post = fit_probe_on_features(
            h_c1.float(), tgt_c1.float(),
            seed=run_seed, epochs=200, hidden=64, val_frac=0.2, lr=1e-3,
        )
        score_c1_post = res_c1_post["score"]

        # R2_c1_raw: probe directly on raw obs for class-1 agents
        obs_c1 = obs_arr[:, :, class1_idx, :].reshape(-1, obs_dim)  # [S*B*n_class1, obs_dim]
        res_c1_raw = train_probe(
            obs_c1.float(), tgt_c1.float(),
            seed=run_seed, epochs=200, hidden=64, val_frac=0.2, lr=1e-3,
        )
        score_c1_raw = res_c1_raw["score"]

    if n_class0 > 0:
        # R2_c0_local: probe on raw obs for class-0 agents (ceiling)
        obs_c0 = obs_arr[:, :, class0_idx, :].reshape(-1, obs_dim)  # [S*B*n_class0, obs_dim]
        tgt_c0 = tgt_arr.unsqueeze(2).expand(S, B, n_class0, 2).reshape(-1, 2)
        res_c0 = train_probe(
            obs_c0.float(), tgt_c0.float(),
            seed=run_seed, epochs=200, hidden=64, val_frac=0.2, lr=1e-3,
        )
        score_c0_local = res_c0["score"]

    # A7: Branch classification
    r2_ceiling = score_c0_local if not (score_c0_local != score_c0_local) else 0.0
    gain = (score_c1_post - score_c1_raw) if not (score_c1_post != score_c1_post or score_c1_raw != score_c1_raw) else float("nan")

    if (
        not (score_c1_post != score_c1_post)
        and not (r2_ceiling != r2_ceiling)
        and r2_ceiling > 0.0
        and score_c1_post >= 0.5 * r2_ceiling
        and gain >= 0.3
    ):
        branch = "HIGH"
    elif (
        not (score_c1_post != score_c1_post)
        and not (gain != gain)
        and score_c1_post < 0.15
        and gain < 0.1
    ):
        branch = "LOW"
    else:
        branch = "INTERMEDIATE"

    print(
        f"[probe-post] R2_c1_post={score_c1_post:.4f} "
        f"R2_c1_raw={score_c1_raw:.4f} "
        f"R2_c0_local(ceil)={score_c0_local:.4f} "
        f"branch={branch}"
    )
    if branch == "INTERMEDIATE":
        print(
            "[probe-post] INTERMEDIATE -> apply capture_k=4 first; "
            "aux-loss only second-line if still intermediate AND capture<0.15"
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
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a model checkpoint (.pt). Required when --probe-mode post.",
    )
    parser.add_argument(
        "--probe-mode",
        choices=("raw", "post"),
        default="raw",
        dest="probe_mode",
        help="'raw': probe raw observations (default). 'post': probe post-comm embeddings.",
    )
    parser.add_argument(
        "--comm-radius",
        type=float,
        default=None,
        dest="comm_radius",
        help="Override comm radius for mask construction in post-mode (optional).",
    )
    args = parser.parse_args()
    main(
        args.config,
        alpha=args.alpha,
        seed=args.seed,
        ckpt=args.ckpt,
        probe_mode=args.probe_mode,
    )
