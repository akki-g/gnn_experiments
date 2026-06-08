"""
MAPPO training on VMAS environments (smoke test: simple_spread).

Usage:
    python -m experiments.train_vmas_mappo --config configs/vmas_simple_spread_mappo.yaml --render off
    python -m experiments.train_vmas_mappo --config configs/vmas_simple_spread_mappo.yaml \\
        --seed 1 --render off
    python -m experiments.train_vmas_mappo --config configs/vmas_simple_spread_mappo.yaml \\
        --resume runs/vmas_simple_spread/<timestamp>/ckpt/final.pt

Artifacts produced in runs/vmas_simple_spread/<timestamp>/:
    ckpt/final.pt        — checkpoint after final iteration
    ckpt/best.pt         — checkpoint at best mean episode return
    ckpt/ckpt_<iter>.pt  — periodic checkpoints
    config.yaml          — copy of config used for this run
    metrics.json         — list of per-iteration metric dicts
    metrics.csv          — same data in CSV format
    console.log          — copy of all stdout printed during training
    policy.mp4/.gif      — rendered only when --render on/auto permits it
"""

import csv
import json
import math
import os
import sys
import argparse
from datetime import datetime

import torch

from backbone.utils import load_yaml_config, set_seed, get_device, make_optimizer
from backbone import MAPPO, RolloutBuffer
from backbone.ppo_update import ppo_update
from comm import IdentityComm, BroadcastComm, AttentionComm, GraphComm
from envs.adapters import VMASAdapter

import yaml


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _build_env(env_cfg: dict, num_envs: int, seed: int, device: str):
    """
    Instantiate the correct environment adapter based on env_cfg.get("kind", "vmas").

    Parameters
    ----------
    env_cfg  : the "env" sub-dict from the YAML config.
    num_envs : B — number of parallel environments (from runtime config).
    seed     : int random seed.
    device   : torch device string.

    Returns an object with the VectorEnvAdapter interface:
        .obs_dim, .action_dim, .state_dim, .num_agents, .B, .device
        .reset() -> (obs, state)
        .step(actions) -> (obs, state, reward, done, info)
    """
    kind = env_cfg.get("kind", "vmas")
    if kind == "vmas":
        return VMASAdapter(
            scenario=env_cfg["scenario"],
            num_envs=num_envs,
            n_agents=env_cfg["num_agents"],
            max_steps=env_cfg["max_steps"],
            seed=seed,
            device=device,
            include_timestep=env_cfg.get("include_timestep_in_state", True),
        )
    elif kind == "pursuit":
        from envs.pursuit_adapter import PursuitAdapter
        return PursuitAdapter(
            scenario=env_cfg.get("name", "pursuit"),
            num_envs=num_envs,
            n_agents=env_cfg["num_agents"],
            max_steps=env_cfg["max_steps"],
            seed=seed,
            device=device,
            include_timestep=env_cfg.get("include_timestep_in_state", True),
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
        )
    else:
        raise ValueError(
            f"Unknown env kind '{kind}'. Supported: 'vmas', 'pursuit'."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Tee:
    """Write-through proxy that copies stdout to both the terminal and a log file."""

    def __init__(self, logfile):
        self._logfile = logfile

    def write(self, s):
        sys.__stdout__.write(s)
        self._logfile.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self._logfile.flush()


def _build_comm_module(comm_cfg: dict, model_cfg: dict):
    """
    Instantiate the comm module named in the config.

    Parameters
    ----------
    comm_cfg  : the "comm" sub-dict from the YAML config.
    model_cfg : the "model" sub-dict (used to read hidden_dim).
    """
    name = comm_cfg.get("module", "identity").lower()
    hidden_dim = model_cfg.get("hidden_dim", 64)
    if name == "identity":
        return IdentityComm()
    elif name == "broadcast":
        return BroadcastComm(hidden_dim, rounds=comm_cfg.get("rounds", 1))
    elif name == "attention":
        return AttentionComm(
            hidden_dim,
            num_heads=comm_cfg.get("num_heads", 1),
            rounds=comm_cfg.get("rounds", 1),
        )
    elif name == "graph":
        return GraphComm(
            hidden_dim,
            num_heads=comm_cfg.get("num_heads", 4),
            num_layers=comm_cfg.get("num_hops", 2),
        )
    else:
        raise ValueError(
            f"Unknown comm module '{name}'. "
            "Supported: 'identity', 'broadcast', 'attention', 'graph'."
        )


def build_radius_mask(positions: "Tensor", radius: float) -> "Tensor":
    """
    Build a boolean comm mask from agent positions and a communication radius.

    Parameters
    ----------
    positions : [B, N, 2]  agent 2D positions.
    radius    : float communication range (inclusive).

    Returns
    -------
    mask : [B, N, N] bool  True if receiver i can receive from sender j,
           i.e. dist(i, j) <= radius.  Diagonal is always False (no self-loop).
    """
    import torch
    assert positions.dim() == 3 and positions.shape[-1] == 2, (
        f"positions must be [B, N, 2], got {positions.shape}"
    )
    # diff[b, i, j, :] = pos[b, i, :] - pos[b, j, :]
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, N, N, 2]
    dist = diff.norm(dim=-1)  # [B, N, N]
    mask = dist <= radius     # [B, N, N] bool
    # Zero diagonal: no self-communication
    N = positions.shape[1]
    idx = torch.arange(N, device=positions.device)
    mask[:, idx, idx] = False
    return mask


def _safe_run_label(label: str | None) -> str:
    """Sanitize an optional run label for use in run directory names."""
    if not label:
        return ""
    safe = []
    previous_separator = False
    for ch in str(label):
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
            previous_separator = ch == "_"
        elif ch in {".", " ", "/", "\\"} and not previous_separator:
            safe.append("_")
            previous_separator = True
    return "".join(safe).strip("_")


def _make_run_dir(log_dir: str, seed: int, run_label: str | None = None) -> str:
    """
    Build a collision-resistant run directory for local runs and SLURM arrays.

    Includes microseconds, seed, pid, and SLURM job/task ids when available.
    """
    for attempt in range(10):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        parts = [timestamp]
        safe_label = _safe_run_label(run_label)
        if safe_label:
            parts.append(safe_label)
        parts.extend([f"seed{seed}", f"pid{os.getpid()}"])
        for env_key, label in (
            ("SLURM_ARRAY_JOB_ID", "arrayjob"),
            ("SLURM_JOB_ID", "job"),
            ("SLURM_ARRAY_TASK_ID", "task"),
        ):
            value = os.environ.get(env_key)
            if value:
                parts.append(f"{label}{value}")
        if attempt:
            parts.append(f"retry{attempt}")
        run_dir = os.path.join(log_dir, "_".join(parts))
        try:
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique run directory under {log_dir}")


def _resolve_render_mode(cfg: dict, override: str = None) -> str:
    """Return one of {'off', 'auto', 'on'} from CLI override or config."""
    mode = override if override is not None else cfg.get("render", {}).get("mode", "off")
    mode = str(mode).lower()
    if mode not in {"off", "auto", "on"}:
        raise ValueError(f"render mode must be one of off/auto/on, got {mode!r}")
    cfg.setdefault("render", {})["mode"] = mode
    return mode


def _is_headless() -> bool:
    """Best-effort display check for VMAS/pyglet rendering."""
    if os.name == "nt":
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _bad_mask_from_info(info: dict, num_agents: int, device: torch.device):
    """Build [B,N] bad_mask where 0.0 marks time-limit truncations."""
    truncated = torch.as_tensor(info["truncated"], device=device, dtype=torch.bool)
    if truncated.dim() == 0:
        truncated = truncated.unsqueeze(0)
    if not bool(truncated.any()):
        return None
    return (~truncated).float().unsqueeze(1).expand(truncated.shape[0], num_agents).clone()


def _terminal_state_from_info(info: dict, fallback_state: torch.Tensor, device: torch.device):
    """Use terminal state for bootstrap when the adapter reset after a done boundary."""
    terminal_state = info.get("terminal_state")
    if terminal_state is None:
        return fallback_state
    truncated = torch.as_tensor(info.get("truncated", False), device=device, dtype=torch.bool)
    if truncated.dim() == 0:
        truncated = truncated.repeat(fallback_state.shape[0])
    if bool(truncated.any()):
        bootstrap_state = fallback_state.clone()
        bootstrap_state[truncated] = terminal_state.to(device)[truncated]
        return bootstrap_state
    return fallback_state


def _linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    """Linearly interpolate start -> end across total_steps calls."""
    if total_steps <= 1:
        return float(end)
    progress = min(max(step / (total_steps - 1), 0.0), 1.0)
    return float(start + progress * (end - start))


def _gap_to_zero(value: float) -> float:
    """For non-positive cost rewards, report the remaining positive gap to zero."""
    if math.isnan(value):
        return float("nan")
    return max(0.0, -float(value))


def _delta_from_start(value: float, start: float | None) -> float:
    """Positive means the reward has improved relative to the run's first value."""
    if start is None or math.isnan(value):
        return float("nan")
    return float(value - start)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set all optimizer parameter groups to the same active learning rate."""
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _task_score(coverage_distance: float, collision_pairs: float) -> float:
    """
    Positive bounded diagnostic score for simple_spread.

    Raw VMAS reward remains the optimization/evaluation target. This score is
    only a monotonic readout: higher means lower coverage distance and fewer
    collisions; 1.0 is the no-distance/no-collision ideal.
    """
    if math.isnan(coverage_distance) or math.isnan(collision_pairs):
        return float("nan")
    coverage_distance = max(0.0, float(coverage_distance))
    collision_pairs = max(0.0, float(collision_pairs))
    return 1.0 / ((1.0 + coverage_distance) * (1.0 + collision_pairs))


def _pursuit_task_score(pursuer_target_distance: float, capture_rate: float) -> float:
    """
    Positive bounded diagnostic score for pursuit.

    Monotone: higher capture_rate and lower pursuer_target_distance → higher score.
    Score ∈ (0, 1].
    Returns 0.0 when capture_rate == 0 or either input is nan/negative.

    Formula: capture_rate * (1 / (1 + pursuer_target_distance))  clamped to [0, 1].
    """
    if math.isnan(pursuer_target_distance) or math.isnan(capture_rate):
        return float("nan")
    capture_rate = max(0.0, float(capture_rate))
    pursuer_target_distance = max(0.0, float(pursuer_target_distance))
    if capture_rate == 0.0:
        return 0.0
    score = capture_rate * (1.0 / (1.0 + pursuer_target_distance))
    return float(min(1.0, max(0.0, score)))


def _evaluate_deterministic_policy(
    model: MAPPO,
    cfg: dict,
    device: torch.device,
) -> dict:
    """Run deterministic no-video evaluation episodes in a fresh env."""
    env_cfg = cfg["env"]
    reward_shift = float(cfg.get("algo", {}).get("reward_shift", 0.0))
    render_cfg = cfg.get("render", {})
    num_episodes = int(render_cfg.get("episodes", 1))
    if num_episodes <= 0:
        raise ValueError("render.episodes must be >= 1 when runtime.eval_interval > 0")

    eval_seed = int(cfg["seed"]) + int(render_cfg.get("eval_seed", 1000))
    eval_env = _build_env(env_cfg, num_envs=1, seed=eval_seed, device=str(device))

    primary_diag_key   = env_cfg.get("primary_diagnostic",   "coverage_distance")
    secondary_diag_key = env_cfg.get("secondary_diagnostic", "collision_pairs")
    task_score_kind    = env_cfg.get("task_score_kind",       "spread")

    model.eval()
    episode_returns = []
    optimized_episode_returns = []
    episode_lengths = []
    primary_diag_vals   = []
    secondary_diag_vals = []
    for _ in range(num_episodes):
        obs, state = eval_env.reset()
        ep_return = 0.0
        ep_len = 0
        for _ in range(env_cfg["max_steps"]):
            out = model.act(obs.to(device), state.to(device), deterministic=True)
            obs, state, reward, done, info = eval_env.step(out["action"].cpu())
            ep_return += reward[0, 0].item()
            ep_len += 1
            if primary_diag_key in info:
                primary_diag_vals.append(
                    torch.as_tensor(info[primary_diag_key], device=device).float().mean().item()
                )
            if secondary_diag_key in info:
                secondary_diag_vals.append(
                    torch.as_tensor(info[secondary_diag_key], device=device).float().mean().item()
                )
            if bool(done[0].any()):
                break
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        optimized_episode_returns.append(ep_return + reward_shift * ep_len)

    mean_return = sum(episode_returns) / len(episode_returns)
    optimized_mean_return = (
        sum(optimized_episode_returns) / len(optimized_episode_returns)
    )
    mean_length = sum(episode_lengths) / len(episode_lengths)
    mean_step_reward = mean_return / max(mean_length, 1.0)
    optimized_mean_step_reward = optimized_mean_return / max(mean_length, 1.0)
    mean_primary_diag = (
        sum(primary_diag_vals) / len(primary_diag_vals)
        if primary_diag_vals
        else float("nan")
    )
    mean_secondary_diag = (
        sum(secondary_diag_vals) / len(secondary_diag_vals)
        if secondary_diag_vals
        else float("nan")
    )

    if task_score_kind == "pursuit":
        eval_task_score = _pursuit_task_score(mean_primary_diag, mean_secondary_diag)
    else:
        eval_task_score = _task_score(mean_primary_diag, mean_secondary_diag)

    # Build return dict — use the actual diagnostic key names so VMAS rows keep
    # "eval_coverage_distance"/"eval_collision_pairs" and pursuit rows get
    # "eval_pursuer_target_distance"/"eval_capture".
    result = {
        "eval_mean_return": float(mean_return),
        "eval_mean_step_reward": float(mean_step_reward),
        "eval_optimized_mean_return": float(optimized_mean_return),
        "eval_optimized_mean_step_reward": float(optimized_mean_step_reward),
        "eval_mean_episode_length": float(mean_length),
        f"eval_{primary_diag_key}": float(mean_primary_diag),
        f"eval_{secondary_diag_key}": float(mean_secondary_diag),
        "eval_task_score": eval_task_score,
    }
    return result


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main(
    config_path: str,
    resume: str = None,
    seed: int = None,
    render: str = None,
) -> str:
    """
    Train MAPPO on a VMAS environment.

    Parameters
    ----------
    config_path : path to YAML config file.
    resume      : optional path to a checkpoint to resume from.
    seed        : optional integer seed override (overrides cfg["seed"]).
    render      : optional render mode override: "off", "auto", or "on".

    Returns
    -------
    run_dir : path to the run directory containing all artifacts.
    """
    cfg = load_yaml_config(config_path)

    # Seed override must happen BEFORE set_seed and run-dir creation
    if seed is not None:
        cfg["seed"] = int(seed)

    set_seed(cfg["seed"])
    device = get_device(prefer_cuda=(cfg["device"] != "cpu"))
    render_mode = _resolve_render_mode(cfg, render)

    log_cfg = cfg.setdefault("logging", {})
    run_dir = log_cfg.get("run_dir")
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
    else:
        # Collision-resistant run directory for local runs and SLURM arrays.
        run_dir = _make_run_dir(
            log_cfg["log_dir"],
            cfg["seed"],
            log_cfg.get("run_label"),
        )

    save_checkpoints = bool(log_cfg.get("save_checkpoints", True))
    save_config = bool(log_cfg.get("save_config", True))
    save_console_log = bool(log_cfg.get("save_console_log", True))
    ckpt_dir = os.path.join(run_dir, "ckpt")
    if save_checkpoints:
        os.makedirs(ckpt_dir, exist_ok=True)

    if save_config:
        config_copy_path = os.path.join(run_dir, "config.yaml")
        with open(config_copy_path, "w") as f:
            yaml.safe_dump(cfg, f)

    if save_console_log:
        log_file = open(os.path.join(run_dir, "console.log"), "w", buffering=1)
        tee = _Tee(log_file)
        sys.stdout = tee
        try:
            _train(cfg, device, run_dir, ckpt_dir, resume, render_mode)
        finally:
            sys.stdout = sys.__stdout__
            log_file.close()
    else:
        _train(cfg, device, run_dir, ckpt_dir, resume, render_mode)

    return run_dir


def _train(cfg, device, run_dir, ckpt_dir, resume, render_mode: str):
    """Inner training body — called inside the finally-guarded stdout tee."""

    env_cfg = cfg["env"]
    model_cfg = cfg.get("model", {})
    algo_cfg = cfg["algo"]
    runtime_cfg = cfg["runtime"]
    log_cfg = cfg["logging"]
    save_checkpoints = bool(log_cfg.get("save_checkpoints", True))
    save_metrics_json = bool(log_cfg.get("save_metrics_json", True))

    # ----- Environment -----
    env = _build_env(env_cfg, num_envs=runtime_cfg["num_envs"], seed=cfg["seed"], device=str(device))
    print(
        f"[env] kind={env_cfg.get('kind','vmas')} name={env_cfg.get('name', env_cfg.get('scenario','?'))} "
        f"n_agents={env.num_agents} obs_dim={env.obs_dim} action_dim={env.action_dim} "
        f"state_dim={env.state_dim} num_envs={env.B}"
    )

    # ----- Model -----
    model = MAPPO(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_agents=env.num_agents,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        comm_module=_build_comm_module(cfg.get("comm", {}), model_cfg),
        share_encoder=algo_cfg["share_encoder"],
        encoder_layers=model_cfg.get("encoder_layers", 2),
        actor_layers=model_cfg.get("actor_layers", 1),
        critic_layers=model_cfg.get("critic_layers", 2),
        device=str(device),
    )

    optimizer = make_optimizer(model.parameters(), lr=algo_cfg["lr"])

    T = runtime_cfg["rollout_length"]
    B = runtime_cfg["num_envs"]
    if T != env_cfg["max_steps"]:
        raise ValueError(
            "VMAS smoke training currently requires runtime.rollout_length == "
            "env.max_steps so time-limit bootstraps use terminal_state metadata "
            f"(got rollout_length={T}, max_steps={env_cfg['max_steps']})."
        )

    buffer = RolloutBuffer(
        rollout_length=T,
        num_envs=B,
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=str(device),
    )

    # ----- Resume -----
    global_step = 0
    if resume:
        ckpt = model.load(resume)
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        print(f"[resume] Loaded checkpoint from {resume} at global_step={global_step}")

    # ----- Training loop setup -----
    total_iters = max(1, runtime_cfg["total_env_steps"] // (T * B))
    checkpoint_interval = runtime_cfg["checkpoint_interval"]
    console_log_interval = log_cfg["console_log_interval"]
    eval_interval = int(runtime_cfg.get("eval_interval", 0))

    gamma = algo_cfg["gamma"]
    gae_lambda = algo_cfg["gae_lambda"]
    ppo_epochs = algo_cfg["ppo_epochs"]
    num_minibatches = algo_cfg["num_minibatches"]
    clip_eps = algo_cfg["clip_eps"]
    value_clip_eps = algo_cfg["value_clip_eps"]
    lr_start = algo_cfg["lr"]
    lr_final = algo_cfg.get("lr_final", lr_start)
    entropy_coef_start = algo_cfg["entropy_coef"]
    entropy_coef_final = algo_cfg.get("entropy_coef_final", entropy_coef_start)
    value_coef = algo_cfg["value_coef"]
    max_grad_norm = algo_cfg["max_grad_norm"]
    normalize_advantages = algo_cfg["normalize_advantages"]
    reward_shift = float(algo_cfg.get("reward_shift", 0.0))
    use_value_norm = algo_cfg.get("use_value_norm", True)
    value_normalizer = model.value_normalizer if use_value_norm else None

    obs, state = env.reset()
    obs, state = obs.to(device), state.to(device)

    ep_return = torch.zeros(B, device=device)
    optimized_ep_return = torch.zeros(B, device=device)
    completed_returns = []
    completed_optimized_returns = []
    best_return = -math.inf
    best_eval_return = -math.inf
    metrics_rows = []
    start_ep_return = None
    start_step_reward = None
    start_eval_return = None

    print(
        f"[train] Starting: total_iters={total_iters} T={T} B={B} "
        f"device={device} seed={cfg['seed']} render={render_mode}"
    )

    last_step_info = None

    # Diagnostic key names from config (VMAS uses hardcoded names; pursuit uses its own).
    primary_diag_key   = env_cfg.get("primary_diagnostic",   "coverage_distance")
    secondary_diag_key = env_cfg.get("secondary_diagnostic", "collision_pairs")
    task_score_kind    = env_cfg.get("task_score_kind",       "spread")
    env_kind           = env_cfg.get("kind", "vmas")

    # Comm radius for Option B dynamic radius-mask plumbing (None = full comm)
    comm_radius = cfg.get("comm", {}).get("comm_radius", None)

    for it in range(total_iters):
        buffer.clear()
        rollout_step_rewards = []
        rollout_optimized_step_rewards = []
        rollout_primary_diag_vals   = []
        rollout_secondary_diag_vals = []
        # pursuit: capture_rate tracking (sum over envs at each truncation)
        rollout_capture_rate_nums   = []

        # ------------------------------------------------------------------ #
        # Rollout collection (eval mode — no dropout / batchnorm training)   #
        # ------------------------------------------------------------------ #
        model.eval()
        for t in range(T):
            # Build radius-based comm mask if comm_radius is set and env exposes positions.
            positions = getattr(env, "agent_positions", None)
            if comm_radius is not None and positions is not None:
                comm_mask = build_radius_mask(positions.to(device), comm_radius)
            else:
                comm_mask = None

            out = model.act(obs, state, mask=comm_mask)
            next_obs, next_state, reward, done, info = env.step(out["action"].cpu())
            reward = reward.to(device)
            optimized_reward = reward + reward_shift
            done = done.to(device)

            # bad_mask=0 only for envs that ended by time-limit truncation.
            bad_mask = _bad_mask_from_info(info, env.num_agents, device)

            buffer.insert(
                obs=obs,
                state=state,
                action=out["action"],
                log_prob=out["log_prob"],
                value=out["value"],
                reward=optimized_reward,
                done=done,
                bad_mask=bad_mask,
                comm_mask=comm_mask,
            )

            # Track per-env episode return using the first agent's reward (team reward)
            ep_return += reward[:, 0]
            optimized_ep_return += optimized_reward[:, 0]
            rollout_step_rewards.append(reward[:, 0].mean().item())
            rollout_optimized_step_rewards.append(optimized_reward[:, 0].mean().item())
            if primary_diag_key in info:
                rollout_primary_diag_vals.append(
                    torch.as_tensor(info[primary_diag_key], device=device).float().mean().item()
                )
            if secondary_diag_key in info:
                rollout_secondary_diag_vals.append(
                    torch.as_tensor(info[secondary_diag_key], device=device).float().mean().item()
                )

            # Collect completed episode returns on truncation
            truncated_env = torch.as_tensor(info["truncated"], device=device, dtype=torch.bool)
            if truncated_env.dim() == 0:
                truncated_env = truncated_env.repeat(B)
            if bool(truncated_env.any()):
                completed_returns.extend(ep_return[truncated_env].tolist())
                completed_optimized_returns.extend(
                    optimized_ep_return[truncated_env].tolist()
                )
                ep_return[truncated_env] = 0.0
                optimized_ep_return[truncated_env] = 0.0
                # Pursuit: log capture_rate at episode boundary (captured_episode = ever captured)
                if env_kind == "pursuit" and "captured_episode" in info:
                    cap_ep = torch.as_tensor(info["captured_episode"], device=device).float()
                    rollout_capture_rate_nums.append(cap_ep[truncated_env].mean().item())

            global_step += B
            obs, state = next_obs.to(device), next_state.to(device)
            last_step_info = info

        # ------------------------------------------------------------------ #
        # Bootstrap: value of the state after the last rollout step           #
        # ------------------------------------------------------------------ #
        model.eval()
        with torch.no_grad():
            bootstrap_state = _terminal_state_from_info(
                last_step_info or {}, state, device
            )
            last_value = model.critic(bootstrap_state)  # [B, N]

        buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda, value_normalizer=value_normalizer)

        # ------------------------------------------------------------------ #
        # PPO update (model.train() called inside ppo_update)                 #
        # ------------------------------------------------------------------ #
        entropy_coef = _linear_schedule(
            entropy_coef_start,
            entropy_coef_final,
            it,
            total_iters,
        )
        active_lr = _linear_schedule(lr_start, lr_final, it, total_iters)
        _set_optimizer_lr(optimizer, active_lr)
        metrics = ppo_update(
            model,
            optimizer,
            buffer,
            ppo_epochs=ppo_epochs,
            num_minibatches=num_minibatches,
            clip_eps=clip_eps,
            value_clip_eps=value_clip_eps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            normalize_advantages=normalize_advantages,
            value_normalizer=value_normalizer,
        )

        # NaN guards on float metrics
        for k, v in metrics.items():
            if isinstance(v, float) and math.isnan(v):
                print(f"[WARN] NaN detected in metric '{k}' at iter={it}")

        # ------------------------------------------------------------------ #
        # Logging                                                              #
        # ------------------------------------------------------------------ #
        recent = completed_returns[-B:] if completed_returns else [float("nan")]
        mean_ep_ret = sum(r for r in recent if not math.isnan(r)) / max(
            sum(1 for r in recent if not math.isnan(r)), 1
        )
        mean_step_rew = (
            sum(rollout_step_rewards) / len(rollout_step_rewards)
            if rollout_step_rewards
            else float("nan")
        )
        optimized_recent = (
            completed_optimized_returns[-B:]
            if completed_optimized_returns
            else [float("nan")]
        )
        mean_optimized_ep_ret = sum(
            r for r in optimized_recent if not math.isnan(r)
        ) / max(sum(1 for r in optimized_recent if not math.isnan(r)), 1)
        mean_optimized_step_rew = (
            sum(rollout_optimized_step_rewards) / len(rollout_optimized_step_rewards)
            if rollout_optimized_step_rewards
            else float("nan")
        )
        mean_primary_diag = (
            sum(rollout_primary_diag_vals) / len(rollout_primary_diag_vals)
            if rollout_primary_diag_vals
            else float("nan")
        )
        mean_secondary_diag = (
            sum(rollout_secondary_diag_vals) / len(rollout_secondary_diag_vals)
            if rollout_secondary_diag_vals
            else float("nan")
        )
        if task_score_kind == "pursuit":
            mean_task_score = _pursuit_task_score(mean_primary_diag, mean_secondary_diag)
        else:
            mean_task_score = _task_score(mean_primary_diag, mean_secondary_diag)
        # Pursuit: capture_rate (fraction of episodes where capture occurred)
        mean_capture_rate = (
            sum(rollout_capture_rate_nums) / len(rollout_capture_rate_nums)
            if rollout_capture_rate_nums
            else float("nan")
        )
        if start_ep_return is None and not math.isnan(mean_ep_ret):
            start_ep_return = mean_ep_ret
        if start_step_reward is None and not math.isnan(mean_step_rew):
            start_step_reward = mean_step_rew

        # Default eval_metrics stub — keys use the env-specific diagnostic names so
        # VMAS rows keep "eval_coverage_distance"/"eval_collision_pairs" byte-identical,
        # while pursuit rows get "eval_pursuer_target_distance"/"eval_capture".
        eval_metrics = {
            "eval_mean_return": float("nan"),
            "eval_mean_step_reward": float("nan"),
            "eval_optimized_mean_return": float("nan"),
            "eval_optimized_mean_step_reward": float("nan"),
            "eval_mean_episode_length": float("nan"),
            f"eval_{primary_diag_key}": float("nan"),
            f"eval_{secondary_diag_key}": float("nan"),
            "eval_task_score": float("nan"),
        }
        should_eval = eval_interval > 0 and (
            it % eval_interval == 0 or it == total_iters - 1
        )
        if should_eval:
            eval_metrics = _evaluate_deterministic_policy(model, cfg, device)
            if start_eval_return is None:
                start_eval_return = eval_metrics["eval_mean_return"]
            if eval_metrics["eval_mean_return"] > best_eval_return:
                best_eval_return = eval_metrics["eval_mean_return"]
                if save_checkpoints:
                    model.save(
                        os.path.join(ckpt_dir, "best_eval.pt"),
                        optimizer=optimizer,
                        global_step=global_step,
                        extra={"eval_mean_return": best_eval_return},
                    )

        # Row dict: diagnostic columns use the env-specific key names so
        # VMAS CSV schema ("coverage_distance", "collision_pairs", etc.) is
        # preserved byte-for-byte, and pursuit gets its own column names.
        row = {
            "iter": it,
            "global_step": global_step,
            "episode_return": mean_ep_ret,
            "mean_step_reward": mean_step_rew,
            "optimized_episode_return": mean_optimized_ep_ret,
            "optimized_mean_step_reward": mean_optimized_step_rew,
            "reward_shift": reward_shift,
            "episode_return_gap_to_zero": _gap_to_zero(mean_ep_ret),
            "mean_step_reward_gap_to_zero": _gap_to_zero(mean_step_rew),
            "episode_return_delta_from_start": _delta_from_start(mean_ep_ret, start_ep_return),
            "mean_step_reward_delta_from_start": _delta_from_start(mean_step_rew, start_step_reward),
            primary_diag_key: mean_primary_diag,
            secondary_diag_key: mean_secondary_diag,
            "task_score": mean_task_score,
            "eval_mean_return": eval_metrics["eval_mean_return"],
            "eval_mean_step_reward": eval_metrics["eval_mean_step_reward"],
            "eval_optimized_mean_return": eval_metrics["eval_optimized_mean_return"],
            "eval_optimized_mean_step_reward": eval_metrics["eval_optimized_mean_step_reward"],
            "eval_mean_episode_length": eval_metrics["eval_mean_episode_length"],
            f"eval_{primary_diag_key}": eval_metrics[f"eval_{primary_diag_key}"],
            f"eval_{secondary_diag_key}": eval_metrics[f"eval_{secondary_diag_key}"],
            "eval_task_score": eval_metrics["eval_task_score"],
            "eval_return_gap_to_zero": _gap_to_zero(eval_metrics["eval_mean_return"]),
            "eval_return_delta_from_start": _delta_from_start(
                eval_metrics["eval_mean_return"], start_eval_return
            ),
            "episodes_completed": len(completed_returns),
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "total_loss": metrics["total_loss"],
            "grad_norm": metrics["grad_norm"],
            "actor_grad_norm": metrics["actor_grad_norm"],
            "critic_grad_norm": metrics["critic_grad_norm"],
            "approx_kl": metrics["approx_kl"],
            "clipfrac": metrics["clipfrac"],
            "explained_variance": metrics["explained_variance"],
            "mean_ratio_epoch0": metrics["mean_ratio_epoch0"],
            "entropy_coef": entropy_coef,
            "lr": active_lr,
            "seed": cfg["seed"],
            "alpha": env_cfg.get("alpha", float("nan")),
            "env_alpha": env_cfg.get("alpha", float("nan")),
        }
        # Pursuit-only extra column: per-episode capture rate (absent for VMAS).
        if env_kind == "pursuit":
            row["capture_rate"] = mean_capture_rate
        metrics_rows.append(row)

        if it % console_log_interval == 0:
            eval_return = eval_metrics["eval_mean_return"]
            eval_return_str = "n/a" if math.isnan(eval_return) else f"{eval_return:+.3f}"
            _pd = "nan" if math.isnan(mean_primary_diag) else f"{mean_primary_diag:.3f}"
            _sd = "nan" if math.isnan(mean_secondary_diag) else f"{mean_secondary_diag:.3f}"
            _sc = "nan" if math.isnan(mean_task_score) else f"{mean_task_score:.3f}"
            print(
                f"iter={it:4d} step={global_step:7d} ep_ret={mean_ep_ret:+.3f} "
                f"opt_ret={mean_optimized_ep_ret:+.3f} "
                f"step_rew={mean_step_rew:.4f} "
                f"opt_step={mean_optimized_step_rew:.4f} "
                f"gap0={_gap_to_zero(mean_ep_ret):.3f} "
                f"dret={_delta_from_start(mean_ep_ret, start_ep_return):+.3f} "
                f"eval_ret={eval_return_str} "
                f"{primary_diag_key}={_pd} "
                f"{secondary_diag_key}={_sd} "
                f"score={_sc} "
                f"pol={metrics['policy_loss']:+.4f} val={metrics['value_loss']:.4f} "
                f"ent={metrics['entropy']:.4f} tot={metrics['total_loss']:+.4f} "
                f"agnorm={metrics['actor_grad_norm']:.3f} cgnorm={metrics['critic_grad_norm']:.3f} "
                f"kl={metrics['approx_kl']:.4f} "
                f"clipf={metrics['clipfrac']:.3f} "
                f"ev={metrics['explained_variance']:+.3f} "
                f"r0={metrics['mean_ratio_epoch0']:.4f} "
                f"entcoef={entropy_coef:.2e} "
                f"lr={active_lr:.1e} seed={cfg['seed']}"
            )

        # ------------------------------------------------------------------ #
        # Checkpointing                                                        #
        # ------------------------------------------------------------------ #
        # Best checkpoint
        if not math.isnan(mean_ep_ret) and mean_ep_ret > best_return:
            best_return = mean_ep_ret
            if save_checkpoints:
                best_ckpt_path = os.path.join(ckpt_dir, "best.pt")
                model.save(
                    best_ckpt_path,
                    optimizer=optimizer,
                    global_step=global_step,
                    extra={"episode_return": best_return},
                )

        # Periodic checkpoint
        if save_checkpoints and it > 0 and it % checkpoint_interval == 0:
            model.save(
                os.path.join(ckpt_dir, f"ckpt_{it}.pt"),
                optimizer=optimizer,
                global_step=global_step,
            )

    # ---------------------------------------------------------------------- #
    # End of training: save final checkpoint and write artifact files         #
    # ---------------------------------------------------------------------- #
    final_ckpt_path = os.path.join(ckpt_dir, "final.pt")
    if save_checkpoints:
        model.save(final_ckpt_path, optimizer=optimizer, global_step=global_step)
        print(f"[train] Final checkpoint saved to {final_ckpt_path}")
    else:
        print("[train] Checkpoint saving disabled.")

    # metrics.json
    if save_metrics_json:
        metrics_json_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_rows, f, indent=2)

    # metrics.csv — field order matches the row dict key insertion order above
    metrics_csv_path = os.path.join(run_dir, "metrics.csv")
    if metrics_rows:
        fieldnames = list(metrics_rows[0].keys())
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)

    print(f"[train] Run artifacts in {run_dir}")

    # ---------------------------------------------------------------------- #
    # Optional post-train render. HPC jobs should use --render off.           #
    # ---------------------------------------------------------------------- #
    if render_mode == "off":
        print("[render] Skipped post-train render (mode=off).")
        return
    if not save_checkpoints:
        print("[render] Skipped post-train render (checkpoint saving disabled).")
        return
    if render_mode == "auto" and _is_headless():
        print("[render] Skipped post-train render (mode=auto, no display detected).")
        return

    from experiments.render_trained_policy import render_trained_policy

    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    render_ckpt = best_ckpt if os.path.exists(best_ckpt) else final_ckpt_path
    try:
        render_trained_policy(ckpt_path=render_ckpt, out_dir=run_dir, cfg=cfg)
    except Exception as e:
        if render_mode == "on":
            raise
        print(
            f"[WARN] Post-train render failed: {e}. "
            "All training artifacts are intact."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MAPPO on a VMAS environment."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed override (overrides config seed).",
    )
    parser.add_argument(
        "--render",
        choices=("off", "auto", "on"),
        default=None,
        help=(
            "Post-training render mode. Defaults to config render.mode. "
            "Use off for SLURM/Newton jobs."
        ),
    )
    args = parser.parse_args()
    main(args.config, resume=args.resume, seed=args.seed, render=args.render)
