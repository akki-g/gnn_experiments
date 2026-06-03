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
from comm import IdentityComm
from envs.adapters import VMASAdapter

import yaml


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


def _build_comm_module(comm_cfg: dict):
    """Instantiate the comm module named in the config (identity only for Phase 1/2)."""
    name = comm_cfg.get("module", "identity").lower()
    if name == "identity":
        return IdentityComm()
    raise ValueError(
        f"Unknown comm module '{name}'. "
        "Phase 2 supports only 'identity'. "
        "Future modules (broadcast, attention, graph) are not yet implemented."
    )


def _make_run_dir(log_dir: str, seed: int) -> str:
    """
    Build a collision-resistant run directory for local runs and SLURM arrays.

    Includes microseconds, seed, pid, and SLURM job/task ids when available.
    """
    for attempt in range(10):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        parts = [timestamp, f"seed{seed}", f"pid{os.getpid()}"]
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

    # Collision-resistant run directory for SLURM arrays.
    run_dir = _make_run_dir(cfg["logging"]["log_dir"], cfg["seed"])
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Copy config into run dir
    config_copy_path = os.path.join(run_dir, "config.yaml")
    with open(config_copy_path, "w") as f:
        yaml.safe_dump(cfg, f)

    # Console tee: all print() output goes to both terminal and console.log
    log_file = open(os.path.join(run_dir, "console.log"), "w", buffering=1)
    tee = _Tee(log_file)
    sys.stdout = tee

    try:
        _train(cfg, device, run_dir, ckpt_dir, resume, render_mode)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()

    return run_dir


def _train(cfg, device, run_dir, ckpt_dir, resume, render_mode: str):
    """Inner training body — called inside the finally-guarded stdout tee."""

    env_cfg = cfg["env"]
    model_cfg = cfg.get("model", {})
    algo_cfg = cfg["algo"]
    runtime_cfg = cfg["runtime"]
    log_cfg = cfg["logging"]

    # ----- Environment -----
    env = VMASAdapter(
        scenario=env_cfg["scenario"],
        num_envs=runtime_cfg["num_envs"],
        n_agents=env_cfg["num_agents"],
        max_steps=env_cfg["max_steps"],
        seed=cfg["seed"],
        device=str(device),
        include_timestep=env_cfg.get("include_timestep_in_state", True),
    )
    print(
        f"[env] scenario={env_cfg['scenario']} n_agents={env.num_agents} "
        f"obs_dim={env.obs_dim} action_dim={env.action_dim} "
        f"state_dim={env.state_dim} num_envs={env.B}"
    )

    # ----- Model -----
    model = MAPPO(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_agents=env.num_agents,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        comm_module=_build_comm_module(cfg.get("comm", {})),
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

    gamma = algo_cfg["gamma"]
    gae_lambda = algo_cfg["gae_lambda"]
    ppo_epochs = algo_cfg["ppo_epochs"]
    num_minibatches = algo_cfg["num_minibatches"]
    clip_eps = algo_cfg["clip_eps"]
    value_clip_eps = algo_cfg["value_clip_eps"]
    entropy_coef = algo_cfg["entropy_coef"]
    value_coef = algo_cfg["value_coef"]
    max_grad_norm = algo_cfg["max_grad_norm"]
    normalize_advantages = algo_cfg["normalize_advantages"]
    use_value_norm = algo_cfg.get("use_value_norm", True)
    value_normalizer = model.value_normalizer if use_value_norm else None

    obs, state = env.reset()
    obs, state = obs.to(device), state.to(device)

    ep_return = torch.zeros(B, device=device)
    completed_returns = []
    best_return = -math.inf
    metrics_rows = []

    print(
        f"[train] Starting: total_iters={total_iters} T={T} B={B} "
        f"device={device} seed={cfg['seed']} render={render_mode}"
    )

    last_step_info = None

    for it in range(total_iters):
        buffer.clear()
        rollout_step_rewards = []

        # ------------------------------------------------------------------ #
        # Rollout collection (eval mode — no dropout / batchnorm training)   #
        # ------------------------------------------------------------------ #
        model.eval()
        for t in range(T):
            out = model.act(obs, state)
            next_obs, next_state, reward, done, info = env.step(out["action"].cpu())
            reward = reward.to(device)
            done = done.to(device)

            # bad_mask=0 only for envs that ended by time-limit truncation.
            bad_mask = _bad_mask_from_info(info, env.num_agents, device)

            buffer.insert(
                obs=obs,
                state=state,
                action=out["action"],
                log_prob=out["log_prob"],
                value=out["value"],
                reward=reward,
                done=done,
                bad_mask=bad_mask,
            )

            # Track per-env episode return using the first agent's reward (team reward)
            ep_return += reward[:, 0]
            rollout_step_rewards.append(reward[:, 0].mean().item())

            # Collect completed episode returns on truncation
            truncated_env = torch.as_tensor(info["truncated"], device=device, dtype=torch.bool)
            if truncated_env.dim() == 0:
                truncated_env = truncated_env.repeat(B)
            if bool(truncated_env.any()):
                completed_returns.extend(ep_return[truncated_env].tolist())
                ep_return[truncated_env] = 0.0

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

        row = {
            "iter": it,
            "global_step": global_step,
            "episode_return": mean_ep_ret,
            "mean_step_reward": mean_step_rew,
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
            "lr": algo_cfg["lr"],
            "seed": cfg["seed"],
        }
        metrics_rows.append(row)

        if it % console_log_interval == 0:
            print(
                f"iter={it:4d} step={global_step:7d} ep_ret={mean_ep_ret:+.3f} "
                f"step_rew={mean_step_rew:.4f} "
                f"pol={metrics['policy_loss']:+.4f} val={metrics['value_loss']:.4f} "
                f"ent={metrics['entropy']:.4f} tot={metrics['total_loss']:+.4f} "
                f"agnorm={metrics['actor_grad_norm']:.3f} cgnorm={metrics['critic_grad_norm']:.3f} "
                f"kl={metrics['approx_kl']:.4f} "
                f"clipf={metrics['clipfrac']:.3f} "
                f"ev={metrics['explained_variance']:+.3f} "
                f"r0={metrics['mean_ratio_epoch0']:.4f} "
                f"lr={algo_cfg['lr']:.1e} seed={cfg['seed']}"
            )

        # ------------------------------------------------------------------ #
        # Checkpointing                                                        #
        # ------------------------------------------------------------------ #
        # Best checkpoint
        if not math.isnan(mean_ep_ret) and mean_ep_ret > best_return:
            best_return = mean_ep_ret
            best_ckpt_path = os.path.join(ckpt_dir, "best.pt")
            model.save(
                best_ckpt_path,
                optimizer=optimizer,
                global_step=global_step,
                extra={"episode_return": best_return},
            )

        # Periodic checkpoint
        if it > 0 and it % checkpoint_interval == 0:
            model.save(
                os.path.join(ckpt_dir, f"ckpt_{it}.pt"),
                optimizer=optimizer,
                global_step=global_step,
            )

    # ---------------------------------------------------------------------- #
    # End of training: save final checkpoint and write artifact files         #
    # ---------------------------------------------------------------------- #
    final_ckpt_path = os.path.join(ckpt_dir, "final.pt")
    model.save(final_ckpt_path, optimizer=optimizer, global_step=global_step)
    print(f"[train] Final checkpoint saved to {final_ckpt_path}")

    # metrics.json
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
