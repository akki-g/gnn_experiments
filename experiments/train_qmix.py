"""
QMIX (feedforward, double-Q) off-policy training on pursuit / PCP / PP environments.

This is the SECOND backbone for the comm-necessity study. It deliberately mirrors
`experiments/train_vmas_mappo.py`: identical env factory, comm slot, kNN/radius mask
construction, deterministic eval, run-directory layout, and metrics/checkpoint
conventions. ONLY the learning algorithm changes (off-policy value decomposition
instead of on-policy PPO), so the probe + eval-sweep run on QMIX checkpoints UNCHANGED
via `backbone.build_agent_from_ckpt` (dispatch on the saved `config["algo"]`).

Isolation guarantee (mirrors MAPPO's "critic never sees comm"): the comm slot feeds
ONLY the per-agent Q-utilities; the mixer reads RAW state. So a QMIX comm advantage is
an information / decentralized-execution effect, not a mixer artifact.

Usage:
    python -m experiments.train_qmix --config configs/qmix_pcp.yaml --render off
    python -m experiments.train_qmix --config configs/qmix_pcp.yaml --seed 1 --render off
    python -m experiments.train_qmix --config configs/qmix_pcp.yaml \
        --resume runs/qmix_pcp/<ts>/ckpt/final.pt

Artifacts (same layout as the MAPPO trainer):
    ckpt/final.pt, ckpt/best.pt, ckpt/best_eval.pt, ckpt/ckpt_<step>.pt,
    config.yaml, metrics.csv, metrics.json, console.log

Cadence units (IMPORTANT — three different clocks):
    runtime.{total_env_steps,warmup_steps,eval_interval,checkpoint_interval} and
        algo.eps_anneal_steps are ENV-steps (gated on global_env_step += num_envs;
        warmup is compared against len(buffer), i.e. transitions == env-steps).
    algo.target_update_interval is UPDATE-steps (gated on the learner update count).
    The collection loop index `it` counts env-step BATCHES (env-steps / num_envs) and
        must never be compared directly against an env-step value.
"""

import csv
import json
import math
import os
import sys
import argparse

import torch
import yaml

from backbone.utils import load_yaml_config, set_seed, get_device
from backbone.q_agent import QAgent
from backbone.mixer import QMixer
from backbone.qmix_learner import QMIXLearner
from backbone.replay_buffer import ReplayBuffer

# Reuse the MAPPO trainer's helpers VERBATIM (parity principle): env factory, comm
# builder, mask builders, deterministic eval, run-dir + logging helpers, schedules.
from experiments.train_vmas_mappo import (
    _build_env,
    _build_comm_module,
    build_knn_mask,
    build_radius_mask,
    _evaluate_deterministic_policy,
    _make_run_dir,
    _resolve_render_mode,
    _gap_to_zero,
    _delta_from_start,
    _linear_schedule,
    _pursuit_task_score,
    _task_score,
    _Tee,
)


def _full_offdiag_mask(B: int, N: int, device) -> torch.Tensor:
    """All-to-all comm mask (off-diagonal True). Used when the env exposes no positions
    or topology is 'full', so the buffer always stores a concrete bool [B,N,N] mask."""
    eye = torch.eye(N, dtype=torch.bool, device=device)
    return (~eye).unsqueeze(0).expand(B, N, N).contiguous()


def _mean(xs):
    xs = [x for x in xs if isinstance(x, float) and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


# ---------------------------------------------------------------------------
# Main entry point (mirrors train_vmas_mappo.main)
# ---------------------------------------------------------------------------

def main(config_path: str, resume: str = None, seed: int = None, render: str = None) -> str:
    cfg = load_yaml_config(config_path)

    if seed is not None:
        cfg["seed"] = int(seed)

    set_seed(cfg["seed"])
    device = get_device(prefer_cuda=(cfg.get("device", "auto") != "cpu"))
    render_mode = _resolve_render_mode(cfg, render)

    log_cfg = cfg.setdefault("logging", {})
    run_dir = log_cfg.get("run_dir")
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = _make_run_dir(log_cfg["log_dir"], cfg["seed"], log_cfg.get("run_label"))

    save_checkpoints = bool(log_cfg.get("save_checkpoints", True))
    save_config = bool(log_cfg.get("save_config", True))
    save_console_log = bool(log_cfg.get("save_console_log", True))
    ckpt_dir = os.path.join(run_dir, "ckpt")
    if save_checkpoints:
        os.makedirs(ckpt_dir, exist_ok=True)

    if save_config:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)

    if save_console_log:
        log_file = open(os.path.join(run_dir, "console.log"), "w", buffering=1)
        sys.stdout = _Tee(log_file)
        try:
            _train_qmix(cfg, device, run_dir, ckpt_dir, resume, render_mode)
        finally:
            sys.stdout = sys.__stdout__
            log_file.close()
    else:
        _train_qmix(cfg, device, run_dir, ckpt_dir, resume, render_mode)

    return run_dir


# ---------------------------------------------------------------------------
# Off-policy training loop
# ---------------------------------------------------------------------------

def _train_qmix(cfg, device, run_dir, ckpt_dir, resume, render_mode: str):
    env_cfg = cfg["env"]
    model_cfg = cfg.get("model", {})
    algo_cfg = cfg["algo"]
    runtime_cfg = cfg["runtime"]
    log_cfg = cfg["logging"]
    comm_cfg = cfg.get("comm", {})
    save_checkpoints = bool(log_cfg.get("save_checkpoints", True))
    save_metrics_json = bool(log_cfg.get("save_metrics_json", True))

    B = int(runtime_cfg["num_envs"])

    # ----- Environment -----
    env = _build_env(env_cfg, num_envs=B, seed=cfg["seed"], device=str(device))
    N = env.num_agents
    print(
        f"[env] kind={env_cfg.get('kind','vmas')} name={env_cfg.get('name', env_cfg.get('scenario','?'))} "
        f"n_agents={N} obs_dim={env.obs_dim} action_dim={env.action_dim} "
        f"state_dim={env.state_dim} num_envs={env.B}"
    )

    # ----- Agent (decentralized; uses comm) + Mixer (centralized; raw state) -----
    comm = _build_comm_module(comm_cfg, model_cfg)
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    # head_layers defaults to actor_layers for capacity parity with the MAPPO actor.
    head_layers = int(model_cfg.get("head_layers", model_cfg.get("actor_layers", 1)))
    agent = QAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        num_agents=N,
        hidden_dim=hidden_dim,
        comm_module=comm,
        encoder_layers=int(model_cfg.get("encoder_layers", 2)),
        head_layers=head_layers,
        state_dim=env.state_dim,
        device=str(device),
    )
    mixer = QMixer(
        N,
        env.state_dim,
        embed_dim=int(algo_cfg.get("mixing_embed_dim", 32)),
        hypernet_embed=int(algo_cfg.get("hypernet_embed", 64)),
        device=str(device),
    )

    env_class_id = getattr(env, "class_id", None)
    if env_class_id is not None:
        env_class_id = env_class_id.to(device)

    learner = QMIXLearner(
        agent,
        mixer,
        env_class_id,
        lr=float(algo_cfg["lr"]),
        gamma=float(algo_cfg["gamma"]),
        grad_clip=float(algo_cfg["grad_clip"]),
        double_q=bool(algo_cfg.get("double_q", True)),
        device=str(device),
    )

    buffer = ReplayBuffer(
        capacity=int(runtime_cfg["replay_capacity"]),
        num_agents=N,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        device=str(device),
        seed=cfg["seed"],
    )

    # ----- Resume -----
    global_env_step = 0
    update_count = 0
    if resume:
        ck = agent.load(resume, map_location=device)
        if ck.get("mixer") is not None:
            mixer.load_state_dict(ck["mixer"])
        if ck.get("optimizer") is not None:
            learner.opt.load_state_dict(ck["optimizer"])
        learner.sync_targets()
        global_env_step = int(ck.get("global_step", 0))
        print(f"[resume] Loaded {resume} at global_step={global_env_step}")

    # ----- Cadence (mind the units — see module docstring) -----
    total_env_steps = int(runtime_cfg["total_env_steps"])
    total_iters = max(1, total_env_steps // B)
    warmup_steps = int(runtime_cfg.get("warmup_steps", 0))
    train_interval = int(runtime_cfg.get("train_interval", 1))
    batch_size = int(runtime_cfg["batch_size"])
    eval_interval = int(runtime_cfg.get("eval_interval", 0))          # env-steps
    checkpoint_interval = int(runtime_cfg.get("checkpoint_interval", 0))  # env-steps
    target_update_interval = int(algo_cfg.get("target_update_interval", 200))  # UPDATE-steps
    eps_start = float(algo_cfg.get("eps_start", 1.0))
    eps_end = float(algo_cfg.get("eps_end", 0.05))
    eps_anneal_steps = int(algo_cfg.get("eps_anneal_steps", 50000))   # env-steps
    console_log_interval = int(log_cfg.get("console_log_interval", 1))

    # ----- Comm topology (read once) -----
    topology = comm_cfg.get("topology", "full")
    knn_k = int(comm_cfg.get("knn_k", 4))
    symmetrize = comm_cfg.get("symmetrize", "or")
    comm_radius = comm_cfg.get("comm_radius", None)

    def build_mask(positions):
        """Concrete bool [B,N,N] comm mask from positions ([B,N,2] tensor or None).
        Same construction as the MAPPO trainer; full fallback so the buffer never
        stores None (knn is what the PCP/PP necessity sweeps actually use)."""
        if positions is None:
            return _full_offdiag_mask(B, N, device)
        positions = positions.to(device)
        if topology == "knn":
            return build_knn_mask(positions, knn_k, symmetrize)
        if topology == "radius" or comm_radius is not None:
            return build_radius_mask(positions, float(comm_radius))
        return _full_offdiag_mask(B, N, device)

    # ----- Diagnostic key names (kept identical to MAPPO for apples-to-apples CSV) -----
    primary_diag_key = env_cfg.get("primary_diagnostic", "coverage_distance")
    secondary_diag_key = env_cfg.get("secondary_diagnostic", "collision_pairs")
    task_score_kind = env_cfg.get("task_score_kind", "spread")
    env_kind = env_cfg.get("kind", "vmas")

    obs, state = env.reset()
    obs, state = obs.to(device), state.to(device)

    ep_return = torch.zeros(B, device=device)
    completed_returns = []
    best_return = -math.inf
    best_eval_return = -math.inf
    metrics_rows = []
    start_ep_return = None
    start_eval_return = None

    def _new_window():
        return {
            "step_rewards": [], "primary": [], "secondary": [], "capture": [],
            "td_loss": [], "q_tot_mean": [], "target_mean": [], "grad_norm": [],
            "mixer_w_min": [],
            # PCP-specific
            "detection_event_rate": [], "cap_without_recent": [],
            "window_open": [], "steps_since_detection": [],
        }

    win = _new_window()

    print(
        f"[train] QMIX start: total_iters={total_iters} B={B} device={device} "
        f"seed={cfg['seed']} warmup={warmup_steps} batch={batch_size} "
        f"target_update(updates)={target_update_interval} render={render_mode}"
    )

    for it in range(total_iters):
        # epsilon by ENV-steps seen so far (eps_start at step 0 -> eps_end after anneal).
        eps = _linear_schedule(eps_start, eps_end, global_env_step, eps_anneal_steps)

        positions = getattr(env, "agent_positions", None)
        comm_mask = build_mask(positions)

        action = agent.act(obs, state, mask=comm_mask, class_id=env_class_id, epsilon=eps)["action"]
        next_obs, next_state, reward, done, info = env.step(action.cpu())
        reward = reward.to(device)
        next_obs = next_obs.to(device)
        next_state = next_state.to(device)

        # ---- Truncation-correct next-step targets (mode A: fixed-length, auto-reset) ----
        trunc = torch.as_tensor(info["truncated"], device=device, dtype=torch.bool)
        if trunc.dim() == 0:
            trunc = trunc.repeat(B)
        terminated = torch.as_tensor(
            info.get("terminated", torch.zeros(B)), device=device, dtype=torch.float32
        ).reshape(B, 1)

        any_trunc = bool(trunc.any())
        if any_trunc and "terminal_obs" in info:
            term_obs = info["terminal_obs"].to(device)        # [B,N,obs_dim] (pre-reset)
            term_state = info["terminal_state"].to(device)    # [B,state_dim] (pre-reset)
            store_next_obs = torch.where(trunc.view(B, 1, 1), term_obs, next_obs)
            store_next_state = torch.where(trunc.view(B, 1), term_state, next_state)
        else:
            store_next_obs, store_next_state = next_obs, next_state

        # next comm mask: use TERMINAL positions at truncation (agent_positions has
        # already auto-reset), else post-step positions. Terminal pursuer positions are
        # the first 2N entries of terminal_state ([pursuer_pos(N*2), target(2), (t)]).
        post_positions = getattr(env, "agent_positions", None)
        if post_positions is not None and any_trunc and "terminal_state" in info:
            term_state = info["terminal_state"].to(device)
            term_pos = term_state[:, : 2 * N].reshape(B, N, 2)
            post_pos = post_positions.to(device)
            next_pos = torch.where(trunc.view(B, 1, 1), term_pos, post_pos)
            next_comm_mask = build_mask(next_pos)
        else:
            next_comm_mask = build_mask(post_positions)

        team_reward = reward[:, 0:1]  # [B,1] team scalar (broadcast across agents in env)

        buffer.insert_batch(
            obs=obs,
            next_obs=store_next_obs,
            state=state,
            next_state=store_next_state,
            actions=action.to(device),
            reward=team_reward,
            terminated=terminated,
            comm_mask=comm_mask,
            next_comm_mask=next_comm_mask,
        )

        # ---- episode-return + diagnostic accumulation (window-averaged into rows) ----
        ep_return += reward[:, 0]
        win["step_rewards"].append(reward[:, 0].mean().item())
        if primary_diag_key in info:
            win["primary"].append(torch.as_tensor(info[primary_diag_key], device=device).float().mean().item())
        if secondary_diag_key in info:
            win["secondary"].append(torch.as_tensor(info[secondary_diag_key], device=device).float().mean().item())
        if env_kind == "pcp":
            for key, acc in (
                ("detection_event", "detection_event_rate"),
                ("capture_without_recent_detection", "cap_without_recent"),
                ("window_open", "window_open"),
                ("steps_since_detection", "steps_since_detection"),
            ):
                if key in info:
                    win[acc].append(torch.as_tensor(info[key], device=device).float().mean().item())

        if any_trunc:
            completed_returns.extend(ep_return[trunc].tolist())
            ep_return[trunc] = 0.0
            if env_kind in ("pursuit", "pcp") and "captured_episode" in info:
                cap_ep = torch.as_tensor(info["captured_episode"], device=device).float()
                win["capture"].append(cap_ep[trunc].mean().item())

        obs, state = next_obs, next_state
        global_env_step += B

        # ---- learner update (once warm) ----
        if len(buffer) >= warmup_steps and (it % train_interval == 0):
            m = learner.update(buffer.sample(batch_size))
            update_count += 1
            win["td_loss"].append(m["td_loss"])
            win["q_tot_mean"].append(m["q_tot_mean"])
            win["target_mean"].append(m["target_mean"])
            win["grad_norm"].append(m["grad_norm"])
            win["mixer_w_min"].append(m["mixer_w_min"])
            if math.isnan(m["td_loss"]):
                print(f"[WARN] NaN td_loss at env_step={global_env_step}")
            if target_update_interval > 0 and update_count % target_update_interval == 0:
                learner.sync_targets()

        # ---- row / eval / checkpoint at ENV-step cadence (crossing detection) ----
        is_last = it == total_iters - 1
        crossed_eval = eval_interval > 0 and (
            global_env_step // eval_interval != (global_env_step - B) // eval_interval
        )
        crossed_ckpt = checkpoint_interval > 0 and (
            global_env_step // checkpoint_interval != (global_env_step - B) // checkpoint_interval
        )

        if crossed_eval or is_last:
            eval_metrics = _evaluate_deterministic_policy(agent, cfg, device)
            if start_eval_return is None:
                start_eval_return = eval_metrics["eval_mean_return"]
            if eval_metrics["eval_mean_return"] > best_eval_return:
                best_eval_return = eval_metrics["eval_mean_return"]
                if save_checkpoints:
                    agent.save(os.path.join(ckpt_dir, "best_eval.pt"),
                               optimizer=learner.opt, global_step=global_env_step,
                               mixer=mixer, extra={"eval_mean_return": best_eval_return})

            recent = completed_returns[-B:] if completed_returns else [float("nan")]
            mean_ep_ret = _mean([float(r) for r in recent])
            mean_step_rew = _mean(win["step_rewards"])
            mean_primary = _mean(win["primary"])
            mean_secondary = _mean(win["secondary"])
            mean_capture = _mean(win["capture"])
            if task_score_kind == "pursuit":
                mean_task_score = _pursuit_task_score(mean_primary, mean_secondary)
            else:
                mean_task_score = _task_score(mean_primary, mean_secondary)
            if start_ep_return is None and not math.isnan(mean_ep_ret):
                start_ep_return = mean_ep_ret

            row = {
                "iter": it,
                "global_step": global_env_step,
                "episode_return": mean_ep_ret,
                "mean_step_reward": mean_step_rew,
                "episode_return_gap_to_zero": _gap_to_zero(mean_ep_ret),
                "episode_return_delta_from_start": _delta_from_start(mean_ep_ret, start_ep_return),
                primary_diag_key: mean_primary,
                secondary_diag_key: mean_secondary,
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
                "eval_return_delta_from_start": _delta_from_start(eval_metrics["eval_mean_return"], start_eval_return),
                "episodes_completed": len(completed_returns),
                # QMIX TD columns (replace PPO-only columns)
                "td_loss": _mean(win["td_loss"]),
                "q_tot_mean": _mean(win["q_tot_mean"]),
                "target_mean": _mean(win["target_mean"]),
                "grad_norm": _mean(win["grad_norm"]),
                "mixer_w_min": min([v for v in win["mixer_w_min"] if not math.isnan(v)], default=float("nan")),
                "epsilon": eps,
                "updates": update_count,
                "buffer_size": len(buffer),
                "lr": float(algo_cfg["lr"]),
                "seed": cfg["seed"],
                "alpha": env_cfg.get("alpha", float("nan")),
                "env_alpha": env_cfg.get("alpha", float("nan")),
            }
            if env_kind in ("pursuit", "pcp"):
                row["capture_rate"] = mean_capture
            if env_kind == "pcp":
                row["detection_event_rate"] = _mean(win["detection_event_rate"])
                row["capture_without_recent_detection_rate"] = _mean(win["cap_without_recent"])
                row["window_open"] = _mean(win["window_open"])
                row["steps_since_detection"] = _mean(win["steps_since_detection"])
            metrics_rows.append(row)

            if (len(metrics_rows) - 1) % console_log_interval == 0:
                er = eval_metrics["eval_mean_return"]
                er_s = "n/a" if math.isnan(er) else f"{er:+.3f}"
                tl = row["td_loss"]
                tl_s = "n/a" if math.isnan(tl) else f"{tl:.4f}"
                _pd = "nan" if math.isnan(mean_primary) else f"{mean_primary:.3f}"
                _sd = "nan" if math.isnan(mean_secondary) else f"{mean_secondary:.3f}"
                print(
                    f"step={global_env_step:8d} ep_ret={mean_ep_ret:+.3f} "
                    f"eval_ret={er_s} {primary_diag_key}={_pd} {secondary_diag_key}={_sd} "
                    f"cap={mean_capture:.3f} td_loss={tl_s} q_tot={row['q_tot_mean']:+.3f} "
                    f"tgt={row['target_mean']:+.3f} gnorm={row['grad_norm']:.2f} "
                    f"w_min={row['mixer_w_min']:.2e} eps={eps:.3f} buf={len(buffer)} "
                    f"upd={update_count} seed={cfg['seed']}"
                )

            # best-by-train-return checkpoint
            if not math.isnan(mean_ep_ret) and mean_ep_ret > best_return:
                best_return = mean_ep_ret
                if save_checkpoints:
                    agent.save(os.path.join(ckpt_dir, "best.pt"), optimizer=learner.opt,
                               global_step=global_env_step, mixer=mixer,
                               extra={"episode_return": best_return})
            win = _new_window()

        if save_checkpoints and (crossed_ckpt or is_last):
            agent.save(os.path.join(ckpt_dir, f"ckpt_{global_env_step}.pt"),
                       optimizer=learner.opt, global_step=global_env_step, mixer=mixer)

    # ----- Final artifacts -----
    final_ckpt_path = os.path.join(ckpt_dir, "final.pt")
    if save_checkpoints:
        agent.save(final_ckpt_path, optimizer=learner.opt, global_step=global_env_step, mixer=mixer)
        print(f"[train] Final checkpoint saved to {final_ckpt_path}")
    else:
        print("[train] Checkpoint saving disabled.")

    if save_metrics_json:
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics_rows, f, indent=2)

    if metrics_rows:
        fieldnames = list(metrics_rows[0].keys())
        metrics_csv_path = os.path.join(run_dir, "metrics.csv")
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)

    print(f"[train] Run artifacts in {run_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QMIX on a pursuit/PCP/PP environment.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override (overrides config seed).")
    parser.add_argument("--render", choices=("off", "auto", "on"), default=None,
                        help="Post-training render mode. Use off for SLURM jobs.")
    args = parser.parse_args()
    main(args.config, resume=args.resume, seed=args.seed, render=args.render)
