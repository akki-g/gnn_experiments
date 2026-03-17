#!/usr/bin/env python3
"""
Unified entry point for single-config training runs.
Used by the SLURM job array - each job calls this with its own args.

Usage:
  python train.py --algo gnn --k 2 --r_comm 1.0
  python train.py --algo ippo
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend - required on cluster nodes
import matplotlib.pyplot as plt
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from environments.guarded_territory import GuardedTerritoryAdapter

# -- Shared hyperparameters -------------------------------------
hidden_dim   = 64
F_dim        = 64
G_dim        = 64
lr           = 3e-4
gamma        = 0.99
gae_lambda   = 0.85
clip_eps     = 0.2
value_coef   = 0.5
entropy_coef = 0.01

ROLLOUT_LENGTH  = 256
ITERATIONS      = 2000
BATCH_SIZE      = 256
NUM_EPOCHS      = 10
SEED            = 42
LOG_EVERY       = 20

NUM_ENVS        = 256
MAX_STEPS       = 200
N_SCOUTS        = 3
N_INTERCEPTORS  = 3
N_INTRUDERS     = 3
N_ZONES         = 2
WORLD_SIZE      = 2.0
SCOUT_FOV       = 0.9
INTERCEPTOR_FOV = 0.6
INTRUDER_SPEED  = 0.3
DEFENDER_SPEED  = 0.8
TAG_RADIUS      = 0.15


def plot_and_save(metrics_history, title, output_dir, filename):
    metrics_to_plot = [
        "policy_loss", "value_loss", "entropy", "mean_bellman_error", "explained_var",
        "mean_episode_return", "mean_episode_rewards", "scout_reward", "interceptor_reward",
        "comm_saliency",
    ]
    fig, axes = plt.subplots(5, 2, figsize=(14, 20))
    axes = axes.flatten()
    for i, name in enumerate(metrics_to_plot):
        vals = metrics_history.get(name, [])
        if len(vals) > 0:
            axes[i].plot(range(1, len(vals) + 1), vals, linewidth=1.5)
            axes[i].set_title(name)
            axes[i].set_xlabel("Iteration")
            axes[i].grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}", flush=True)


def make_adapter(device):
    """Shared adapter constructor used by both GNN and IPPO."""
    return GuardedTerritoryAdapter(
        num_envs=NUM_ENVS,
        device=device,
        n_scouts=N_SCOUTS,
        n_interceptors=N_INTERCEPTORS,
        n_intruders=N_INTRUDERS,
        n_zones=N_ZONES,
        max_steps=MAX_STEPS,
        world_size=WORLD_SIZE,
        scout_fov=SCOUT_FOV,
        interceptor_fov=INTERCEPTOR_FOV,
        intruder_speed=INTRUDER_SPEED,
        defender_speed=DEFENDER_SPEED,
        tag_radius=TAG_RADIUS,
        intruder_skill=0.0,
    )


# ---------------------------------------------------------------------
def run_gnn(k, r_comm, device, output_dir):
    from gnn.trainer_multienv import GNNTrainerMultiEnv

    config_name = f"K{k}_r{r_comm:.1f}"
    print(f"\n=== GNN | K={k}, r_comm={r_comm} | device={device} ===", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    adapter = make_adapter(device)

    trainer = GNNTrainerMultiEnv(
        adapter=adapter,
        hidden_dim=hidden_dim,
        F_feat=F_dim,
        G_feat=G_dim,
        K=k,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        device=device,
        r_comm=r_comm,
        total_timesteps=ITERATIONS * ROLLOUT_LENGTH,
        rollout_length=ROLLOUT_LENGTH,
    )

    steps_done = 0
    for iteration in range(1, ITERATIONS + 1):
        rollout_steps = min(ROLLOUT_LENGTH, ITERATIONS * ROLLOUT_LENGTH - steps_done)
        last_obs, rollout_metrics = trainer.collect_rollouts(num_steps=rollout_steps)
        update_metrics = trainer.update(
            last_obs,
            num_actor_epochs=NUM_EPOCHS,
            num_critic_epochs=3,
            B=BATCH_SIZE,
        )
        steps_done += rollout_steps

        progress = iteration / ITERATIONS
        trainer.adapter.env.scenario.intruder_skill = min(1.0, progress / 0.6)

        if iteration == 1 or iteration % LOG_EVERY == 0 or iteration == ITERATIONS:
            print(
                f"[{config_name}] iter={iteration}/{ITERATIONS} steps={steps_done} | "
                f"pi={update_metrics['policy_loss']:.4f} "
                f"v={update_metrics['value_loss']:.4f} "
                f"ent={update_metrics['entropy']:.4f} "
                f"ret={rollout_metrics['mean_episode_return']:.3f} "
                f"exp_var={update_metrics['explained_var']:.3f}",
                flush=True,
            )

    plot_and_save(
        trainer.metrics_history,
        f"GNN {config_name}",
        output_dir,
        f"gnn_{config_name}.png",
    )
    print(
        f"\n[{config_name}] Final return: "
        f"{trainer.metrics_history['mean_episode_return'][-1]:.3f}",
        flush=True,
    )


# ---------------------------------------------------------------------
def run_ippo(device, output_dir):
    from baseline.trainer import IPPOTrainer

    print(f"\n=== IPPO baseline | {NUM_ENVS} envs | device={device} ===", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    adapter = make_adapter(device)

    trainer = IPPOTrainer(
        adapter=adapter,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        device=device,
    )

    total_steps = ITERATIONS * ROLLOUT_LENGTH
    steps_done  = 0
    iteration   = 0

    while steps_done < total_steps:
        rollout_steps = min(ROLLOUT_LENGTH, total_steps - steps_done)
        last_obs, rollout_metrics = trainer.collect_rollouts(num_steps=rollout_steps)
        update_metrics = trainer.update(
            last_obs,
            num_actor_epochs=NUM_EPOCHS,
            num_critic_epochs=3,
            B=BATCH_SIZE,
        )
        steps_done += rollout_steps
        iteration  += 1

        progress = steps_done / total_steps
        trainer.adapter.env.scenario.intruder_skill = min(1.0, progress / 0.6)

        if iteration == 1 or iteration % LOG_EVERY == 0 or steps_done >= total_steps:
            print(
                f"[IPPO] iter={iteration} steps={steps_done}/{total_steps} | "
                f"pi={update_metrics['policy_loss']:.4f} "
                f"v={update_metrics['value_loss']:.4f} "
                f"ent={update_metrics['entropy']:.4f} "
                f"ret={rollout_metrics['mean_episode_return']:.3f} "
                f"exp_var={update_metrics['explained_var']:.3f}",
                flush=True,
            )

    plot_and_save(
        trainer.metrics_history,
        "IPPO Baseline",
        output_dir,
        "ippo_baseline.png",
    )
    print(
        f"\n[IPPO] Final return: "
        f"{trainer.metrics_history['mean_episode_return'][-1]:.3f}",
        flush=True,
    )


# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["gnn", "ippo"], required=True)
    parser.add_argument("--k",      type=int,   default=2,   help="GNN hops (gnn only)")
    parser.add_argument("--r_comm", type=float, default=1.0, help="Comm radius (gnn only)")
    parser.add_argument("--output_dir", type=str, default="outputs/cluster_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if args.algo == "gnn":
        run_gnn(args.k, args.r_comm, device, args.output_dir)
    else:
        run_ippo(device, args.output_dir)


if __name__ == "__main__":
    main()