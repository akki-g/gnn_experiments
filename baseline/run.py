import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from baseline.trainer import IPPOTrainer
from environments.guarded_territory import GuardedTerritoryAdapter

# Hyperparameters
hidden_dim = 64
lr = 3e-4
gamma = 0.99
gae_lambda = 0.85
clip_eps = 0.2
value_coef = 0.5
entropy_coef = 0.01

TOTAL_TIMESTEPS = 1003520
ROLLOUT_LENGTH = 2048
BATCH_SIZE = 64
NUM_EPOCHS = 10
SEED = 42
LOG_EVERY = 5

# Environment config
NUM_ENVS = 1
MAX_STEPS = 200
N_SCOUTS = 3
N_INTERCEPTORS = 3
N_INTRUDERS = 3
N_ZONES = 2
WORLD_SIZE = 2.0

SCOUT_FOV = 0.9
INTERCEPTOR_FOV = 0.6
INTRUDER_SPEED = 0.3
DEFENDER_SPEED = 0.8
TAG_RADIUS = 0.15

OUTPUT_DIR = "outputs/ippo_guarded_territory"
os.makedirs(OUTPUT_DIR, exist_ok=True)

metrics_to_plot = [
    "policy_loss",
    "value_loss",
    "entropy",
    "mean_bellman_error",
    "mean_episode_return",
    "mean_episode_rewards",
    "scout_reward",
    "interceptor_reward",
]


def plot_metrics(metrics_history, title, save_path):
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    for i, name in enumerate(metrics_to_plot):
        vals = metrics_history.get(name, [])
        axes[i].plot(range(1, len(vals) + 1), vals, linewidth=1.8)
        axes[i].set_title(name)
        axes[i].set_xlabel("Iteration")
        axes[i].grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}, ROLLOUT_LENGTH={ROLLOUT_LENGTH}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    adapter = GuardedTerritoryAdapter(
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

    steps_done = 0
    iteration = 0

    while steps_done < TOTAL_TIMESTEPS:
        rollout_steps = min(ROLLOUT_LENGTH, TOTAL_TIMESTEPS - steps_done)
        last_obs, rollout_metrics = trainer.collect_rollouts(num_steps=rollout_steps)
        update_metrics = trainer.update(
            last_obs,
            num_actor_epochs=NUM_EPOCHS,
            num_critic_epochs=NUM_EPOCHS,
            B=BATCH_SIZE,
        )

        steps_done += rollout_steps
        iteration += 1

        progress = steps_done / TOTAL_TIMESTEPS
        new_skill = min(1.0, progress)
        trainer.adapter.env.scenario.intruder_skill = new_skill

        if iteration == 1 or iteration % LOG_EVERY == 0 or steps_done >= TOTAL_TIMESTEPS:
            print(
                f"[IPPO] Iter {iteration} | "
                f"steps={steps_done}/{TOTAL_TIMESTEPS} | "
                f"skill={new_skill:.2f} | "
                f"pi_loss={update_metrics['policy_loss']:.4f} | "
                f"v_loss={update_metrics['value_loss']:.4f} | "
                f"ent={update_metrics['entropy']:.4f} | "
                f"bellman={update_metrics['mean_bellman_error']:.4f} | "
                f"ep_ret={rollout_metrics['mean_episode_return']:.4f} | "
                f"ep_rew={rollout_metrics['mean_episode_rewards']:.4f} | "
                f"scout={rollout_metrics['mean_scout_rewards']:.4f} | "
                f"inter={rollout_metrics['mean_inter_rewards']:.4f}"
            )

    plot_path = os.path.join(OUTPUT_DIR, "ippo_metrics.png")
    plot_metrics(
        trainer.metrics_history,
        title="IPPO Training Metrics (Guarded Territory)",
        save_path=plot_path,
    )
    print(f"\nSaved: {plot_path}")

    final_return = trainer.metrics_history["mean_episode_return"][-1]
    final_reward = trainer.metrics_history["mean_episode_rewards"][-1]
    print("\nFinal IPPO results:")
    print(f"  Episode return: {final_return:.2f}")
    print(f"  Episode reward: {final_reward:.4f}")
    print(f"  Scout reward:   {trainer.metrics_history['scout_reward'][-1]:.4f}")
    print(f"  Interceptor reward: {trainer.metrics_history['interceptor_reward'][-1]:.4f}")


if __name__ == "__main__":
    main()
