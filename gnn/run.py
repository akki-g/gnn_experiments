import os
import matplotlib.pyplot as plt
import torch
import numpy as np

from gnn.trainer_multienv import GNNTrainerMultiEnv
from environments.guarded_territory import GuardedTerritoryAdapter

# -- Hyperparameters ----------------------------------------------
F_dim = 64
G_dim = 64
hidden_dim = 64

lr = 3e-4
gamma = 0.99
gae_lambda = 0.85
clip_eps = 0.2
value_coef = 0.5
entropy_coef = 0.01

ITERATIONS = 1000
ROLLOUT_LENGTH = 512
TOTAL_TIMESTEPS = ITERATIONS * ROLLOUT_LENGTH
BATCH_SIZE = 256
NUM_EPOCHS = 10

K_VALUES = [2, 3, 5]
R_COMM_VALUES = [1.0, 1.5, 2.0]
SEED = 42
LOG_EVERY = 5

# -- Environment config --------------------------------------------
NUM_ENVS = 256      
MAX_STEPS = 200
N_SCOUTS = 3
N_INTERCEPTORS = 3
N_INTRUDERS = 3
N_ZONES = 2
WORLD_SIZE = 2.0

# -- Agent Congfig -------------
SCOUT_FOV = 0.9
INTERCEPTOR_FOV = 0.6
INTRUDER_SPEED = 0.3
DEFENDER_SPEED = 0.8
TAG_RADIUS = .15

OUTPUT_DIR = "outputs/experiments_k_r"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Sweep: K={K_VALUES}, r_comm={R_COMM_VALUES}")
print(f"TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}, ROLLOUT_LENGTH={ROLLOUT_LENGTH}")

# -- Plotting helper -----------------------------------------------
metrics_to_plot = [
    "policy_loss", "value_loss", "entropy",
    "mean_bellman_error", "mean_episode_return", "mean_episode_rewards",
    "scout_reward", "interceptor_reward",
    "explained_var"
]

def plot_metrics(metrics_history, title, save_path):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
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

# -- Sweep ---------------------------------------------------------
results = {}
summary_rows = []

for K_hops in K_VALUES:
    for R_COMM in R_COMM_VALUES:
        config_name = f"K={K_hops}, r_comm={R_COMM}"
        print(f"\n=== Running {config_name} ===")

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
            intruder_skill=0.0
        )

        trainer = GNNTrainerMultiEnv(
            adapter=adapter,
            hidden_dim=hidden_dim,
            F_feat=F_dim,
            G_feat=G_dim,
            K=K_hops,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            device=device,
            r_comm=R_COMM,
            total_timesteps=TOTAL_TIMESTEPS,
            rollout_length=ROLLOUT_LENGTH,
        )

        steps_done = 0
        iteration = 0

        while iteration < ITERATIONS:
            rollout_steps = min(ROLLOUT_LENGTH, ROLLOUT_LENGTH*ITERATIONS - steps_done)
            last_obs, rollout_metrics = trainer.collect_rollouts(num_steps=rollout_steps)
            update_metrics = trainer.update(last_obs, num_actor_epochs=NUM_EPOCHS, num_critic_epochs=NUM_EPOCHS, B=BATCH_SIZE)

            steps_done += rollout_steps
            iteration += 1

            progress = iteration / ITERATIONS
            new_skill = min(1.0, progress/0.6)  # linear skill increase
            trainer.adapter.env.scenario.intruder_skill = new_skill

            if iteration == 1 or iteration % LOG_EVERY == 0 or steps_done >= ITERATIONS*ROLLOUT_LENGTH:
                print(
                    f"[{config_name}] Iter {iteration}/{ITERATIONS} | "
                    f"steps={steps_done}/{ITERATIONS*ROLLOUT_LENGTH} | "
                    f"pi_loss={update_metrics['policy_loss']:.4f} | "
                    f"v_loss={update_metrics['value_loss']:.4f} | "
                    f"ent={update_metrics['entropy']:.4f} | "
                    f"bellman={update_metrics['mean_bellman_error']:.4f} | "
                    f"ep_ret={rollout_metrics['mean_episode_return']:.4f} | "
                    f"ep_rew={rollout_metrics['mean_episode_rewards']:.4f} |"
                    f"ep_scout_rew={rollout_metrics['mean_scout_rewards']:4f} |"
                    f"ep_inter_rew: {rollout_metrics['mean_inter_rewards']:.4f} |"
                    f"explained_var: {update_metrics['explained_var']:.4f}"
                )


        results[(K_hops, R_COMM)] = trainer.metrics_history

        final_return = trainer.metrics_history["mean_episode_return"][-1]
        final_reward = trainer.metrics_history["mean_episode_rewards"][-1]
        summary_rows.append((K_hops, R_COMM, final_return, final_reward))

        plot_path = os.path.join(OUTPUT_DIR, f"metrics_K{K_hops}_r{R_COMM:.1f}.png")
        plot_metrics(
            trainer.metrics_history,
            title=f"Training Metrics ({config_name})",
            save_path=plot_path,
        )
        print(f"Saved: {plot_path}")

        # Clean up for next config
        del trainer, adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
