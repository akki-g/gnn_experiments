"""
Unified entry point for Guided Coverage training runs
Used for all our current testing algos: IPPO, "GNN-MAPPO", HetNet
Guided Coverage Env:
    - No intruders unlike guarded territory
    - Shared reward for both classes based on coverage

Usage:
    python guided_coverage_train.py --algo ippo
    python guided_coverage_train.py --algo gnn --k 2 --r_comm 1.0
    python guided_coverage_train.py --algo hetnet --r_comm 1.0 
"""

import argparse
import os 
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from environments.guided_coverage import GuidedCoverageAdapter


# shared hyper params
hidden_dim = 64
F_dim = 64
G_dim = 64
lr = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_eps = 0.2
value_coef = 0.5
entropy_coef = 0.01   # entropy regularization — prevents policy collapse in continuous MARL

ROLLOUT_LENGTH = 265
ITERATIONS = 1000
BATCH_SIZE = 256
NUM_EPOCHS = 5
SEED = 42
LOG_EVERY = 20

# env config: simpler env so smaller world, no intruders, and fewer steps
NUM_ENVS = 256
MAX_STEPS = 100
N_SCOUTS = 2
N_INTERCEPTORS = 2
N_ZONES = 3             # more zones than intruders forces them to rely on scout comm
WORLD_SIZE = 3.0    
SCOUT_FOV = 1.5
INTERCEPTOR_FOV = 0.3

# hetnet specifics
HETNET_N_HEADS = 4
HETNET_HEAD_DIM = 16
HETNET_N_LAYERS = 3
HETNET_SSN_DIM = 5

def plot_and_save(metrics_history, title, output_dir, filename):
    populated = {k: v for k, v in metrics_history.items() if len(v) > 0}
    n_metrics = len(populated)
    if n_metrics == 0:
        return
    nrows = max((n_metrics + 1) // 2, 1)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows))
    if n_metrics <= 2:
        axes = np.array(axes).flatten()
    else:
        axes = axes.flatten()
    for i, (name, vals) in enumerate(populated.items()):
        if i < len(axes):
            axes[i].plot(range(1, len(vals) + 1), vals, linewidth=1.5)
            axes[i].set_title(name)
            axes[i].set_xlabel("Iteration")
            axes[i].grid(True, alpha=0.3)
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}", flush=True)


#adapter construction
def make_adapter(device):
    return GuidedCoverageAdapter(
        num_envs=NUM_ENVS,
        device=device,
        n_scouts=N_SCOUTS,
        n_intercs=N_INTERCEPTORS,
        n_zones=N_ZONES,
        max_steps=MAX_STEPS,
        world_size=WORLD_SIZE,
        scout_fov=SCOUT_FOV,
        interc_fov=INTERCEPTOR_FOV,
    )

# ippo training
def run_ippo(device, output_dir = "ippo_gc_run"):
    from baseline.trainer import IPPOTrainer
    print(f"\n=== IPPO baseline (Guided Coverage) | {NUM_ENVS} envs | "
          f"device={device} ===", flush=True)

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
    steps_done = 0
    iteration = 0

    while steps_done < total_steps:
        rollout_steps = min(ROLLOUT_LENGTH, total_steps-steps_done)
        last_obs, rollout_metrics = trainer.collect_rollouts(num_steps=ROLLOUT_LENGTH)
        update_metrics = trainer.update(
            last_obs, 
            num_actor_epochs=NUM_EPOCHS,
            num_critic_epochs=3,
            B=BATCH_SIZE,
        )

        steps_done += rollout_steps
        iteration += 1

        if iteration == 1 or iteration % LOG_EVERY == 0 or steps_done >= total_steps:
            print(
                f"[IPPO] iter={iteration} steps={steps_done}/{total_steps} | "
                f"pi={update_metrics['policy_loss']:.4f} "
                f"v={update_metrics['value_loss']:.4f} "
                f"ent={update_metrics['entropy']:.4f} "
                f"ret={rollout_metrics['mean_episode_return']:.3f} "
                f"exp_var={update_metrics.get('explained_var', 0.0):.3f}",
                flush=True,
            )

    plot_and_save(trainer.metrics_history,
                  "IPPO - Giuded Coverage",
                  output_dir,
                  "gc_ippo.png")
    
    ckpt_path = os.path.join(output_dir, "gc_ippo_final.pt")
    torch.save({
        "policy_state_dict": trainer.policy.state_dict(),
        "critic_state_dict": trainer.critic.state_dict(),
        "metrics_history": trainer.metrics_history,
    }, ckpt_path)

    print(f"Saved checkpoint: {ckpt_path}", flush=True)
    print(
        f"\n[IPPO] Final Return:"
        f"{trainer.metrics_history['mean_episode_return'][-1]:.3f}",
        flush=True
    )


# gnn-mappo
def run_gnn(k, r_comm, device, output_dir):
    from gnn.trainer_multienv import GNNTrainerMultiEnv

    config_name = f"K{k}_r{r_comm:.1f}"
    print(f"\n=== GNN-MAPPO | K={k}, r_comm={r_comm} (Guided Coverage) | "
          f"device={device} ===", flush=True)
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    adapter = make_adapter(device)

    total_timesteps = ITERATIONS * ROLLOUT_LENGTH
    steps_done = 0

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
        total_timesteps=ITERATIONS*ROLLOUT_LENGTH,
        rollout_length=ROLLOUT_LENGTH
    )
    
    for i in range(1, ITERATIONS+1):
        rollout_steps = min(ROLLOUT_LENGTH, total_timesteps - steps_done)
        last_obs, rollout_metrics = trainer.collect_rollouts(num_steps=rollout_steps)
        update_metrics = trainer.update(
            last_obs=last_obs,
            num_actor_epochs=NUM_EPOCHS,
            num_critic_epochs=3,
            B=BATCH_SIZE,
        )

        steps_done += rollout_steps

        if i == 1 or i % LOG_EVERY == 0 or i == ITERATIONS:
            print(
                f"[GNN {config_name}] iter={i}/{ITERATIONS} "
                f"steps={steps_done} | "
                f"pi={update_metrics['policy_loss']:.4f} "
                f"v={update_metrics['value_loss']:.4f} "
                f"ent={update_metrics['entropy']:.4f} "
                f"ret={rollout_metrics['mean_episode_return']:.3f} "
                f"exp_var={update_metrics.get('explained_var', 0.0):.3f}",
                flush=True,
            )

    plot_and_save(
        trainer.metrics_history,
        f"GNN-MAPPO {config_name} — Guided Coverage",
        output_dir,
        f"gc_gnn_{config_name}.png",
    )
    # save checkpoint
    ckpt_path = os.path.join(output_dir, f"gc_gnn_{config_name}_final.pt")
    torch.save({
        "comm_policy_state_dict": trainer.comm_policy.state_dict(),
        "critic_state_dict":      trainer.critic.state_dict(),
        "metrics_history":        trainer.metrics_history,
        "config": {"K": k, "r_comm": r_comm},
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}", flush=True)
    print(
        f"\n[GNN {config_name}] Final return: "
        f"{trainer.metrics_history['mean_episode_return'][-1]:.3f}",
        flush=True,
    )

# hetnet 
def run_hetnet(r_comm, device, output_dir):

    from HetGAT.hetnetPolicy import HetNetPolicy
    from HetGAT.critic import HetNetCritic
    from HetGAT.mahac import MAHAC
    from HetGAT.rollout import MAHACBuffer
    from HetGAT.utils import collect_rollout

    print(f"\n=== HetNet/MAHAC | r_comm={r_comm} (Guided Coverage) | "
          f"device={device} ===", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    adapter = make_adapter(device=device)

    # dims
    obs_dim_s = adapter.obs_portion_dim
    obs_dim_i = adapter.obs_portion_dim
    state_dim = adapter.state_dim
    act_dim = adapter.action_dim
    n_s = adapter.n_scouts
    n_i = adapter.n_interceptors

    # ssn final embedding dim after multihead concat
    ssn_embed_dim = HETNET_N_HEADS * HETNET_HEAD_DIM

    policy = HetNetPolicy(
        obs_dim_scout=obs_dim_s,
        obs_dim_interc=obs_dim_i,
        state_dim_scout=state_dim,
        state_dim_interc=state_dim,
        action_dim=act_dim,
        hidden_dim=hidden_dim,
        n_heads=HETNET_N_HEADS,
        head_dim=HETNET_HEAD_DIM,
        n_layers=HETNET_N_LAYERS,
        ssn_input_dim=HETNET_SSN_DIM,
        r_comm=r_comm,
    ).to(device)

    # build critic (per-class) both read only from ssn
    critic = HetNetCritic(
        ssn_dim=ssn_embed_dim,
        agent_dim=0,
        hidden_dim=hidden_dim,
        mode="per_class",
    ).to(device=device)

    # build MAHAC (owns actor_optim and critic_optim)
    mahac = MAHAC(
        policy=policy,
        critic=critic,
        lr_actor=lr,
        lr_critic=lr,
        critic_epochs=3,
        gamma=gamma,
        lam=gae_lambda,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=0.5,
        ppo_epochs=4,
        log_std_min=-1.5
    )

    # metrics history from mahac.update()   
    metrics_history = {
        "mean_episode_return": [], "mean_episode_reward": [],
        "policy_loss_scout": [], "policy_loss_interc": [],
        "value_loss": [], 
        "entropy_scout": [], "entropy_interc": [],
        "ev_scout": [], "ev_interc": [],
        "ev_scout_post": [], "ev_interc_post": [],
        "msbe_scout": [], "msbe_interc": [],
        "ratio_scout_mean": [], "ratio_interc_mean": [],
        "log_std_scout": [], "log_std_interc": [],
    }

    total_steps = ITERATIONS * ROLLOUT_LENGTH   
    steps_done = 0

    #training loop
    for i in range(1, ITERATIONS+1):
        rollout_steps = min(ROLLOUT_LENGTH, total_steps - steps_done)

        #preallocate buffer
        buffer = MAHACBuffer(
            T=rollout_steps,
            B=NUM_ENVS,
            n_scouts=n_s,
            n_interc=n_i,
            obs_dim_scout=obs_dim_s,
            obs_dim_interc=obs_dim_i,
            state_dim_scout=state_dim,
            state_dim_interc=state_dim,
            action_dim=act_dim,
            ssn_dim=HETNET_SSN_DIM,
            hidden_dim=hidden_dim,
            device=device,
        )

        # collect T steps of experience
        # collect_rollout():
        #   calls adapter.hetnet_reset() for init obs
        #   loops T steps: policy forward -> squash actions -> critic -> buffer store -> hetnet step
        #   resets LSTM hidden for done envs
        #   returns bootstrap_vals dict

        bootstrap_values = collect_rollout(
            env_adapter=adapter,
            policy=policy,
            critic=critic,
            buffer=buffer,
            n_scouts=n_s,
            n_interc=n_i
        )

        # compute ep returns from buffer
        with torch.no_grad():
            ep_returns = []
            running = torch.zeros(NUM_ENVS, device=device)
            for t in range(buffer.T):
                running += buffer.rewards[t]
                done_mask = buffer.dones[t].bool()
                if done_mask.any():
                    for e in done_mask.nonzero(as_tuple=True)[0].tolist():
                        ep_returns.append(running[e].item())
                    running[done_mask] = 0.0

            mean_ep_return = (
                float(np.mean(ep_returns)) if ep_returns
                else running.mean().item()
            )

            mean_ep_reward = buffer.rewards.mean().item()

        # ppo update (backprop through time over T timesteps)
        # mahac.update(...) -> metrics dict
        # actor: ppo_epochs outer loops * T timesteps inner loop
        # critic: critic_epochs outer loop * T inner, w value clipping

        update_metrics = mahac.update(buffer, bootstrap_values)
        steps_done += rollout_steps

        # record
        metrics_history["mean_episode_return"].append(mean_ep_return)
        metrics_history["mean_episode_reward"].append(mean_ep_reward)
        for key in metrics_history:
            if key not in ("mean_episode_return", "mean_episode_reward"):
                metrics_history[key].append(update_metrics.get(key,0.0))
        
        if i == 1 or i % LOG_EVERY == 0 or steps_done >= total_steps:
            print(
                f"[HetNet] iter={i}/{ITERATIONS} "
                f"steps={steps_done}/{total_steps} | "
                f"pi_s={update_metrics['policy_loss_scout']:.4f} "
                f"pi_i={update_metrics['policy_loss_interc']:.4f} "
                f"v={update_metrics['value_loss']:.4f} "
                f"ent_s={update_metrics['entropy_scout']:.3f} "
                f"ent_i={update_metrics['entropy_interc']:.3f} "
                f"ret={mean_ep_return:.3f} "
                f"ev_s_post={update_metrics['ev_scout_post']:.3f} "
                f"ev_i_post={update_metrics['ev_interc_post']:.3f} "
                f"r_s={update_metrics['ratio_scout_mean']:.3f} "
                f"r_i={update_metrics['ratio_interc_mean']:.3f} "
                f"log_std_s={update_metrics['log_std_scout']:.2f} "  # FIX-P2-B3: label matches value
                f"log_std_i={update_metrics['log_std_interc']:.2f}",  # FIX-P2-B3
                flush=True,
            )
 
    plot_and_save(
        metrics_history,
        f"HetNet/MAHAC r_comm={r_comm} — Guided Coverage",
        output_dir,
        f"gc_hetnet_r{r_comm:.1f}.png",
    )
    ckpt_path = os.path.join(output_dir, f"gc_hetnet_r{r_comm:.1f}_final.pt")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "metrics_history":   metrics_history,
        "config": {"r_comm": r_comm},
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}", flush=True)
    print(f"\n[HetNet] Final return: {metrics_history['mean_episode_return'][-1]:.3f}",
          flush=True)
 

def main():
    parser = argparse.ArgumentParser(
        description="Train on Guided Coverage"
    )
    parser.add_argument("--algo", choices=["gnn", "ippo", "hetnet"], required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--r_comm", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="outputs/guided_coverage")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"Env: Guided Coverage | {N_SCOUTS}S + {N_INTERCEPTORS}I -> "
          f"{N_ZONES} zones | world={WORLD_SIZE}", flush=True)

    if args.algo == "gnn":
        run_gnn(args.k, args.r_comm, device, args.output_dir)
    elif args.algo == "ippo":
        run_ippo(device, args.output_dir)
    else:
        run_hetnet(args.r_comm, device, args.output_dir)


if __name__ == "__main__":
    main()
