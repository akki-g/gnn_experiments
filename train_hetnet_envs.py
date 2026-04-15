"""
Unified training script for PP and PCP environments
Supports all three algorithms: IPPO, GNN-MAPPO, HetNet/MAHAC

Usage:
    python train_hetnet_envs.py --env pp --algo ippo
    python train_hetnet_envs.py --env pcp --algo gnn --k 2 --r_comm 1.5
    python train_hetnet_envs.py --env pcp --algo hetnet --r_comm 1.5
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

from environments.predator_prey import PredatorPreyAdapter
from environments.predator_capture_prey import PredatorCapturePreyAdapter


# shared hyper params
hidden_dim = 64
F_dim = 64
G_dim = 64
lr = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_eps = 0.2
value_coef = 0.5
entropy_coef = 0.01

# hetnet specifics
HETNET_N_HEADS = 4
HETNET_HEAD_DIM = 16
HETNET_N_LAYERS = 3
HETNET_SSN_DIM = 5

LOG_EVERY = 20


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


def make_adapter(args, device):
    if args.env == "pp":
        return PredatorPreyAdapter(
            num_envs=args.n_envs,
            device=device,
            n_scouts=args.n_scouts,
            n_interceptors=args.n_interc,
            max_steps=args.max_steps,
        )
    else:
        return PredatorCapturePreyAdapter(
            num_envs=args.n_envs,
            device=device,
            n_scouts=args.n_scouts,
            n_interceptors=args.n_interc,
            max_steps=args.max_steps,
        )


def run_ippo(args, device):
    from baseline.trainer import IPPOTrainer

    print(f"\n=== IPPO baseline ({args.env.upper()}) | {args.n_envs} envs | "
          f"device={device} ===", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    adapter = make_adapter(args, device)

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

    total_steps = args.iters * args.rollout_length
    steps_done = 0
    iteration = 0

    while steps_done < total_steps:
        rollout_steps = min(args.rollout_length, total_steps - steps_done)
        last_obs, rollout_metrics = trainer.collect_rollouts(num_steps=rollout_steps)
        update_metrics = trainer.update(
            last_obs,
            num_actor_epochs=args.n_epochs,
            num_critic_epochs=3,
            B=args.batch_size,
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

    tag = f"{args.env}_ippo"
    plot_and_save(trainer.metrics_history, f"IPPO — {args.env.upper()}", args.output_dir, f"{tag}.png")
    ckpt_path = os.path.join(args.output_dir, f"{tag}_final.pt")
    torch.save({
        "policy_state_dict": trainer.policy.state_dict(),
        "critic_state_dict": trainer.critic.state_dict(),
        "metrics_history": trainer.metrics_history,
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}", flush=True)
    print(
        f"\n[IPPO] Final return: "
        f"{trainer.metrics_history['mean_episode_return'][-1]:.3f}",
        flush=True,
    )


def run_gnn(args, device):
    from gnn.trainer_multienv import GNNTrainerMultiEnv

    config_name = f"K{args.k}_r{args.r_comm:.1f}"
    print(f"\n=== GNN-MAPPO | K={args.k}, r_comm={args.r_comm} ({args.env.upper()}) | "
          f"device={device} ===", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    adapter = make_adapter(args, device)

    total_steps = args.iters * args.rollout_length
    steps_done = 0
    iteration = 0

    trainer = GNNTrainerMultiEnv(
        adapter=adapter,
        hidden_dim=hidden_dim,
        F_feat=F_dim,
        G_feat=G_dim,
        K=args.k,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        device=device,
        r_comm=args.r_comm,
        total_timesteps=total_steps,
        rollout_length=args.rollout_length,
    )

    while steps_done < total_steps:
        rollout_steps = min(args.rollout_length, total_steps - steps_done)
        last_obs, rollout_metrics = trainer.collect_rollouts(num_steps=rollout_steps)
        update_metrics = trainer.update(
            last_obs=last_obs,
            num_actor_epochs=args.n_epochs,
            num_critic_epochs=3,
            B=args.batch_size,
        )
        steps_done += rollout_steps
        iteration += 1

        if iteration == 1 or iteration % LOG_EVERY == 0 or steps_done >= total_steps:
            print(
                f"[GNN {config_name}] iter={iteration} "
                f"steps={steps_done}/{total_steps} | "
                f"pi={update_metrics['policy_loss']:.4f} "
                f"v={update_metrics['value_loss']:.4f} "
                f"ent={update_metrics['entropy']:.4f} "
                f"ret={rollout_metrics['mean_episode_return']:.3f} "
                f"exp_var={update_metrics.get('explained_var', 0.0):.3f}",
                flush=True,
            )

    tag = f"{args.env}_gnn_{config_name}"
    plot_and_save(
        trainer.metrics_history,
        f"GNN-MAPPO {config_name} — {args.env.upper()}",
        args.output_dir,
        f"{tag}.png",
    )
    ckpt_path = os.path.join(args.output_dir, f"{tag}_final.pt")
    torch.save({
        "comm_policy_state_dict": trainer.comm_policy.state_dict(),
        "critic_state_dict": trainer.critic.state_dict(),
        "metrics_history": trainer.metrics_history,
        "config": {"K": args.k, "r_comm": args.r_comm},
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}", flush=True)
    print(
        f"\n[GNN {config_name}] Final return: "
        f"{trainer.metrics_history['mean_episode_return'][-1]:.3f}",
        flush=True,
    )


def run_hetnet(args, device):
    from HetGAT.hetnetPolicy import HetNetPolicy
    from HetGAT.critic import HetNetCritic
    from HetGAT.mahac import MAHAC
    from HetGAT.rollout import MAHACBuffer
    from HetGAT.utils import collect_rollout

    print(f"\n=== HetNet/MAHAC | r_comm={args.r_comm} ({args.env.upper()}) | "
          f"device={device} ===", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    adapter = make_adapter(args, device)

    obs_dim_s = adapter.obs_portion_dim
    obs_dim_i = adapter.obs_portion_dim
    state_dim = adapter.state_dim
    act_dim = adapter.action_dim
    n_s = adapter.n_scouts
    n_i = adapter.n_interceptors

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
        r_comm=args.r_comm,
    ).to(device)

    critic = HetNetCritic(
        ssn_dim=ssn_embed_dim,
        agent_dim=0,
        hidden_dim=hidden_dim,
        mode="per_class",
    ).to(device=device)

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
        log_std_min=-1.5,
    )

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

    total_steps = args.iters * args.rollout_length
    steps_done = 0

    for i in range(1, args.iters + 1):
        rollout_steps = min(args.rollout_length, total_steps - steps_done)

        buffer = MAHACBuffer(
            T=rollout_steps,
            B=args.n_envs,
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

        bootstrap_values = collect_rollout(
            env_adapter=adapter,
            policy=policy,
            critic=critic,
            buffer=buffer,
            n_scouts=n_s,
            n_interc=n_i,
        )

        with torch.no_grad():
            ep_returns = []
            running = torch.zeros(args.n_envs, device=device)
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

        update_metrics = mahac.update(buffer, bootstrap_values)
        steps_done += rollout_steps

        metrics_history["mean_episode_return"].append(mean_ep_return)
        metrics_history["mean_episode_reward"].append(mean_ep_reward)
        for key in metrics_history:
            if key not in ("mean_episode_return", "mean_episode_reward"):
                metrics_history[key].append(update_metrics.get(key, 0.0))

        if i == 1 or i % LOG_EVERY == 0 or steps_done >= total_steps:
            print(
                f"[HetNet] iter={i}/{args.iters} "
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
                f"std_s={update_metrics['log_std_scout']:.2f} "
                f"std_i={update_metrics['log_std_interc']:.2f}",
                flush=True,
            )

    tag = f"{args.env}_hetnet_r{args.r_comm:.1f}_s{n_s}i{n_i}"
    plot_and_save(
        metrics_history,
        f"HetNet/MAHAC r_comm={args.r_comm} — {args.env.upper()}",
        args.output_dir,
        f"{tag}.png",
    )
    ckpt_path = os.path.join(args.output_dir, f"{tag}_final.pt")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "metrics_history": metrics_history,
        "config": {"r_comm": args.r_comm, "n_scouts": n_s, "n_interc": n_i},
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}", flush=True)
    print(
        f"\n[HetNet] Final return: {metrics_history['mean_episode_return'][-1]:.3f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train on PP or PCP environments")
    parser.add_argument("--env", choices=["pp", "pcp"], required=True)
    parser.add_argument("--algo", choices=["ippo", "gnn", "hetnet"], required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--r_comm", type=float, default=1.5)
    parser.add_argument("--n_scouts", type=int, default=2)
    parser.add_argument("--n_interc", type=int, default=None)
    parser.add_argument("--n_envs", type=int, default=256)
    parser.add_argument("--rollout_length", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/hetnet_envs")
    args = parser.parse_args()

    # auto-set pcp default: n_interc=2 if not explicitly set
    if args.n_interc is None:
        args.n_interc = 2 if args.env == "pcp" else 1

    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(
        f"Env: {args.env.upper()} | {args.n_scouts}S + {args.n_interc}I | "
        f"world=5.0 | max_steps={args.max_steps}",
        flush=True,
    )

    if args.algo == "ippo":
        run_ippo(args, device)
    elif args.algo == "gnn":
        run_gnn(args, device)
    else:
        run_hetnet(args, device)


if __name__ == "__main__":
    main()
