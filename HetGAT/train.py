import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import os
import json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from HetGAT.hetnetPolicy import HetNetPolicy
from HetGAT.critic import HetNetCritic
from HetGAT.mahac import MAHAC
from HetGAT.rollout import MAHACBuffer
from HetGAT.utils import collect_rollout, build_ssn_input

from environments.guarded_territory import GuardedTerritoryAdapter

def make_config():
    parser = argparse.ArgumentParser(description="HetNet MAHAC Training")

    parser.add_argument("--n_envs", type=int, default=64)
    parser.add_argument("--n_scouts", type=int, default=3)
    parser.add_argument("--n_interc", type=int, default=3)
    parser.add_argument("--n_intruders", type=int, default=2)
    parser.add_argument("--n_zones", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--world_size", type=float, default=5.0)
    parser.add_argument("--scout_fov", type=float, default=1.0)
    parser.add_argument("--interceptor_fov", type=float, default=0.5)
    parser.add_argument("--tag_radius", type=float, default=0.1)
    parser.add_argument("--intruder_skill", type=float, default=0.5)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=16)
    parser.add_argument("--n_gat_layers", type=int, default=3)
    parser.add_argument("--r_comm", type=float, default=1.5)

    parser.add_argument("--n_iterations", type=int, default=2000)
    parser.add_argument("--rollout_length", type=int, default=200)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--log_std_min", type=float, default=-1.5)

    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", default=False,
                        help="Run a rendered eval episode after training (local only, not for SLURM)")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to a checkpoint .pt file to load for rendering")
 
    args = parser.parse_args()
    return args

@torch.no_grad()
def evaluate(env_adapter, policy, n_scouts, n_interc, device, n_episodes=10):
    """run deterministic eval eps, returns mean ep rew"""
    B = env_adapter.num_envs
    total_rewards = torch.zeros(B, device=device)
    episode_counts = torch.zeros(B, device=device)
    step_count = torch.zeros(B, device=device)

    obs_s, state_s, obs_i, state_i, positions = env_adapter.hetnet_reset()
    hidden_s = policy.preprocess_scout.init_hidden(B, n_scouts, device)
    hidden_i = policy.preprocess_interc.init_hidden(B, n_interc, device)

    max_eval_steps = env_adapter.env.max_steps if hasattr(env_adapter.env, 'max_steps') else 150
    for step in range(max_eval_steps * n_episodes):
        ssn_input = build_ssn_input(
            n_scouts, n_interc, env_adapter.n_intruders,
            env_adapter.world_size, step / max_eval_steps, B, device,
        )
 
        scout_dist, interc_dist, _, hidden_s, hidden_i = policy(
            obs_s, state_s, obs_i, state_i,
            positions, ssn_input, hidden_s, hidden_i,
            n_scouts, n_interc,
        )

        scout_actions = torch.tanh(scout_dist.mean)
        interc_actions = torch.tanh(interc_dist.mean)
        all_actions = torch.cat([scout_actions, interc_actions], dim=1)
 
        reward, done, info = env_adapter.hetnet_step(all_actions)
        obs_s, state_s, obs_i, state_i, positions = env_adapter.get_obs()
 
        total_rewards += reward
        step_count += 1

        if done.any():
            episode_counts += done.float()
            # Reset hidden states for terminated envs
            done_mask_s = done.unsqueeze(1).expand(-1, n_scouts).reshape(-1).bool()
            done_mask_i = done.unsqueeze(1).expand(-1, n_interc).reshape(-1).bool()
            for key in hidden_s:
                hidden_s[key][done_mask_s] = 0.0
            for key in hidden_i:
                hidden_i[key][done_mask_i] = 0.0
 
        # Stop when enough episodes completed
        if episode_counts.min() >= n_episodes:
            break
        
    mean_reward = (total_rewards / step_count.clamp(min=1)).mean().item()
    mean_episodes = episode_counts.mean().item()
    return mean_reward, mean_episodes


def plot_metrics(log_history, save_dir, run_name, log_std_min):
    """Generate training_curves.png with a 3x2 subplot grid."""
    if not log_history:
        return

    iters = [e['iter'] for e in log_history]

    def get(key):
        return [e.get(key, float('nan')) for e in log_history]

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(run_name, fontsize=14)

    # (0,0) Policy loss per class
    ax = axes[0, 0]
    ax.plot(iters, get('policy_loss_scout'), label='scout', color='tab:blue')
    ax.plot(iters, get('policy_loss_interc'), label='interceptor', color='tab:orange')
    ax.set_title('Policy Loss')
    ax.set_xlabel('Iteration')
    ax.legend()

    # (0,1) Value loss (combined)
    ax = axes[0, 1]
    ax.plot(iters, get('value_loss'), color='tab:green')
    ax.set_title('Value Loss')
    ax.set_xlabel('Iteration')

    # (1,0) Entropy per class
    ax = axes[1, 0]
    ax.plot(iters, get('entropy_scout'), label='scout', color='tab:blue')
    ax.plot(iters, get('entropy_interc'), label='interceptor', color='tab:orange')
    # entropy floor: 0.5 * D * (1 + ln(2*pi) + 2*log_std_min), D=2
    entropy_floor = 0.5 * 2 * (1 + np.log(2 * np.pi) + 2 * log_std_min)
    ax.axhline(entropy_floor, linestyle='--', color='gray', linewidth=0.8, label=f'floor ({entropy_floor:.2f})')
    ax.set_title('Entropy')
    ax.set_xlabel('Iteration')
    ax.legend()

    # (1,1) Explained variance per class
    ax = axes[1, 1]
    ax.plot(iters, get('ev_scout'), label='scout', color='tab:blue')
    ax.plot(iters, get('ev_interc'), label='interceptor', color='tab:orange')
    ax.axhline(0.0, linestyle='--', color='red', linewidth=0.8, label='useless (0)')
    ax.axhline(1.0, linestyle='--', color='green', linewidth=0.8, label='perfect (1)')
    ax.set_title('Explained Variance')
    ax.set_xlabel('Iteration')
    ax.legend()

    # (2,0) log_std per class
    ax = axes[2, 0]
    ax.plot(iters, get('log_std_scout'), label='scout', color='tab:blue')
    ax.plot(iters, get('log_std_interc'), label='interceptor', color='tab:orange')
    ax.axhline(log_std_min, linestyle='--', color='gray', linewidth=0.8, label=f'min ({log_std_min})')
    ax.set_title('log_std')
    ax.set_xlabel('Iteration')
    ax.legend()

    # (2,1) Advantage statistics with +/-1 std shading
    ax = axes[2, 1]
    mean_s = np.array(get('advantage_mean_scout'))
    std_s = np.array(get('advantage_std_scout'))
    mean_i = np.array(get('advantage_mean_interc'))
    std_i = np.array(get('advantage_std_interc'))
    iters_arr = np.array(iters)

    ax.plot(iters_arr, mean_s, label='scout mean', color='tab:blue')
    ax.fill_between(iters_arr, mean_s - std_s, mean_s + std_s, alpha=0.2, color='tab:blue')
    ax.plot(iters_arr, mean_i, label='interceptor mean', color='tab:orange')
    ax.fill_between(iters_arr, mean_i - std_i, mean_i + std_i, alpha=0.2, color='tab:orange')
    ax.axhline(0.0, linestyle='--', color='gray', linewidth=0.8)
    ax.set_title('Advantage Statistics (pre-norm)')
    ax.set_xlabel('Iteration')
    ax.legend()

    if log_history and 'intruder_skill' in log_history[0]:
        ax2 = ax.twinx()
        ax2.plot(iters_arr, get('intruder_skill'), color='gray', linestyle=':', linewidth=1.5, label='intruder skill')
        ax2.set_ylabel('Intruder Skill', color='gray')
        ax2.set_ylim(-0.05, 1.1)
        ax2.legend(loc='upper left')

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'training_curves.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> saved training curves: {out_path}")


@torch.no_grad()
def render_evaluation(args, policy, save_dir):
    """
    Run a single deterministic episode and save as a GIF.
    Requires --render flag and imageio installed.
    Only intended for local runs, not SLURM.
    """
    try:
        import imageio
    except ImportError:
        print("imageio not installed -- skipping render. pip install imageio")
        return

    device = torch.device('cpu')
    policy = policy.to(device)
    policy.eval()

    env_adapter = GuardedTerritoryAdapter(
        num_envs=1,
        device='cpu',
        n_scouts=args.n_scouts,
        n_interceptors=args.n_interc,
        n_intruders=args.n_intruders,
        n_zones=args.n_zones,
        max_steps=args.max_steps,
        world_size=args.world_size,
        scout_fov=args.scout_fov,
        interceptor_fov=args.interceptor_fov,
        tag_radius=args.tag_radius,
        intruder_skill=args.intruder_skill,
    )

    obs_s, state_s, obs_i, state_i, positions = env_adapter.hetnet_reset()
    hidden_s = policy.preprocess_scout.init_hidden(1, args.n_scouts, device)
    hidden_i = policy.preprocess_interc.init_hidden(1, args.n_interc, device)

    frames = []
    total_reward = 0.0
    step = 0

    for step in range(args.max_steps):
        ssn_input = build_ssn_input(
            args.n_scouts, args.n_interc, args.n_intruders,
            args.world_size, step / args.max_steps, 1, device,
        )

        scout_dist, interc_dist, _, hidden_s, hidden_i = policy(
            obs_s, state_s, obs_i, state_i,
            positions, ssn_input, hidden_s, hidden_i,
            args.n_scouts, args.n_interc,
        )

        all_actions = torch.cat([torch.tanh(scout_dist.mean), torch.tanh(interc_dist.mean)], dim=1)
        reward, done, _ = env_adapter.hetnet_step(all_actions)
        obs_s, state_s, obs_i, state_i, positions = env_adapter.get_obs()

        total_reward += reward.item()

        # collect frame
        frame = env_adapter.env.render(mode='rgb_array')
        if frame is not None:
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            frames.append(frame)

        if done.any():
            break

    print(f"Render eval: total_reward={total_reward:.3f}, steps={step + 1}")

    if frames:
        gif_path = os.path.join(save_dir, 'eval_episode.gif')
        imageio.mimsave(gif_path, frames, fps=15)
        print(f"  -> saved eval gif: {gif_path}")
    else:
        print("  -> no frames captured (env may not support rgb_array rendering)")


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    if args.run_name is None:
        args.run_name = f"hetnet_s{args.n_scouts}_i{args.n_interc}_e{args.n_envs}_{int(time.time())}"
    save_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(save_dir, exist_ok=True)


    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run: {args.run_name}")
    print(f"Save dir: {save_dir}")
    print(f"Device: {device}")

    env_adapter = GuardedTerritoryAdapter(
        num_envs=args.n_envs,
        device=str(device),
        n_scouts=args.n_scouts,
        n_interceptors=args.n_interc,
        n_intruders=args.n_intruders,
        n_zones=args.n_zones,
        max_steps=args.max_steps,
        world_size=args.world_size,
        scout_fov=args.scout_fov,
        interceptor_fov=args.interceptor_fov,
        tag_radius=args.tag_radius,
        intruder_skill=0.0,  # curriculum starts at 0; --intruder_skill is only used for --render eval
    )

    obs_portion_dim = env_adapter.obs_portion_dim
    state_dim = env_adapter.state_dim
    print(f"Env: {args.n_scouts}s/{args.n_interc}i/{args.n_intruders}intruders")
    print(f"  obs_portion_dim={obs_portion_dim}, state_dim={state_dim}")
    print(f"  total_obs_dim={env_adapter.obs_dim}, action_dim={env_adapter.action_dim}")


    policy = HetNetPolicy(
        obs_dim_scout=obs_portion_dim,
        obs_dim_interc=obs_portion_dim,
        state_dim_scout=state_dim,
        state_dim_interc=state_dim,
        action_dim=env_adapter.action_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        n_layers=args.n_gat_layers,
        ssn_input_dim=5,
        r_comm=args.r_comm,
    ).to(device)

    ssn_out_dim = args.n_heads * args.head_dim  # after concat
    critic = HetNetCritic(
        ssn_dim=ssn_out_dim,
        mode='per_class',
    ).to(device)

    n_params_policy = sum(p.numel() for p in policy.parameters())
    n_params_critic = sum(p.numel() for p in critic.parameters())
    print(f"Policy params: {n_params_policy:,}")
    print(f"Critic params: {n_params_critic:,}")
    print(f"Total params:  {n_params_policy + n_params_critic:,}")

    trainer = MAHAC(
        policy=policy,
        critic=critic,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        lam=args.lam,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        log_std_min=args.log_std_min,
    )

    buffer = MAHACBuffer(
        T=args.rollout_length,
        B=args.n_envs,
        n_scouts=args.n_scouts,
        n_interc=args.n_interc,
        obs_dim_scout=obs_portion_dim,
        obs_dim_interc=obs_portion_dim,
        state_dim_scout=state_dim,
        state_dim_interc=state_dim,
        action_dim=env_adapter.action_dim,
        ssn_dim=5,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    print(f"\nStarting training: {args.n_iterations} iterations")
    print(f"  rollout_length={args.rollout_length}, ppo_epochs={args.ppo_epochs}")
    print(f"  Total env steps per iter: {args.rollout_length * args.n_envs}")
    print("=" * 70)

    log_history = []
    best_eval_reward = float("-inf")

    for iteration in range(1, args.n_iterations + 1):
        t_start = time.time()

        bootstrap_values = collect_rollout(
            env_adapter, policy, critic, buffer,
            args.n_scouts, args.n_interc,
        )

        # --- Intruder skill curriculum (match GNN-MAPPO baseline) ---
        # Ramps from 0.0 to 1.0 over the first 60% of training,
        # then holds at 1.0 for the remaining 40%.
        progress = iteration / args.n_iterations
        new_skill = min(1.0, progress / 0.6)
        env_adapter.env.scenario.intruder_skill = new_skill

        metrics = trainer.update(buffer, bootstrap_values)

        t_elapsed = time.time() - t_start
        fps = (args.rollout_length * args.n_envs) / t_elapsed

        if iteration % args.log_interval == 0:
            log_entry = {
                "iter": iteration,
                "fps": fps,
                "intruder_skill": new_skill,
                **metrics,
            }
            log_history.append(log_entry)
 
            print(
                f"[{iteration:5d}/{args.n_iterations}] "
                f"fps={fps:.0f}  "
                f"p_loss_s={metrics['policy_loss_scout']:.4f}  "
                f"p_loss_i={metrics['policy_loss_interc']:.4f}  "
                f"v_loss={metrics['value_loss']:.4f}  "
                f"ent_s={metrics['entropy_scout']:.3f}  "
                f"ent_i={metrics['entropy_interc']:.3f}  "
                f"std_s={metrics['log_std_scout']:.3f}  "
                f"std_i={metrics['log_std_interc']:.3f}  "
                f"skill={new_skill:.2f}"
            )

        if iteration % args.save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"ckpt_{iteration}.pt")
            torch.save({
                "iteration": iteration,
                "policy_state_dict": policy.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "metrics": metrics,
            }, ckpt_path)

            log_path = os.path.join(save_dir, "log.json")
            with open(log_path, "w") as f:
                json.dump(log_history, f, indent=2)
 
            print(f"  -> saved checkpoint: {ckpt_path}")


    final_path = os.path.join(save_dir, "final.pt")
    torch.save({
        "iteration": args.n_iterations,
        "policy_state_dict": policy.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
    }, final_path)
 
    log_path = os.path.join(save_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)
 
    print("=" * 70)
    print(f"Training complete. Final checkpoint: {final_path}")

    plot_metrics(log_history, save_dir, args.run_name, args.log_std_min)

    if args.render:
        render_evaluation(args, policy, save_dir)


if __name__ == "__main__":
    args = make_config()

    if args.render and args.load_checkpoint is not None:
        # Render-only mode: load checkpoint and skip training
        device = torch.device('cpu')
        ckpt = torch.load(args.load_checkpoint, map_location=device)

        if args.run_name is None:
            args.run_name = os.path.basename(os.path.dirname(args.load_checkpoint))
        save_dir = os.path.join(args.save_dir, args.run_name)
        os.makedirs(save_dir, exist_ok=True)

        env_adapter = GuardedTerritoryAdapter(
            num_envs=1, device='cpu',
            n_scouts=args.n_scouts, n_interceptors=args.n_interc,
            n_intruders=args.n_intruders, n_zones=args.n_zones,
            max_steps=args.max_steps, world_size=args.world_size,
            scout_fov=args.scout_fov, interceptor_fov=args.interceptor_fov,
            tag_radius=args.tag_radius, intruder_skill=args.intruder_skill,
        )
        policy = HetNetPolicy(
            obs_dim_scout=env_adapter.obs_portion_dim,
            obs_dim_interc=env_adapter.obs_portion_dim,
            state_dim_scout=env_adapter.state_dim,
            state_dim_interc=env_adapter.state_dim,
            action_dim=env_adapter.action_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            n_layers=args.n_gat_layers,
            ssn_input_dim=5,
            r_comm=args.r_comm,
        )
        policy.load_state_dict(ckpt['policy_state_dict'])
        render_evaluation(args, policy, save_dir)
    else:
        train(args)