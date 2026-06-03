"""
MAPPO training entry point.

Usage:
    python -m experiments.train_mappo --config configs/toy_sanity.yaml
    python -m experiments.train_mappo --config configs/toy_sanity.yaml --resume runs/toy_sanity/ckpt/final.pt
"""

import os
import argparse
import torch
from backbone.utils import load_yaml_config, set_seed, get_device, make_optimizer
from backbone import MAPPO, RolloutBuffer
from backbone.ppo_update import ppo_update
from comm import IdentityComm
from envs import ToyMultiAgentEnv


def _build_comm_module(comm_cfg: dict):
    """Instantiate the comm module named in the config."""
    name = comm_cfg.get("module", "identity").lower()
    if name == "identity":
        return IdentityComm()
    raise ValueError(
        f"Unknown comm module '{name}'. "
        "Phase 1 supports only 'identity'. "
        "Future modules (broadcast, attention, graph) are not yet implemented."
    )


def main(config_path: str, resume: str = None) -> None:
    """Train MAPPO on the configured environment."""
    cfg = load_yaml_config(config_path)
    set_seed(cfg["seed"])
    device = get_device(prefer_cuda=(cfg["device"] != "cpu"))

    # Defensive fallback: mappo_base.yaml has no "env" block
    env_cfg = cfg.get("env") or {"num_agents": 2, "episode_length": 32}

    env = ToyMultiAgentEnv(
        num_envs=cfg["runtime"]["num_envs"],
        num_agents=env_cfg["num_agents"],
        episode_length=env_cfg["episode_length"],
        seed=cfg["seed"],
    )

    model_cfg = cfg.get("model", {})
    model = MAPPO(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_agents=env.num_agents,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        comm_module=_build_comm_module(cfg.get("comm", {})),
        share_encoder=cfg["algo"]["share_encoder"],
        encoder_layers=model_cfg.get("encoder_layers", 2),
        actor_layers=model_cfg.get("actor_layers", 1),
        critic_layers=model_cfg.get("critic_layers", 2),
        device=str(device),
    )
    optimizer = make_optimizer(model.parameters(), lr=cfg["algo"]["lr"])
    use_value_norm = cfg["algo"].get("use_value_norm", False)
    value_normalizer = model.value_normalizer if use_value_norm else None
    buffer = RolloutBuffer(
        rollout_length=cfg["runtime"]["rollout_length"],
        num_envs=cfg["runtime"]["num_envs"],
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=str(device),
    )

    global_step = 0
    if resume:
        ckpt = model.load(resume)
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        print(f"resumed from {resume} at global_step={global_step}")

    obs, state = env.reset()
    obs, state = obs.to(device), state.to(device)

    T = cfg["runtime"]["rollout_length"]
    B = cfg["runtime"]["num_envs"]
    total_iters = max(1, cfg["runtime"]["total_env_steps"] // (T * B))

    ep_return = torch.zeros(B, device=device)
    ep_success = torch.zeros(B, device=device)
    ep_count = 0
    completed_returns = []
    completed_success = []

    for it in range(total_iters):
        buffer.clear()

        # ---- Rollout collection: eval mode so future dropout/batchnorm are inactive ----
        model.eval()
        for t in range(T):
            out = model.act(obs, state)
            next_obs, next_state, reward, done, info = env.step(out["action"].cpu())
            reward = reward.to(device)
            done = done.to(device)
            # Toy env has no time-limit truncation: bad_masks = all-ones (default)
            # Toy env has no dead agents: active_masks = all-ones (default)
            buffer.insert(
                obs=obs,
                state=state,
                action=out["action"],
                log_prob=out["log_prob"],
                value=out["value"],
                reward=reward,
                done=done,
                # bad_mask and active_mask omitted → buffer defaults to all-ones
            )
            ep_return += reward.mean(dim=1)
            ep_success += info["success"].to(device)
            # Reset per-env return when episode ends (done is [B, N], same across N)
            env_done = done[:, 0].bool()
            if env_done.any():
                for i in torch.nonzero(env_done, as_tuple=False).flatten().tolist():
                    completed_returns.append(ep_return[i].item())
                    completed_success.append(
                        ep_success[i].item() / env_cfg["episode_length"]
                    )
                ep_return[env_done] = 0.0
                ep_success[env_done] = 0.0
                ep_count += int(env_done.sum().item())
            global_step += B
            obs, state = next_obs.to(device), next_state.to(device)

        # Bootstrap value for the state after the last collected step
        model.eval()
        with torch.no_grad():
            last_value = model.critic(state)

        buffer.compute_returns_and_advantages(
            last_value, cfg["algo"]["gamma"], cfg["algo"]["gae_lambda"],
            value_normalizer=value_normalizer,
        )

        # ---- PPO update: model.train() is called inside ppo_update() ----
        metrics = ppo_update(
            model,
            optimizer,
            buffer,
            ppo_epochs=cfg["algo"]["ppo_epochs"],
            num_minibatches=cfg["algo"]["num_minibatches"],
            clip_eps=cfg["algo"]["clip_eps"],
            value_clip_eps=cfg["algo"]["value_clip_eps"],
            entropy_coef=cfg["algo"]["entropy_coef"],
            value_coef=cfg["algo"]["value_coef"],
            max_grad_norm=cfg["algo"]["max_grad_norm"],
            normalize_advantages=cfg["algo"]["normalize_advantages"],
            value_normalizer=value_normalizer,
        )

        # NaN guards
        for k, v in metrics.items():
            if isinstance(v, float):
                assert v == v, f"NaN in metric {k}"

        if it % cfg["logging"]["console_log_interval"] == 0:
            recent_ret = sum(completed_returns[-32:]) / max(len(completed_returns[-32:]), 1)
            recent_suc = sum(completed_success[-32:]) / max(len(completed_success[-32:]), 1)
            print(
                f"iter={it:4d} step={global_step:6d} ep_ret={recent_ret:.3f} "
                f"ep_succ={recent_suc:.3f} pol={metrics['policy_loss']:+.4f} "
                f"val={metrics['value_loss']:.4f} ent={metrics['entropy']:.4f} "
                f"tot={metrics['total_loss']:+.4f} gnorm={metrics['grad_norm']:.3f} "
                f"kl={metrics['approx_kl']:.4f} clipf={metrics['clipfrac']:.3f} "
                f"ev={metrics['explained_variance']:+.3f} "
                f"r0={metrics['mean_ratio_epoch0']:.4f} lr={cfg['algo']['lr']:.1e} seed={cfg['seed']}"
            )

        if it > 0 and it % cfg["runtime"]["checkpoint_interval"] == 0:
            os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
            ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"], f"ckpt_{it}.pt")
            model.save(ckpt_path, optimizer=optimizer, global_step=global_step)

    os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
    model.save(
        os.path.join(cfg["logging"]["checkpoint_dir"], "final.pt"),
        optimizer=optimizer,
        global_step=global_step,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPPO backbone.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    args = parser.parse_args()
    main(args.config, resume=args.resume)
