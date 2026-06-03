"""
MAPPO evaluation entry point.

Usage:
    python -m experiments.evaluate_mappo --config configs/toy_sanity.yaml --checkpoint runs/toy_sanity/ckpt/final.pt
"""

import argparse
import torch
from backbone.utils import load_yaml_config, get_device
from backbone import MAPPO
from envs import ToyMultiAgentEnv


def main(config_path: str, checkpoint_path: str) -> None:
    """Evaluate a saved MAPPO checkpoint."""
    cfg = load_yaml_config(config_path)
    device = get_device(prefer_cuda=(cfg["device"] != "cpu"))
    env_cfg = cfg.get("env") or {"num_agents": 2, "episode_length": 32}
    env = ToyMultiAgentEnv(
        num_envs=cfg["runtime"]["num_envs"],
        num_agents=env_cfg["num_agents"],
        episode_length=env_cfg["episode_length"],
        seed=cfg["seed"],
    )
    model = MAPPO(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_agents=env.num_agents,
        hidden_dim=cfg["model"]["hidden_dim"],
        share_encoder=cfg["algo"]["share_encoder"],
        device=str(device),
    )
    model.load(checkpoint_path)
    model.eval()

    obs, state = env.reset()
    obs, state = obs.to(device), state.to(device)
    n_steps = env_cfg["episode_length"] * 4  # 4 deterministic eval episodes
    total_reward = 0.0
    total_success = 0.0
    for _ in range(n_steps):
        out = model.act(obs, state, deterministic=True)
        next_obs, next_state, reward, done, info = env.step(out["action"].cpu())
        total_reward += reward.mean().item()
        total_success += info["success"].mean().item()
        obs, state = next_obs.to(device), next_state.to(device)
    print(f"eval reward (mean per step) = {total_reward / n_steps:.4f}")
    print(f"eval success (mean per step) = {total_success / n_steps:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
