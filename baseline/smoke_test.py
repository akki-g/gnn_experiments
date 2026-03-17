from __future__ import annotations

import ast
import importlib
import sys
import traceback
from pathlib import Path

import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from baseline.critic import CriticNetwork
from baseline.policy import IPPOPolicy
from baseline.rollout_buffer import IPPORolloutBuffer
from baseline.trainer import IPPOTrainer


class FakeAdapter:
    def __init__(
        self,
        num_envs: int = 1,
        n_scouts: int = 2,
        n_interceptors: int = 1,
        n_intruders: int = 2,
        n_zones: int = 1,
        obs_dim: int = 10,
        action_dim: int = 2,
        done_every: int = 4,
        device: str = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.n_scouts = n_scouts
        self.n_interceptors = n_interceptors
        self.n_defenders = n_scouts + n_interceptors
        self.n_intruders = n_intruders
        self.n_zones = n_zones
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.done_every = done_every
        self.device = device
        self._step_count = 0

    def _make_obs_pos(self):
        obs = torch.randn(self.num_envs, self.n_defenders, self.obs_dim, device=self.device)
        pos = torch.randn(self.num_envs, self.n_defenders, 2, device=self.device)
        obs[:, :, 2:4] = pos
        return obs, pos

    def reset(self):
        self._step_count = 0
        return self._make_obs_pos()

    def reset_env(self):
        return self._make_obs_pos()

    def step(self, actions):
        self._step_count += 1
        obs, pos = self._make_obs_pos()
        rewards = -actions.pow(2).sum(dim=-1)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.done_every > 0 and self._step_count % self.done_every == 0:
            dones[:] = True
        info = {
            "n_tagged": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "n_breached": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
        }
        return obs, rewards, dones, info, pos


def test_imports() -> None:
    for name in [
        "baseline.policy",
        "baseline.critic",
        "baseline.rollout_buffer",
        "baseline.trainer",
    ]:
        importlib.import_module(name)


def test_run_syntax() -> None:
    run_path = Path(__file__).with_name("run.py")
    ast.parse(run_path.read_text(encoding="utf-8"), filename=str(run_path))


def test_policy() -> None:
    policy = IPPOPolicy(obs_dim=10, hidden_dim=16, action_dim=2)
    obs = torch.randn(4, 3, 10)
    mean = policy(obs)
    actions, log_prob, entropy = policy.get_actions(obs)
    eval_lp, eval_entropy = policy.evaluate_actions(obs, actions)
    assert mean.shape == (4, 3, 2)
    assert actions.shape == (4, 3, 2)
    assert log_prob.shape == (4, 3)
    assert entropy.shape == (4, 3)
    assert eval_lp.shape == (4, 3)
    assert eval_entropy.shape == (4, 3)


def test_critic() -> None:
    critic = CriticNetwork(obs_dim=10, hidden_dim=16, device="cpu")
    out = critic(torch.randn(6, 10))
    assert out.shape == (6, 1)
    assert torch.isfinite(out).all()


def test_rollout_buffer() -> None:
    buffer = IPPORolloutBuffer(gamma=0.99, gae_lambda=0.95, device="cpu")
    for _ in range(5):
        buffer.add_timestep(
            obs=torch.randn(3, 10),
            actions=torch.randn(3, 2).clamp(-1, 1),
            rewards=torch.randn(3),
            dones=torch.zeros(3),
            log_probs=torch.randn(3),
            values=torch.randn(3),
        )
    buffer.compute_advantages(last_values=torch.randn(3))
    assert buffer.advantages.shape == (5, 3)
    assert buffer.returns.shape == (5, 3)
    saw_batch = False
    for obs, actions, log_probs, advantages, returns in buffer.get_batches(2):
        saw_batch = True
        assert obs.shape[1:] == (3, 10)
        assert actions.shape[1:] == (3, 2)
        assert log_probs.shape[1:] == (3,)
        assert advantages.shape[1:] == (3,)
        assert returns.shape[1:] == (3,)
    assert saw_batch
    buffer.clear()
    assert buffer.advantages is None


def test_trainer() -> None:
    adapter = FakeAdapter(num_envs=1, done_every=3)
    trainer = IPPOTrainer(
        adapter=adapter,
        hidden_dim=16,
        lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device="cpu",
    )
    last_obs, rollout_metrics = trainer.collect_rollouts(num_steps=4)
    update_metrics = trainer.update(last_obs, num_actor_epochs=1, num_critic_epochs=1, B=2)
    assert last_obs.shape == (adapter.n_defenders, adapter.obs_dim)
    assert "mean_episode_rewards" in rollout_metrics
    assert "policy_loss" in update_metrics
    assert "explained_var" in update_metrics


def main() -> int:
    tests = [
        ("imports", test_imports),
        ("run syntax", test_run_syntax),
        ("policy", test_policy),
        ("critic", test_critic),
        ("rollout buffer", test_rollout_buffer),
        ("trainer", test_trainer),
    ]
    failures = []

    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:
            failures.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()

    if failures:
        print("\nBaseline smoke tests failed:")
        for name in failures:
            print(f"- {name}")
        return 1

    print("\nAll baseline smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
