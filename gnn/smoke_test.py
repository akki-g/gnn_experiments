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

from gnn.action import ActionHead
from gnn.batch_rollout import GNNRolloutBuffer
from gnn.comm import CommPolicy
from gnn.critic import CriticNetwork
from gnn.graph_conv import GraphConv
from gnn.obs_encode import ObservationEncoder
from gnn.trainer import GNNTrainer
from gnn.trainer_multienv import GNNTrainerMultiEnv


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

    def _make_obs_and_pos(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs = torch.randn(
            self.num_envs,
            self.n_defenders,
            self.obs_dim,
            dtype=torch.float32,
            device=self.device,
        )
        positions = torch.randn(
            self.num_envs,
            self.n_defenders,
            2,
            dtype=torch.float32,
            device=self.device,
        )
        obs[:, :, 2:4] = positions
        return obs, positions

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._step_count = 0
        return self._make_obs_and_pos()

    def reset_env(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._make_obs_and_pos()

    def build_adj(self, positions: torch.Tensor, r_comm: float) -> torch.Tensor:
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist = torch.linalg.vector_norm(diff, dim=-1)
        adj = (dist <= max(r_comm, 1e-3)).float()
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return adj / deg

    def step(
        self, defender_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        self._step_count += 1
        obs, positions = self._make_obs_and_pos()
        rewards = -defender_actions.pow(2).sum(dim=-1)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.done_every > 0 and self._step_count % self.done_every == 0:
            dones[:] = True
        info = {
            "n_tagged": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "n_breached": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
        }
        return obs, rewards, dones, info, positions


def _assert_finite(tensor: torch.Tensor, msg: str) -> None:
    assert torch.isfinite(tensor).all(), msg


def test_imports() -> None:
    module_names = [
        "gnn.action",
        "gnn.batch_rollout",
        "gnn.comm",
        "gnn.critic",
        "gnn.graph_conv",
        "gnn.obs_encode",
        "gnn.trainer",
        "gnn.trainer_multienv",
    ]
    for module_name in module_names:
        importlib.import_module(module_name)


def test_run_py_syntax_only() -> None:
    run_path = Path(__file__).with_name("run.py")
    ast.parse(run_path.read_text(encoding="utf-8"), filename=str(run_path))


def test_action_head() -> None:
    model = ActionHead(input_dim=8, hidden_dim=16, output_dim=2)
    x = torch.randn(3, 4, 8)
    out = model(x)
    assert out.shape == (3, 4, 2)
    _assert_finite(out, "ActionHead output contains non-finite values")


def test_observation_encoder() -> None:
    model = ObservationEncoder(input_dim=10, hidden_dim=32, output_dim=8)
    x = torch.randn(5, 10)
    out = model(x)
    assert out.shape == (5, 8)
    _assert_finite(out, "ObservationEncoder output contains non-finite values")


def test_graph_conv() -> None:
    model = GraphConv(F_in=8, G_out=6, K=3)
    x = torch.randn(2, 5, 8)
    s = torch.eye(5).unsqueeze(0).repeat(2, 1, 1)
    out = model(x, s)
    assert out.shape == (2, 5, 6)
    _assert_finite(out, "GraphConv output contains non-finite values")


def test_comm_policy() -> None:
    policy = CommPolicy(obs_dim=10, hidden_dim=16, action_dim=2, F_feat=8, G_feat=8, K=2)
    obs = torch.randn(2, 4, 10)
    s = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)

    mean = policy(obs, s)
    actions, log_prob, entropy = policy.get_actions(obs, s)
    eval_log_prob, eval_entropy, eval_mean = policy.evaluate_actions(obs, s, actions)

    assert mean.shape == (2, 4, 2)
    assert actions.shape == (2, 4, 2)
    assert log_prob.shape == (2, 4)
    assert entropy.shape == (2, 4)
    assert eval_log_prob.shape == (2, 4)
    assert eval_entropy.shape == (2, 4)
    assert eval_mean.shape == (2, 4, 2)
    assert (actions.abs() <= 1.0 + 1e-6).all()


def test_critic_network() -> None:
    critic = CriticNetwork(obs_dim=10, hidden_dim=32, device="cpu")
    obs = torch.randn(7, 10)
    values = critic(obs)
    assert values.shape == (7, 1)
    _assert_finite(values, "Critic output contains non-finite values")


def test_rollout_buffer() -> None:
    torch.manual_seed(0)
    buffer = GNNRolloutBuffer(gamma=0.99, gae_lambda=0.95, device="cpu")
    t, e, n, obs_dim, act_dim = 4, 2, 3, 10, 2
    for _ in range(t):
        buffer.add_timestep(
            obs=torch.randn(e, n, obs_dim),
            actions=torch.randn(e, n, act_dim).clamp(-1.0, 1.0),
            rewards=torch.randn(e, n),
            dones=torch.zeros(e, n),
            log_probs=torch.randn(e, n),
            values=torch.randn(e, n),
            A=torch.eye(n).unsqueeze(0).repeat(e, 1, 1),
        )

    buffer.compute_advantages(last_values=torch.randn(e, n))
    assert buffer.advantages is not None
    assert buffer.returns is not None
    assert buffer.advantages.shape == (t, e, n)
    assert buffer.returns.shape == (t, e, n)

    saw_batch = False
    for obs, actions, log_probs, adv, ret, adj in buffer.get_batches(B=2):
        saw_batch = True
        assert obs.shape[1:] == (n, obs_dim)
        assert actions.shape[1:] == (n, act_dim)
        assert log_probs.shape[1:] == (n,)
        assert adv.shape[1:] == (n,)
        assert ret.shape[1:] == (n,)
        assert adj.shape[1:] == (n, n)
    assert saw_batch

    buffer.clear()
    assert len(buffer.obs) == 0
    assert buffer.advantages is None
    assert buffer.returns is None


def test_trainer_single_env() -> None:
    adapter = FakeAdapter(num_envs=1, done_every=3)
    trainer = GNNTrainer(
        adapter=adapter,
        hidden_dim=16,
        F_feat=8,
        G_feat=8,
        K=2,
        lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device="cpu",
        r_comm=1.0,
    )

    last_obs, rollout_metrics = trainer.collect_rollouts(num_steps=4)
    update_metrics = trainer.update(last_obs, num_actor_epochs=1, num_critic_epochs=1, B=2)

    assert last_obs.shape == (adapter.n_defenders, adapter.obs_dim)
    assert "mean_episode_rewards" in rollout_metrics
    assert "policy_loss" in update_metrics
    assert "value_loss" in update_metrics


def test_trainer_multi_env() -> None:
    adapter = FakeAdapter(num_envs=2, done_every=3)
    trainer = GNNTrainerMultiEnv(
        adapter=adapter,
        hidden_dim=16,
        F_feat=8,
        G_feat=8,
        K=2,
        lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device="cpu",
        r_comm=1.0,
        total_timesteps=24,
        rollout_length=4,
    )

    last_obs, rollout_metrics = trainer.collect_rollouts(num_steps=4)
    update_metrics = trainer.update(last_obs, num_actor_epochs=1, num_critic_epochs=1, B=2)

    assert last_obs.shape == (adapter.num_envs, adapter.n_defenders, adapter.obs_dim)
    assert "mean_episode_rewards" in rollout_metrics
    assert "policy_loss" in update_metrics
    assert "value_loss" in update_metrics
    assert "explained_var" in update_metrics


def main() -> int:
    torch.manual_seed(0)

    tests = [
        ("imports", test_imports),
        ("run.py syntax", test_run_py_syntax_only),
        ("action", test_action_head),
        ("obs_encode", test_observation_encoder),
        ("graph_conv", test_graph_conv),
        ("comm", test_comm_policy),
        ("critic", test_critic_network),
        ("batch_rollout", test_rollout_buffer),
        ("trainer", test_trainer_single_env),
        ("trainer_multienv", test_trainer_multi_env),
    ]

    failures: list[str] = []
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {name}")
        except Exception:
            failures.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()

    if failures:
        print("\nSmoke tests failed:")
        for name in failures:
            print(f"- {name}")
        return 1

    print("\nAll gnn smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
