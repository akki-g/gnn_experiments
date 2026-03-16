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


def test_syntax() -> None:
    src = Path(__file__).with_name("guarded_territory.py").read_text(encoding="utf-8")
    ast.parse(src, filename=str(Path(__file__).with_name("guarded_territory.py")))


def test_import() -> None:
    importlib.import_module("environments.guarded_territory")


def test_adapter_runtime() -> None:
    guarded = importlib.import_module("environments.guarded_territory")
    GuardedTerritoryAdapter = guarded.GuardedTerritoryAdapter

    adapter = GuardedTerritoryAdapter(
        num_envs=1,
        device="cpu",
        n_scouts=2,
        n_interceptors=1,
        n_intruders=1,
        n_zones=1,
        max_steps=8,
        world_size=1.5,
        scout_fov=0.8,
        interceptor_fov=0.6,
        intruder_speed=0.25,
        defender_speed=0.7,
        tag_radius=0.12,
        intruder_skill=0.0,
    )

    obs, pos = adapter.reset()
    assert obs.shape == (1, adapter.n_defenders, adapter.obs_dim)
    assert pos.shape == (1, adapter.n_defenders, 2)
    assert torch.isfinite(obs).all()
    assert torch.isfinite(pos).all()

    adj = adapter.build_adj(pos, r_comm=1.0)
    assert adj.shape == (1, adapter.n_defenders, adapter.n_defenders)
    row_sums = adj.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    actions = torch.zeros(1, adapter.n_defenders, adapter.action_dim, dtype=torch.float32)
    next_obs, rewards, dones, info, next_pos = adapter.step(actions)
    assert next_obs.shape == (1, adapter.n_defenders, adapter.obs_dim)
    assert rewards.shape == (1, adapter.n_defenders)
    assert dones.shape == (1,)
    assert next_pos.shape == (1, adapter.n_defenders, 2)
    assert isinstance(info, dict)

    reset_obs, reset_pos = adapter.reset_env()
    assert reset_obs.shape == (1, adapter.n_defenders, adapter.obs_dim)
    assert reset_pos.shape == (1, adapter.n_defenders, 2)


def main() -> int:
    tests = [
        ("syntax", test_syntax),
        ("import", test_import),
        ("adapter runtime", test_adapter_runtime),
    ]
    failures: list[str] = []

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {name}")
        except ModuleNotFoundError as exc:
            if exc.name == "vmas":
                print("[FAIL] vmas dependency is missing. Install vmas to run env smoke tests.")
                return 2
            failures.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()
        except Exception:
            failures.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()

    if failures:
        print("\nEnvironment smoke tests failed:")
        for name in failures:
            print(f"- {name}")
        return 1

    print("\nAll environment smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
