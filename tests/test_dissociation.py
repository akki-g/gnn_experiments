"""
Smoke tests for the Phase 3 communication-necessity dissociation work order.

Tests:
    test_probe_extraction_linear_recovery  -- fit_probe_on_features R2 >= 0.98 on linear target
    test_knn_adjacency                     -- build_knn_mask correctness
    test_message_ablation_per_module       -- zero_messages isolates teammate info
    test_isolation_invariant               -- positions never enter obs; mask build does not mutate env
"""

import copy

import pytest
import torch
import torch.nn as nn

from probes.predictability import fit_probe_on_features, train_probe
from experiments.train_vmas_mappo import build_knn_mask
from comm.utils import full_comm_mask
from comm.broadcast import BroadcastComm
from comm.attention import AttentionComm
from comm.graph import GraphComm
from envs.pursuit import PursuitEnv


# ---------------------------------------------------------------------------
# Test 1: probe extraction on linear data
# ---------------------------------------------------------------------------

def test_probe_extraction_linear_recovery():
    """
    fit_probe_on_features recovers target from linearly embedded features (R2 >= 0.98).
    """
    torch.manual_seed(42)
    M = 2000
    D = 8
    target = torch.randn(M, 2)
    W = torch.randn(2, D)
    features = target @ W + 1e-4 * torch.randn(M, D)
    features = features.float()
    target = target.float()

    res = fit_probe_on_features(features, target, seed=0, epochs=400, lr=1e-2)

    assert res["score"] >= 0.98, (
        f"Expected R2 >= 0.98 on linear target, got {res['score']:.4f}"
    )
    # Shape checks
    assert features.shape == (M, D), f"features shape {features.shape}"
    assert features.dtype == torch.float32
    pred = res["model"](features)
    assert pred.shape == (M, 2), f"probe prediction shape {pred.shape}"

    # Control: small PursuitEnv, alpha=1 (full-sight), collect raw obs for class-0 agents
    # and verify probe trains without crashing and achieves >= 0.5 (weak sanity check).
    env = PursuitEnv(
        num_envs=4,
        num_agents=3,
        max_steps=50,
        seed=0,
        alpha=1.0,
        num_full_sight=2,
        world_size=1.0,
        capture_radius=0.1,
    )
    env.reset()
    obs_steps = []
    tgt_steps = []
    for _ in range(50):
        actions = torch.randint(0, 5, (4, 3))
        obs, _, _, _, info = env.step(actions)
        # class-0 agents are indices 0 and 1
        obs_steps.append(obs[:, 0, :].cpu())   # [4, obs_dim]
        tgt_steps.append(info["target_pos"].cpu())  # [4, 2]

    obs_flat = torch.cat(obs_steps, dim=0).float()   # [200, obs_dim]
    tgt_flat = torch.cat(tgt_steps, dim=0).float()   # [200, 2]
    ctrl_res = train_probe(obs_flat, tgt_flat, seed=0, epochs=200, hidden=64)
    assert ctrl_res["score"] >= 0.5, (
        f"Control probe (class-0, alpha=1) expected R2 >= 0.5, got {ctrl_res['score']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: kNN adjacency correctness
# ---------------------------------------------------------------------------

def test_knn_adjacency():
    """
    Verify build_knn_mask for a 1D arrangement of 5 agents.
    x positions: [0, 1, 2, 10, 11].  Agents 3 and 4 are far from 0,1,2.
    """
    positions = torch.zeros(1, 5, 2)
    positions[0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0])

    # k=2, directed (no symmetrize)
    mask_dir = build_knn_mask(positions, k=2, symmetrize="none")  # [1, 5, 5]
    assert mask_dir.shape == (1, 5, 5), f"shape {mask_dir.shape}"
    assert mask_dir.dtype == torch.bool

    # Agent 0's 2 nearest are agents 1 (dist=1) and 2 (dist=2)
    row0 = mask_dir[0, 0]   # [5] -- which senders agent0 listens to
    assert row0[1].item() and row0[2].item(), "agent0 should have agents 1,2 as kNN"
    assert not row0[0].item(), "agent0 diagonal must be False"
    assert not row0[3].item() and not row0[4].item(), \
        "agent0 should NOT include far agents 3,4 (k=2)"
    assert row0.sum().item() == 2, f"agent0 kNN row sum should be 2, got {row0.sum().item()}"

    # Diagonal must be all False
    for i in range(5):
        assert not mask_dir[0, i, i].item(), f"diagonal [{i},{i}] must be False"

    # symmetrize="or"
    mask_or = build_knn_mask(positions, k=2, symmetrize="or")
    # symmetry check: mask[i,j] == mask[j,i]
    assert torch.equal(mask_or[0], mask_or[0].T), "or-symmetrized mask must be symmetric"
    # degree >= 2 for all nodes
    degree = mask_or[0].long().sum(dim=1)
    assert (degree >= 2).all(), f"all agents should have degree >= 2, got {degree}"

    # deterministic: two calls produce identical masks
    mask_a = build_knn_mask(positions, k=2, symmetrize="or")
    mask_b = build_knn_mask(positions, k=2, symmetrize="or")
    assert torch.equal(mask_a, mask_b), "build_knn_mask must be deterministic"

    # agents 3 and 4 should be mutual neighbors (they are each other's only nearby agent)
    assert mask_or[0, 3, 4].item() and mask_or[0, 4, 3].item(), \
        "agents 3 and 4 should be mutual kNN neighbors"

    # agent 0's kNN (undirected) are agents 1 and 2
    assert mask_or[0, 0, 1].item() and mask_or[0, 0, 2].item(), \
        "agent 0 undirected kNN should include agents 1 and 2"


# ---------------------------------------------------------------------------
# Test 5: zero_messages ablation per module
# ---------------------------------------------------------------------------

def _make_full_mask(B: int, N: int) -> torch.Tensor:
    """Return [B, N, N] full off-diagonal comm mask (True for all i!=j)."""
    base = full_comm_mask(N, device=torch.device("cpu"))  # [N, N]
    return base.unsqueeze(0).expand(B, -1, -1).clone()


@pytest.mark.parametrize("module_name", ["broadcast", "attention", "graph"])
def test_message_ablation_per_module(module_name: str):
    """
    zero_messages=True zeroes teammate message contribution:
        - Perturbing a teammate's embedding does NOT change target agent output.
    zero_messages=False (default):
        - Perturbing a teammate DOES change the target agent output.

    For BOTH flags:
        - Perturbing own embedding (agent0 self) changes agent0 output (residual/own-path).
    """
    B, N, D = 4, 3, 16
    num_heads = 4 if module_name == "graph" else 1

    def build_module(zero: bool) -> nn.Module:
        torch.manual_seed(7)
        if module_name == "broadcast":
            return BroadcastComm(D, rounds=1, zero_messages=zero)
        elif module_name == "attention":
            return AttentionComm(D, num_heads=num_heads, rounds=1, zero_messages=zero)
        else:
            return GraphComm(D, num_heads=num_heads, num_layers=2, zero_messages=zero)

    mod_zero  = build_module(zero=True)
    # Rebuild with same seed to get identical initial weights
    mod_norm  = build_module(zero=False)

    # Copy weights from mod_zero to mod_norm so both have identical weights
    mod_norm.load_state_dict(mod_zero.state_dict())

    torch.manual_seed(0)
    h_base = torch.randn(B, N, D)
    mask = _make_full_mask(B, N)

    # Evaluate both modules on base input
    with torch.no_grad():
        out_zero_base  = mod_zero(h_base, mask)   # [B, N, D]
        out_norm_base  = mod_norm(h_base, mask)   # [B, N, D]

    # Perturb teammate (agent1) -- target is agent0
    h_pert_teammate = h_base.clone()
    h_pert_teammate[:, 1, :] += 1.0

    with torch.no_grad():
        out_zero_pert_tm = mod_zero(h_pert_teammate, mask)
        out_norm_pert_tm = mod_norm(h_pert_teammate, mask)

    # zero=True: agent0 output must be unchanged when only agent1 changes
    assert torch.allclose(out_zero_base[:, 0, :], out_zero_pert_tm[:, 0, :], atol=1e-6), (
        f"[{module_name}] zero_messages=True: agent0 output should not change when agent1 is perturbed"
    )
    # zero=False: agent0 output MUST change when agent1 changes
    max_diff = (out_norm_base[:, 0, :] - out_norm_pert_tm[:, 0, :]).abs().max().item()
    assert max_diff > 1e-4, (
        f"[{module_name}] zero_messages=False: agent0 output should change when agent1 is perturbed "
        f"(max_diff={max_diff:.6f})"
    )

    # Perturb own embedding (agent0 self) -- both modes must change agent0 output
    h_pert_self = h_base.clone()
    h_pert_self[:, 0, :] += 1.0

    with torch.no_grad():
        out_zero_pert_self = mod_zero(h_pert_self, mask)
        out_norm_pert_self = mod_norm(h_pert_self, mask)

    diff_zero_self = (out_zero_base[:, 0, :] - out_zero_pert_self[:, 0, :]).abs().max().item()
    diff_norm_self = (out_norm_base[:, 0, :] - out_norm_pert_self[:, 0, :]).abs().max().item()

    assert diff_zero_self > 1e-4, (
        f"[{module_name}] zero_messages=True: agent0 output should change when its own h is perturbed "
        f"(own-path/residual must be preserved). max_diff={diff_zero_self:.6f}"
    )
    assert diff_norm_self > 1e-4, (
        f"[{module_name}] zero_messages=False: agent0 output should change when its own h is perturbed. "
        f"max_diff={diff_norm_self:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 6: isolation invariant
# ---------------------------------------------------------------------------

def test_isolation_invariant():
    """
    build_knn_mask consumes positions and returns [B, N, N] bool mask.
    Building the mask:
        - does NOT mutate pursuer positions (positions equal before/after)
    Two envs with different capture_k / knn_k but identical seed have identical obs.
    """
    env = PursuitEnv(
        num_envs=2,
        num_agents=4,
        max_steps=5,
        seed=0,
        observe_teammates=False,
        alpha=0.0,
        num_full_sight=2,
        world_size=1.0,
        capture_radius=0.1,
    )
    obs, _ = env.reset()
    assert obs.shape == (2, 4, env.obs_dim), f"obs shape {obs.shape}"

    actions = torch.zeros(2, 4, dtype=torch.long)
    env.step(actions)

    # Use env.pursuer_pos directly (PursuitEnv attribute; PursuitAdapter wraps it
    # as agent_positions but PursuitEnv does not have that property)
    positions_before = env.pursuer_pos.clone()

    # build_knn_mask accepts raw positions [B, N, 2]
    mask = build_knn_mask(positions_before, k=4)
    assert mask.shape == (2, 4, 4), f"mask shape {mask.shape}"
    assert mask.dtype == torch.bool

    # Positions must be unchanged after building the mask
    positions_after = env.pursuer_pos.clone()
    assert torch.equal(positions_before, positions_after), \
        "build_knn_mask must not mutate env.pursuer_pos"

    # Two envs with different capture_k/knn_k but same seed should have same obs_dim
    # and byte-identical obs at reset (obs not influenced by capture_k or knn settings)
    env_a = PursuitEnv(
        num_envs=2, num_agents=4, max_steps=5, seed=0,
        observe_teammates=False, alpha=0.0, num_full_sight=2,
        world_size=1.0, capture_radius=0.1, capture_k=1,
    )
    env_b = PursuitEnv(
        num_envs=2, num_agents=4, max_steps=5, seed=0,
        observe_teammates=False, alpha=0.0, num_full_sight=2,
        world_size=1.0, capture_radius=0.1, capture_k=4,
    )
    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()

    assert env_a.obs_dim == env_b.obs_dim, \
        f"obs_dim must be identical regardless of capture_k: {env_a.obs_dim} vs {env_b.obs_dim}"
    assert torch.equal(obs_a, obs_b), \
        "obs must be identical for same seed regardless of capture_k"
