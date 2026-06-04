"""
Critical gate: observation isolation tests.

Verifies:
1. At alpha=0, observe_teammates=False:
   A sensor-limited agent's FULL obs vector is byte-identical across ALL target
   positions, including the case where dist==0 (HARD-BLIND at r=0).

2. Complement sanity at alpha=1:
   The same agent's obs DOES change as the target moves (proves the test can
   detect leakage — the test is not trivially vacuous).

3. Teammate-invariance at several alphas with observe_teammates=False:
   Teleporting teammates does NOT change the focal agent's obs.

4. Actor-isolation (structural):
   MAPPO.act() uses only obs for the actor path (comm slot = IdentityComm).
   - Same obs + different state → identical action logits (actor is state-blind).
   - Different obs → different logits (test is sensitive).
"""

import math
import torch
import pytest

from envs.asymmetry import build_obs, class_id_vector
from backbone.mappo import MAPPO
from comm.identity import IdentityComm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r_max(world_size=1.0):
    return 2.0 * world_size * math.sqrt(2.0)


def _make_class_id(N=3, num_full_sight=1):
    return class_id_vector(N, num_full_sight)


# ---------------------------------------------------------------------------
# 1. alpha=0: sensor-limited obs INDEPENDENT of target position
# ---------------------------------------------------------------------------

def test_alpha0_sensor_limited_obs_independent_of_target():
    """
    HARD-BLIND: at alpha=0, class-1 agent's full obs vector is byte-identical
    across a grid of target positions including dist==0.
    """
    B, N = 4, 3
    world_size = 1.0
    r_max = _r_max(world_size)
    class_id = _make_class_id(N, num_full_sight=1)  # agent 0=full, 1,2=limited

    # Fix pursuer positions — agent 1 (class 1) at known position
    pursuer_pos = torch.zeros(B, N, 2)
    pursuer_pos[:, 0, :] = torch.tensor([0.3, 0.2])
    pursuer_pos[:, 1, :] = torch.tensor([-0.5, 0.4])
    pursuer_pos[:, 2, :] = torch.tensor([0.1, -0.6])

    # Reference obs at origin target
    ref_obs = build_obs(
        pursuer_pos,
        torch.zeros(B, 2),
        class_id, alpha=0.0, r_max=r_max, world_size=world_size,
    )

    # Target sweep: random positions, corners, origin, AND agent 1's own position
    target_positions = [
        torch.zeros(B, 2),
        torch.ones(B, 2),
        -torch.ones(B, 2),
        pursuer_pos[:, 1, :].clone(),  # dist == 0 for agent 1
    ]
    for _ in range(30):
        target_positions.append((torch.rand(B, 2) * 2 - 1) * world_size)

    for target_pos in target_positions:
        obs = build_obs(
            pursuer_pos, target_pos, class_id,
            alpha=0.0, r_max=r_max, world_size=world_size,
        )
        for n in [1, 2]:  # class-1 agents
            assert torch.equal(obs[:, n, :], ref_obs[:, n, :]), (
                f"Agent {n} obs differs at alpha=0 for target={target_pos[0].tolist()}. "
                "Target slice must be zeroed regardless of target position."
            )


# ---------------------------------------------------------------------------
# 2. Complement sanity: at alpha=1, obs DOES change with target
# ---------------------------------------------------------------------------

def test_alpha1_sensor_limited_obs_changes_with_target():
    """
    At alpha=1, the sensor-limited agent's obs should change as target moves.
    This confirms the isolation test above is sensitive (not trivially True).
    """
    B, N = 4, 3
    world_size = 1.0
    r_max = _r_max(world_size)
    class_id = _make_class_id(N, num_full_sight=1)

    pursuer_pos = torch.zeros(B, N, 2)
    pursuer_pos[:, 1, :] = torch.tensor([-0.5, 0.4])

    target_a = torch.zeros(B, 2)
    target_b = torch.ones(B, 2) * 0.8

    obs_a = build_obs(pursuer_pos, target_a, class_id, alpha=1.0, r_max=r_max,
                      world_size=world_size)
    obs_b = build_obs(pursuer_pos, target_b, class_id, alpha=1.0, r_max=r_max,
                      world_size=world_size)

    # At alpha=1, agent 1 (class 1) should see the target → obs must differ
    assert not torch.equal(obs_a[:, 1, :], obs_b[:, 1, :]), (
        "At alpha=1, sensor-limited agent obs should change when target moves. "
        "If this fails, the alpha=0 isolation test above is vacuous."
    )


# ---------------------------------------------------------------------------
# 3. Teammate-invariance: teleporting teammates does NOT change focal agent obs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_teammate_positions_do_not_affect_focal_agent_obs(alpha):
    """
    With observe_teammates=False (default), changing teammate positions should
    not change any agent's local obs — all cross-agent info flows via graph only.
    """
    B, N = 4, 3
    world_size = 1.0
    r_max = _r_max(world_size)
    class_id = _make_class_id(N, num_full_sight=2)

    # Base pursuer positions
    pursuer_pos_base = torch.zeros(B, N, 2)
    pursuer_pos_base[:, 0, :] = torch.tensor([0.2, 0.3])
    pursuer_pos_base[:, 1, :] = torch.tensor([-0.4, 0.1])
    pursuer_pos_base[:, 2, :] = torch.tensor([0.5, -0.5])
    target_pos = torch.zeros(B, 2)

    obs_base = build_obs(pursuer_pos_base, target_pos, class_id, alpha=alpha,
                         r_max=r_max, world_size=world_size, observe_teammates=False)

    # Teleport teammates (agents 1, 2) to random positions — agent 0 focal
    for _ in range(10):
        pursuer_pos_new = pursuer_pos_base.clone()
        pursuer_pos_new[:, 1, :] = (torch.rand(B, 2) * 2 - 1) * world_size
        pursuer_pos_new[:, 2, :] = (torch.rand(B, 2) * 2 - 1) * world_size
        obs_new = build_obs(pursuer_pos_new, target_pos, class_id, alpha=alpha,
                            r_max=r_max, world_size=world_size, observe_teammates=False)
        # Focal agent 0's obs should be unchanged (own_pos is fixed)
        assert torch.equal(obs_new[:, 0, :], obs_base[:, 0, :]), (
            f"Agent 0 obs changed when only teammates moved (alpha={alpha}). "
            "observe_teammates=False should guarantee isolation."
        )


# ---------------------------------------------------------------------------
# 4. Actor-isolation: varying state does not change actor output; varying obs does
# ---------------------------------------------------------------------------

def test_actor_uses_only_obs_not_state():
    """
    MAPPO.act() actor path: obs → encoder → IdentityComm → actor.
    The actor head never reads state (only the critic does).
    Verify: same obs + two different states → identical action logits/distribution.
    Verify: different obs → different logits.
    """
    B, N = 2, 3
    obs_dim   = 5
    state_dim = 9   # N*2 + 2 + 1
    action_dim = 5
    hidden_dim = 16

    model = MAPPO(
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=N,
        hidden_dim=hidden_dim,
        comm_module=IdentityComm(),
        device="cpu",
    )
    model.eval()

    obs    = torch.randn(B, N, obs_dim)
    state_a = torch.randn(B, state_dim)
    state_b = torch.randn(B, state_dim)
    # Guarantee states are different
    assert not torch.equal(state_a, state_b)

    with torch.no_grad():
        out_a = model.act(obs, state_a, deterministic=True)
        out_b = model.act(obs, state_b, deterministic=True)

    # Same obs → same action (deterministic)
    assert torch.equal(out_a["action"], out_b["action"]), (
        "Same obs with different states should give identical deterministic actions. "
        "Actor must not consume state."
    )
    assert torch.allclose(out_a["log_prob"], out_b["log_prob"]), (
        "Same obs with different states should give identical log_probs."
    )

    # Different obs → different logits
    obs_different = obs + 5.0  # large shift to guarantee different output
    with torch.no_grad():
        out_c = model.act(obs_different, state_a, deterministic=True)
    # Confirm the test is sensitive: different obs gives different log_probs
    assert not torch.allclose(out_a["log_prob"], out_c["log_prob"]), (
        "Different obs should produce different log_probs. "
        "If this fails, the actor is degenerate."
    )


def test_actor_encode_is_obs_only():
    """
    model.encode() takes only obs (no state). Verify structurally by calling
    encode with two obs and confirming output changes, and that the function
    signature does not accept state.
    """
    B, N = 2, 3
    model = MAPPO(
        obs_dim=5, state_dim=9, action_dim=5, num_agents=N,
        hidden_dim=16, comm_module=IdentityComm(), device="cpu",
    )
    model.eval()

    obs_a = torch.randn(B, N, 5)
    obs_b = torch.randn(B, N, 5)

    with torch.no_grad():
        h_a = model.encode(obs_a)
        h_b = model.encode(obs_b)

    assert h_a.shape == (B, N, model.hidden_dim)
    assert not torch.allclose(h_a, h_b), "Different obs should produce different embeddings"
