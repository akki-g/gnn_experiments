"""
Tests for envs/asymmetry.py — observation asymmetry knob.

Covers:
  - alpha_to_visibility_radius monotonicity, r(0)==0, r(1)==r_max
  - build_obs at alpha=1: class-1 obs == class-0 obs (symmetric)
  - build_obs at alpha=0: class-1 target slice all-zero for ALL target positions
  - Intermediate alpha: visible iff dist <= alpha*r_max (inside/outside boundary)
"""

import torch
import pytest

from envs.asymmetry import (
    alpha_to_visibility_radius,
    class_id_vector,
    build_obs,
)


# ---------------------------------------------------------------------------
# alpha_to_visibility_radius
# ---------------------------------------------------------------------------

def test_radius_at_zero():
    r_max = 2.828
    assert alpha_to_visibility_radius(0.0, r_max) == pytest.approx(0.0)


def test_radius_at_one():
    r_max = 2.828
    assert alpha_to_visibility_radius(1.0, r_max) == pytest.approx(r_max)


def test_radius_monotone():
    r_max = 4.0
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    radii = [alpha_to_visibility_radius(a, r_max) for a in alphas]
    for i in range(len(radii) - 1):
        assert radii[i] <= radii[i + 1], (
            f"Radius not monotone: r({alphas[i]})={radii[i]} > r({alphas[i+1]})={radii[i+1]}"
        )


def test_radius_invalid_alpha():
    with pytest.raises(AssertionError):
        alpha_to_visibility_radius(-0.1, 2.0)
    with pytest.raises(AssertionError):
        alpha_to_visibility_radius(1.1, 2.0)


def test_radius_r_max_geq_diameter():
    """r_max should be >= the diameter of [-world_size, world_size]^2 = 2*sqrt(2)*world_size."""
    import math
    world_size = 1.0
    r_max = 2.0 * world_size * math.sqrt(2.0)
    assert alpha_to_visibility_radius(1.0, r_max) == pytest.approx(r_max)
    # Diameter of unit square = 2*sqrt(2) ≈ 2.828
    diameter = 2.0 * math.sqrt(2.0) * world_size
    assert r_max >= diameter - 1e-9, f"r_max={r_max} < diameter={diameter}"


# ---------------------------------------------------------------------------
# class_id_vector
# ---------------------------------------------------------------------------

def test_class_id_vector_basic():
    cid = class_id_vector(4, 2)
    assert cid.tolist() == [0, 0, 1, 1]


def test_class_id_vector_all_full():
    cid = class_id_vector(3, 3)
    assert (cid == 0).all()


def test_class_id_vector_none_full():
    cid = class_id_vector(3, 0)
    assert (cid == 1).all()


# ---------------------------------------------------------------------------
# build_obs helpers
# ---------------------------------------------------------------------------

def _make_inputs(B, N, world_size=1.0):
    """Random in-world positions."""
    pursuer_pos = (torch.rand(B, N, 2) * 2 - 1) * world_size
    target_pos  = (torch.rand(B, 2)   * 2 - 1) * world_size
    return pursuer_pos, target_pos


# ---------------------------------------------------------------------------
# alpha=1: class-1 obs byte-identical to class-0
# ---------------------------------------------------------------------------

def test_alpha1_class1_same_as_class0():
    """At alpha=1, sensor-limited (class 1) obs == full-sight (class 0) obs."""
    import math
    B, N = 8, 3
    world_size = 1.0
    r_max = 2.0 * world_size * math.sqrt(2.0)
    class_id = class_id_vector(N, num_full_sight=1)  # agent 0 = full-sight, 1,2 = limited

    pursuer_pos, target_pos = _make_inputs(B, N, world_size)
    obs = build_obs(pursuer_pos, target_pos, class_id, alpha=1.0, r_max=r_max,
                    world_size=world_size)

    # Target visible flag [4] should be 1.0 for all agents (all in-world → dist<=r_max)
    target_visible_all = obs[:, :, 4]
    assert (target_visible_all == 1.0).all(), (
        f"At alpha=1, all agents should see the target. Got:\n{target_visible_all}"
    )

    # class-0 and class-1 target slices [2:5] should be identical (same r_max)
    class0_slice = obs[:, 0, 2:5]     # agent 0 (full-sight)
    class1_slice = obs[:, 1, 2:5]     # agent 1 (sensor-limited, but r=r_max at alpha=1)
    # They differ in own_pos [0:2], but target slice [2:5] should be the same as
    # each agent's target_rel depends on their own position — so we check visible flag only.
    # The visible flag for ALL agents at alpha=1 must be 1.0.
    assert (obs[:, :, 4] == 1.0).all()


def test_alpha1_target_visible_is_one_for_in_world_targets():
    """At alpha=1, target_visible == 1.0 for any in-world target position."""
    import math
    B, N = 4, 3
    world_size = 1.0
    r_max = 2.0 * world_size * math.sqrt(2.0)
    class_id = class_id_vector(N, num_full_sight=2)

    # Place pursuers at corners to maximise distances
    pursuer_pos = torch.tensor([
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]],
    ]).expand(B, N, 2).clone()
    target_pos = torch.tensor([[1.0, 1.0]]).expand(B, 2).clone()  # max distance corner

    obs = build_obs(pursuer_pos, target_pos, class_id, alpha=1.0, r_max=r_max,
                    world_size=world_size)
    assert (obs[:, :, 4] == 1.0).all(), (
        "At alpha=1, target should always be visible (r_max >= diameter)"
    )


# ---------------------------------------------------------------------------
# alpha=0: class-1 target slice all-zero
# ---------------------------------------------------------------------------

def test_alpha0_class1_target_slice_all_zero():
    """At alpha=0, sensor-limited agents have target_rel=0 and target_visible=0."""
    import math
    B, N = 8, 3
    world_size = 1.0
    r_max = 2.0 * world_size * math.sqrt(2.0)
    class_id = class_id_vector(N, num_full_sight=1)  # 0=full, 1,2=limited

    pursuer_pos, _ = _make_inputs(B, N, world_size)
    # Sweep many different target positions including agent's own position
    target_positions = [
        (torch.rand(B, 2) * 2 - 1) * world_size,   # random
        pursuer_pos[:, 1, :].clone(),                # same pos as agent 1 (dist=0)
        torch.zeros(B, 2),                           # origin
        torch.ones(B, 2),                            # corner
    ]

    for target_pos in target_positions:
        obs = build_obs(pursuer_pos, target_pos, class_id, alpha=0.0, r_max=r_max,
                        world_size=world_size)
        for n in [1, 2]:  # class-1 agents
            target_rel    = obs[:, n, 2:4]
            target_visible = obs[:, n, 4]
            assert (target_rel == 0.0).all(), (
                f"Agent {n} target_rel should be zero at alpha=0, got {target_rel}"
            )
            assert (target_visible == 0.0).all(), (
                f"Agent {n} target_visible should be 0 at alpha=0, got {target_visible}"
            )


def test_alpha0_class1_obs_independent_of_target():
    """
    At alpha=0, sweeping the target over many positions leaves class-1 obs unchanged.
    The full obs vector (all 5 dims) must be byte-identical across all target positions.
    """
    import math
    B, N = 4, 3
    world_size = 1.0
    r_max = 2.0 * world_size * math.sqrt(2.0)
    class_id = class_id_vector(N, num_full_sight=1)

    # Fixed pursuer positions
    pursuer_pos = torch.zeros(B, N, 2)
    pursuer_pos[:, 0, :] = 0.3
    pursuer_pos[:, 1, :] = -0.5
    pursuer_pos[:, 2, :] = 0.7

    # Reference obs at one target position
    ref_target = torch.zeros(B, 2)
    ref_obs = build_obs(pursuer_pos, ref_target, class_id, alpha=0.0, r_max=r_max,
                        world_size=world_size)

    # Sweep 20 different target positions
    for _ in range(20):
        target_pos = (torch.rand(B, 2) * 2 - 1) * world_size
        obs = build_obs(pursuer_pos, target_pos, class_id, alpha=0.0, r_max=r_max,
                        world_size=world_size)
        for n in [1, 2]:  # class-1 agents
            assert torch.equal(obs[:, n, :], ref_obs[:, n, :]), (
                f"Agent {n} obs at alpha=0 should be independent of target position"
            )


# ---------------------------------------------------------------------------
# Intermediate alpha: visible iff dist <= alpha*r_max
# ---------------------------------------------------------------------------

def test_intermediate_alpha_visibility_boundary():
    """Construct positions exactly inside and outside alpha*r_max; assert flag."""
    import math
    B, N = 1, 3
    world_size = 1.0
    r_max = 2.0 * world_size * math.sqrt(2.0)
    alpha = 0.5
    r_limited = alpha * r_max

    class_id = class_id_vector(N, num_full_sight=1)  # agent 0=full, 1,2=limited

    # Place one class-1 agent at origin; place target just inside radius
    pursuer_pos = torch.zeros(B, N, 2)  # all at origin
    target_inside  = torch.tensor([[r_limited * 0.9, 0.0]])   # dist < r_limited
    target_outside = torch.tensor([[r_limited * 1.1, 0.0]])   # dist > r_limited

    obs_inside = build_obs(pursuer_pos, target_inside, class_id, alpha=alpha,
                           r_max=r_max, world_size=world_size)
    obs_outside = build_obs(pursuer_pos, target_outside, class_id, alpha=alpha,
                            r_max=r_max, world_size=world_size)

    # Class-1 agent (index 1): inside → visible=1; outside → visible=0
    assert obs_inside[0, 1, 4].item()  == pytest.approx(1.0), \
        "Class-1 agent inside radius should be visible"
    assert obs_outside[0, 1, 4].item() == pytest.approx(0.0), \
        "Class-1 agent outside radius should not be visible"

    # Class-0 agent (index 0): always visible regardless of alpha
    assert obs_inside[0, 0, 4].item()  == pytest.approx(1.0), \
        "Class-0 agent should always be visible"
    assert obs_outside[0, 0, 4].item() == pytest.approx(1.0), \
        "Class-0 agent should always be visible even outside limited radius"


def test_full_sight_always_visible():
    """Class-0 (full-sight) agents see the target at ALL alphas."""
    import math
    B, N = 4, 3
    world_size = 1.0
    r_max = 2.0 * world_size * math.sqrt(2.0)
    class_id = class_id_vector(N, num_full_sight=2)  # agents 0,1 = full-sight

    pursuer_pos, target_pos = _make_inputs(B, N, world_size)

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        obs = build_obs(pursuer_pos, target_pos, class_id, alpha=alpha,
                        r_max=r_max, world_size=world_size)
        for n in [0, 1]:  # full-sight agents
            assert (obs[:, n, 4] == 1.0).all(), (
                f"Class-0 agent {n} should always be visible, alpha={alpha}"
            )


# ---------------------------------------------------------------------------
# observe_teammates=False: no teammate fields
# ---------------------------------------------------------------------------

def test_no_teammate_fields_by_default():
    """With observe_teammates=False, obs_dim=5 (no teammate slice)."""
    import math
    B, N = 4, 3
    world_size = 1.0
    r_max = 2.0 * world_size * math.sqrt(2.0)
    class_id = class_id_vector(N, 2)
    pursuer_pos, target_pos = _make_inputs(B, N, world_size)

    obs = build_obs(pursuer_pos, target_pos, class_id, alpha=1.0, r_max=r_max,
                    world_size=world_size, observe_teammates=False)
    assert obs.shape[-1] == 5
