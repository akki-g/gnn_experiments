"""
Observation asymmetry utilities for the pursuit environment.

These are pure functions with no environment construction — testable in isolation.

Alpha semantics
---------------
alpha in [0, 1] controls how far sensor-limited agents (class 1) can see the target:
    r = alpha * r_max
    alpha=1 → r = r_max  → sensor-limited sees as far as full-sight (symmetric)
    alpha=0 → r = 0      → sensor-limited is completely blind to the target

Full-sight agents (class 0) ALWAYS use r = r_max regardless of alpha.

Observation layout (per pursuer, obs_dim fixed across both classes)
--------------------------------------------------------------------
Without observe_teammates (default, obs_dim=5):
    [0,1]  own_pos       (x, y)                       — never masked
    [2,3]  target_rel    (tx-ox, ty-oy) * visible     — zeroed when not visible
    [4]    target_visible (0.0 or 1.0)

With observe_teammates (obs_dim = 5 + 2*(N-1)):
    [0,1]  own_pos
    [2,3]  target_rel * visible
    [4]    target_visible
    [5+2k, 5+2k+1]  teammate k relative position (k=0..N-2, skipping self)

The target slice stays at fixed indices [2,3,4] regardless of observe_teammates.
Teammate fields carry NO masking (full observation isolation: all cross-agent
information flows through the GNN graph edges only — never in local obs).

HARD-BLIND short-circuit
-------------------------
If a class's radius r == 0.0 (alpha=0 for class 1), visible is unconditionally
False even when dist == 0. This makes alpha=0 ⟹ target slice always zeros for
class-1 agents, making their observations INDEPENDENT of target position.
"""

from typing import Optional

import torch
from torch import Tensor


def alpha_to_visibility_radius(alpha: float, r_max: float) -> float:
    """
    Convert alpha ∈ [0, 1] to a visibility radius for sensor-limited agents.

    Parameters
    ----------
    alpha : float in [0, 1]. Controls sensor-limited agent sight range.
    r_max : float. Maximum possible distance in the world (diameter).

    Returns
    -------
    float. Visibility radius = alpha * r_max.
        alpha=0 → 0.0  (blind)
        alpha=1 → r_max (full-sight, same as class 0)
    """
    assert 0.0 <= alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"
    return float(alpha) * float(r_max)


def class_id_vector(num_agents: int, num_full_sight: int) -> Tensor:
    """
    Build a [N] long tensor of agent class IDs.

    Class 0 = full-sight: the first `num_full_sight` agents.
    Class 1 = sensor-limited: the remaining agents.

    Parameters
    ----------
    num_agents     : total number of agents N.
    num_full_sight : how many agents are full-sight (class 0).

    Returns
    -------
    class_id : LongTensor of shape [N].
    """
    assert 0 <= num_full_sight <= num_agents, (
        f"num_full_sight={num_full_sight} must be in [0, {num_agents}]"
    )
    ids = torch.ones(num_agents, dtype=torch.long)  # default class 1
    ids[:num_full_sight] = 0  # class 0 = full-sight
    return ids


def build_obs(
    pursuer_pos: Tensor,
    target_pos: Tensor,
    class_id: Tensor,
    alpha: float,
    r_max: float,
    world_size: float,
    observe_teammates: bool = False,
) -> Tensor:
    """
    Build per-agent observations with visibility masking applied.

    This is the SOLE masking point. PursuitEnv.reset/step call this function.
    Masking NEVER happens in the trainer.

    Parameters
    ----------
    pursuer_pos     : [B, N, 2]  pursuer positions in [-world_size, world_size]^2.
    target_pos      : [B, 2]     target position (ground truth — masked in obs).
    class_id        : [N] long   0 = full-sight, 1 = sensor-limited.
    alpha           : float ∈ [0,1]. Controls sensor-limited visibility radius.
    r_max           : float. Maximum world diameter; full-sight radius.
    world_size      : float. World half-extent (kept for potential future use).
    observe_teammates: bool. If True, append teammate relative positions after index 4.
                      Default False = NO teammate fields in obs (isolation guarantee).

    Returns
    -------
    obs : [B, N, obs_dim]  where obs_dim = 5 (or 5 + 2*(N-1) if observe_teammates).

    Visibility rule (HARD-BLIND at r==0):
        full-sight  (class 0): r = r_max     → always visible (r=r_max >= any dist)
        sensor-limited(class 1): r = alpha*r_max
            if r == 0.0: unconditionally blind (visible = False, even at dist=0)
            else:        visible = (dist <= r)
    """
    B, N, _ = pursuer_pos.shape
    assert target_pos.shape == (B, 2), (
        f"target_pos shape {target_pos.shape} != ({B}, 2)"
    )
    assert class_id.shape == (N,), (
        f"class_id shape {class_id.shape} != ({N},)"
    )

    # target relative to each pursuer: [B, N, 2]
    target_expanded = target_pos.unsqueeze(1).expand(B, N, 2)  # [B, N, 2]
    target_rel = target_expanded - pursuer_pos                  # [B, N, 2]

    # Euclidean distance to target: [B, N]
    dist = torch.linalg.vector_norm(target_rel, dim=-1)        # [B, N]

    # Per-class visibility radii
    r_full = float(r_max)                            # class 0: always r_max
    r_limited = alpha_to_visibility_radius(alpha, r_max)  # class 1: alpha * r_max

    # Visibility mask [B, N] bool
    # Using a single comparison op for both classes (deterministic boundary).
    # HARD-BLIND: if r_limited == 0.0, class-1 agents are ALWAYS blind.
    visible = torch.zeros(B, N, dtype=torch.bool, device=pursuer_pos.device)

    full_sight_mask = (class_id == 0)   # [N] bool
    limited_mask    = (class_id == 1)   # [N] bool

    if full_sight_mask.any():
        # Full-sight: r = r_max (>= any possible distance in world)
        # Use r_full comparison — always True for in-world positions
        visible[:, full_sight_mask] = (dist[:, full_sight_mask] <= r_full)

    if limited_mask.any():
        if r_limited == 0.0:
            # HARD-BLIND: unconditionally False (stays zeros from init)
            pass
        else:
            visible[:, limited_mask] = (dist[:, limited_mask] <= r_limited)

    # Apply mask to target_rel: zero out non-visible entries
    vis_float = visible.float().unsqueeze(-1)   # [B, N, 1]
    target_rel_masked = target_rel * vis_float  # [B, N, 2]
    target_visible = visible.float()            # [B, N]

    # Build observation tensor [B, N, obs_dim]
    # Indices [0,1]: own_pos (never masked)
    # Indices [2,3]: target_rel_masked
    # Index  [4]:   target_visible
    obs_parts = [
        pursuer_pos,                          # [B, N, 2]  — own pos
        target_rel_masked,                    # [B, N, 2]  — masked target relative
        target_visible.unsqueeze(-1),         # [B, N, 1]  — visibility flag
    ]

    if observe_teammates:
        # Build [B, N, 2*(N-1)] teammate relative positions
        # For agent j: concat over k!=j of (pursuer_pos[:,k,:] - pursuer_pos[:,j,:])
        tm_rel_list = []
        for k in range(N):
            rel_k = pursuer_pos[:, k:k+1, :].expand(B, N, 2) - pursuer_pos  # [B, N, 2]
            # Zero out self-distance (diagonal)
            diag = torch.arange(N, device=pursuer_pos.device)
            rel_k[:, diag == k, :] = 0.0
            tm_rel_list.append(rel_k)

        # Stack and reshape: remove self entry for each agent
        # Result: [B, N, (N-1)*2]
        # Build index lists excluding self
        all_rel = torch.stack(tm_rel_list, dim=2)  # [B, N, N, 2]
        # For agent j, collect indices k != j
        teammate_rel = []
        for j in range(N):
            others = [k for k in range(N) if k != j]
            tm_j = all_rel[:, j, others, :]  # [B, N-1, 2]
            tm_j_flat = tm_j.reshape(B, (N-1)*2)  # [B, 2*(N-1)]
            teammate_rel.append(tm_j_flat)
        # Stack: [B, N, 2*(N-1)]
        tm_tensor = torch.stack(teammate_rel, dim=1)  # [B, N, 2*(N-1)]
        obs_parts.append(tm_tensor)

    obs = torch.cat(obs_parts, dim=-1)  # [B, N, obs_dim]

    # Assert obs_dim matches expected
    if observe_teammates:
        expected_obs_dim = 5 + 2 * (N - 1)
    else:
        expected_obs_dim = 5
    assert obs.shape == (B, N, expected_obs_dim), (
        f"obs shape {obs.shape} != ({B}, {N}, {expected_obs_dim})"
    )
    return obs
