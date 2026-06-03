"""
PPO loss functions.

All input tensors are expected to be [T*B, N] after the minibatch flatten
(researcher decision 1: T and B collapsed, N kept).

active_masks
------------
Optional [T*B, N] float tensor (0=dead agent, 1=alive).  When provided, loss
terms are computed as a masked mean over alive agent steps only.  For Phase 1
(toy env, all agents always alive), active_masks is always all-ones, so the
masked mean equals the regular mean.

Functions
---------
ppo_policy_loss    : clipped surrogate objective → (loss scalar, clip_frac scalar)
value_loss         : clipped value loss
entropy_loss       : mean entropy (returns positive scalar; caller negates if needed)
explained_variance : fraction of return variance explained by the value fn
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


def _masked_mean(x: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Mean of x, optionally restricted to entries where mask > 0."""
    if mask is None:
        return x.mean()
    denom = mask.sum().clamp(min=1.0)
    return (x * mask).sum() / denom


def ppo_policy_loss(
    ratio: Tensor,
    advantages: Tensor,
    clip_eps: float,
    active_masks: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Clipped PPO surrogate policy loss.

    Parameters
    ----------
    ratio        : [T*B, N]  exp(new_log_prob - old_log_prob).
    advantages   : [T*B, N]  normalised GAE advantages.
    clip_eps     : clipping epsilon.
    active_masks : [T*B, N]  optional float mask; 0=dead agent, 1=alive.

    Returns
    -------
    loss     : scalar — mean clipped surrogate loss (to be minimised, i.e. negated).
    clipfrac : scalar — fraction of active samples where clip was active.
    """
    assert ratio.shape == advantages.shape, (
        f"ratio shape {ratio.shape} != advantages shape {advantages.shape}"
    )
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    loss = -_masked_mean(torch.min(surr1, surr2), active_masks)
    clipfrac = _masked_mean((ratio - 1).abs().gt(clip_eps).float(), active_masks)
    return loss, clipfrac


def value_loss(
    values_new: Tensor,
    values_old: Tensor,
    returns: Tensor,
    value_clip_eps: float,
    active_masks: Optional[Tensor] = None,
) -> Tensor:
    """
    Value loss with optional clipping (researcher decision 4).

    value_clip_eps controls the clipping range around values_old.

    Parameters
    ----------
    values_new     : [T*B, N]  new value estimates from current critic.
    values_old     : [T*B, N]  stored value estimates (detached).
    returns        : [T*B, N]  computed GAE returns.
    value_clip_eps : clipping range (matches config key algo.value_clip_eps).
    active_masks   : [T*B, N]  optional float mask; 0=dead agent, 1=alive.

    Returns
    -------
    loss : scalar.
    """
    assert values_new.shape == values_old.shape == returns.shape, (
        f"Shape mismatch: values_new={values_new.shape}, "
        f"values_old={values_old.shape}, returns={returns.shape}"
    )
    v_clipped = values_old + (values_new - values_old).clamp(-value_clip_eps, value_clip_eps)
    vl_unc = (values_new - returns).pow(2)
    vl_clp = (v_clipped - returns).pow(2)
    return 0.5 * _masked_mean(torch.max(vl_unc, vl_clp), active_masks)


def entropy_loss(
    entropy: Tensor,
    active_masks: Optional[Tensor] = None,
) -> Tensor:
    """
    Mean entropy over all active agents and minibatch entries.

    Parameters
    ----------
    entropy      : [T*B, N]  per-agent entropy.
    active_masks : [T*B, N]  optional float mask; 0=dead agent, 1=alive.

    Returns
    -------
    loss : scalar (positive; caller multiplies by -entropy_coef to add bonus).
    """
    return _masked_mean(entropy, active_masks)


def explained_variance(values: Tensor, returns: Tensor) -> Tensor:
    """
    Fraction of return variance explained by the value function.

    EV = 1 - Var(returns - values) / Var(returns)

    Parameters
    ----------
    values  : flat tensor  value estimates.
    returns : flat tensor  GAE returns.

    Returns
    -------
    ev : scalar in (-inf, 1].  Values close to 1.0 indicate a good critic.
    """
    var_y = returns.var()
    if var_y.item() < 1e-8:
        return torch.zeros((), device=values.device)
    return 1 - (returns - values).var() / (var_y + 1e-8)
