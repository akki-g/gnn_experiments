"""
PPO update entry point.

Loop order (inner):
    for epoch in range(ppo_epochs):
        for minibatch in buffer.get_minibatches(num_minibatches):
            out = model.evaluate(obs_mb, state_mb, actions_mb)
            new_log_probs = out["log_prob"]   # [T*B, N]
            old_log_probs = minibatch["log_probs"]   # already stored, detached
            ratio = torch.exp(new_log_probs - old_log_probs)
            # NOTE: old_log_probs are NEVER recomputed — they come from the buffer.
            # On epoch 0 before any optimizer step, ratio ≈ 1.0 for all (b,n).
            <compute policy_loss, value_loss, entropy_loss using active_masks>
            total_loss.backward()
            optimizer.step()

active_masks
------------
The minibatch dict from RolloutBuffer includes "active_masks" [T*B, N]
(0=dead agent, 1=alive).  These are forwarded to every loss function so that
dead-agent steps do not contribute to policy gradients or critic updates.
For Phase 1 (toy env, no agent deaths), active_masks is always all-ones and the
masked mean equals the regular mean.

Returned metrics:
    policy_loss         : mean clipped surrogate loss
    value_loss          : raw mean value MSE *before* multiplying by value_coef
                          (total_loss correctly applies value_coef; this field
                          is logged unscaled for diagnostic use)
    entropy             : mean entropy
    total_loss          : weighted sum (pol + value_coef*val - entropy_coef*ent)
    actor_grad_norm     : L2 norm of encoder+comm+actor gradients BEFORE clipping
    critic_grad_norm    : L2 norm of critic gradients BEFORE clipping
    grad_norm           : alias for actor_grad_norm (kept for backward compat);
                          clip_grad_norm_ returns the pre-clip norm, so logged
                          values can exceed max_grad_norm — the applied grads
                          ARE clipped to max_grad_norm
    approx_kl           : 0.5 * E[(log π_new - log π_old)²] — this is the
                          squared-log-ratio approximation to KL, not the
                          standard first-order Schulman estimator
                          (old - new log-probs).mean(); both are valid proxies
    clipfrac            : fraction of active samples where clip was active
    mean_ratio_epoch0   : mean ratio on the FIRST MINIBATCH of epoch 0 only
                          (not the full epoch); should be ≈ 1.0 as a sanity
                          check that old log-probs were not recomputed
    explained_variance  : fraction of return variance explained by the value fn;
                          computed in raw reward space (denormalized when
                          value_normalizer is provided)
"""

from typing import Dict

import torch
import torch.nn as nn

from backbone.losses import ppo_policy_loss, value_loss, entropy_loss, explained_variance


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer,
    *,
    ppo_epochs: int,
    num_minibatches: int,
    clip_eps: float,
    value_clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    normalize_advantages: bool = True,
    value_normalizer=None,
) -> Dict[str, float]:
    """
    Run PPO mini-batch update over the filled rollout buffer.

    Calls model.train() at the start so any future dropout/batchnorm/attention
    layers inside encoder, actor, or critic are in training mode.

    Parameters
    ----------
    model              : MAPPO instance.
    optimizer          : single joint optimizer over encoder + actor + critic (decision 3).
    buffer             : filled RolloutBuffer with computed advantages/returns.
    ppo_epochs         : number of passes over the buffer.
    num_minibatches    : number of minibatches to split T*B into.
    clip_eps           : PPO clipping epsilon for the policy ratio.
    value_clip_eps     : PPO clipping epsilon for the value loss (decision 4).
    entropy_coef       : weight for entropy bonus.
    value_coef         : weight for value loss.
    max_grad_norm      : gradient clip norm.
    normalize_advantages: normalize advantages to zero mean, unit variance per minibatch.

    Returns
    -------
    dict of scalar float metrics (keys listed in module docstring).
    """
    # Ensure training mode so future layers (dropout, batchnorm, attention) behave correctly
    model.train()

    if value_normalizer is not None:
        value_normalizer.update(buffer.returns)

    metrics = {
        "policy_loss": [], "value_loss": [], "entropy": [], "total_loss": [],
        "grad_norm": [], "actor_grad_norm": [], "critic_grad_norm": [],
        "approx_kl": [], "clipfrac": [],
    }
    mean_ratio_epoch0 = None

    for epoch in range(ppo_epochs):
        first_mb = True
        for mb in buffer.get_minibatches(num_minibatches):
            active_masks = mb["active_masks"]  # [T*B, N] — always ones for toy env

            out = model.evaluate(mb["obs"], mb["state"], mb["actions"])
            new_lp = out["log_prob"]
            old_lp = mb["log_probs"]
            ratio = torch.exp(new_lp - old_lp)

            if epoch == 0 and first_mb:
                # Sampled from first minibatch of epoch 0 only (not the full epoch).
                # Should be ≈ 1.0; confirms old log-probs were not recomputed.
                mean_ratio_epoch0 = ratio.mean().item()
                first_mb = False

            adv = mb["advantages"]
            if normalize_advantages:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            pol_loss, clipfrac = ppo_policy_loss(ratio, adv, clip_eps, active_masks)
            # val_loss is raw MSE (unscaled); value_coef is applied in total only.
            if value_normalizer is not None:
                returns_target = value_normalizer.normalize(mb["returns"]).detach()
            else:
                returns_target = mb["returns"]
            val_loss = value_loss(out["value"], mb["values"], returns_target, value_clip_eps, active_masks)
            ent = entropy_loss(out["entropy"], active_masks)
            total = pol_loss + value_coef * val_loss - entropy_coef * ent

            optimizer.zero_grad()
            total.backward()
            actor_gnorm = nn.utils.clip_grad_norm_(model.actor_parameters(), max_grad_norm).item()
            critic_gnorm = nn.utils.clip_grad_norm_(model.critic_parameters(), max_grad_norm).item()
            optimizer.step()

            # Squared-log-ratio KL approximation: 0.5·E[(log r)²].
            # Not the Schulman (old-new).mean() estimator, but a valid proxy.
            approx_kl = 0.5 * ((new_lp - old_lp) ** 2).mean().item()
            metrics["policy_loss"].append(pol_loss.item())
            metrics["value_loss"].append(val_loss.item())
            metrics["entropy"].append(ent.item())
            metrics["total_loss"].append(total.item())
            metrics["actor_grad_norm"].append(actor_gnorm)
            metrics["critic_grad_norm"].append(critic_gnorm)
            metrics["grad_norm"].append(actor_gnorm)  # alias for backward compat
            metrics["approx_kl"].append(approx_kl)
            metrics["clipfrac"].append(clipfrac.item())

    if value_normalizer is not None:
        ev_values = value_normalizer.denormalize(buffer.values).flatten()
    else:
        ev_values = buffer.values.flatten()
    ev = explained_variance(ev_values, buffer.returns.flatten()).item()
    return {
        "policy_loss": sum(metrics["policy_loss"]) / len(metrics["policy_loss"]),
        "value_loss": sum(metrics["value_loss"]) / len(metrics["value_loss"]),
        "entropy": sum(metrics["entropy"]) / len(metrics["entropy"]),
        "total_loss": sum(metrics["total_loss"]) / len(metrics["total_loss"]),
        "actor_grad_norm": sum(metrics["actor_grad_norm"]) / len(metrics["actor_grad_norm"]),
        "critic_grad_norm": sum(metrics["critic_grad_norm"]) / len(metrics["critic_grad_norm"]),
        "grad_norm": sum(metrics["grad_norm"]) / len(metrics["grad_norm"]),
        "approx_kl": sum(metrics["approx_kl"]) / len(metrics["approx_kl"]),
        "clipfrac": sum(metrics["clipfrac"]) / len(metrics["clipfrac"]),
        "mean_ratio_epoch0": mean_ratio_epoch0,
        "explained_variance": ev,
    }
