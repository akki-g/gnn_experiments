"""
Shared utility functions for communication modules.

These helpers handle masked aggregation, masked softmax, self-loop addition,
and full-communication mask construction.  They are imported by broadcast.py,
attention.py, and graph.py.

Mask convention (matches backbone plan):
    mask[..., i, j] = True  =>  receiver i may receive from sender j.
    Diagonal is always False (no self-message by default).
    mask = None => full off-diagonal (all-to-all minus self) — built inside each module.
"""

import torch
from torch import Tensor


def masked_mean(x: Tensor, mask: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """
    Compute the mean of `x` over `dim`, counting only positions where mask is True.

    All-False rows (no valid senders) yield zero (not NaN).

    Parameters
    ----------
    x    : arbitrary shape tensor.
    mask : bool tensor broadcastable to x.
    dim  : dimension to reduce.

    Returns
    -------
    Tensor with `dim` removed (or kept if keepdim=True), same dtype as x.
    """
    x_masked = x * mask.float()
    sum_x = x_masked.sum(dim=dim, keepdim=keepdim)
    count = mask.float().sum(dim=dim, keepdim=keepdim).clamp(min=1.0)
    return sum_x / count


def masked_softmax(scores: Tensor, mask: Tensor, dim: int) -> Tensor:
    """
    Softmax over `dim` restricted to positions where mask is True.

    All-masked rows (mask.any(dim)==False) yield all-zero weights (no NaN).

    Parameters
    ----------
    scores : logit tensor of any shape.
    mask   : bool tensor, same shape as scores.
              True = valid position; False = masked out.
    dim    : dimension to softmax over (the "senders" axis).

    Returns
    -------
    Tensor of same shape as scores, non-negative, zero at masked positions.
    """
    # Fill masked positions with -inf before softmax
    filled = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(filled, dim=dim)
    # Any row that was entirely masked will be NaN after softmax(-inf,...);
    # replace with 0 so all-masked rows produce a zero aggregation.
    # mask.any(dim) == False identifies all-masked rows.
    all_masked = ~mask.any(dim=dim, keepdim=True)  # True where NO valid sender
    weights = weights.masked_fill(all_masked, 0.0)
    return weights


def add_self_loop(mask: Tensor) -> Tensor:
    """
    Return a NEW mask tensor with the diagonal set to True (self-loop added).

    Does NOT modify the input in-place.

    Parameters
    ----------
    mask : [..., N, N] bool tensor.

    Returns
    -------
    New [..., N, N] bool tensor with diagonal True.
    """
    N = mask.shape[-1]
    result = mask.clone()
    idx = torch.arange(N, device=mask.device)
    result[..., idx, idx] = True
    return result


def full_comm_mask(n: int, device: torch.device, dtype: torch.dtype = torch.bool) -> Tensor:
    """
    Build an [N, N] full off-diagonal communication mask.

    mask[i, j] = True  for i != j
    mask[i, i] = False

    Parameters
    ----------
    n      : number of agents.
    device : target device.
    dtype  : output dtype (default bool).

    Returns
    -------
    [N, N] bool tensor.
    """
    mask = torch.ones(n, n, dtype=dtype, device=device)
    idx = torch.arange(n, device=device)
    mask[idx, idx] = False
    return mask.bool()
