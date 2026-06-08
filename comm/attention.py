"""
Attention-based communication module (multi-head, multi-round).

Architecture (per round, weights SHARED across rounds):
    Q  = W_q(h)               [B, N, D]
    K  = W_k(h)               [B, N, D]
    V  = W_v(h)               [B, N, D]
    split Q, K, V into H heads of size d_k = D // H
    score[b, i, j, head] = Q[b, i, head] · K[b, j, head] / sqrt(d_k)
    alpha[b, i, j, head] = masked_softmax(score, mask, dim=j)  — safe all-zero for all-masked rows
    c[b, i, head]        = sum_j alpha[b, i, j, head] * V[b, j, head]
    c_concat[b, i]       = concat(c over heads)                [B, N, D]
    h'                   = ln(W_o(concat([h, c_concat], dim=-1)))  [B, N, D]

Shape contract:
    Input  h    : [B, N, D]  or  [T, B, N, D]
    Input  mask : [B, N, N]  or  [T, B, N, N]  bool
                  True = receiver i may receive from sender j.
                  None => full off-diagonal (built inside).
    Output      : same shape as h.

Init:
    W_q, W_k, W_v : Linear(D, D, bias=False), orthogonal gain=sqrt(2)
    W_o            : Linear(2*D, D, bias=False), orthogonal gain=0.01 (near pass-through)
    ln             : LayerNorm(D)  — mode-invariant, no dropout/batchnorm.

message_bits = rounds * (d_k + d_v) * 32
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from comm.base import CommModule
from comm.utils import masked_softmax, full_comm_mask


class AttentionComm(CommModule):
    """
    Multi-head attention comm module with masked aggregation.

    Parameters
    ----------
    hidden_dim : D — embedding dimension. Must be divisible by num_heads.
    num_heads  : H — number of attention heads.
    rounds     : number of message-passing rounds (weights shared).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 1, rounds: int = 1):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.rounds = rounds
        self.d_k = hidden_dim // num_heads
        self.d_v = hidden_dim // num_heads  # d_v == d_k (symmetric head split)

        # Projections SHARED across rounds
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Output projection: concat [h, c] -> hidden_dim
        self.W_o = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.ln = nn.LayerNorm(hidden_dim)

        nn.init.orthogonal_(self.W_q.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.W_k.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.W_v.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.W_o.weight, gain=0.01)

    def forward(
        self,
        h: Tensor,
        mask: Optional[Tensor] = None,
        class_id: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        h        : [B, N, D] or [T, B, N, D]
        mask     : [B, N, N] or [T, B, N, N] bool, or None (full comm)
        class_id : Optional [N] — accepted, carried, unused this phase.

        Returns
        -------
        Tensor of same shape as h.
        """
        assert h.dim() in (3, 4), (
            f"AttentionComm expects h with 3 or 4 dims, got {h.dim()}"
        )
        assert h.shape[-1] == self.hidden_dim, (
            f"AttentionComm: h.shape[-1]={h.shape[-1]} != hidden_dim={self.hidden_dim}"
        )

        # Handle [T, B, N, D] by reshaping to [T*B, N, D]
        reshaped = False
        if h.dim() == 4:
            T, B_orig, N, D = h.shape
            h = h.reshape(T * B_orig, N, D)
            if mask is not None:
                assert mask.shape == (T, B_orig, N, N), (
                    f"mask shape {mask.shape} != ({T}, {B_orig}, {N}, {N})"
                )
                mask = mask.reshape(T * B_orig, N, N)
            reshaped = True
        else:
            B_orig = None

        # After possible reshape, read actual batch size from h
        B, N, D = h.shape

        if mask is None:
            mask = full_comm_mask(N, device=h.device).unsqueeze(0).expand(B, -1, -1)
        else:
            assert mask.shape[-2:] == (N, N), (
                f"mask.shape[-2:]={mask.shape[-2:]} != ({N}, {N})"
            )
            assert mask.shape[0] == B, (
                f"mask.shape[0]={mask.shape[0]} != B={B}"
            )

        H = self.num_heads
        d_k = self.d_k

        for _ in range(self.rounds):
            # Project: [B, N, D]
            Q = self.W_q(h)  # [B, N, D]
            K = self.W_k(h)  # [B, N, D]
            V = self.W_v(h)  # [B, N, D]

            # Split heads: [B, N, H, d_k]
            Q = Q.reshape(B, N, H, d_k)
            K = K.reshape(B, N, H, d_k)
            V = V.reshape(B, N, H, d_k)

            # scores[b, i, j, h] = Q[b,i,h] · K[b,j,h] / sqrt(d_k)
            # Q: [B, N, H, d_k] => [B, N, 1, H, d_k]
            # K: [B, N, H, d_k] => [B, 1, N, H, d_k]
            Q_e = Q.unsqueeze(2)  # [B, N, 1, H, d_k]
            K_e = K.unsqueeze(1)  # [B, 1, N, H, d_k]
            scores = (Q_e * K_e).sum(dim=-1) / math.sqrt(d_k)  # [B, N, N, H]

            # mask: [B, N, N] => [B, N, N, 1] => [B, N, N, H]
            mask_4d = mask.unsqueeze(-1).expand_as(scores)  # [B, N, N, H]

            # masked_softmax over senders (dim=2)
            alpha = masked_softmax(scores, mask_4d, dim=2)  # [B, N, N, H]

            # Aggregate: c[b, i, h] = sum_j alpha[b,i,j,h] * V[b,j,h]
            # V: [B, N, H, d_k] => [B, 1, N, H, d_k]
            V_e = V.unsqueeze(1)  # [B, 1, N, H, d_k]
            # alpha: [B, N, N, H] => [B, N, N, H, 1]
            alpha_e = alpha.unsqueeze(-1)             # [B, N, N, H, 1]
            c = (alpha_e * V_e).sum(dim=2)            # [B, N, H, d_k]
            c = c.reshape(B, N, H * d_k)             # [B, N, D]  (D = H * d_k)

            # h' = ln(W_o([h, c]))
            h = self.ln(self.W_o(torch.cat([h, c], dim=-1)))  # [B, N, D]

        if reshaped:
            h = h.reshape(T, B_orig, N, D)
        return h

    def message_bits(self) -> int:
        """Total bits a receiver ingests per sender across all rounds and heads."""
        # d_k + d_v per round (both are D//H each, combined = 2*D//H per head, times H = 2*D)
        # But definition is per-sender payload = (d_k + d_v) floats across all heads
        # = (self.d_k + self.d_v) * num_heads * rounds = D + D = 2*D per round... however
        # the plan says: rounds * (d_k + d_v) * 32, where d_k and d_v are per-head sizes.
        return self.rounds * (self.d_k + self.d_v) * 32
