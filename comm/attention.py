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
    hidden_dim  : D — embedding dimension. Must be divisible by num_heads.
    num_heads   : H — number of attention heads.
    rounds      : number of message-passing rounds (weights shared).
    num_classes : number of agent class types for per-class conditioning (default 2).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        rounds: int = 1,
        zero_messages: bool = False,
        num_classes: int = 2,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.rounds = rounds
        self.zero_messages = zero_messages
        self.d_k = hidden_dim // num_heads
        self.d_v = hidden_dim // num_heads  # d_v == d_k (symmetric head split)
        self.num_classes = num_classes

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

        # Per-class embedding [num_classes, D]: shifts h by a class-specific offset
        # before Q/K/V projections so each agent class sees a distinct embedding.
        # Initialized near-zero (std=0.01) so at init output ≈ pre-change behavior,
        # preserving mean_ratio_epoch0 ≈ 1.0. Present regardless of zero_messages
        # (it is an input feature shift, not a cross-agent message) so _zm twin has
        # identical parameter count.
        self.class_emb = nn.Embedding(num_classes, hidden_dim)
        nn.init.normal_(self.class_emb.weight, std=0.01)

        # Per-class-pair bias on attention logits [num_classes, num_classes].
        # Added to scores[b, i, j, head] via class_bias[class_id[i], class_id[j]]
        # BEFORE masked_softmax, so masked edges remain masked (bias cannot resurrect
        # a masked entry — masked_fill(-inf) applied after).
        # Zeros init is critical: at init adds nothing, preserving ratio ≈ 1.0.
        # Present regardless of zero_messages so _zm twin has identical parameters.
        self.class_bias = nn.Parameter(torch.zeros(num_classes, num_classes))

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
        class_id : Optional [N] long — per-agent class index (0 or 1).
                   If provided:
                     1. Adds a learned per-class embedding to h before Q/K/V projection.
                     2. Adds a learned class-pair bias to attention logits before softmax.
                   If None, behaves exactly as before (backward compat).

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

        # Apply per-class embedding BEFORE reshape so we handle both [B,N,D] and
        # [T,B,N,D] uniformly.  emb: [N, D] broadcast-added across all leading dims.
        if class_id is not None:
            emb = self.class_emb(class_id)  # [N, D]
            if h.dim() == 4:
                emb = emb.reshape(1, 1, emb.shape[0], emb.shape[1])  # [1,1,N,D]
            else:
                emb = emb.unsqueeze(0)  # [1, N, D]
            h = h + emb

        # Precompute per-agent-pair class bias [N, N] for use inside the loop.
        # class_bias[class_id[i], class_id[j]] — scalar bias for each (i,j) pair.
        # Built once per forward pass (outside the rounds loop) since class_id is fixed.
        pair_bias = None
        if class_id is not None:
            # class_bias: [C, C]; index by sender and receiver class ids
            # pair_bias[i, j] = class_bias[class_id[i], class_id[j]]
            pair_bias = self.class_bias[class_id[:, None], class_id[None, :]]  # [N, N]

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

            # Add class-pair bias to logits BEFORE masked_softmax.
            # This ensures masked entries (set to -inf by masked_softmax) are not
            # resurrected by the bias — the softmax masking is applied AFTER this add.
            # pair_bias: [N, N] -> broadcast to [1, N, N, 1] -> [B, N, N, H]
            if pair_bias is not None:
                scores = scores + pair_bias.unsqueeze(0).unsqueeze(-1)  # [B, N, N, H]

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

            # D3: zero_messages zeros aggregated teammate messages BEFORE concat
            if self.zero_messages:
                c = torch.zeros_like(c)

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
