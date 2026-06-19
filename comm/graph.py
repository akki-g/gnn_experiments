"""
Graph neural network communication module (multi-head, multi-layer GNN).

Architecture (per layer, PER-LAYER weight matrices, heads SHARED within a layer):
    graph_mask = add_self_loop(mask)          self-loop ALWAYS present
    For each layer l:
        Q = W_q_l(h)  K = W_k_l(h)  V = W_v_l(h)   [B, N, D]
        Split Q, K, V into H heads of size d_k = D // H
        For each head head_idx:
            score_ij = tau * Q_i · K_j               scalar per (i, j) pair
            alpha_ij = softmax(score, dim=j) over graph_mask neighborhood
            c_i      = sum_j alpha_ij * V_j           [B, N, d_k]
        concat heads -> delta [B, N, D]
        delta = proj_l(concat) -> tanh(delta)
        h     = ln_l(h + delta)
    return h at final layer (no cross-layer concat)

Shape contract:
    Input  h    : [B, N, D]  or  [T, B, N, D]
    Input  mask : [B, N, N]  or  [T, B, N, N]  bool
                  True = receiver i may receive from sender j.
                  None => full off-diagonal (built inside, plus self-loop in graph_mask).
    Output      : same shape as h, same D.

Init:
    Per-layer W_q, W_k, W_v  : orthogonal gain=sqrt(2), no bias
    Per-layer proj            : orthogonal gain=0.01 (near pass-through), no bias
    Per-layer ln              : LayerNorm(D)
    tau = 1 / sqrt(d_k)

message_bits = num_layers * 2 * D * 32
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from comm.base import CommModule
from comm.utils import masked_softmax, add_self_loop, full_comm_mask


class GraphComm(CommModule):
    """
    Graph neural network comm module.

    Parameters
    ----------
    hidden_dim  : D — embedding dimension. Must be divisible by num_heads.
    num_heads   : H — number of attention heads (shared within each layer).
    num_layers  : L — number of GNN message-passing layers.
    num_classes : number of agent class types for per-class conditioning (default 2).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        zero_messages: bool = False,
        num_classes: int = 2,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.zero_messages = zero_messages
        self.d_k = hidden_dim // num_heads
        self.tau = 1.0 / math.sqrt(self.d_k)
        self.num_classes = num_classes

        # PER-LAYER weight matrices (nn.ModuleList length L)
        self.W_q_layers = nn.ModuleList()
        self.W_k_layers = nn.ModuleList()
        self.W_v_layers = nn.ModuleList()
        self.proj_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()

        for _ in range(num_layers):
            W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
            W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
            W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
            proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            ln_l = nn.LayerNorm(hidden_dim)

            nn.init.orthogonal_(W_q.weight, gain=math.sqrt(2))
            nn.init.orthogonal_(W_k.weight, gain=math.sqrt(2))
            nn.init.orthogonal_(W_v.weight, gain=math.sqrt(2))
            nn.init.orthogonal_(proj.weight, gain=0.01)

            self.W_q_layers.append(W_q)
            self.W_k_layers.append(W_k)
            self.W_v_layers.append(W_v)
            self.proj_layers.append(proj)
            self.ln_layers.append(ln_l)

        # Per-class embedding [num_classes, D]: shifts h by a class-specific offset
        # before Q/K/V projections so each agent class sees a distinct initial embedding.
        # Initialized near-zero (std=0.01) so at init output ≈ pre-change behavior,
        # preserving mean_ratio_epoch0 ≈ 1.0. Present regardless of zero_messages so
        # _zm twin has identical parameter count.
        self.class_emb = nn.Embedding(num_classes, hidden_dim)
        nn.init.normal_(self.class_emb.weight, std=0.01)

        # Per-class-pair bias on graph attention logits [num_classes, num_classes].
        # Added to scores[b, i, j, head] for each (i,j) pair based on their class ids,
        # BEFORE masked_softmax so masked edges remain masked (masked_fill(-inf) applied
        # after this add). Zeros init is critical: at init adds nothing, preserving
        # ratio ≈ 1.0. Present regardless of zero_messages (same-param _zm twin).
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
                     1. Adds a learned per-class embedding to h at the start of each
                        GNN layer (h is re-conditioned per layer so the embedding is
                        added once before the per-layer Q/K/V projections).
                     2. Adds a learned class-pair bias to attention logits before softmax.
                   If None, behaves exactly as before (backward compat).

        Returns
        -------
        Tensor of same shape as h.
        """
        assert h.dim() in (3, 4), (
            f"GraphComm expects h with 3 or 4 dims, got {h.dim()}"
        )
        assert h.shape[-1] == self.hidden_dim, (
            f"GraphComm: h.shape[-1]={h.shape[-1]} != hidden_dim={self.hidden_dim}"
        )

        # Apply per-class embedding BEFORE reshape so we handle both [B,N,D] and
        # [T,B,N,D] uniformly.  emb: [N, D] broadcast-added across all leading dims.
        # Applied once at the start (before any GNN layer) to the input h.
        if class_id is not None:
            emb = self.class_emb(class_id)  # [N, D]
            if h.dim() == 4:
                emb = emb.reshape(1, 1, emb.shape[0], emb.shape[1])  # [1,1,N,D]
            else:
                emb = emb.unsqueeze(0)  # [1, N, D]
            h = h + emb

        # Precompute class-pair bias [N, N] for use inside the layer loop.
        # class_bias[class_id[i], class_id[j]] — scalar added to scores[b,i,j,head].
        pair_bias = None
        if class_id is not None:
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
            base_mask = full_comm_mask(N, device=h.device).unsqueeze(0).expand(B, -1, -1)
        else:
            assert mask.shape[-2:] == (N, N), (
                f"mask.shape[-2:]={mask.shape[-2:]} != ({N}, {N})"
            )
            assert mask.shape[0] == B, (
                f"mask.shape[0]={mask.shape[0]} != B={B}"
            )
            base_mask = mask

        # Self-loop ALWAYS present in graph_mask
        graph_mask = add_self_loop(base_mask)  # [B, N, N] True including self

        H = self.num_heads
        d_k = self.d_k

        for l in range(self.num_layers):
            Q = self.W_q_layers[l](h)  # [B, N, D]
            K = self.W_k_layers[l](h)  # [B, N, D]
            V = self.W_v_layers[l](h)  # [B, N, D]

            # Split heads: [B, N, H, d_k]
            Q = Q.reshape(B, N, H, d_k)
            K = K.reshape(B, N, H, d_k)
            V = V.reshape(B, N, H, d_k)

            # scores[b, i, j, head] = tau * Q[b,i,head] · K[b,j,head]
            Q_e = Q.unsqueeze(2)  # [B, N, 1, H, d_k]
            K_e = K.unsqueeze(1)  # [B, 1, N, H, d_k]
            scores = self.tau * (Q_e * K_e).sum(dim=-1)  # [B, N, N, H]

            # Add class-pair bias to scores BEFORE masked_softmax so masked entries
            # (filled to -inf in masked_softmax) are not resurrected by the bias.
            # pair_bias: [N, N] -> [1, N, N, 1] -> broadcast to [B, N, N, H]
            if pair_bias is not None:
                scores = scores + pair_bias.unsqueeze(0).unsqueeze(-1)

            # Expand graph_mask to [B, N, N, H]
            graph_mask_4d = graph_mask.unsqueeze(-1).expand_as(scores)

            # Softmax over neighborhood (senders axis = dim=2)
            alpha = masked_softmax(scores, graph_mask_4d, dim=2)  # [B, N, N, H]

            # D4: zero_messages replaces the attention distribution with a self-only
            # identity distribution (weight=1 on self, 0 on all teammates), completely
            # decoupling agent i's message from its neighbors.  This preserves the
            # own-path capacity through the residual (h + delta) and self V-projection.
            # We build a [B, N, N, H] identity distribution rather than multiplying
            # the softmax alpha by a diagonal selector, because the softmax alpha[i,i]
            # varies with neighbor embeddings and would leak information.
            if self.zero_messages:
                alpha = torch.zeros_like(alpha)
                idx_diag = torch.arange(N, device=alpha.device)
                alpha[:, idx_diag, idx_diag, :] = 1.0  # self-only: weight=1 on self

            # Aggregate: c[b, i, h] = sum_j alpha[b,i,j,h] * V[b,j,h]
            V_e = V.unsqueeze(1)        # [B, 1, N, H, d_k]
            alpha_e = alpha.unsqueeze(-1)  # [B, N, N, H, 1]
            c = (alpha_e * V_e).sum(dim=2)  # [B, N, H, d_k]
            c = c.reshape(B, N, H * d_k)   # [B, N, D]

            # Residual: proj -> tanh = delta; h = ln(h + delta)
            delta = torch.tanh(self.proj_layers[l](c))  # [B, N, D]
            h = self.ln_layers[l](h + delta)

        if reshaped:
            h = h.reshape(T, B_orig, N, D)
        return h

    def message_bits(self) -> int:
        """Total bits a receiver ingests per sender across all layers."""
        return self.num_layers * 2 * self.hidden_dim * 32
