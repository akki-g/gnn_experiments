"""
Broadcast communication module.

Each agent broadcasts its embedding; receivers aggregate neighbor embeddings
with a masked mean and update their state with a learned gating.

Architecture (per round, weights SHARED across rounds):
    c_i = masked_mean over senders j of h_j      [B, N, D]
    h   = tanh(W_H(h) + W_C(c))                  [B, N, D]

Shape contract:
    Input  h    : [B, N, D]  or  [T, B, N, D]
    Input  mask : [B, N, N]  or  [T, B, N, N]  bool
                  True = receiver i may receive from sender j.
                  None => full off-diagonal (built inside).
    Output      : same shape as h.

Init:
    W_H  : Linear(D, D, bias=False), orthogonal gain=sqrt(2)
    W_C  : Linear(D, D, bias=False), orthogonal gain=0.01 (near pass-through)
    Activation: tanh

No dropout / batchnorm — module is mode-invariant (eval and train behave identically).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from comm.base import CommModule
from comm.utils import masked_mean, full_comm_mask


class BroadcastComm(CommModule):
    """
    Broadcast aggregation comm module.

    Parameters
    ----------
    hidden_dim  : D — embedding dimension.
    rounds      : number of message-passing rounds (weights shared).
    num_classes : number of agent class types for per-class embedding (default 2).
    """

    def __init__(
        self,
        hidden_dim: int,
        rounds: int = 1,
        zero_messages: bool = False,
        num_classes: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rounds = rounds
        self.zero_messages = zero_messages
        self.num_classes = num_classes

        # Weights SHARED across rounds
        self.W_H = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_C = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Orthogonal init: W_H with gain=sqrt(2), W_C with small gain (near pass-through)
        nn.init.orthogonal_(self.W_H.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.W_C.weight, gain=0.01)

        # Per-class embedding [num_classes, D]: added to h before message passing so
        # each agent class gets a distinct bias in embedding space.
        # Initialized near-zero (std=0.01) so at init the module output is close to
        # the pre-class_id behavior, preserving the mean_ratio_epoch0 ≈ 1.0 property.
        # Present regardless of zero_messages (class_emb is an input feature shift,
        # not a cross-agent message) so the _zm twin has identical parameter count.
        self.class_emb = nn.Embedding(num_classes, hidden_dim)
        nn.init.normal_(self.class_emb.weight, std=0.01)

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
                   If provided, adds a learned per-class embedding to h before
                   message passing. If None, behaves exactly as before (backward compat).

        Returns
        -------
        Tensor of same shape as h.
        """
        assert h.dim() in (3, 4), (
            f"BroadcastComm expects h with 3 or 4 dims, got {h.dim()}"
        )
        assert h.shape[-1] == self.hidden_dim, (
            f"BroadcastComm: h.shape[-1]={h.shape[-1]} != hidden_dim={self.hidden_dim}"
        )

        # Apply per-class embedding BEFORE reshape so we handle both [B,N,D] and
        # [T,B,N,D] uniformly: emb is [N, D], broadcast-add across all leading dims.
        if class_id is not None:
            # class_id: [N] long -> emb: [N, D]
            emb = self.class_emb(class_id)  # [N, D]
            # Reshape to match leading dims of h
            if h.dim() == 4:
                emb = emb.reshape(1, 1, emb.shape[0], emb.shape[1])  # [1,1,N,D]
            else:
                emb = emb.unsqueeze(0)  # [1, N, D]
            h = h + emb  # broadcast-add across batch/time dims

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
            # Full off-diagonal: [N, N] broadcast to [B, N, N]
            mask = full_comm_mask(N, device=h.device).unsqueeze(0).expand(B, -1, -1)
        else:
            assert mask.shape[-2:] == (N, N), (
                f"mask.shape[-2:]={mask.shape[-2:]} != ({N}, {N})"
            )
            assert mask.shape[0] == B, (
                f"mask.shape[0]={mask.shape[0]} != B={B}"
            )

        for _ in range(self.rounds):
            # mask[b, i, j] = True => i receives from j
            # h[b, j, :] are the "senders"
            # For receiver i: gather all h[:, j, :] where mask[:, i, j] is True
            # h_expand: [B, 1, N, D] broadcast to [B, N, N, D]
            h_expand = h.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
            # mask_expand: [B, N, N, 1] for masked_mean over dim=2
            mask_expand = mask.unsqueeze(-1)  # [B, N, N, 1]
            # c[b, i, :] = mean over j of h_expand[b, i, j, :] where mask[b, i, j]
            c = masked_mean(h_expand, mask_expand, dim=2)  # [B, N, D]
            # D2: zero_messages zeros the aggregated teammate message (ablation)
            if self.zero_messages:
                c = torch.zeros_like(c)
            h = torch.tanh(self.W_H(h) + self.W_C(c))

        if reshaped:
            h = h.reshape(T, B_orig, N, D)
        return h

    def message_bits(self) -> int:
        """Total bits a receiver ingests per sender across all rounds.
        Note: class_emb is an input-feature shift, not a cross-agent message.
        """
        return self.rounds * self.hidden_dim * 32
