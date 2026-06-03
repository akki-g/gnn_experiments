"""
Abstract base class for swappable communication modules.

Shape contract
--------------
h        : Tensor [B, N, D] or [T, B, N, D]
             per-agent embeddings produced by the shared encoder.
mask     : Optional Tensor [B, N, N] or [T, B, N, N]  bool
             True where agent i is allowed to receive from agent j.
class_id : Optional Tensor [N]  long
             agent-class identifier for heterogeneous comm modules.

return   : Tensor with identical shape to h.

The comm slot sits between the shared encoder and the actor head.
It must NOT be used as part of the centralized critic input path.
All future communication modules (Broadcast, Attention, Graph, …)
must subclass CommModule and implement forward().
"""

import torch.nn as nn


class CommModule(nn.Module):
    """
    Future communication plugin interface.

    h:        Tensor shaped [B, N, D] or [T, B, N, D]
              per-agent embeddings from the shared encoder
    mask:     Optional Tensor shaped [B, N, N] or [T, B, N, N]
              bool communication mask
    class_id: Optional Tensor shaped [N]
              future agent-class id vector

    returns: Tensor with the same leading dimensions and embedding size as h
    """

    def forward(self, h, mask=None, class_id=None):
        raise NotImplementedError

    def message_bits(self) -> int:
        return 0
