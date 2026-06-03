"""
Identity (no-op) communication module.

Returns the input tensor unchanged — no message passing, no transformation.
This is the default comm slot for Phase 1 and the baseline for ablations.
"""

from comm.base import CommModule


class IdentityComm(CommModule):
    """
    Pass-through comm module.  Acts as if there is no communication.

    forward returns h unchanged (not a copy, the same object) so the
    ablation baseline adds exactly zero computation.
    """

    def forward(self, h, mask=None, class_id=None):
        assert h.dim() in (3, 4), (
            f"IdentityComm expects h with 3 or 4 dimensions, got {h.dim()}"
        )
        return h

    def message_bits(self) -> int:
        return 0
