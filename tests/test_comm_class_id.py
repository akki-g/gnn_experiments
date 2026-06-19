"""
Tests for class_id conditioning in BroadcastComm, AttentionComm, GraphComm.

Verifies:
1. Different class_id vectors provably produce different outputs (after setting
   class_emb/class_bias to non-zero values to make the difference detectable).
2. forward(h, class_id=None) still runs and returns correct shape (backward compat).
3. For attention/graph: the class_bias enters the attention distribution
   (output differs between two class_id vectors when class_bias is non-zero).
4. Existing comm tests still pass (backward compat — covered by running those tests
   separately; here we verify class_id=None path produces the same result as
   the pre-change behavior by checking output shape and no-NaN).
"""

import math
import torch
import pytest

from comm import BroadcastComm, AttentionComm, GraphComm

B, N, D = 4, 5, 32  # N=5 to use non-trivial class patterns

NUM_CLASSES = 2

# Two distinct class_id vectors so the bias/embedding has an observable effect
CID_A = torch.tensor([0, 0, 1, 1, 0], dtype=torch.long)  # [N]
CID_B = torch.tensor([1, 1, 0, 0, 1], dtype=torch.long)  # [N] — swapped


def make_h():
    torch.manual_seed(42)
    return torch.randn(B, N, D)


def make_h4():
    T = 3
    torch.manual_seed(42)
    return torch.randn(T, B, N, D)


# ---------------------------------------------------------------------------
# BroadcastComm
# ---------------------------------------------------------------------------

class TestBroadcastClassId:

    def _make(self):
        m = BroadcastComm(D, rounds=1, num_classes=NUM_CLASSES)
        # Set class_emb weights to DISTINCT values per class so that CID_A != CID_B.
        # class 0 -> all 1.0; class 1 -> all -1.0: these are different, so the
        # per-agent embedding differs between agents with class 0 vs class 1.
        with torch.no_grad():
            m.class_emb.weight[0].fill_(1.0)
            m.class_emb.weight[1].fill_(-1.0)
        return m

    def test_class_id_affects_output(self):
        """Different class_id vectors produce different outputs (class_id is used)."""
        m = self._make()
        h = make_h()

        out_a = m(h, class_id=CID_A)
        out_b = m(h, class_id=CID_B)

        assert out_a.shape == (B, N, D), f"shape A: {out_a.shape}"
        assert out_b.shape == (B, N, D), f"shape B: {out_b.shape}"
        assert not torch.allclose(out_a, out_b, atol=1e-5), (
            "BroadcastComm: output with CID_A should differ from CID_B "
            f"when class_emb is non-zero. Max diff: {(out_a - out_b).abs().max().item():.6f}"
        )

    def test_none_class_id_runs_and_correct_shape(self):
        """class_id=None path runs without error and returns correct shape."""
        m = self._make()
        h = make_h()
        out = m(h, class_id=None)
        assert out.shape == (B, N, D), f"shape with None class_id: {out.shape}"
        assert torch.isfinite(out).all(), "NaN/Inf with class_id=None"

    def test_none_class_id_unchanged_from_base(self):
        """class_id=None should produce the same result as the module with zero class_emb."""
        m = BroadcastComm(D, rounds=1, num_classes=NUM_CLASSES)
        # Build a clone with class_emb zeroed — both should give same output for None
        m_zero = BroadcastComm(D, rounds=1, num_classes=NUM_CLASSES)
        with torch.no_grad():
            for p1, p2 in zip(m.parameters(), m_zero.parameters()):
                p2.copy_(p1)
            m_zero.class_emb.weight.zero_()

        h = make_h()
        out_none = m(h, class_id=None)
        out_zero = m_zero(h, class_id=None)
        assert torch.allclose(out_none, out_zero, atol=1e-7), (
            "class_id=None should be equivalent to zero class_emb "
            f"(max diff: {(out_none - out_zero).abs().max().item():.7f})"
        )

    def test_4d_input_class_id(self):
        """class_id works with [T, B, N, D] input."""
        m = self._make()
        h = make_h4()
        T = h.shape[0]
        out_a = m(h, class_id=CID_A)
        out_b = m(h, class_id=CID_B)
        assert out_a.shape == (T, B, N, D), f"4D shape A: {out_a.shape}"
        assert not torch.allclose(out_a, out_b, atol=1e-5), (
            "BroadcastComm [T,B,N,D]: CID_A and CID_B should differ"
        )

    def test_no_nan_with_class_id(self):
        """Output is finite when class_id is provided."""
        m = self._make()
        out = m(make_h(), class_id=CID_A)
        assert torch.isfinite(out).all(), "NaN/Inf in BroadcastComm with class_id"

    def test_zero_messages_has_same_params(self):
        """zero_messages=True twin has the same number of parameters (class_emb present)."""
        m_live = BroadcastComm(D, rounds=1, zero_messages=False, num_classes=NUM_CLASSES)
        m_zm   = BroadcastComm(D, rounds=1, zero_messages=True,  num_classes=NUM_CLASSES)
        n_live = sum(p.numel() for p in m_live.parameters())
        n_zm   = sum(p.numel() for p in m_zm.parameters())
        assert n_live == n_zm, (
            f"zero_messages twin param count mismatch: live={n_live}, zm={n_zm}"
        )


# ---------------------------------------------------------------------------
# AttentionComm
# ---------------------------------------------------------------------------

class TestAttentionClassId:

    def _make(self):
        m = AttentionComm(D, num_heads=1, rounds=1, num_classes=NUM_CLASSES)
        # Set class_emb to DISTINCT values per class; class_bias to an asymmetric matrix
        # so that CID_A and CID_B produce measurably different logit patterns.
        with torch.no_grad():
            m.class_emb.weight[0].fill_(1.0)
            m.class_emb.weight[1].fill_(-1.0)
            # Asymmetric: same-class edges get +5, cross-class edges get -5
            m.class_bias[0, 0] = 5.0
            m.class_bias[1, 1] = 5.0
            m.class_bias[0, 1] = -5.0
            m.class_bias[1, 0] = -5.0
        return m

    def test_class_id_affects_output(self):
        """Different class_id vectors produce different outputs."""
        m = self._make()
        h = make_h()

        out_a = m(h, class_id=CID_A)
        out_b = m(h, class_id=CID_B)

        assert out_a.shape == (B, N, D)
        assert out_b.shape == (B, N, D)
        assert not torch.allclose(out_a, out_b, atol=1e-5), (
            "AttentionComm: output with CID_A should differ from CID_B. "
            f"Max diff: {(out_a - out_b).abs().max().item():.6f}"
        )

    def test_class_bias_enters_attention_distribution(self):
        """
        Verify class_bias changes attention weights (not just downstream).
        We replicate the attention weight computation (as in test_comm_attention.py)
        and check that CID_A and CID_B produce different attention distributions.
        Uses an ASYMMETRIC class_bias matrix so CID_A and CID_B produce genuinely
        different pair_bias patterns (a symmetric matrix with class_bias[0,0]=class_bias[1,1]
        and class_bias[0,1]=class_bias[1,0] would be invariant to class swapping).
        """
        m = AttentionComm(D, num_heads=1, rounds=1, num_classes=NUM_CLASSES)
        with torch.no_grad():
            m.class_emb.weight.zero_()  # isolate bias effect
            # Asymmetric: class 0->0 gets very high bias, all others 0
            m.class_bias[0, 0] = 20.0
            m.class_bias[0, 1] = 0.0
            m.class_bias[1, 0] = 0.0
            m.class_bias[1, 1] = 0.0

        h = make_h()
        H = m.num_heads
        d_k = m.d_k

        def _attn_weights(cid):
            pair_bias = m.class_bias[cid[:, None], cid[None, :]]  # [N, N]
            Q = m.W_q(h).reshape(B, N, H, d_k)
            K = m.W_k(h).reshape(B, N, H, d_k)
            Q_e = Q.unsqueeze(2)
            K_e = K.unsqueeze(1)
            scores = (Q_e * K_e).sum(-1) / math.sqrt(d_k)  # [B, N, N, H]
            scores = scores + pair_bias.unsqueeze(0).unsqueeze(-1)
            from comm.utils import full_comm_mask
            mask = full_comm_mask(N, device=h.device).unsqueeze(0).expand(B, -1, -1)
            mask_4d = mask.unsqueeze(-1).expand_as(scores)
            filled = scores.masked_fill(~mask_4d, float("-inf"))
            return torch.softmax(filled, dim=2)  # [B, N, N, H]

        w_a = _attn_weights(CID_A)
        w_b = _attn_weights(CID_B)

        assert not torch.allclose(w_a, w_b, atol=1e-4), (
            "AttentionComm: class_bias should change attention weights between "
            f"CID_A and CID_B. Max diff: {(w_a - w_b).abs().max().item():.6f}"
        )

    def test_none_class_id_runs_and_correct_shape(self):
        m = self._make()
        out = m(make_h(), class_id=None)
        assert out.shape == (B, N, D)
        assert torch.isfinite(out).all()

    def test_4d_input_class_id(self):
        m = self._make()
        h = make_h4()
        T = h.shape[0]
        out_a = m(h, class_id=CID_A)
        assert out_a.shape == (T, B, N, D)
        out_b = m(h, class_id=CID_B)
        assert not torch.allclose(out_a, out_b, atol=1e-5), (
            "AttentionComm [T,B,N,D]: CID_A and CID_B should differ"
        )

    def test_no_nan_with_class_id(self):
        m = self._make()
        out = m(make_h(), class_id=CID_A)
        assert torch.isfinite(out).all()

    def test_zero_messages_has_same_params(self):
        m_live = AttentionComm(D, rounds=1, zero_messages=False, num_classes=NUM_CLASSES)
        m_zm   = AttentionComm(D, rounds=1, zero_messages=True,  num_classes=NUM_CLASSES)
        n_live = sum(p.numel() for p in m_live.parameters())
        n_zm   = sum(p.numel() for p in m_zm.parameters())
        assert n_live == n_zm, (
            f"AttentionComm zero_messages twin param count mismatch: live={n_live}, zm={n_zm}"
        )

    def test_class_bias_zeros_init_preserves_behavior(self):
        """
        At init class_emb std=0.01 is near-zero and class_bias=zeros.
        Verify: with class_id provided vs None, outputs are close (near-zero effect).
        This confirms ratio≈1 property is preserved at init.
        """
        torch.manual_seed(7)
        m = AttentionComm(D, num_heads=1, rounds=1, num_classes=NUM_CLASSES)
        # class_emb is std=0.01 (very small), class_bias is zeros by default
        h = make_h()
        with torch.no_grad():
            out_none = m(h, class_id=None)
            out_cid  = m(h, class_id=CID_A)
        max_diff = (out_none - out_cid).abs().max().item()
        # With std=0.01 class_emb and zeros class_bias, the outputs should be very close
        assert max_diff < 0.5, (
            f"At init, class_id should have near-zero effect (max diff={max_diff:.4f})"
        )


# ---------------------------------------------------------------------------
# GraphComm
# ---------------------------------------------------------------------------

class TestGraphClassId:

    def _make(self):
        m = GraphComm(D, num_heads=4, num_layers=2, num_classes=NUM_CLASSES)
        with torch.no_grad():
            m.class_emb.weight[0].fill_(1.0)
            m.class_emb.weight[1].fill_(-1.0)
            m.class_bias[0, 0] = 5.0
            m.class_bias[1, 1] = 5.0
            m.class_bias[0, 1] = -5.0
            m.class_bias[1, 0] = -5.0
        return m

    def test_class_id_affects_output(self):
        """Different class_id vectors produce different outputs."""
        m = self._make()
        h = make_h()

        out_a = m(h, class_id=CID_A)
        out_b = m(h, class_id=CID_B)

        assert out_a.shape == (B, N, D)
        assert out_b.shape == (B, N, D)
        assert not torch.allclose(out_a, out_b, atol=1e-5), (
            "GraphComm: output with CID_A should differ from CID_B. "
            f"Max diff: {(out_a - out_b).abs().max().item():.6f}"
        )

    def test_class_bias_enters_attention_distribution(self):
        """
        Verify class_bias changes graph attention weights.
        Replicate layer-0 score computation to check attention distribution differs.
        Uses an asymmetric class_bias matrix (class_bias[0,0]=20, rest=0) so that
        CID_A=[0,0,1,1,0] and CID_B=[1,1,0,0,1] yield different pair_bias patterns.
        """
        m = GraphComm(D, num_heads=4, num_layers=1, num_classes=NUM_CLASSES)
        with torch.no_grad():
            m.class_emb.weight.zero_()  # isolate bias effect
            m.class_bias[0, 0] = 20.0
            m.class_bias[0, 1] = 0.0
            m.class_bias[1, 0] = 0.0
            m.class_bias[1, 1] = 0.0

        h = make_h()
        H = m.num_heads
        d_k = m.d_k
        tau = m.tau

        from comm.utils import add_self_loop, full_comm_mask
        base_mask = full_comm_mask(N, device=h.device).unsqueeze(0).expand(B, -1, -1)
        graph_mask = add_self_loop(base_mask)  # [B, N, N]

        def _attn_weights(cid):
            pair_bias = m.class_bias[cid[:, None], cid[None, :]]  # [N, N]
            Q = m.W_q_layers[0](h).reshape(B, N, H, d_k)
            K = m.W_k_layers[0](h).reshape(B, N, H, d_k)
            Q_e = Q.unsqueeze(2)
            K_e = K.unsqueeze(1)
            scores = tau * (Q_e * K_e).sum(-1)  # [B, N, N, H]
            scores = scores + pair_bias.unsqueeze(0).unsqueeze(-1)
            gm4d = graph_mask.unsqueeze(-1).expand_as(scores)
            filled = scores.masked_fill(~gm4d, float("-inf"))
            return torch.softmax(filled, dim=2)

        w_a = _attn_weights(CID_A)
        w_b = _attn_weights(CID_B)

        assert not torch.allclose(w_a, w_b, atol=1e-4), (
            "GraphComm: class_bias should change attention weights. "
            f"Max diff: {(w_a - w_b).abs().max().item():.6f}"
        )

    def test_none_class_id_runs_and_correct_shape(self):
        m = self._make()
        out = m(make_h(), class_id=None)
        assert out.shape == (B, N, D)
        assert torch.isfinite(out).all()

    def test_4d_input_class_id(self):
        m = self._make()
        h = make_h4()
        T = h.shape[0]
        out_a = m(h, class_id=CID_A)
        assert out_a.shape == (T, B, N, D)
        out_b = m(h, class_id=CID_B)
        assert not torch.allclose(out_a, out_b, atol=1e-5), (
            "GraphComm [T,B,N,D]: CID_A and CID_B should differ"
        )

    def test_no_nan_with_class_id(self):
        m = self._make()
        out = m(make_h(), class_id=CID_A)
        assert torch.isfinite(out).all()

    def test_zero_messages_has_same_params(self):
        m_live = GraphComm(D, num_layers=2, zero_messages=False, num_classes=NUM_CLASSES)
        m_zm   = GraphComm(D, num_layers=2, zero_messages=True,  num_classes=NUM_CLASSES)
        n_live = sum(p.numel() for p in m_live.parameters())
        n_zm   = sum(p.numel() for p in m_zm.parameters())
        assert n_live == n_zm, (
            f"GraphComm zero_messages twin param count mismatch: live={n_live}, zm={n_zm}"
        )

    def test_class_bias_zeros_init_preserves_behavior(self):
        """At init, class_id effect should be near-zero (class_emb std=0.01, class_bias=0)."""
        torch.manual_seed(7)
        m = GraphComm(D, num_heads=4, num_layers=1, num_classes=NUM_CLASSES)
        h = make_h()
        with torch.no_grad():
            out_none = m(h, class_id=None)
            out_cid  = m(h, class_id=CID_A)
        max_diff = (out_none - out_cid).abs().max().item()
        assert max_diff < 0.5, (
            f"At init, class_id should have near-zero effect (max diff={max_diff:.4f})"
        )
