# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""recurrent_kda_leapquant kernel vs a pure-torch fp64 reference.

The KDA recurrence differs from GDN in exactly one line — the decay is a
per-K-column vector ``g [B, T, HV, K]`` instead of a per-head scalar
(``(I − βkk^T)Diag(α)S + βkv^T`` ≡ ``Diag(α)S + βk(v − S^T(α⊙k))^T``,
paper appendix A) — so this suite pins the same contracts as the GDN one
(test_leapquant_recurrent.py) against a per-K-decay reference:

* steady state (``w_in``): the fp32 W workspace row holds the running state;
* cold row (``payload``): the kernel reconstructs
  ``S0 = q^T u + codes * (s_k/127) * s_v`` (eq. 8-9) in-tile from the M3.2
  flush encoder's checkpoint;
* NULL rows/slots (sentinel ``-1``) start from a zero state and skip the
  workspace write;
* padded (non-contiguous row stride) W pools;
* the host-side gate expansion ``kda_expand_gate`` against the fla formula
  (softplus + lower_bound safe variant, both fp32-exact vs manual math).

Reference (fp64 accumulation; Kimi-Linear is H == HV):

    S *= exp(g)[None, :]; v~ = v - (S k); v~ *= beta; S += v~ k^T; o = (S q) * scale

Run on an Atlas A3 NPU (directory suffix routes to a3_2 CI runners).
"""

import unittest

import torch

from vllm_ascend.ops.triton.leapquant import (
    kda_expand_gate,
    leapquant_flush_triton,
    recurrent_kda_leapquant,
)
from vllm_ascend.quantization.methods.mamba_state.leapquant.payload import (
    LeapQuantPayloadPool,
)

DEVICE = torch.device("npu:0")
RANK = 4


def _reference(s0, q, k, v, g, beta, scale):
    """s0 [B, HV, V, K] fp32; q/k [B,T,H,K]; v [B,T,HV,V] -> (o fp32, s_final fp32)."""
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    rep = HV // H
    s = s0.double().clone()
    o = torch.empty(B, T, HV, V, dtype=torch.float32, device=s0.device)
    for t in range(T):
        s *= torch.exp(g[:, t].double())[:, :, None, :]  # per-K-column decay
        kt = k[:, t].double().repeat_interleave(rep, dim=1)  # [B, HV, K]
        qt = q[:, t].double().repeat_interleave(rep, dim=1)
        vt = v[:, t].double()
        bt = beta[:, t].double()[:, :, None]
        sk = torch.einsum("bhvk,bhk->bhv", s, kt)
        vtilde = (vt - sk) * bt
        s = s + vtilde[:, :, :, None] * kt[:, :, None, :]
        ot = torch.einsum("bhvk,bhk->bhv", s, qt)
        o[:, t] = (ot * scale).float()
    return o, s.float()


def _l2norm(x):
    """Host l2norm matching fla's l2norm_fwd (eps 1e-6 added to the square)."""
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


class TestRecurrentKDALeapQuant(unittest.TestCase):
    def _make_case(self, B, T, H, HV, V, K):
        torch.manual_seed(31)
        scale = K**-0.5
        q = _l2norm(torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32))
        k = _l2norm(torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32))
        v = torch.randn(B, T, HV, V, device=DEVICE, dtype=torch.float32)
        # log-space per-K decay, mildly negative like a real KDA gate
        g = -0.05 * torch.rand(B, T, HV, K, device=DEVICE, dtype=torch.float32) - 0.01
        beta = torch.rand(B, T, HV, device=DEVICE, dtype=torch.float32)
        s0 = 0.3 * torch.randn(B, HV, V, K, device=DEVICE, dtype=torch.float32)
        return q, k, v, g, beta, s0, scale

    def _check(self, B, T, H, HV, V, K):
        q, k, v, g, beta, s0, scale = self._make_case(B, T, H, HV, V, K)

        # ---- steady path: W rows seeded with the exact fp32 s0 ----------
        w = torch.zeros(B * 2, HV, V, K, device=DEVICE, dtype=torch.float32)
        w_rows = torch.arange(B, device=DEVICE, dtype=torch.int64)
        w[w_rows] = s0
        o = recurrent_kda_leapquant(
            q, k, v, g, beta, scale=scale, w_in=w, w_out=w, w_rows=w_rows
        )
        torch.npu.synchronize()
        ref_o, ref_s = _reference(s0, q, k, v, g, beta, scale)
        self.assertLess((o - ref_o).abs().max().item(), 2e-4, "steady: output drift")
        self.assertLess((w[w_rows] - ref_s).abs().max().item(), 2e-4, "steady: state drift")
        # non-assigned workspace rows stay zero
        self.assertEqual(w[B:].abs().max().item(), 0.0)

        # ---- cold path: payload checkpoint -> in-kernel reconstruct ------
        pool = LeapQuantPayloadPool.allocate(B, HV, V, K, RANK, device=DEVICE)
        slot_ids = torch.arange(B, device=DEVICE, dtype=torch.int64)
        leapquant_flush_triton(
            s0.clone(), slot_ids, compensator_tokens=RANK,
            payload=pool, rewrite_state=False,
        )
        torch.npu.synchronize()
        s0_recon = pool.reconstruct_rows(slot_ids)  # [B, HV, V, K] fp32
        # the flush dual-write check: reconstruction close to the original
        self.assertLess((s0_recon - s0).norm().item() / (s0.norm().item() + 1e-30), 0.02)

        w2 = torch.zeros(B, HV, V, K, device=DEVICE, dtype=torch.float32)
        w_rows2 = torch.arange(B, device=DEVICE, dtype=torch.int64)
        o2 = recurrent_kda_leapquant(
            q, k, v, g, beta, scale=scale,
            w_out=w2, w_rows=w_rows2, payload=pool, slot_ids=slot_ids,
        )
        torch.npu.synchronize()
        ref_o2, ref_s2 = _reference(s0_recon, q, k, v, g, beta, scale)
        # looser than steady: the kernel reconstructs S0 in-tile (rank-by-rank
        # fp32 accumulation) while the reference starts from torch's fused
        # reconstruction of the same fp16 planes — fp32 rounding-order noise.
        self.assertLess((o2 - ref_o2).abs().max().item(), 4e-4, "payload-init: output drift")
        self.assertLess((w2 - ref_s2).abs().max().item(), 4e-4, "payload-init: state drift")
        # cold and steady runs must agree up to the checkpoint round-trip:
        # (o2 vs ref_o) differs only by the flush quantization error, already
        # bounded above; pin the relative gap explicitly.
        self.assertLess(
            (o2 - ref_o).norm().item() / (ref_o.norm().item() + 1e-30), 0.02,
            "payload-init vs exact-start divergence beyond flush error",
        )

    def test_decode_step_kda_geometry(self):
        """Kimi-Linear head geometry H=HV=4, V=K=128, single decode token."""
        self._check(B=4, T=1, H=4, HV=4, V=128, K=128)

    def test_multi_token(self):
        """Multi-token decode rows (T>1) on a smaller geometry."""
        self._check(B=2, T=3, H=2, HV=2, V=64, K=64)

    def test_padded_workspace_rows(self):
        """W rows living in a padded (non-contiguous row stride) pool."""
        torch.manual_seed(32)
        B, T, H, HV, V, K = 3, 1, 2, 2, 64, 64
        scale = K**-0.5
        q = _l2norm(torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32))
        k = _l2norm(torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32))
        v = torch.randn(B, T, HV, V, device=DEVICE, dtype=torch.float32)
        g = -0.05 * torch.rand(B, T, HV, K, device=DEVICE, dtype=torch.float32) - 0.01
        beta = torch.rand(B, T, HV, device=DEVICE, dtype=torch.float32)
        s0 = 0.3 * torch.randn(B, HV, V, K, device=DEVICE, dtype=torch.float32)

        wide = torch.zeros(B * 2, HV, V, K, device=DEVICE, dtype=torch.float32)
        w = wide[::2]  # row stride spans B*2 rows
        w_rows = torch.arange(B, device=DEVICE, dtype=torch.int64)
        w[w_rows] = s0
        o = recurrent_kda_leapquant(
            q, k, v, g, beta, scale=scale, w_in=w, w_out=w, w_rows=w_rows
        )
        torch.npu.synchronize()
        ref_o, ref_s = _reference(s0, q, k, v, g, beta, scale)
        self.assertLess((o - ref_o).abs().max().item(), 2e-4)
        self.assertLess((w[w_rows] - ref_s).abs().max().item(), 2e-4)

    def test_null_slot_starts_from_zero(self):
        """slot_id < 0 (NULL) starts the recurrence from a zero state."""
        torch.manual_seed(33)
        B, T, H, HV, V, K = 2, 1, 2, 2, 64, 64
        scale = K**-0.5
        q = _l2norm(torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32))
        k = _l2norm(torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32))
        v = torch.randn(B, T, HV, V, device=DEVICE, dtype=torch.float32)
        g = -0.05 * torch.rand(B, T, HV, K, device=DEVICE, dtype=torch.float32) - 0.01
        beta = torch.rand(B, T, HV, device=DEVICE, dtype=torch.float32)

        pool = LeapQuantPayloadPool.allocate(2, HV, V, K, RANK, device=DEVICE)
        # -1 = NULL (unmapped request): must take the zero-state branch, not
        # read the payload at a bogus offset.
        slot_ids = torch.tensor([-1, 1], device=DEVICE, dtype=torch.int64)
        s1 = 0.3 * torch.randn(1, HV, V, K, device=DEVICE, dtype=torch.float32)
        flush_state = torch.zeros(2, HV, V, K, device=DEVICE, dtype=torch.float32)
        flush_state[1] = s1[0]
        leapquant_flush_triton(
            flush_state, torch.tensor([1], device=DEVICE, dtype=torch.int64),
            compensator_tokens=RANK, payload=pool, rewrite_state=False,
        )
        torch.npu.synchronize()

        w = torch.zeros(B, HV, V, K, device=DEVICE, dtype=torch.float32)
        o = recurrent_kda_leapquant(
            q, k, v, g, beta, scale=scale,
            w_out=w, w_rows=torch.arange(B, device=DEVICE, dtype=torch.int64),
            payload=pool, slot_ids=slot_ids,
        )
        torch.npu.synchronize()
        zero_init = torch.zeros(1, HV, V, K, device=DEVICE, dtype=torch.float32)
        full = torch.cat([zero_init, pool.reconstruct_rows(slot_ids[1:])], 0)
        ref_o, ref_s = _reference(full, q, k, v, g, beta, scale)
        self.assertLess((o - ref_o).abs().max().item(), 2e-4)
        self.assertLess((w - ref_s).abs().max().item(), 2e-4)

    def test_null_row_steady_path(self):
        """w_row = -1 (NULL request): zero state in, no workspace write."""
        torch.manual_seed(34)
        B, T, H, HV, V, K = 2, 1, 2, 2, 64, 64
        scale = K**-0.5
        q = _l2norm(torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32))
        k = _l2norm(torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32))
        v = torch.randn(B, T, HV, V, device=DEVICE, dtype=torch.float32)
        g = -0.05 * torch.rand(B, T, HV, K, device=DEVICE, dtype=torch.float32) - 0.01
        beta = torch.rand(B, T, HV, device=DEVICE, dtype=torch.float32)
        s0 = 0.3 * torch.randn(1, HV, V, K, device=DEVICE, dtype=torch.float32)

        w = torch.zeros(2, HV, V, K, device=DEVICE, dtype=torch.float32)
        w[1] = s0[0]
        rows = torch.tensor([-1, 1], device=DEVICE, dtype=torch.int64)
        o = recurrent_kda_leapquant(
            q, k, v, g, beta, scale=scale, w_in=w, w_out=w, w_rows=rows
        )
        torch.npu.synchronize()
        ref_o, ref_s = _reference(
            torch.cat([torch.zeros(1, HV, V, K, device=DEVICE), s0]), q, k, v, g, beta, scale
        )
        self.assertLess((o - ref_o).abs().max().item(), 2e-4)
        # row -1 never written (row 0 untouched = still zero), row 1 updated
        self.assertEqual(w[0].abs().max().item(), 0.0)
        self.assertLess((w[1] - ref_s[1]).abs().max().item(), 2e-4)

    def test_kda_expand_gate_formula(self):
        """Host gate expansion matches the fla in-kernel formula exactly."""
        torch.manual_seed(35)
        B, T, H, K = 2, 3, 4, 16
        a = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.float32) * 3.0
        a_log = torch.randn(H, device=DEVICE, dtype=torch.float32)
        dtb = torch.randn(H, K, device=DEVICE, dtype=torch.float32)

        x = a + dtb.view(1, 1, H, K)
        softplus = torch.where(
            x <= 20.0, torch.log1p(torch.exp(x)), x
        )  # fla's stable form, log1p variant
        manual = -torch.exp(a_log).view(1, 1, H, 1) * softplus
        g = kda_expand_gate(a, a_log, dtb)
        torch.npu.synchronize()
        self.assertLess((g - manual).abs().max().item(), 1e-6, "softplus variant")

        lb = -3.0
        manual_safe = lb * torch.sigmoid(torch.exp(a_log).view(1, 1, H, 1) * x)
        g_safe = kda_expand_gate(a, a_log, dtb, lower_bound=lb)
        torch.npu.synchronize()
        self.assertLess((g_safe - manual_safe).abs().max().item(), 1e-6, "safe variant")
        # both variants stay non-positive (decay semantics)
        self.assertLessEqual(g.max().item(), 0.0)
        self.assertLessEqual(g_safe.max().item(), 0.0)
        self.assertGreaterEqual(g_safe.min().item(), lb, "safe gate bounded below")

    def test_validation(self):
        q, k, v, g, beta, s0, scale = self._make_case(1, 1, 2, 2, 64, 64)
        with self.assertRaises(ValueError):
            recurrent_kda_leapquant(q, k, v, g, beta)  # w_rows required
        rows = torch.zeros(1, device=DEVICE, dtype=torch.int64)
        with self.assertRaises(ValueError):
            # cold row (no w_in) without payload + slot_ids
            recurrent_kda_leapquant(q, k, v, g, beta, w_rows=rows)
        pool = LeapQuantPayloadPool.allocate(1, 2, 32, 64, RANK, device=DEVICE)
        with self.assertRaises(ValueError):
            # geometry mismatch
            recurrent_kda_leapquant(
                q, k, v, g, beta, w_rows=rows, payload=pool,
                slot_ids=torch.zeros(1, device=DEVICE, dtype=torch.int64),
            )
        with self.assertRaises(ValueError):
            # scalar GDN-shaped gate is rejected (needs [B, T, HV, K])
            recurrent_kda_leapquant(
                q, k, v, g[:, :, :, 0], beta, w_in=s0, w_out=s0, w_rows=rows
            )
        gva_v = torch.randn(1, 1, 4, 32, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(ValueError):
            # GVA (H != HV) is rejected until the gate head mapping exists
            recurrent_kda_leapquant(
                q, k, gva_v, g.repeat(1, 1, 2, 1), beta,
                w_in=torch.zeros(1, 4, 32, 64, device=DEVICE), w_rows=rows,
            )


if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    unittest.main()
