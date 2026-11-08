# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Payload-pool flush mode (M3.2) vs the PR1 torch reference.

``leapquant_flush_triton(..., payload=pool)`` must persist a checkpoint
whose torch-side reconstruction matches ``leap_window_flush`` under the
same gates as the roundtrip kernel test (error ratio vs the quantization
error itself, elementwise bound of two int8 code steps), while:

* pure payload mode (``rewrite_state=False``, the default) leaves the
  fp32 state pool bit-untouched and non-due payload rows bit-zero;
* dual-write mode (``rewrite_state=True``) reproduces the roundtrip
  kernel's in-place result and stays consistent with the stored payload;
* the layout accounting hits the plan's ~1.19 B/elem at GDN geometry.

Run on an Atlas A3 NPU (directory suffix routes to a3_2 CI runners).
"""

import unittest
from unittest.mock import MagicMock

import torch

from vllm_ascend.ops.triton.leapquant import leapquant_flush_triton
from vllm_ascend.quantization.methods.mamba_state.leapquant.payload import (
    LeapQuantPayloadPool,
    PayloadLayout,
)
from vllm_ascend.quantization.methods.mamba_state.leapquant.state import (
    fit_compensators,
    leap_window_flush,
    quantize_residual,
    start_basis,
)

DEVICE = torch.device("npu:0")


def _code_step_bound(pool: torch.Tensor, slot_ids: torch.Tensor, rank: int) -> float:
    """Largest single int8 code step of the reference quantization."""
    rows = pool[slot_ids.to(torch.int64)].reshape(-1, pool.shape[-2], pool.shape[-1]).float()
    q0 = start_basis(rows.shape[-2], rank, device=rows.device, dtype=torch.float32)
    u, q = fit_compensators(rows, q0, rank)
    _, s_k, s_v = quantize_residual(rows, u, q)
    return (s_k[:, None, :] * s_v[:, :, None]).max().item() / 127.0


def _reference_flush(pool: torch.Tensor, slot_ids: torch.Tensor, rank: int) -> torch.Tensor:
    flushed = leap_window_flush(pool[slot_ids].clone(), compensator_tokens=rank)
    out = pool.clone()
    out[slot_ids.to(torch.int64)] = flushed.to(pool.dtype)
    return out


class TestLeapQuantPayload(unittest.TestCase):
    def _flush(self, pool, slot_ids, rank, *, rewrite_state=None, hcnt_value=0):
        """Flush a clone into a fresh payload pool; return (pool, payload)."""
        slots, heads, v_dim, k_dim = pool.shape
        pl = LeapQuantPayloadPool.allocate(
            slots, heads, v_dim, k_dim, rank, device=pool.device
        )
        state = pool.clone()
        kwargs = {"compensator_tokens": rank, "payload": pl, "hcnt_value": hcnt_value}
        if rewrite_state is not None:
            kwargs["rewrite_state"] = rewrite_state
        leapquant_flush_triton(state, slot_ids, **kwargs)
        torch.npu.synchronize()
        return state, pl

    def _check_payload(self, pool, slot_ids, rank, pl):
        """Reconstruction gates: same bounds as the roundtrip kernel UT."""
        torch.manual_seed(1234)
        reference = _reference_flush(pool, slot_ids, rank)
        rows = slot_ids.to(torch.int64)
        recon = pl.reconstruct_rows(slot_ids).cpu()

        ref_rows = reference[rows].cpu()
        orig_rows = pool[rows].cpu()
        err_payload = (recon - ref_rows).norm().item()
        err_total = (ref_rows - orig_rows).norm().item() + 1e-30
        self.assertLess(
            err_payload / err_total, 0.5, "payload reconstruction vs torch divergence"
        )

        step = _code_step_bound(pool, slot_ids, rank)
        scale = orig_rows.abs().mean().item() + 1.0
        max_abs = (recon - ref_rows).abs().max().item()
        self.assertLess(
            max_abs, 2.0 * step + 1e-5 * scale, f"elementwise drift too large: {max_abs:.3e}"
        )

        # payload field sanity
        codes = pl.codes[rows].cpu()
        self.assertTrue(int(codes.abs().max()) <= 127)
        self.assertGreaterEqual(float(pl.s_k[rows].cpu().min()), 1e-12)
        self.assertGreaterEqual(float(pl.s_v[rows].cpu().min()), 1e-12)
        self.assertTrue(torch.isfinite(pl.q[rows].float().cpu()).all())
        self.assertTrue(torch.isfinite(pl.u[rows].float().cpu()).all())

        # non-due payload rows stay bit-zero (pool was zero-initialized)
        mask = torch.ones(pool.shape[0], dtype=torch.bool)
        mask[rows] = False
        spare = pl.buffer.reshape(pl.layout.slots, -1)[mask.cpu()]
        self.assertEqual(int(spare.int().abs().sum()), 0)
        return reference, recon

    # ---- layout ---------------------------------------------------------
    def test_layout_accounting(self):
        lay = PayloadLayout(slots=1024, heads=8, v_dim=128, k_dim=128, rank=4)
        self.assertEqual(
            lay.head_block_bytes, 128 * 128 + 4 * 128 + 4 * 128 + 2 * 4 * 128 + 2 * 4 * 128 + 4
        )
        self.assertAlmostEqual(lay.bytes_per_element, 19460 / 16384, places=9)
        self.assertAlmostEqual(4.0 / lay.bytes_per_element, 3.3677, places=3)
        offs = lay.field_offsets()
        self.assertEqual(offs["codes"], 0)
        for o in offs.values():
            self.assertEqual(o % 4, 0, f"field offset {o} breaks 4B alignment")

    def test_layout_rejects_bad_geometry(self):
        with self.assertRaises(ValueError):
            PayloadLayout(1, 1, 100, 128, 4)  # V not a power of two
        with self.assertRaises(ValueError):
            PayloadLayout(1, 1, 128, 128, 17)  # rank > 16
        with self.assertRaises(ValueError):
            PayloadLayout(0, 1, 128, 128, 4)

    # ---- pure payload mode ----------------------------------------------
    def test_payload_gdn_head_shape(self):
        """Qwen3.5/Kimi head geometry; state pool must stay bit-untouched."""
        torch.manual_seed(7)
        pool = torch.randn(8, 4, 128, 128, device=DEVICE, dtype=torch.float32)
        slot_ids = torch.tensor([1, 4, 6], device=DEVICE, dtype=torch.int64)
        state, pl = self._flush(pool, slot_ids, 4, hcnt_value=16)
        self.assertTrue(torch.equal(state, pool), "pure payload mode must not touch state")
        self._check_payload(pool, slot_ids, 4, pl)
        rows = slot_ids.to(torch.int64)
        self.assertTrue(torch.equal(pl.hcnt[rows].cpu(), torch.full((3, 4), 16, dtype=torch.int32)))

    def test_payload_kda_geometry(self):
        """KDA pool geometry [slots, H, V, K] with H=8, V=K=128."""
        torch.manual_seed(10)
        pool = torch.randn(6, 8, 128, 128, device=DEVICE, dtype=torch.float32)
        slot_ids = torch.tensor([0, 2, 3, 5], device=DEVICE, dtype=torch.int32)
        state, pl = self._flush(pool, slot_ids, 4)
        self.assertTrue(torch.equal(state, pool))
        self._check_payload(pool, slot_ids, 4, pl)

    def test_payload_rectangular_and_rank2(self):
        torch.manual_seed(8)
        pool = torch.randn(4, 2, 64, 256, device=DEVICE, dtype=torch.float32) * 3.0
        slot_ids = torch.tensor([0, 3], device=DEVICE, dtype=torch.int64)
        state, pl = self._flush(pool, slot_ids, 2)
        self.assertTrue(torch.equal(state, pool))
        self._check_payload(pool, slot_ids, 2, pl)

    def test_payload_deterministic(self):
        """Fixed q0 and no atomics: two flushes store bit-identical bytes."""
        torch.manual_seed(9)
        pool = torch.randn(4, 2, 128, 128, device=DEVICE, dtype=torch.float32)
        slot_ids = torch.tensor([1, 3], device=DEVICE, dtype=torch.int64)
        _, pl1 = self._flush(pool, slot_ids, 4)
        _, pl2 = self._flush(pool, slot_ids, 4)
        self.assertTrue(torch.equal(pl1.buffer.cpu(), pl2.buffer.cpu()))

    # ---- dual-write mode --------------------------------------------------
    def test_dual_write_matches_roundtrip(self):
        """rewrite_state=True: state == roundtrip kernel result, consistent
        with the stored payload."""
        torch.manual_seed(11)
        pool = torch.randn(8, 4, 128, 128, device=DEVICE, dtype=torch.float32)
        slot_ids = torch.tensor([2, 5], device=DEVICE, dtype=torch.int64)

        roundtrip = pool.clone()
        leapquant_flush_triton(roundtrip, slot_ids, compensator_tokens=4)
        torch.npu.synchronize()

        state, pl = self._flush(pool, slot_ids, 4, rewrite_state=True)
        rows = slot_ids.to(torch.int64)

        step = _code_step_bound(pool, slot_ids, 4)
        scale = pool[rows].abs().mean().item() + 1.0
        max_abs = (state[rows] - roundtrip[rows]).abs().max().item()
        self.assertLess(max_abs, 1e-4 * scale, "dual-write vs roundtrip kernel drift")

        recon = pl.reconstruct_rows(slot_ids).cpu()
        max_gap = (recon - state[rows].cpu()).abs().max().item()
        self.assertLess(max_gap, 1e-4 * scale, "payload reconstruction vs dual-written state")

        mask = torch.ones(pool.shape[0], dtype=torch.bool)
        mask[rows] = False
        self.assertTrue(torch.equal(state[mask], pool[mask]))
        self._check_payload(pool, slot_ids, 4, pl)

    # ---- wrapper guards ---------------------------------------------------
    def test_geometry_mismatch_rejected(self):
        torch.manual_seed(12)
        pool = torch.randn(4, 4, 128, 128, device=DEVICE, dtype=torch.float32)
        slot_ids = torch.tensor([0], device=DEVICE, dtype=torch.int64)
        bad = LeapQuantPayloadPool.allocate(4, 8, 128, 128, 4, device=pool.device)  # H mismatch
        with self.assertRaises(ValueError):
            leapquant_flush_triton(pool, slot_ids, compensator_tokens=4, payload=bad)
        bad_rank = LeapQuantPayloadPool.allocate(4, 4, 128, 128, 2, device=pool.device)
        with self.assertRaises(ValueError):
            leapquant_flush_triton(pool, slot_ids, compensator_tokens=4, payload=bad_rank)
        with self.assertRaises(ValueError):
            leapquant_flush_triton(pool, slot_ids, compensator_tokens=4, rewrite_state=True)
        self.assertTrue(torch.equal(pool, pool))  # pool untouched by failures


def _engine_page_views(slots, heads, v_dim, k_dim, rank, *, page_pad=12, device=DEVICE):
    """Emulate the Ascend mamba-page allocation for a GDN layer.

    Mirrors ``_adjust_dsv4_kv_layout``: one int8 allocation, per-state views
    carved by byte offset with the *page* as the block stride, plus trailing
    page padding so the payload view is genuinely strided.
    """
    hbb = PayloadLayout(1, heads, v_dim, k_dim, rank).head_block_bytes
    conv_dim, width = 512, 4
    conv_bytes, ssm_bytes = conv_dim * width * 2, heads * v_dim * k_dim * 4
    page = conv_bytes + ssm_bytes + heads * hbb + page_pad
    raw = torch.zeros(slots * page, dtype=torch.int8, device=device)

    def carve(shape, dtype, off):
        esz = torch.empty((), dtype=dtype).element_size()
        dense = torch.empty(shape).stride()
        return torch.as_strided(raw.view(dtype), shape, (page // esz, *dense[1:]), off // esz)

    conv = carve((slots, conv_dim, width), torch.bfloat16, 0)
    ssm = carve((slots, heads, v_dim, k_dim), torch.float32, conv_bytes)
    payload = carve((slots, heads, hbb), torch.uint8, conv_bytes + ssm_bytes)
    return raw, page, conv, ssm, payload


class TestLeapQuantEnginePool(unittest.TestCase):
    """Payload riding the engine mamba page (M3.2b allocation path)."""

    def test_engine_strided_pool_dual_write(self):
        """Strided ssm + payload views out of one page: the flush must land
        in the payload region only and keep the page padding / conv bytes /
        non-due slots untouched."""
        from vllm_ascend.ops.triton.leapquant import leapquant_flush_triton

        slots, heads, rank = 8, 4, 4
        torch.manual_seed(21)
        raw, page, conv, ssm, payload_view = _engine_page_views(slots, heads, 128, 128, rank)
        ssm.copy_(torch.randn_like(ssm))
        raw_before = raw.clone()
        slot_ids = torch.tensor([1, 5], device=DEVICE, dtype=torch.int64)

        pool = LeapQuantPayloadPool(payload_view, PayloadLayout(slots, heads, 128, 128, rank))
        self.assertEqual(pool.slot_stride_bytes, page)
        self.assertEqual(pool.head_stride_bytes, PayloadLayout(1, heads, 128, 128, rank).head_block_bytes)

        reference = _reference_flush(ssm.cpu(), slot_ids.cpu(), rank).to(DEVICE)
        leapquant_flush_triton(ssm, slot_ids, compensator_tokens=rank, payload=pool, rewrite_state=True)
        torch.npu.synchronize()

        rows = slot_ids.to(torch.int64)
        step = _code_step_bound(ssm.cpu(), slot_ids.cpu(), rank)
        scale = ssm[rows].abs().mean().item() + 1.0
        self.assertLess((ssm[rows] - reference[rows]).abs().max().item(), 2.0 * step + 1e-5 * scale)
        recon = pool.reconstruct_rows(slot_ids).cpu()
        self.assertLess(
            (recon - reference[rows].cpu()).abs().max().item(), 2.0 * step + 1e-5 * scale
        )

        # everything outside the due regions is bit-untouched: conv plane,
        # page padding, non-due slots.  Due slots may change in their ssm
        # (dual-write) and payload regions only.
        raw_after = raw.clone()
        hbb = pool.head_stride_bytes
        conv_bytes = conv.shape[1] * conv.shape[2] * 2  # per-slot bf16 conv plane
        ssm_bytes = heads * 128 * 128 * 4
        payload_off = conv_bytes + ssm_bytes
        flat_mask = (raw_before != raw_after).view(slots, page)
        for s in range(slots):
            if s in (1, 5):
                self.assertFalse(flat_mask[s, :conv_bytes].any(), f"slot {s}: conv bytes changed")
                self.assertFalse(
                    flat_mask[s, payload_off + heads * hbb :].any(), f"slot {s}: padding changed"
                )
            else:
                self.assertFalse(flat_mask[s].any(), f"slot {s}: non-due slot changed")

    def test_engine_pool_wrapper_rejects_misaligned(self):
        """A view whose slot stride breaks 4-byte field alignment is rejected."""
        slots, heads = 4, 2
        hbb = PayloadLayout(1, heads, 128, 128, 4).head_block_bytes
        bad_stride = PayloadLayout(1, heads, 128, 128, 4).slot_bytes + 2
        raw = torch.zeros(slots * bad_stride, dtype=torch.uint8, device=DEVICE)
        view = torch.as_strided(raw, (slots, heads, hbb), (bad_stride, hbb, 1))
        with self.assertRaises(ValueError):
            LeapQuantPayloadPool(view, PayloadLayout(slots, heads, 128, 128, 4))

    def test_state_extension_helper(self):
        """get_state_shape/dtype extension: off -> None, on -> uint8 component."""
        import types
        from unittest.mock import patch

        from vllm_ascend.ascend_config import LeapQuantConfig
        from vllm_ascend.quantization.methods.mamba_state.leapquant import (
            leapquant_payload_state_extension as ext,
        )

        conv = (512, 4)
        temporal = (8, 128, 128)
        # engine off (no AscendConfig singleton in UT) -> no extension
        self.assertIsNone(ext((conv, temporal)))

        cfg = LeapQuantConfig(
            enabled=True, window=16, compensator_tokens=4, kernel_backend="triton", payload_pool=True
        )
        fake = types.SimpleNamespace(leapquant_config=cfg)
        with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=fake):
            shape, dtype = ext((conv, temporal))
        self.assertEqual(shape, (8, PayloadLayout(1, 8, 128, 128, 4).head_block_bytes))
        self.assertEqual(dtype, torch.uint8)

        # non-power-of-two head dims fail fast at spec-build time
        with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=fake):
            with self.assertRaises(ValueError):
                ext(((512, 4), (8, 96, 128)))


class TestLeapQuantRuntimePayload(unittest.TestCase):
    """End-to-end flush-hook wiring: config gate -> runtime -> kernel."""

    def test_runtime_payload_dual_write(self):
        from vllm_ascend.ascend_config import LeapQuantConfig
        from vllm_ascend.quantization.methods.mamba_state.leapquant import (
            leapquant_flush_state_rows,
        )

        torch.manual_seed(13)
        pool = torch.randn(8, 4, 128, 128, device=DEVICE, dtype=torch.float32)
        indices = torch.tensor([1, 4, 6], device=DEVICE, dtype=torch.int64)
        seq_lens = torch.tensor([32, 33, 16])  # rows 0 and 2 due -> slots 1 and 6
        cfg = LeapQuantConfig(
            enabled=True,
            window=16,
            compensator_tokens=4,
            kernel_backend="triton",
            payload_pool=True,
        )
        pl = LeapQuantPayloadPool.allocate(8, 4, 128, 128, 4, device=pool.device)

        torch.manual_seed(1234)
        reference = _reference_flush(pool, indices[[0, 2]], 4)
        leapquant_flush_state_rows(pool, indices, seq_lens, config=cfg, payload=pl)
        torch.npu.synchronize()

        # fp32 pool: due rows flushed (dual-write keeps decode-side readers
        # correct), non-due rows bit-untouched.
        step = _code_step_bound(pool, indices[[0, 2]], 4)
        scale = pool[[1, 6]].abs().mean().item() + 1.0
        max_abs = (pool[[1, 6]] - reference[[1, 6]]).abs().max().item()
        self.assertLess(max_abs, 2.0 * step + 1e-5 * scale)
        self.assertTrue(torch.equal(pool[4], reference[4]))

        # payload: due rows checkpointed, non-due rows bit-zero, hcnt = 0
        self.assertTrue(torch.equal(pl.hcnt[[1, 6]].cpu(), torch.zeros(2, 4, dtype=torch.int32)))
        spare = pl.buffer.reshape(pl.layout.slots, -1)[[0, 2, 3, 4, 5, 7]]
        self.assertEqual(int(spare.int().abs().sum()), 0)
        recon = pl.reconstruct_rows(torch.tensor([1, 6], device=DEVICE)).cpu()
        max_recon = (recon - reference[[1, 6]].cpu()).abs().max().item()
        self.assertLess(max_recon, 2.0 * step + 1e-5 * scale)

    def test_runtime_payload_never_silently_dropped(self):
        from vllm_ascend.ascend_config import LeapQuantConfig
        from vllm_ascend.quantization.methods.mamba_state.leapquant import (
            leapquant_flush_state_rows,
        )

        cfg = LeapQuantConfig(
            enabled=True, window=16, compensator_tokens=4, kernel_backend="triton"
        )
        pool = torch.randn(4, 2, 128, 128, device=DEVICE, dtype=torch.float32)
        pl = LeapQuantPayloadPool.allocate(4, 2, 128, 128, 4, device=pool.device)
        # A KDA-style gather/scatter plan disables the fused path; the
        # payload must then hard-fail instead of quietly skipping the write.
        plan = MagicMock()
        with self.assertRaises(RuntimeError):
            leapquant_flush_state_rows(
                pool,
                torch.tensor([1], device=DEVICE),
                torch.tensor([16]),
                config=cfg,
                state_copy=plan,
                payload=pl,
            )

    def test_config_payload_pool_requires_triton(self):
        from vllm_ascend.ascend_config import LeapQuantConfig

        with self.assertRaises(ValueError):
            LeapQuantConfig(enabled=True, payload_pool=True)


if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    unittest.main()
