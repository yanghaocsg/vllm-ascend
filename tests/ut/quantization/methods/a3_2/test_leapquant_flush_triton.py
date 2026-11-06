# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused triton flush kernel vs the PR1 torch reference.

The kernel must reproduce ``leap_window_flush`` written back into the state
pool: identical reconstruction up to (a) fp32 summation-order noise in the
subspace iteration and (b) at most a few +-1 int8 code steps at quantization
ties.  The plan's acceptance bound is an error ratio
``||K - S|| / ||torch - S|| <= 1.01``; here we additionally pin a tight
elementwise bound so drift in the fused math is caught early.

Run on an Atlas A3 NPU (directory suffix routes to a3_2 CI runners).
"""

import unittest

import torch

from vllm_ascend.ops.triton.leapquant import leapquant_flush_triton
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
    out = pool.clone()
    flushed = leap_window_flush(pool[slot_ids].clone(), compensator_tokens=rank)
    out[slot_ids.to(torch.int64)] = flushed.to(pool.dtype)
    return out


class TestLeapQuantFlushTriton(unittest.TestCase):
    def _check(self, pool, slot_ids, rank=4, skip_ratio=False):
        torch.manual_seed(1234)
        reference = _reference_flush(pool, slot_ids, rank)

        actual = pool.clone()
        leapquant_flush_triton(actual, slot_ids, compensator_tokens=rank)
        torch.npu.synchronize()

        scale = pool[slot_ids.to(torch.int64)].abs().mean().item() + 1.0
        step = _code_step_bound(pool, slot_ids, rank)
        max_abs = (actual - reference).abs().max().item()
        # Adjacent codes differ by (s_k*s_v)/127; allow two steps for
        # subspace-iteration ulp differences flipping a tie.
        self.assertLess(max_abs, 2.0 * step + 1e-5 * scale, f"elementwise drift too large: {max_abs:.3e}")

        rows = slot_ids.to(torch.int64)
        orig = pool[rows]
        err_kernel = (actual[rows] - reference[rows]).norm().item()
        err_total = (reference[rows] - orig).norm().item() + 1e-30
        # Not bit-exact vs torch (different accumulation orders flip a
        # small fraction of rounding ties), but the aggregate divergence
        # stays well below the quantization error itself.
        if not skip_ratio:
            self.assertLess(err_kernel / err_total, 0.5, "kernel vs torch divergence vs quant error")

        # untouched rows must stay bit-identical
        mask = torch.ones(pool.shape[0], dtype=torch.bool)
        mask[slot_ids.to(torch.int64)] = False
        self.assertTrue(torch.equal(actual[mask], pool[mask]))

        # deterministic across launches (fixed q0, no atomics)
        again = pool.clone()
        leapquant_flush_triton(again, slot_ids, compensator_tokens=rank)
        torch.npu.synchronize()
        self.assertTrue(torch.equal(again[slot_ids.to(torch.int64)], actual[slot_ids.to(torch.int64)]))
        return reference, actual

    def test_gdn_head_shape(self):
        """Qwen3.5/Kimi GDN head: [V, K] = [128, 128]."""
        torch.manual_seed(7)
        pool = torch.randn(8, 4, 128, 128, device=DEVICE, dtype=torch.float32)
        slot_ids = torch.tensor([1, 4, 6], device=DEVICE, dtype=torch.int64)
        self._check(pool, slot_ids)

    def test_rectangular_and_rank2(self):
        torch.manual_seed(8)
        pool = torch.randn(4, 2, 64, 256, device=DEVICE, dtype=torch.float32) * 3.0
        slot_ids = torch.tensor([0, 3], device=DEVICE, dtype=torch.int32)
        self._check(pool, slot_ids, rank=2)

    def test_padded_batch_stride(self):
        """State pools may carry physical page padding in stride(0)."""
        torch.manual_seed(9)
        raw = torch.zeros(16, 4, 128, 128, device=DEVICE, dtype=torch.float32)
        pool = raw[:8]  # stride(0) still spans 16 slots
        pool.copy_(torch.randn(8, 4, 128, 128, device=DEVICE, dtype=torch.float32))
        slot_ids = torch.tensor([2, 5], device=DEVICE, dtype=torch.int64)
        self._check(pool, slot_ids)

    def test_zero_and_tiny_rows(self):
        """Zero states flush to zero; tiny-magnitude rows stay finite."""
        pool = torch.zeros(3, 2, 128, 128, device=DEVICE, dtype=torch.float32)
        pool[0] = 1e-9 * torch.randn_like(pool[0])
        slot_ids = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int64)
        # relative comparisons are meaningless at 1e-9 magnitudes; the
        # elementwise code-step bound plus the zero-row checks below carry
        # the correctness signal for degenerate blocks.
        _, actual = self._check(pool, slot_ids, skip_ratio=True)
        self.assertEqual(actual[1].abs().max().item(), 0.0)
        self.assertTrue(torch.isfinite(actual[0]).all())

    def test_error_ratio_acceptance(self):
        """Plan acceptance: quantization error of the fused kernel within
        1.01x of the torch reference round-trip error."""
        torch.manual_seed(11)
        pool = torch.randn(4, 16, 128, 128, device=DEVICE, dtype=torch.float32)
        slot_ids = torch.arange(4, device=DEVICE, dtype=torch.int64)
        reference = _reference_flush(pool, slot_ids, 4)
        actual = pool.clone()
        leapquant_flush_triton(actual, slot_ids, compensator_tokens=4)
        torch.npu.synchronize()
        orig = pool[slot_ids]
        e_kernel = (actual[slot_ids] - orig).norm().item()
        e_torch = (reference[slot_ids] - orig).norm().item()
        self.assertLess(e_kernel / e_torch, 1.01)


if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    unittest.main()
