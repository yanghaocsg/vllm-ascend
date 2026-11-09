# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical unit tests for the LeapQuant window-flush state operators.

These tests pin the paper's two headline properties on CPU:

1. Near-exact reconstruction when the state is dominated by a rank-``r``
   subspace (compensator tokens carry it losslessly out of int8).
2. Bounded error under a long recurrent stream versus diverging per-step
   int8 quantization (the "error accumulation" fix, paper Fig. 3).
"""

import unittest

import torch

from vllm_ascend.quantization.methods.mamba_state.leapquant import (
    crossed_boundary_mask,
    decode_flush_mask,
    fit_compensators,
    leap_window_flush,
    prefill_flush_mask,
    quantize_residual,
    start_basis,
)


def _absmax_int8(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().max() / 127.0
    return (x / scale).round().clamp(-127, 127) * scale


class TestLeapQuantState(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_reconstruction_error_random_state(self):
        state = torch.randn(4, 8, 128, 128) * 0.2
        flushed = leap_window_flush(state, compensator_tokens=4)
        rel = (flushed - state).norm() / state.norm()
        self.assertLess(rel.item(), 0.02)

    def test_codes_and_scales_contract(self):
        blocks = torch.randn(16, 128, 128) * 0.3
        basis = start_basis(128, 4, device=blocks.device)
        u, q = fit_compensators(blocks, basis, 4)
        self.assertEqual(u.dtype, torch.float16)
        self.assertEqual(q.dtype, torch.float16)
        self.assertEqual(tuple(u.shape), (16, 4, 128))
        self.assertEqual(tuple(q.shape), (16, 4, 128))
        codes, s_k, s_v = quantize_residual(blocks, u, q)
        self.assertEqual(codes.dtype, torch.int8)
        self.assertGreaterEqual(int(codes.min()), -127)
        self.assertLessEqual(int(codes.max()), 127)
        self.assertTrue(bool((s_k > 0).all()))
        self.assertTrue(bool((s_v > 0).all()))

    def test_low_rank_state_near_exact(self):
        basis = torch.randn(128, 4)
        low_rank = basis @ torch.randn(4, 128)
        state = low_rank.unsqueeze(0).unsqueeze(0).repeat(2, 8, 1, 1)
        flushed = leap_window_flush(state, compensator_tokens=4)
        rel = (flushed - state).norm() / state.norm()
        self.assertLess(rel.item(), 1e-4)

    def test_flush_deterministic(self):
        state = torch.randn(2, 4, 64, 64)
        first = leap_window_flush(state, compensator_tokens=4)
        second = leap_window_flush(state, compensator_tokens=4)
        self.assertTrue(torch.equal(first, second))

    def test_flush_preserves_shape_and_dtype(self):
        state = torch.randn(3, 5, 64, 48)
        flushed = leap_window_flush(state, compensator_tokens=4)
        self.assertEqual(flushed.shape, state.shape)
        self.assertEqual(flushed.dtype, state.dtype)

    def test_stream_error_bounded_versus_per_step(self):
        # GDN-like stream: S_t = beta * S_{t-1} + update. Per-step absmax
        # int8 diverges with T; LeapQuant (flush every 16 tokens) stays
        # bounded at a small multiple of the window-entry error.
        steps, window, beta = 128, 16, 0.99
        reference = torch.zeros(64, 64, dtype=torch.float64)
        leap = torch.zeros(64, 64)
        per_step = torch.zeros(64, 64)
        leap_err, step_err = [], []
        for t in range(steps):
            update = 0.5 * torch.randn(64, 64) * 0.05
            reference = beta * reference + update.double()
            leap = beta * leap + update
            per_step = _absmax_int8(beta * per_step + update)
            if (t + 1) % window == 0:
                leap = leap_window_flush(leap.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            leap_err.append((leap.double() - reference).abs().mean().item())
            step_err.append((per_step.double() - reference).abs().mean().item())
        # LeapQuant beats per-step quantization by a wide margin at the end.
        self.assertGreater(step_err[-1] / max(leap_err[-1], 1e-30), 3.0)
        # Its own error saturates: window flushes inject a fixed residual each
        # boundary, decaying geometrically, so doubling the stream length
        # multiplies the error by well under 2 (per-step quantization instead
        # tracks the unbounded fp64-reference gap).
        self.assertLess(leap_err[-1] / max(leap_err[steps // 2 - 1], 1e-30), 1.8)

    def test_window_masks(self):
        # decode: end = context + 1
        contexts = torch.tensor([0, 15, 31, 47, 12])
        self.assertEqual(
            decode_flush_mask(contexts, 16).tolist(),
            [False, True, True, True, False],
        )
        # prefill: end = context + query_len
        ctx = torch.tensor([0, 8, 40, 100])
        query = torch.tensor([64, 8, 8, 20])
        self.assertEqual(prefill_flush_mask(ctx, query, 16).tolist(), [True, True, True, False])
        # seq_len-based condition (used by the runtime) must agree: a request
        # with (context, query) has seq_len = context + query after the step.
        for c, q in zip(ctx.tolist(), query.tolist()):
            self.assertEqual((c + q) % 16 == 0, prefill_flush_mask(torch.tensor([c]), torch.tensor([q]), 16).item())

    def test_crossed_boundary_mask(self):
        w = 16
        # decode equivalence: query_len == 1 reduces exactly to seq % w == 0.
        # (seq=0 — an empty sequence, no state to flush — is excluded, same
        # as the `ends > 0` guard in decode_flush_mask.)
        seqs = torch.arange(1, 70)
        ones = torch.ones_like(seqs)
        self.assertTrue(torch.equal(crossed_boundary_mask(seqs, ones, w), seqs % w == 0))
        # interior crossing (the M1 gap): chunk [10, 20) crosses boundary 16
        # without ending on it -> due under M3.3, missed by seq % w == 0
        self.assertTrue(crossed_boundary_mask(torch.tensor([20]), torch.tensor([10]), w).item())
        self.assertNotEqual(20 % w, 0)
        # multi-boundary chunk [5, 50) crosses 16/32/48 -> due (one flush)
        self.assertTrue(crossed_boundary_mask(torch.tensor([50]), torch.tensor([45]), w).item())
        # no crossing: [20, 30) lives inside window 2
        self.assertFalse(crossed_boundary_mask(torch.tensor([30]), torch.tensor([10]), w).item())
        # boundary-aligned chunk end [8, 32) still due
        self.assertTrue(crossed_boundary_mask(torch.tensor([32]), torch.tensor([24]), w).item())
        # degenerate query > seq clamps the start to 0 (defensive only)
        self.assertFalse(crossed_boundary_mask(torch.tensor([5]), torch.tensor([10]), w).item())
        # no boundary is ever skipped: walking [0, 100) in steps of 7 flushes
        # exactly once per crossed boundary (16, 32, 48, 64, 80, 96)
        seq, q = 100, 7
        due_ends = [
            min(s + q, seq)
            for s in range(0, seq, q)
            if crossed_boundary_mask(
                torch.tensor([min(s + q, seq)]),
                torch.tensor([q if s + q <= seq else seq - s]),
                w,
            ).item()
        ]
        self.assertEqual(due_ends, [21, 35, 49, 70, 84, 98])


if __name__ == "__main__":
    unittest.main()
