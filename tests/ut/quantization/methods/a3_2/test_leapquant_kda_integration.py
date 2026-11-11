# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""NPU integration smoke: KDA AscendC recurrent op vs the LeapQuant triton path.

Runs the real production decode call (``vllm_ascend.ops.kda.run_recurrent_kda``
-> fla_npu AscendC ``recurrent_kda``: in-kernel gate expansion + l2norm) token
by token on a state pool, against the M3.4a KDA replacement (host-side
``kda_expand_gate`` + fla ``l2norm_fwd`` + ``recurrent_kda_leapquant`` on the
shadow W pool). Pins the numerics contract of the call-site swap: the host
reproduction of the in-kernel gate/l2norm must track the binary at bf16
noise, and both gate variants (softplus = Kimi-Linear, ``lower_bound`` safe =
bounded decay) are exercised. Window-boundary flushes ride both trajectories
like the GDN integration test.

AscendC op constraints pinned while writing this test (die-14/15 probe
matrix, see TRITON_M3_DEV_PLAN M3.4a notes):

* **decode-only**: every ``cu_seqlens`` span must be exactly 1 token —
  multi-token spans fail aclnnRecurrentKdaGetWorkspaceSize 161002 (in
  production multi-token work goes through ``run_chunk_kda``);
* **multi-request state writes are unreliable on this CANN stack**: with two
  sequences per call the second sequence's state row is dropped and its
  output row reads garbage (nan/inf), depending on the process's NPU
  allocation history in ways that defy control from the test (values, input
  layout, allocation order/counts, import order, allocator warm-up shapes
  were all bisected). Single-sequence calls are robust, so the reference
  trajectory issues **one request per call** (two live slots 1/3 on the same
  pool), which still exercises the full M3.4a contract: multi-slot pools,
  both gate variants, flushes, and the host gate/l2norm reproduction.
  This is an fla_npu op fragility worth reporting upstream — the triton
  kernel replaces exactly this call, so M3.4a is also the fix;
* per-step inputs are materialized fresh in the production ``[1, T, H, D]``
  layout (``rearrange("n (h d) -> 1 n h d")`` projection form) and held for
  the whole trajectory: the binary runs on a side stream, and letting the
  caching allocator reuse their memory mid-read yields transient nan/inf.

Run on an Atlas A3 NPU (directory suffix routes to a3_2 CI runners).
"""

import unittest

import torch

from vllm_ascend.ascend_config import LeapQuantConfig
from vllm_ascend.ops.kda import run_recurrent_kda
from vllm_ascend.quantization.methods.mamba_state.leapquant import leapquant_flush_state_rows

# The AscendC recurrent op requires head_dim == 128 on both K and V sides.
H, DK, DV = 4, 128, 128
STEPS, WINDOW = 64, 16
SLOTS = 8
SEQS = 2
LIVE_SLOTS = [1, 3]  # two live rows of the padded state pool


def _draw_step(device, t, seq):
    """Deterministic bf16 inputs for (step, sequence): fresh [1, 1, H, D]
    tensors in the production layout, reseeded per (t, seq) so both
    trajectories see identical values regardless of call order."""
    torch.manual_seed(52 + 2 * t + seq)
    q = torch.randn(1, 1, H, DK, device=device).to(torch.bfloat16)
    k = torch.randn(1, 1, H, DK, device=device).to(torch.bfloat16)
    v = torch.randn(1, 1, H, DV, device=device).to(torch.bfloat16)
    a = (torch.randn(1, 1, H, DK, device=device) * 0.5).to(torch.bfloat16)
    beta = (torch.rand(1, 1, H, device=device) * 0.5 + 0.5).sigmoid()
    return q, k, v, a, beta


def _gate_params(device):
    torch.manual_seed(52)
    a_log = torch.log(torch.rand(H, device=device) + 0.5)
    dt_bias = 0.1 * torch.randn(H, DK, device=device)
    return a_log, dt_bias


def _run_trajectory_ascendc(lower_bound, flush: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Production AscendC decode trajectory (the call being replaced), one
    request per call (see module docstring for why multi-request batches are
    avoided)."""
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    a_log, dt_bias = _gate_params(device)

    state = torch.zeros(SLOTS, H, DV, DK, device=device, dtype=torch.float32)
    indices = torch.tensor(LIVE_SLOTS, device=device, dtype=torch.int32)
    cu = torch.tensor([0, 1], device=device, dtype=torch.int32)
    seq_lens = torch.zeros(SEQS, dtype=torch.int64)
    config = LeapQuantConfig(enabled=True, window=WINDOW, compensator_tokens=4)
    outputs = []  # [STEPS][SEQS, H, V]
    held = []
    for t in range(STEPS):
        step_out = []
        for s in range(SEQS):
            q_t, k_t, v_t, a_t, beta_t = _draw_step(device, t, s)
            held += [q_t, k_t, v_t, a_t, beta_t]
            out = run_recurrent_kda(
                q_t, k_t, v_t, a_t, beta_t,
                state,
                cu,
                indices[s:s + 1],
                a_log,
                dt_bias,
                lower_bound=lower_bound,
            )
            torch.npu.synchronize()
            step_out.append(out[0, 0].detach().clone())
        outputs.append(torch.stack(step_out))
        seq_lens += 1
        if flush:
            leapquant_flush_state_rows(state, indices, seq_lens, config=config)
    return torch.stack(outputs), state


def _run_trajectory_triton(lower_bound, flush: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """M3.4a KDA replacement trajectory (the _leapquant_kda_recurrent math):
    fla l2norm + host gate expansion + payload-capable triton kernel on the
    shadow W pool, one request per call to mirror the AscendC reference."""
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    from vllm_ascend.ops.triton.leapquant import kda_expand_gate, recurrent_kda_leapquant

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    a_log, dt_bias = _gate_params(device)

    w = torch.zeros(SLOTS, H, DV, DK, device=device, dtype=torch.float32)
    indices = torch.tensor(LIVE_SLOTS, device=device, dtype=torch.int64)
    seq_lens = torch.zeros(SEQS, dtype=torch.int64)
    config = LeapQuantConfig(enabled=True, window=WINDOW, compensator_tokens=4)
    outputs = []
    for t in range(STEPS):
        step_out = []
        for s in range(SEQS):
            q_t, k_t, v_t, a_t, beta_t = _draw_step(device, t, s)
            # exact _leapquant_kda_recurrent preprocessing: contiguous bf16
            # in, fla l2norm, then the [1,T,H,K] raw gate expanded on host.
            q_n = l2norm_fwd(q_t.contiguous())
            k_n = l2norm_fwd(k_t.contiguous())
            g = kda_expand_gate(a_t, a_log, dt_bias, lower_bound)
            out = recurrent_kda_leapquant(
                q_n,
                k_n,
                v_t,
                g,
                beta_t,
                scale=DK**-0.5,
                w_in=w,
                w_out=w,
                w_rows=indices[s:s + 1],
            )
            step_out.append(out[0, 0].detach().clone())
        outputs.append(torch.stack(step_out))
        seq_lens += 1
        if flush:
            leapquant_flush_state_rows(w, indices, seq_lens, config=config)
    return torch.stack(outputs), w


class TestLeapQuantKDAIntegration(unittest.TestCase):
    # Hold reference tensors as class attributes: keeps the allocation alive
    # for the comparison and avoids interleaved NPU-memory reuse.
    ref_output: dict[float | None, torch.Tensor] = {}
    ref_state: dict[float | None, torch.Tensor] = {}

    @classmethod
    def setUpClass(cls):
        for lb in (None, -3.0):
            cls.ref_output[lb], cls.ref_state[lb] = _run_trajectory_ascendc(lb, flush=False)

    def test_triton_decode_matches_ascendc(self):
        """M3.4a KDA: the triton replacement (host gate + l2norm) must
        reproduce the AscendC in-kernel trajectory at bf16 noise, for both
        gate variants."""
        for lb in (None, -3.0):
            with self.subTest(lower_bound=lb):
                tri_out, tri_state = _run_trajectory_triton(lb, flush=False)
                ref_out, ref_state = self.ref_output[lb], self.ref_state[lb]
                out_rel = ((tri_out - ref_out).norm() / ref_out.norm()).item()
                state_rel = ((tri_state[LIVE_SLOTS] - ref_state[LIVE_SLOTS]).norm()
                             / ref_state[LIVE_SLOTS].norm()).item()
                self.assertLess(out_rel, 0.02, f"triton vs AscendC output: {out_rel}")
                self.assertLess(state_rel, 0.02, f"triton vs AscendC state: {state_rel}")
                # Untouched slots must remain exactly zero.
                untouched = [i for i in range(SLOTS) if i not in LIVE_SLOTS]
                self.assertEqual(int(tri_state[untouched].abs().sum()), 0)

    def test_triton_flushed_trajectory(self):
        """Triton decode + window flush stays within the same envelope as the
        GDN integration bound (4 int8 flushes on a 64-token trajectory)."""
        for lb in (None, -3.0):
            with self.subTest(lower_bound=lb):
                flushed_out, flushed_state = _run_trajectory_triton(lb, flush=True)
                ref_out, _ = _run_trajectory_triton(lb, flush=False)
                out_rel = ((flushed_out - ref_out).norm() / ref_out.norm()).item()
                self.assertLess(out_rel, 0.05, f"output drift too large: {out_rel}")
                untouched = [i for i in range(SLOTS) if i not in LIVE_SLOTS]
                self.assertEqual(int(flushed_state[untouched].abs().sum()), 0)

    def test_ascendc_flushed_envelope(self):
        """Sanity anchor: the AscendC trajectory itself tolerates the flush
        (proves any triton-vs-AscendC gap is not flush-induced)."""
        for lb in (None, -3.0):
            with self.subTest(lower_bound=lb):
                flushed_out, _ = _run_trajectory_ascendc(lb, flush=True)
                ref_out = self.ref_output[lb]
                out_rel = ((flushed_out - ref_out).norm() / ref_out.norm()).item()
                self.assertLess(out_rel, 0.05, f"AscendC flushed drift: {out_rel}")


if __name__ == "__main__":
    torch.npu.set_device(torch.device("npu:0"))
    unittest.main()
