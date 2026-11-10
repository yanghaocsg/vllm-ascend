# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""M3.4a workspace tests: the shadow fp32 pool W with identity slot mapping.

The consistency argument is determinism: the decode kernel and the flush are
per-row (no cross-row interaction), so a trajectory that runs decode+flush on
W rows — with W kept in lockstep by the prefill mirror — must be **bit-equal**
to the same trajectory running directly on the engine pool rows.  That
equality is what lets gdn.py swap the state pool for W without changing
numerics; the mirror-heal test pins the slot-reuse recovery path.

Run on an Atlas A3 NPU (directory suffix routes to a3_2 CI runners).
"""

import unittest

import torch

from vllm_ascend.ascend_config import LeapQuantConfig
from vllm_ascend.quantization.methods.mamba_state.leapquant import (
    leapquant_flush_state_rows,
    leapquant_workspace_for,
)

H, DK, DV = 4, 128, 128
STEPS, WINDOW = 64, 16
SLOTS = 8
LIVE = [1, 3]


def _mirror(state: torch.Tensor, w: torch.Tensor, indices: torch.Tensor) -> None:
    """The §2.3 gdn.py mirror site: gather post-prefill rows from the engine
    pool and scatter them into W (all-true flags: every prefilled slot)."""
    from vllm_ascend.ops.triton.mamba.state_index import (
        gather_ssm_states,
        scatter_ssm_states_,
    )

    present = torch.ones(indices.shape[0], dtype=torch.bool, device=indices.device)
    scatter_ssm_states_(w, indices, gather_ssm_states(state, indices, present))


def _trajectory(pool_is_workspace: bool, mid_prefill: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode+flush trajectory on either the pool directly (M3.4) or the W
    shadow kept in lockstep by mirrors (M3.4a). Identical inputs/ops."""
    from vllm_ascend.ops.triton.leapquant import recurrent_gdn_leapquant

    torch.manual_seed(42)
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    q = torch.randn(STEPS, len(LIVE), H, DK, device=device, dtype=torch.float32)
    k = torch.randn(STEPS, len(LIVE), H, DK, device=device, dtype=torch.float32)
    v = torch.randn(STEPS, len(LIVE), H, DV, device=device, dtype=torch.float32)
    g = -torch.rand(STEPS, len(LIVE), H, device=device, dtype=torch.float32) * 0.1
    beta = (torch.rand(STEPS, len(LIVE), H, device=device, dtype=torch.float32) * 0.5 + 0.5).to(torch.bfloat16)
    q_n = torch.nn.functional.normalize(q, dim=-1).to(torch.bfloat16)
    k_n = torch.nn.functional.normalize(k, dim=-1).to(torch.bfloat16)

    state = torch.zeros(SLOTS, H, DV, DK, device=device, dtype=torch.float32)
    # Simulated prefill: seed the live rows in the engine pool, then mirror.
    state[LIVE] = torch.randn(len(LIVE), H, DV, DK, device=device) * 0.2
    w = leapquant_workspace_for(state) if pool_is_workspace else state
    if pool_is_workspace:
        _mirror(state, w, torch.tensor(LIVE, device=device, dtype=torch.int64))

    indices = torch.tensor(LIVE, device=device, dtype=torch.int64)
    seq_lens = torch.zeros(len(LIVE), dtype=torch.int64)
    config = LeapQuantConfig(enabled=True, window=WINDOW, compensator_tokens=4)
    outputs = []

    for t in range(STEPS):
        out = recurrent_gdn_leapquant(
            q_n[t][:, None].contiguous(),
            k_n[t][:, None].contiguous(),
            v[t].to(torch.bfloat16)[:, None].contiguous(),
            g[t][:, None].contiguous(),
            beta[t][:, None].contiguous(),
            scale=DK**-0.5,
            w_in=w,
            w_out=w,
            w_rows=indices,
        )
        outputs.append(out[:, 0].detach().clone())
        seq_lens += 1
        if flush_due := bool((seq_lens % WINDOW == 0).any()):
            leapquant_flush_state_rows(w, indices, seq_lens, config=config)
        if mid_prefill and t == STEPS // 2 - 2:
            # A chunked-prefill continuation writes engine rows mid-flight.
            # It writes a value computed from the prompt only (chunked prefill
            # never interleaves with decode for the same request), so both
            # modes must install the same fresh row; the mirror propagates
            # the engine write into W.  (t = STEPS//2 - 2 lands off the flush
            # boundary, so the write is not immediately overwritten.)
            fresh = torch.randn(len(LIVE), H, DV, DK, device=device) * 0.2
            state[LIVE] = fresh
            if pool_is_workspace:
                _mirror(state, w, torch.tensor(LIVE, device=device, dtype=torch.int64))
    return torch.stack(outputs), w


class TestLeapQuantWorkspace(unittest.TestCase):
    def test_workspace_alloc_and_cache(self):
        state = torch.randn(4, 2, 32, 64, device="npu:0", dtype=torch.float32)
        w = leapquant_workspace_for(state)
        self.assertEqual(tuple(w.shape), tuple(state.shape))
        self.assertEqual(w.dtype, torch.float32)
        self.assertTrue(w.is_contiguous())
        self.assertTrue(torch.equal(w, state))  # initial full sync
        self.assertIs(leapquant_workspace_for(state), w)  # cached per pool
        other = torch.randn(4, 2, 32, 64, device="npu:0", dtype=torch.float32)
        self.assertIsNot(leapquant_workspace_for(other), w)

    def test_workspace_validation(self):
        with self.assertRaises(ValueError):
            leapquant_workspace_for(torch.zeros(4, 2, 32))
        with self.assertRaises(TypeError):
            leapquant_workspace_for(torch.zeros(4, 2, 32, 64, device="npu:0", dtype=torch.float16))

    def test_prefill_mirror_heals_stale_row(self):
        """Slot reuse: a new request's prefill overwrites the engine row; the
        mirror must make W follow exactly (stale shadow content is gone)."""
        state = torch.randn(4, 2, 32, 64, device="npu:0", dtype=torch.float32)
        w = leapquant_workspace_for(state)
        w[2] = float("nan")  # stale/garbage shadow row
        state[2] = torch.randn_like(state[2])
        _mirror(state, w, torch.tensor([2], device="npu:0", dtype=torch.int64))
        self.assertTrue(torch.equal(w[2], state[2]))
        self.assertFalse(torch.isnan(w).any())

    def test_w_trajectory_bitwise_matches_pool(self):
        """The M3.4a invariant: decode+flush on W (mirrored after prefill)
        reproduces the on-pool trajectory bit for bit."""
        pool_out, pool_state = _trajectory(pool_is_workspace=False, mid_prefill=False)
        w_out, w_state = _trajectory(pool_is_workspace=True, mid_prefill=False)
        self.assertTrue(torch.equal(pool_out, w_out), "outputs must be bit-equal")
        self.assertTrue(torch.equal(pool_state, w_state), "states must be bit-equal")
        untouched = [i for i in range(SLOTS) if i not in LIVE]
        self.assertEqual(int(w_state[untouched].abs().sum()), 0)

    def test_w_trajectory_with_midflight_prefill(self):
        """Mixed steps: a mid-trajectory prefill (engine-row write + mirror)
        keeps W and the pool trajectory bit-equal."""
        pool_out, pool_state = _trajectory(pool_is_workspace=False, mid_prefill=True)
        w_out, w_state = _trajectory(pool_is_workspace=True, mid_prefill=True)
        self.assertTrue(torch.equal(pool_out, w_out))
        self.assertTrue(torch.equal(pool_state, w_state))


if __name__ == "__main__":
    unittest.main()
