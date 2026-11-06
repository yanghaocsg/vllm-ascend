# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Fused LeapQuant window-flush kernels (triton-ascend, RFC vllm-ascend#21097 M3).

One triton program per (due slot, head) rewrites its ``[V, K]`` fp32 state
block in place with the LeapQuant window-boundary round-trip of PR1
(``state.leap_window_flush``):

1. one round of subspace iteration (2-pass MGS) -> top-``R`` compensator
   factors; the *stored* factors are rounded to fp16 exactly like the
   torch reference, while the subspace iteration itself runs in fp32
   (``uk = S^T qv`` uses the un-rounded ``qv``);
2. smoothed two-scale int8 quantization of the residual
   (``s_k = sqrt(mean_V |E|)`` per key column, ``s_v = absmax_K(|E|/s_k)``
   per value row, round-half-to-even codes clamped to [-127, 127]);
3. the block is replaced by the reconstruction
   ``S' = q^T u + codes * (s_k/127) * s_v``.

The numerics are a line-by-line port of
``vllm_ascend/quantization/methods/mamba_state/leapquant/state.py``; the
unit test pins the kernel to that reference (error-ratio <= 1.01 plus an
elementwise bound of two int8 code steps).

triton-ascend notes (all learned the hard way on this build):

* The whole ``[V, K]`` block cannot live in one program: the AIV unified
  buffer is 192 KB and full-block math multi-buffers ~700 KB.  The passes
  stream the block in ``BLOCK_V x K`` / ``V x BLOCK_K`` tiles.
* R-fold unrolled reductions per tile make the CANN (BiShengHIR) frontend
  take ~an hour per kernel; runtime loops with small loop-carried
  accumulators keep the IR small.
* Two loops containing structurally identical ``tl.dot`` bodies abort the
  BiShengIR fusion pipeline (SIGABRT in ConvertLinalgIRToBinary).
* ``tl.where(cond[:, None] == r, block[D, 16], 0)``-style axis-0
  extractions and ``trans(where(...))`` compositions silently mis-lower
  to zeros.  Only the axis-1 pattern ``tl.sum(tl.where(offs_r[None, :] ==
  r, block, 0.0), axis=1)`` on ``[D, 16]`` blocks is reliable, so every
  subspace vector lives in a contiguous ``[D, 16]`` plane and is read in
  that orientation only; the residual low-rank product is an unrolled
  rank-R outer product instead of a dot.

Split into two launches (scales, then reconstruct): the second kernel
reuses the stored ``s_k``/``s_v``, which also pins both passes to
bit-identical scales.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.quantization.methods.mamba_state.leapquant.state import start_basis

__all__ = ["leapquant_flush_triton"]

# Rank-padded block width (N >= 16 for tl.dot on triton-ascend).
_LEAPQUANT_RANK_PAD = 16

_Q0_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


@triton.jit
def _round_half_even(x):
    """torch.round parity (ties to even); inputs are <= 127 in magnitude."""
    n0 = tl.floor(x)
    frac = x - n0
    n1 = n0 + 1.0
    n0_even = (n0 - 2.0 * tl.floor(n0 * 0.5)) == 0.0
    tie = tl.where(n0_even, n0, n1)
    return tl.where(frac == 0.5, tie, tl.where(frac > 0.5, n1, n0))


@triton.jit
def _mgs_cols(block, offs_r, R: tl.constexpr, PASSES: tl.constexpr):
    """2-pass modified Gram-Schmidt over the first R columns of ``block``.

    ``block`` is ``[D, 16]`` with padding columns (>= R) exactly zero;
    they stay zero so subsequent ``tl.dot``s are unpolluted.  Matches
    ``state.mgs_orthonormal_columns`` including the 1e-30 norm clamp.
    """
    for j in tl.static_range(R):
        v_j = tl.sum(tl.where(offs_r[None, :] == j, block, 0.0), axis=1)
        for _ in tl.static_range(PASSES):
            for i in tl.static_range(R):
                if i < j:
                    c_i = tl.sum(tl.where(offs_r[None, :] == i, block, 0.0), axis=1)
                    v_j = v_j - c_i * tl.sum(c_i * v_j)
        nrm = tl.sqrt(tl.sum(v_j * v_j))
        v_j = v_j / tl.maximum(nrm, 1e-30)
        block = tl.where(offs_r[None, :] == j, v_j[:, None], block)
    return block


@triton.jit
def _leapquant_flush_scales_kernel(
    state_ptr,  # fp32 [slots, H, V, K]
    due_ptr,  # int32/int64 [D] slot ids to flush
    q0_ptr,  # fp32 [V, R] fixed orthonormal start basis (contiguous)
    qk_ptr,  # fp32 [D*H, K, 16] key-side factor plane
    qv_ptr,  # fp32 [D*H, V, 16] value-side factor plane
    uk_ptr,  # fp32 [D*H, K, 16] key-side coefficient plane
    sk_ptr,  # fp32 [D*H, K] per-key-column residual scales (out)
    sv_ptr,  # fp32 [D*H, V] per-value-row residual scales (out)
    stride_slot: tl.int64,
    stride_head: tl.int64,
    stride_v: tl.int64,
    stride_k: tl.int64,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    d = pid // H
    h = pid % H
    slot = tl.load(due_ptr + d).to(tl.int64)
    base = state_ptr + slot * stride_slot + h * stride_head

    offs_v = tl.arange(0, V)
    offs_k = tl.arange(0, K)
    offs_bv = tl.arange(0, BLOCK_V)
    offs_bk = tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, 16)

    # ---- qk = mgs(S^T q0): stream V tiles, accumulate [K, 16] ----------
    qk16 = tl.zeros([K, 16], dtype=tl.float32)
    for vt in range(0, V, BLOCK_V):
        s_t = tl.load(base + (vt + offs_bv)[:, None] * stride_v + offs_k[None, :] * stride_k)
        q0_t = tl.load(
            q0_ptr + (vt + offs_bv)[:, None] * R + offs_r[None, :],
            mask=offs_r[None, :] < R,
            other=0.0,
        )
        qk16 = qk16 + tl.dot(tl.trans(s_t), q0_t, input_precision="ieee")
    qk16 = _mgs_cols(qk16, offs_r, R, 2)
    tl.store(qk_ptr + pid.to(tl.int64) * (K * 16) + offs_k[:, None] * 16 + offs_r[None, :], qk16)

    # ---- qv = mgs(S qk): stream K tiles, accumulate [V, 16] ------------
    qv16 = tl.zeros([V, 16], dtype=tl.float32)
    for kt in range(0, K, BLOCK_K):
        s_t = tl.load(base + offs_v[:, None] * stride_v + (kt + offs_bk)[None, :] * stride_k)
        qk_t = tl.load(qk_ptr + pid.to(tl.int64) * (K * 16) + (kt + offs_bk)[:, None] * 16 + offs_r[None, :])
        qv16 = qv16 + tl.dot(s_t, qk_t, input_precision="ieee")
    qv16 = _mgs_cols(qv16, offs_r, R, 2)
    # store RAW qv first: uk = S^T qv uses the un-rounded factors.
    tl.store(qv_ptr + pid.to(tl.int64) * (V * 16) + offs_v[:, None] * 16 + offs_r[None, :], qv16)

    # ---- uk = S^T qv (raw): stream V tiles, accumulate [K, 16] ---------
    uk16 = tl.zeros([K, 16], dtype=tl.float32)
    for vt in range(0, V, BLOCK_V):
        s_t = tl.load(base + (vt + offs_bv)[:, None] * stride_v + offs_k[None, :] * stride_k)
        qv_t = tl.load(qv_ptr + pid.to(tl.int64) * (V * 16) + (vt + offs_bv)[:, None] * 16 + offs_r[None, :])
        uk16 = uk16 + tl.dot(tl.trans(s_t), qv_t, input_precision="ieee")
    # Checkpoint factors are fp16-rounded (reference casts q/u before the
    # residual and reconstruction use them).
    uk16 = uk16.to(tl.float16).to(tl.float32)
    qv_f16 = qv16.to(tl.float16).to(tl.float32)
    tl.store(uk_ptr + pid.to(tl.int64) * (K * 16) + offs_k[:, None] * 16 + offs_r[None, :], uk16)
    tl.store(qv_ptr + pid.to(tl.int64) * (V * 16) + offs_v[:, None] * 16 + offs_r[None, :], qv_f16)

    # ---- residual scales: s_k per K tile, s_v running max over K -------
    qv_f = tl.load(qv_ptr + pid.to(tl.int64) * (V * 16) + offs_v[:, None] * 16 + offs_r[None, :])  # [V, 16]
    # rank-R factor columns as [V] vectors (axis-1 extraction only; see
    # module notes) for the unrolled outer-product low-rank tile.
    q_cols = ()
    for r in tl.static_range(R):
        q_cols = q_cols + (tl.sum(tl.where(offs_r[None, :] == r, qv_f, 0.0), axis=1),)

    s_v = tl.zeros([V], dtype=tl.float32)
    for kt in range(0, K, BLOCK_K):
        s_t = tl.load(base + offs_v[:, None] * stride_v + (kt + offs_bk)[None, :] * stride_k)
        uk_t = tl.load(uk_ptr + pid.to(tl.int64) * (K * 16) + (kt + offs_bk)[:, None] * 16 + offs_r[None, :])  # [BK, 16]
        lr_t = tl.zeros([V, BLOCK_K], dtype=tl.float32)
        for r in tl.static_range(R):
            u_row = tl.sum(tl.where(offs_r[None, :] == r, uk_t, 0.0), axis=1)
            lr_t += q_cols[r][:, None] * u_row[None, :]
        a_t = tl.abs(s_t - lr_t)
        s_k_t = tl.maximum(tl.sqrt(tl.sum(a_t, axis=0) / V), 1e-12)
        tl.store(sk_ptr + pid.to(tl.int64) * K + kt + offs_bk, s_k_t)
        s_v = tl.maximum(s_v, tl.max(a_t / s_k_t[None, :], axis=1))
    s_v = tl.maximum(s_v, 1e-12)
    tl.store(sv_ptr + pid.to(tl.int64) * V + offs_v, s_v)


@triton.jit
def _leapquant_reconstruct_kernel(
    state_ptr,  # fp32 [slots, H, V, K]
    due_ptr,  # int32/int64 [D] slot ids to flush
    qv_ptr,  # fp32 [D*H, V, 16] (fp16-rounded value-side factors)
    uk_ptr,  # fp32 [D*H, K, 16] (fp16-rounded key-side coefficients)
    sk_ptr,  # fp32 [D*H, K]
    sv_ptr,  # fp32 [D*H, V]
    stride_slot: tl.int64,
    stride_head: tl.int64,
    stride_v: tl.int64,
    stride_k: tl.int64,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    d = pid // H
    h = pid % H
    slot = tl.load(due_ptr + d).to(tl.int64)
    base = state_ptr + slot * stride_slot + h * stride_head

    offs_v = tl.arange(0, V)
    offs_bk = tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, 16)

    qv_f = tl.load(qv_ptr + pid.to(tl.int64) * (V * 16) + offs_v[:, None] * 16 + offs_r[None, :])  # [V, 16]
    s_v = tl.load(sv_ptr + pid.to(tl.int64) * V + offs_v)  # [V]
    q_cols = ()
    for r in tl.static_range(R):
        q_cols = q_cols + (tl.sum(tl.where(offs_r[None, :] == r, qv_f, 0.0), axis=1),)

    # ---- codes + reconstruction, in-place rewrite ----------------------
    for kt in range(0, K, BLOCK_K):
        s_t = tl.load(base + offs_v[:, None] * stride_v + (kt + offs_bk)[None, :] * stride_k)
        uk_t = tl.load(uk_ptr + pid.to(tl.int64) * (K * 16) + (kt + offs_bk)[:, None] * 16 + offs_r[None, :])  # [BK, 16]
        lr_t = tl.zeros([V, BLOCK_K], dtype=tl.float32)
        for r in tl.static_range(R):
            u_row = tl.sum(tl.where(offs_r[None, :] == r, uk_t, 0.0), axis=1)
            lr_t += q_cols[r][:, None] * u_row[None, :]
        e_t = s_t - lr_t
        a_t = tl.abs(e_t)
        s_k_t = tl.load(sk_ptr + pid.to(tl.int64) * K + kt + offs_bk)
        mag = _round_half_even(a_t / (s_k_t[None, :] * s_v[:, None]) * 127.0)
        sgn = tl.where(e_t >= 0.0, 1.0, -1.0)
        codes = tl.minimum(tl.maximum(sgn * mag, -127.0), 127.0)
        out = lr_t + codes * (s_k_t[None, :] / 127.0) * s_v[:, None]
        tl.store(base + offs_v[:, None] * stride_v + (kt + offs_bk)[None, :] * stride_k, out)


def _get_q0(v_dim: int, rank: int, device: torch.device) -> torch.Tensor:
    key = (v_dim, rank, str(device))
    cached = _Q0_CACHE.get(key)
    if cached is None:
        cached = start_basis(v_dim, rank, device=device, dtype=torch.float32)
        _Q0_CACHE[key] = cached
    return cached


def leapquant_flush_triton(
    state: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    compensator_tokens: int = 4,
) -> None:
    """Fused in-place LeapQuant flush of ``slot_ids`` rows of a state pool.

    Mirrors ``leap_window_flush(state[slot_ids])`` written back into the
    pool, with the gather / flush / scatter fused into two kernel
    launches (PR1 runs them as three host-side steps plus ~15 torch ops).

    Args:
        state: fp32 pool ``[slots, H, V, K]``; ``V``/``K`` must be powers
            of two (Qwen3.5/Kimi GDN heads are 128x128).  Arbitrary batch
            and head strides are allowed.
        slot_ids: 1D int32/int64 tensor of slot rows whose sequences just
            crossed a window boundary.
        compensator_tokens: rank ``R`` of the compensator subspace (1..16).
    """
    if state.ndim != 4:
        raise ValueError(f"state must be [slots, H, V, K], got ndim={state.ndim}")
    if state.dtype != torch.float32:
        raise TypeError(f"state must be fp32, got {state.dtype}")
    _, heads, v_dim, k_dim = state.shape
    if v_dim & (v_dim - 1) or k_dim & (k_dim - 1):
        raise ValueError(f"V/K must be powers of two, got V={v_dim}, K={k_dim}")
    if not 1 <= compensator_tokens <= _LEAPQUANT_RANK_PAD:
        raise ValueError(f"compensator_tokens must be 1..16, got {compensator_tokens}")
    if slot_ids.ndim != 1 or slot_ids.numel() == 0:
        raise ValueError("slot_ids must be a non-empty 1D tensor")
    if slot_ids.device != state.device:
        raise ValueError("slot_ids and state must be on the same device")

    due = slot_ids.to(torch.int64) if slot_ids.dtype != torch.int64 else slot_ids
    q0 = _get_q0(v_dim, compensator_tokens, state.device)
    blocks = due.numel() * heads
    qk_s = torch.empty((blocks, k_dim, 16), dtype=torch.float32, device=state.device)
    qv_s = torch.empty((blocks, v_dim, 16), dtype=torch.float32, device=state.device)
    uk_s = torch.empty((blocks, k_dim, 16), dtype=torch.float32, device=state.device)
    sk = torch.empty((blocks, k_dim), dtype=torch.float32, device=state.device)
    sv = torch.empty((blocks, v_dim), dtype=torch.float32, device=state.device)
    grid = (blocks,)
    _leapquant_flush_scales_kernel[grid](
        state,
        due,
        q0,
        qk_s,
        qv_s,
        uk_s,
        sk,
        sv,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        H=heads,
        V=v_dim,
        K=k_dim,
        R=compensator_tokens,
        BLOCK_V=32,
        BLOCK_K=32,
        num_warps=8,
    )
    _leapquant_reconstruct_kernel[grid](
        state,
        due,
        qv_s,
        uk_s,
        sk,
        sv,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        H=heads,
        V=v_dim,
        K=k_dim,
        R=compensator_tokens,
        BLOCK_K=32,
        num_warps=8,
    )
