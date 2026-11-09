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
* ``x.to(tl.float16).to(tl.float32)`` in registers compiles to a NO-OP
  on this build (fp16 appears to be emulated in fp32 registers), so the
  fp16 rounding of the compensator factors cannot be done with casts —
  it happens at an fp16 *store*.  The ``qv``/``uk`` scratch planes are
  therefore fp16 memory (every consumer loads truly rounded values,
  matching the torch reference), while a separate fp32 ``qvr`` plane
  keeps the raw factors the ``uk = S^T qv`` dot needs.

Split into two launches (scales, then reconstruct): the second kernel
reuses the stored ``s_k``/``s_v``, which also pins both passes to
bit-identical scales.

``payload=`` mode (M3.2) keeps the scales launch but swaps the second
launch for ``_leapquant_payload_encode_kernel``: instead of rebuilding
fp32 in place, it persists the LeapQuant checkpoint per (slot, head) —
int8 codes plus the scales and fp16 compensator planes — into a
``LeapQuantPayloadPool`` (~1.19 B/elem vs 4 B fp32).  With
``rewrite_state=True`` the reconstruct kernel runs as a third launch
(dual-write, the migration-safe gate); the codes math is identical in
all modes.
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
    qvr_ptr,  # fp32 [D*H, V, 16] RAW value-side factors (uk-dot input only)
    qv_ptr,  # fp16 [D*H, V, 16] fp16-rounded value-side factors
    uk_ptr,  # fp16 [D*H, K, 16] fp16-rounded key-side coefficients
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
    # RAW plane for the uk dot (register casts cannot round on this build;
    # see module notes) plus the fp16-rounded checkpoint plane.
    tl.store(qvr_ptr + pid.to(tl.int64) * (V * 16) + offs_v[:, None] * 16 + offs_r[None, :], qv16)
    tl.store(qv_ptr + pid.to(tl.int64) * (V * 16) + offs_v[:, None] * 16 + offs_r[None, :],
             qv16.to(qv_ptr.dtype.element_ty))

    # ---- uk = S^T qv (raw): stream V tiles, accumulate [K, 16] ---------
    uk16 = tl.zeros([K, 16], dtype=tl.float32)
    for vt in range(0, V, BLOCK_V):
        s_t = tl.load(base + (vt + offs_bv)[:, None] * stride_v + offs_k[None, :] * stride_k)
        qv_t = tl.load(qvr_ptr + pid.to(tl.int64) * (V * 16) + (vt + offs_bv)[:, None] * 16 + offs_r[None, :])
        uk16 = uk16 + tl.dot(tl.trans(s_t), qv_t, input_precision="ieee")
    # Checkpoint factors are fp16-rounded via the fp16 store (the reference
    # casts q/u before the residual and reconstruction use them).
    tl.store(uk_ptr + pid.to(tl.int64) * (K * 16) + offs_k[:, None] * 16 + offs_r[None, :],
             uk16.to(uk_ptr.dtype.element_ty))

    # ---- residual scales: s_k per K tile, s_v running max over K -------
    qv_f = tl.load(qv_ptr + pid.to(tl.int64) * (V * 16) + offs_v[:, None] * 16 + offs_r[None, :]).to(tl.float32)  # [V, 16]
    # rank-R factor columns as [V] vectors (axis-1 extraction only; see
    # module notes) for the unrolled outer-product low-rank tile.
    q_cols = ()
    for r in tl.static_range(R):
        q_cols = q_cols + (tl.sum(tl.where(offs_r[None, :] == r, qv_f, 0.0), axis=1),)

    s_v = tl.zeros([V], dtype=tl.float32)
    for kt in range(0, K, BLOCK_K):
        s_t = tl.load(base + offs_v[:, None] * stride_v + (kt + offs_bk)[None, :] * stride_k)
        uk_t = tl.load(uk_ptr + pid.to(tl.int64) * (K * 16) + (kt + offs_bk)[:, None] * 16 + offs_r[None, :]).to(tl.float32)  # [BK, 16]
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
    qv_ptr,  # fp16 [D*H, V, 16] (fp16-rounded value-side factors)
    uk_ptr,  # fp16 [D*H, K, 16] (fp16-rounded key-side coefficients)
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

    qv_f = tl.load(qv_ptr + pid.to(tl.int64) * (V * 16) + offs_v[:, None] * 16 + offs_r[None, :]).to(tl.float32)  # [V, 16]
    s_v = tl.load(sv_ptr + pid.to(tl.int64) * V + offs_v)  # [V]
    q_cols = ()
    for r in tl.static_range(R):
        q_cols = q_cols + (tl.sum(tl.where(offs_r[None, :] == r, qv_f, 0.0), axis=1),)

    # ---- codes + reconstruction, in-place rewrite ----------------------
    for kt in range(0, K, BLOCK_K):
        s_t = tl.load(base + offs_v[:, None] * stride_v + (kt + offs_bk)[None, :] * stride_k)
        uk_t = tl.load(uk_ptr + pid.to(tl.int64) * (K * 16) + (kt + offs_bk)[:, None] * 16 + offs_r[None, :]).to(tl.float32)  # [BK, 16]
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


@triton.jit
def _leapquant_payload_encode_kernel(
    state_ptr,  # fp32 [slots, H, V, K] (read; rewritten only if REWRITE_STATE)
    due_ptr,  # int32/int64 [D] slot ids to flush
    codes_ptr,  # int8 [slots, H, V, K] payload codes (out)
    sk_ptr,  # fp32 [slots, H, K] payload key scales (out)
    sv_ptr,  # fp32 [slots, H, V] payload value scales (out)
    u_ptr,  # fp16 [slots, H, R, K] payload key-side factors (out)
    q_ptr,  # fp16 [slots, H, R, V] payload value-side factors (out)
    hcnt_ptr,  # int32 [slots, H] window fill counters (out)
    hcnt_vals_ptr,  # int32 [D] per-due-row window fill counter (M3.3: mid-window flush phase)
    qv_ptr,  # fp16 [D*H, V, 16] fp16-rounded value-side factors (scratch)
    uk_ptr,  # fp16 [D*H, K, 16] fp16-rounded key-side coefficients (scratch)
    sk_s_ptr,  # fp32 [D*H, K] residual scales (scratch)
    sv_s_ptr,  # fp32 [D*H, V] residual scales (scratch)
    stride_slot: tl.int64,
    stride_head: tl.int64,
    stride_v: tl.int64,
    stride_k: tl.int64,
    pl_hb_i8: tl.int64,  # payload head-block stride, int8 (codes) elements
    pl_sb_i8: tl.int64,  # payload slot stride, int8 elements
    pl_hb_f16: tl.int64,  # ... fp16 elements (u/q fields)
    pl_sb_f16: tl.int64,
    pl_hb_f32: tl.int64,  # ... fp32 elements (s_k/s_v/hcnt fields)
    pl_sb_f32: tl.int64,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Persist the LeapQuant checkpoint (codes + scales + fp16 factors).

    Identical math to ``_leapquant_reconstruct_kernel`` — codes are cut
    from the *original* state read — but the outputs go to the payload
    pool and the fp32 state is never rewritten.  Dual-write mode (state
    pool and payload both refreshed) is expressed host-side as this
    kernel followed by ``_leapquant_reconstruct_kernel``: a fused
    REWRITE_STATE branch keeps both store sets live and overflows the AIV
    unified buffer at 128x128/R=4.  q/u are stored as ``[D, 16]``-plane
    slices in the same orientation as the scratch buffers (see module
    notes on transposed access patterns).
    """
    pid = tl.program_id(0)
    d = pid // H
    h = pid % H
    slot = tl.load(due_ptr + d).to(tl.int64)
    base = state_ptr + slot * stride_slot + h * stride_head
    pid64 = pid.to(tl.int64)

    offs_v = tl.arange(0, V)
    offs_k = tl.arange(0, K)
    offs_bk = tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, 16)

    qv_f = tl.load(qv_ptr + pid64 * (V * 16) + offs_v[:, None] * 16 + offs_r[None, :]).to(tl.float32)  # [V, 16]
    s_v = tl.load(sv_s_ptr + pid64 * V + offs_v)  # [V]
    q_cols = ()
    for r in tl.static_range(R):
        q_cols = q_cols + (tl.sum(tl.where(offs_r[None, :] == r, qv_f, 0.0), axis=1),)

    # ---- codes from the ORIGINAL state (+ optional dual fp32 rewrite) ----
    codes_base = codes_ptr + slot * pl_sb_i8 + h * pl_hb_i8
    for kt in range(0, K, BLOCK_K):
        s_t = tl.load(base + offs_v[:, None] * stride_v + (kt + offs_bk)[None, :] * stride_k)
        uk_t = tl.load(uk_ptr + pid64 * (K * 16) + (kt + offs_bk)[:, None] * 16 + offs_r[None, :]).to(tl.float32)  # [BK, 16]
        lr_t = tl.zeros([V, BLOCK_K], dtype=tl.float32)
        for r in tl.static_range(R):
            u_row = tl.sum(tl.where(offs_r[None, :] == r, uk_t, 0.0), axis=1)
            lr_t += q_cols[r][:, None] * u_row[None, :]
        e_t = s_t - lr_t
        a_t = tl.abs(e_t)
        s_k_t = tl.load(sk_s_ptr + pid64 * K + kt + offs_bk)
        mag = _round_half_even(a_t / (s_k_t[None, :] * s_v[:, None]) * 127.0)
        sgn = tl.where(e_t >= 0.0, 1.0, -1.0)
        codes = tl.minimum(tl.maximum(sgn * mag, -127.0), 127.0)
        # Values are already clamped to [-127, 127]; saturate is a no-op
        # safety net (swiglu_quant.py precedent for int8 casts).
        codes_i8 = codes.cast(tl.int8, overflow_mode="saturate")
        tl.store(codes_base + offs_v[:, None] * K + (kt + offs_bk)[None, :], codes_i8)

    # ---- scales + compensator planes -> payload fields ------------------
    tl.store(sk_ptr + slot * pl_sb_f32 + h * pl_hb_f32 + offs_k, tl.load(sk_s_ptr + pid64 * K + offs_k))
    tl.store(sv_ptr + slot * pl_sb_f32 + h * pl_hb_f32 + offs_v, s_v)
    uk_f = tl.load(uk_ptr + pid64 * (K * 16) + offs_k[:, None] * 16 + offs_r[None, :])  # [K, 16] fp16
    # The scratch planes are fp16 memory (the only reliable rounding point
    # on this build), so the payload fp16 stores are value-preserving.
    tl.store(u_ptr + slot * pl_sb_f16 + h * pl_hb_f16 + offs_k[:, None] * R + offs_r[None, :],
             uk_f.to(u_ptr.dtype.element_ty), mask=offs_r[None, :] < R)
    tl.store(q_ptr + slot * pl_sb_f16 + h * pl_hb_f16 + offs_v[:, None] * R + offs_r[None, :],
             qv_f.to(q_ptr.dtype.element_ty), mask=offs_r[None, :] < R)
    tl.store(hcnt_ptr + slot * pl_sb_f32 + h * pl_hb_f32, tl.load(hcnt_vals_ptr + d))


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
    payload=None,  # LeapQuantPayloadPool (duck-typed; avoids an import cycle)
    rewrite_state: bool | None = None,
    hcnt_value: int = 0,
    hcnt_values: torch.Tensor | None = None,
) -> None:
    """Fused LeapQuant flush of ``slot_ids`` rows of a state pool.

    Mirrors ``leap_window_flush(state[slot_ids])`` with the gather /
    flush / scatter fused into two kernel launches (PR1 runs them as
    three host-side steps plus ~15 torch ops).  Two modes share the same
    scales launch:

    * ``payload=None`` (roundtrip, M3.1): the flush round-trip is written
      back into the fp32 pool in place.  This is the accuracy baseline
      and the permanent fallback path.
    * ``payload=pool`` (M3.2): the checkpoint (int8 codes, fp32 scales,
      fp16 compensator planes, hcnt) is persisted into ``pool`` instead.
      ``rewrite_state=True`` additionally keeps the fp32 in-place rewrite
      (dual-write, migration gate); the default ``False`` is the pure
      memory-saving mode whose state rows are left untouched.

    Args:
        state: fp32 pool ``[slots, H, V, K]``; ``V``/``K`` must be powers
            of two (Qwen3.5/Kimi GDN heads are 128x128).  Arbitrary batch
            and head strides are allowed.
        slot_ids: 1D int32/int64 tensor of slot rows whose sequences just
            crossed a window boundary.
        compensator_tokens: rank ``R`` of the compensator subspace (1..16).
        payload: ``LeapQuantPayloadPool`` whose geometry must match
            ``state`` and ``compensator_tokens``.
        rewrite_state: payload-mode fp32 rewrite flag (see above); must
            be None in roundtrip mode.
        hcnt_value: window fill counter stored for every due (slot, head);
            used when ``hcnt_values`` is None.
        hcnt_values: per-due-row window fill counters (M3.3).  A flush at
            an exact boundary stores 0 (fresh window); a chunked-prefill
            flush landing mid-window stores ``seq_len % window`` so the
            payload readout knows the checkpoint's window phase.  int32,
            ``[len(slot_ids)]``, on ``state.device``; overrides
            ``hcnt_value``.
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

    if payload is not None:
        lay = payload.layout
        if (lay.heads, lay.v_dim, lay.k_dim) != (heads, v_dim, k_dim):
            raise ValueError(
                f"payload geometry [H={lay.heads}, V={lay.v_dim}, K={lay.k_dim}] "
                f"!= state [{heads}, {v_dim}, {k_dim}]"
            )
        if lay.rank != compensator_tokens:
            raise ValueError(
                f"payload rank {lay.rank} != compensator_tokens {compensator_tokens}"
            )
        if payload.buffer.device != state.device:
            raise ValueError("payload and state must be on the same device")
    else:
        if rewrite_state is not None:
            raise ValueError("rewrite_state is only valid in payload mode")
        rewrite_state = True  # roundtrip: always rewrite in place
    if payload is not None and rewrite_state is None:
        rewrite_state = False  # pure payload mode: leave the fp32 pool alone

    due = slot_ids.to(torch.int64) if slot_ids.dtype != torch.int64 else slot_ids
    if hcnt_values is None:
        hcnt_vals_t = torch.full(
            (due.numel(),), hcnt_value, dtype=torch.int32, device=state.device
        )
    else:
        if hcnt_values.dtype != torch.int32 or hcnt_values.device != state.device:
            raise ValueError("hcnt_values must be an int32 tensor on state.device")
        if hcnt_values.shape != due.shape:
            raise ValueError(f"hcnt_values shape {tuple(hcnt_values.shape)} != slot_ids {tuple(due.shape)}")
        hcnt_vals_t = hcnt_values
    q0 = _get_q0(v_dim, compensator_tokens, state.device)
    blocks = due.numel() * heads
    qk_s = torch.empty((blocks, k_dim, 16), dtype=torch.float32, device=state.device)
    # qv/uk scratch must be fp16 *memory* — in-register fp16 casts fold to
    # no-ops on this build (module notes) — plus a raw fp32 qv plane for the
    # uk = S^T qv subspace step, which runs on un-rounded factors.
    qvr_s = torch.empty((blocks, v_dim, 16), dtype=torch.float32, device=state.device)
    qv_s = torch.empty((blocks, v_dim, 16), dtype=torch.float16, device=state.device)
    uk_s = torch.empty((blocks, k_dim, 16), dtype=torch.float16, device=state.device)
    sk = torch.empty((blocks, k_dim), dtype=torch.float32, device=state.device)
    sv = torch.empty((blocks, v_dim), dtype=torch.float32, device=state.device)
    grid = (blocks,)
    _leapquant_flush_scales_kernel[grid](
        state,
        due,
        q0,
        qk_s,
        qvr_s,
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
    if payload is None:
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
        return

    # Actual byte strides: engine pools are strided views over the mamba page
    # (slot stride = whole conv+ssm+payload page, possibly padded), so take
    # the strides from the pool buffer rather than the contiguous layout.
    hb = payload.head_stride_bytes
    sb = payload.slot_stride_bytes
    _leapquant_payload_encode_kernel[grid](
        state,
        due,
        payload.codes,
        payload.s_k,
        payload.s_v,
        payload.u,
        payload.q,
        payload.hcnt,
        hcnt_vals_t,
        qv_s,
        uk_s,
        sk,
        sv,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        hb,
        sb,
        hb // 2,
        sb // 2,
        hb // 4,
        sb // 4,
        H=heads,
        V=v_dim,
        K=k_dim,
        R=compensator_tokens,
        BLOCK_K=32,
        num_warps=8,
    )
    if rewrite_state:
        # Dual-write gate: refresh the fp32 pool with the identical
        # round-trip (same scratch inputs -> bit-equal to the roundtrip
        # mode's in-place result).
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
