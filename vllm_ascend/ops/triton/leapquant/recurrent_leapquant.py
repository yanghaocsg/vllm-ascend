# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Payload-reading recurrent decode kernel for LeapQuant (RFC vllm-ascend#21097 M3.4).

Replaces the fp32 state-pool read of the production decode path (GDN first,
KDA in ``recurrent_kda_leapquant``) with a **payload readout**: the
window-boundary checkpoint ``S0 = q^T u + codes * (s_k/127) * s_v``
(paper eq. 8-9) is reconstructed directly into each program's ``[BV, BK]``
fp32 tile, after which the standard gated-delta-rule recurrence runs
unchanged:

    b_h *= exp(g)                     # decay: scalar (GDN) / per-K column (KDA)
    v~  = v - (S k)_v                 # delta correction
    v~  *= beta
    S   += v~ k^T                     # outer-product update
    o    = (S q) * scale              # readout

KDA vs GDN is exactly one constexpr away (``PER_K_DECAY``): the recurrence
loads ``g [B, T, HV, K]`` (log space, one column per key channel) instead of
the scalar ``g [B, T, HV]`` — the KDA update
``(I − βkk^T)Diag(α)S + βk v^T`` equals ``Diag(α)S + βk(v − S^T(α⊙k))^T``
(paper appendix A), so after decaying each K column the remaining four lines
are the GDN ones verbatim. The gate expansion itself stays host-side
(``kda_expand_gate``): the fla_npu AscendC ``recurrent_kda`` binary computes
``-exp(A_log)·softplus(a + dt_bias)`` (safe variant
``lower_bound·sigmoid(exp(A_log)·(a + dt_bias))``) in-kernel; since the
binary cannot be modified, the triton path expands the same formula in torch
before launch.

This is the paper's eq. 10 readout staged as "checkpoint reconstruct +
in-window recurrence": within a window the running state lives in the
per-active-request **W workspace** (fp32 ``[rows, HV, V, K]``, ~B/slots of
the old pool), while the *pooled* storage is the int8 payload (~1.19
B/elem, the M3.2 pool) — with the fp32 ssm component of the mamba page
dropped, pool capacity rises ~3.2x.

Kernel contract mirrors the proven triton skeleton
(``ops/triton/kda/fused_recurrent_kda.py`` / upstream fla
``fused_recurrent.py``): one program owns one (K-block, V-block, seq x
v-head) tile for the whole token loop; state per head is ``[V, K]`` (V rows,
K cols) exactly matching the payload field geometry (codes ``[V, K]``,
``s_v`` per V row, ``s_k`` per K column).

Modes (wrapper): steady-state rows pass ``w_in`` (fp32 W workspace); a cold
row (fresh window / resume) omits it and the kernel reconstructs the
initial tile from the payload checkpoint (``USE_PAYLOAD_INIT``). NULL rows
/ slots (sentinel ``-1``, the ``ssm_state_indices`` convention) run on a
zero state and skip the workspace write. The window-boundary re-encode
stays host-side as the M3.2 split launches; fusing it in-kernel is a later
optimization, not a correctness need.

triton-ascend notes (same build constraints as ``flush_kernel.py``): no
``tl.dot`` anywhere (row-dots + outer products, like the stock recurrent
kernels); rank columns extracted only with the axis-1
``tl.sum(tl.where(offs_r == r, block, 0), axis=1)`` pattern; payload
strides passed per field dtype like ``_leapquant_payload_encode_kernel``.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.triton_utils import tl, triton

__all__ = ["kda_expand_gate", "recurrent_gdn_leapquant", "recurrent_kda_leapquant"]


@triton.jit
def _recurrent_gdn_leapquant_kernel(
    q_ptr,  # [B, T, H, K], host-side l2norm'd (GDN convention)
    k_ptr,  # [B, T, H, K]
    v_ptr,  # [B, T, HV, V]
    g_ptr,  # [B, T, HV] fp32 scalar decay (log space); [B, T, HV, K] if PER_K_DECAY
    beta_ptr,  # [B, T, HV] fp32, sigmoid already applied
    o_ptr,  # [B, T, HV, V]
    w_in_ptr,  # fp32 [rows, HV, V, K] workspace read (dead if PAYLOAD_INIT)
    w_out_ptr,  # fp32 [rows, HV, V, K] workspace write
    w_row_ptr,  # int [B] workspace row id per sequence
    codes_ptr,  # int8 payload codes (PAYLOAD_INIT only)
    sk_ptr,  # fp32 payload key scales [slots, H, K]
    sv_ptr,  # fp32 payload value scales [slots, H, V]
    u_ptr,  # fp16 payload key-side factors (physical [K, R] plane)
    qp_ptr,  # fp16 payload value-side factors (physical [V, R] plane)
    slot_ptr,  # int [B] payload slot id per sequence (NULL <= 0)
    stride_qb: tl.int64, stride_qt: tl.int64, stride_qh: tl.int64, stride_qd: tl.int64,
    stride_vb: tl.int64, stride_vt: tl.int64, stride_vh: tl.int64, stride_vd: tl.int64,
    stride_ob: tl.int64, stride_ot: tl.int64, stride_oh: tl.int64, stride_od: tl.int64,
    stride_gb: tl.int64, stride_gt: tl.int64, stride_gh: tl.int64, stride_gk: tl.int64,
    stride_bb: tl.int64, stride_bt: tl.int64, stride_bh: tl.int64,
    stride_wi_row: tl.int64, stride_wi_head: tl.int64, stride_wi_v: tl.int64, stride_wi_k: tl.int64,
    stride_wo_row: tl.int64, stride_wo_head: tl.int64, stride_wo_v: tl.int64, stride_wo_k: tl.int64,
    pl_hb_i8: tl.int64, pl_sb_i8: tl.int64,  # payload strides, codes elements
    pl_hb_f16: tl.int64, pl_sb_f16: tl.int64,  # ... fp16 (u/q fields)
    pl_hb_f32: tl.int64, pl_sb_f32: tl.int64,  # ... fp32 (s_k/s_v fields)
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    R: tl.constexpr,
    USE_PAYLOAD_INIT: tl.constexpr,
    PER_K_DECAY: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_nh = tl.program_id(2)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)  # GVA head mapping

    o_v = i_v * BV + tl.arange(0, BV)
    o_k = i_k * BK + tl.arange(0, BK)
    offs_r = tl.arange(0, 16)

    # workspace row id; NULL rows (sentinel -1, matching ssm_state_indices)
    # skip the workspace read/write entirely — the recurrence still runs on
    # a zero state so the output row stays well-defined.
    row = tl.load(w_row_ptr + i_n).to(tl.int64)

    if USE_PAYLOAD_INIT:
        # ---- S0 = q^T u + codes * (s_k/127) * s_v  (eq. 8-9 readout) ----
        slot = tl.load(slot_ptr + i_n).to(tl.int64)
        if slot < 0:
            b_h = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            codes = tl.load(
                codes_ptr + slot * pl_sb_i8 + i_hv * pl_hb_i8
                + o_v[:, None] * K + o_k[None, :]
            ).to(tl.float32)
            s_k = tl.load(sk_ptr + slot * pl_sb_f32 + i_hv * pl_hb_f32 + o_k)
            s_v = tl.load(sv_ptr + slot * pl_sb_f32 + i_hv * pl_hb_f32 + o_v)
            # low-rank term: rank-R outer products from the fp16 planes;
            # axis-1 extraction only (build constraint, see flush_kernel.py).
            q_plane = tl.load(
                qp_ptr + slot * pl_sb_f16 + i_hv * pl_hb_f16
                + o_v[:, None] * R + offs_r[None, :],
                mask=offs_r[None, :] < R, other=0.0,
            ).to(tl.float32)  # [BV, 16]
            u_plane = tl.load(
                u_ptr + slot * pl_sb_f16 + i_hv * pl_hb_f16
                + o_k[:, None] * R + offs_r[None, :],
                mask=offs_r[None, :] < R, other=0.0,
            ).to(tl.float32)  # [BK, 16]
            b_h = codes * (s_k[None, :] / 127.0) * s_v[:, None]
            for r in tl.static_range(R):
                q_col = tl.sum(tl.where(offs_r[None, :] == r, q_plane, 0.0), axis=1)
                u_col = tl.sum(tl.where(offs_r[None, :] == r, u_plane, 0.0), axis=1)
                b_h += q_col[:, None] * u_col[None, :]
    else:
        if row < 0:
            b_h = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            b_h = tl.load(
                w_in_ptr + row * stride_wi_row + i_hv * stride_wi_head
                + o_v[:, None] * stride_wi_v + o_k[None, :] * stride_wi_k
            )

    # ---- token loop: the stock gated-delta-rule recurrence ---------------
    for i_t in range(0, T):
        b_q = tl.load(
            q_ptr + i_n * stride_qb + i_t * stride_qt + i_h * stride_qh + o_k
        ).to(tl.float32)
        b_k = tl.load(
            k_ptr + i_n * stride_qb + i_t * stride_qt + i_h * stride_qh + o_k
        ).to(tl.float32)
        b_v = tl.load(
            v_ptr + i_n * stride_vb + i_t * stride_vt + i_hv * stride_vh + o_v
        ).to(tl.float32)
        b_g = tl.load(g_ptr + i_n * stride_gb + i_t * stride_gt + i_hv * stride_gh).to(tl.float32)
        b_beta = tl.load(beta_ptr + i_n * stride_bb + i_t * stride_bt + i_hv * stride_bh).to(tl.float32)

        if PER_K_DECAY:
            # KDA: per-K-column decay (g [B, T, HV, K], log space). Loading
            # g's K vector through stride_gk keeps the fla layout contract
            # (gate indexed by the v-head id; Kimi-Linear is H == HV).
            b_gk = tl.load(
                g_ptr + i_n * stride_gb + i_t * stride_gt + i_hv * stride_gh + o_k * stride_gk
            ).to(tl.float32)
            b_h *= tl.exp(b_gk)[None, :]
        else:
            # scalar decay (GDN): whole-state gate
            b_h *= tl.exp(b_g)
        # delta correction: v~ = v - (S k)_v
        b_v = b_v - tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        # outer-product update: S += v~ k^T
        b_h += b_v[:, None] * b_k[None, :]
        # readout: o = (S q) * scale
        b_o = tl.sum(b_h * b_q[None, :], 1) * scale
        tl.store(
            o_ptr + i_n * stride_ob + i_t * stride_ot + i_hv * stride_oh + o_v,
            b_o.to(o_ptr.dtype.element_ty),
        )

    # ---- final state -> W workspace row ----------------------------------
    if row >= 0:
        tl.store(
            w_out_ptr + row * stride_wo_row + i_hv * stride_wo_head
            + o_v[:, None] * stride_wo_v + o_k[None, :] * stride_wo_k,
            b_h,
        )


def _payload_launch_args(
    payload: Any,
    device: torch.device,
    slot_ids: torch.Tensor,
    hv: int,
    v: int,
    k: int,
) -> tuple:
    """Build the payload kernel-argument tail shared by both wrappers.

    Returns ``(codes, s_k, s_v, u_plane, q_plane, slot_ids, hb_i8, sb_i8,
    hb_f16, sb_f16, hb_f32, sb_f32)`` — per-field element strides derived from
    the pool's byte strides (engine pools are strided views over the mamba
    page), exactly the plumbing ``leapquant_flush_triton`` launches with.
    """
    lay = payload.layout
    if (lay.heads, lay.v_dim, lay.k_dim) != (hv, v, k):
        raise ValueError(
            f"payload geometry [H={lay.heads}, V={lay.v_dim}, K={lay.k_dim}] "
            f"!= state [{hv}, {v}, {k}]"
        )
    if payload.buffer.device != device:
        raise ValueError("payload and inputs must be on the same device")
    hb, sb = payload.head_stride_bytes, payload.slot_stride_bytes
    return (
        payload.codes, payload.s_k, payload.s_v,
        payload.u.transpose(-2, -1), payload.q.transpose(-2, -1),  # physical planes
        slot_ids.to(device),
        hb, sb, hb // 2, sb // 2, hb // 4, sb // 4,
    )


def _launch_common(
    *,
    q, k, v, g, beta, out, w_in, w_out, w_rows, pl_args, wi_strides,
    g_strides: tuple,
    per_k_decay: bool,
    hv: int, v_dim: int, k_dim: int, t_len: int, h_cnt: int, rank: int,
    use_payload_init: bool, scale: float,
) -> None:
    """Shared launch tail: validation of strides + the kernel grid call.

    Kept as one function so the GDN and KDA wrappers cannot drift apart on
    anything except the decay geometry (``PER_K_DECAY`` + the g stride set:
    3 strides for the scalar GDN gate, 4 for the per-K-column KDA gate).
    """
    # k shares q's stride set (same [B, T, H, K] geometry; stock recurrent
    # kernels do the same).  Guard instead of silently mis-addressing.
    if tuple(k.stride()) != tuple(q.stride()):
        k = k.contiguous()
    for t_, name_ in ((q, "q"), (k, "k"), (v, "v"), (out, "out"), (g, "g"), (beta, "beta")):
        if t_.stride(-1) != 1:
            raise ValueError(f"{name_} must be contiguous in the last dimension")

    bk = triton.next_power_of_2(k_dim)
    bv = min(triton.next_power_of_2(v_dim), 8)  # proven tile geometry (KDA kernel)
    grid = (k_dim // bk, v_dim // bv, q.shape[0] * hv)
    _recurrent_gdn_leapquant_kernel[grid](
        q, k, v, g, beta, out,
        w_in if w_in is not None else out,  # dead under USE_PAYLOAD_INIT
        w_out,
        w_rows.to(q.device),
        *pl_args[:6],  # codes, s_k, s_v, u, q planes, slot ids
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        *g_strides,
        beta.stride(0), beta.stride(1), beta.stride(2),
        *wi_strides,
        w_out.stride(0), w_out.stride(1), w_out.stride(2), w_out.stride(3),
        *pl_args[6:],  # payload per-field strides
        scale,
        H=h_cnt, HV=hv, K=k_dim, V=v_dim, T=t_len, R=rank,
        USE_PAYLOAD_INIT=use_payload_init,
        PER_K_DECAY=per_k_decay,
        BV=bv, BK=bk,
        num_warps=1,
        num_stages=3,
    )


def recurrent_gdn_leapquant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    w_in: torch.Tensor | None = None,
    w_out: torch.Tensor | None = None,
    w_rows: torch.Tensor | None = None,
    payload: Any | None = None,
    slot_ids: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """GDN decode step reading state from the W workspace / payload checkpoint.

    Args:
        q/k: ``[B, T, H, K]`` — caller has applied l2norm (GDN convention:
            the fla_npu AscendC op gets l2norm'd inputs too).
        v: ``[B, T, HV, V]``.
        g: ``[B, T, HV]`` fp32 scalar decay (log space, applied as ``exp(g)``).
        beta: ``[B, T, HV]`` fp32, sigmoid already applied.
        scale: readout scale; defaults ``K ** -0.5``, applied in-kernel.
        w_in: fp32 ``[rows, HV, V, K]`` running-state workspace (steady path).
        w_out: fp32 destination rows; defaults to ``w_in``.
        w_rows: ``[B]`` int workspace row id per sequence (required).
        payload: ``LeapQuantPayloadPool`` for cold rows (``w_in=None`` mode);
            ``slot_ids`` ``[B]`` gives the pool row (NULL ``< 0``, the
            ``ssm_state_indices`` sentinel, starts from a zero state).
        out: optional preallocated ``[B, T, HV, V]`` output.

    Returns:
        ``o [B, T, HV, V]``; ``w_out`` rows hold the post-step state.
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    if scale is None:
        scale = K**-0.5
    if w_rows is None:
        raise ValueError("w_rows is required (workspace row id per sequence)")
    if w_out is None:
        w_out = w_in
    use_payload_init = w_in is None
    if use_payload_init:
        if payload is None or slot_ids is None:
            raise ValueError("payload + slot_ids are required when w_in is None")
        pl_args = _payload_launch_args(payload, q.device, slot_ids, HV, V, K)
        rank = payload.layout.rank
    else:
        if w_in.dim() != 4 or tuple(w_in.shape[1:]) != (HV, V, K):
            raise ValueError(f"w_in must be [rows, {HV}, {V}, {K}], got {tuple(w_in.shape)}")
        rank = 1
        # dead pointers (USE_PAYLOAD_INIT=False eliminates the loads)
        pl_args = (q, q, q, q, q, w_rows, 0, 0, 0, 0, 0, 0)

    if out is None:
        out = torch.empty_like(v)
    wi_strides = (
        (w_in.stride(0), w_in.stride(1), w_in.stride(2), w_in.stride(3))
        if w_in is not None
        else (0, 0, 0, 0)
    )
    _launch_common(
        q=q, k=k, v=v, g=g, beta=beta, out=out, w_in=w_in, w_out=w_out,
        w_rows=w_rows, pl_args=pl_args, wi_strides=wi_strides,
        g_strides=(g.stride(0), g.stride(1), g.stride(2), 0),
        per_k_decay=False,
        hv=HV, v_dim=V, k_dim=K, t_len=T, h_cnt=H, rank=rank,
        use_payload_init=use_payload_init, scale=scale,
    )
    return out


def kda_expand_gate(
    a: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Expand raw KDA gate activations to log-space per-K decay (fla formula).

    Mirrors the in-kernel expansion of the fla_npu AscendC ``recurrent_kda``
    binary (which cannot be modified), so the triton decode path reproduces
    the production gate numerics:

    * softplus variant (``lower_bound is None``, Kimi-Linear):

        ``g = -exp(A_log) * softplus(a + dt_bias)``
        with the numerically stable ``softplus`` (linear above 20)

    * safe variant (``lower_bound`` in ``[-5, 0)``, bounded decay):

        ``g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))``

    Args:
        a: raw gate ``[B, T, HV, K]`` (the model's ``g1`` after head split).
        a_log: ``[HV]`` fp32 per-head log-amplitude.
        dt_bias: ``[HV, K]`` (or flat ``[HV*K]``) fp32 per-dim bias.
        lower_bound: optional safe-gate bound; ``None`` selects softplus.

    Returns:
        ``[B, T, HV, K]`` fp32 log-space decay consumed by
        ``recurrent_kda_leapquant``.
    """
    hv, k = a_log.numel(), a.shape[-1]
    x = a.float() + dt_bias.float().reshape(1, 1, hv, k)
    a_exp = torch.exp(a_log.float().view(1, 1, hv, 1))
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(a_exp * x)
    return -a_exp * torch.nn.functional.softplus(x, beta=1, threshold=20.0)


def recurrent_kda_leapquant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    w_in: torch.Tensor | None = None,
    w_out: torch.Tensor | None = None,
    w_rows: torch.Tensor | None = None,
    payload: Any | None = None,
    slot_ids: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """KDA decode step reading state from the W workspace / payload checkpoint.

    Same contract as :func:`recurrent_gdn_leapquant` (shared kernel, shared
    payload-init / W / NULL plumbing) with the KDA decay geometry: ``g`` is
    the **per-K-column** log-space decay ``[B, T, HV, K]`` produced by
    :func:`kda_expand_gate`, and each key channel decays independently —
    the only recurrence difference to GDN (paper appendix A).

    Args:
        q/k: ``[B, T, H, K]`` — caller has applied l2norm (Kimi-Linear is
            ``H == HV``; GVA KDA would need the gate head mapping revisited).
        v: ``[B, T, HV, V]``.
        g: ``[B, T, HV, K]`` fp32 log-space per-K decay (``kda_expand_gate``
            output — gate expansion stays host-side because the AscendC
            binary doing it in-kernel cannot be reused for the payload
            readout).
        beta: ``[B, T, HV]`` fp32, sigmoid already applied.
        scale: readout scale; defaults ``K ** -0.5``, applied in-kernel.
        w_in: fp32 ``[rows, HV, V, K]`` running-state workspace (steady path).
        w_out: fp32 destination rows; defaults to ``w_in``.
        w_rows: ``[B]`` int workspace row id per sequence (required).
        payload: ``LeapQuantPayloadPool`` for cold rows (``w_in=None`` mode);
            ``slot_ids`` ``[B]`` gives the pool row (NULL ``< 0`` starts from
            a zero state).
        out: optional preallocated ``[B, T, HV, V]`` output.

    Returns:
        ``o [B, T, HV, V]``; ``w_out`` rows hold the post-step state.
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    if H != HV:
        raise ValueError(
            f"recurrent_kda_leapquant expects H == HV (Kimi-Linear geometry), "
            f"got H={H}, HV={HV}; GVA KDA needs a gate head mapping first"
        )
    if g.dim() != 4 or tuple(g.shape) != (B, T, HV, K):
        raise ValueError(
            f"g must be the per-K decay [{B}, {T}, {HV}, {K}] (kda_expand_gate "
            f"output), got {tuple(g.shape)}"
        )
    if scale is None:
        scale = K**-0.5
    if w_rows is None:
        raise ValueError("w_rows is required (workspace row id per sequence)")
    if w_out is None:
        w_out = w_in
    use_payload_init = w_in is None
    if use_payload_init:
        if payload is None or slot_ids is None:
            raise ValueError("payload + slot_ids are required when w_in is None")
        pl_args = _payload_launch_args(payload, q.device, slot_ids, HV, V, K)
        rank = payload.layout.rank
    else:
        if w_in.dim() != 4 or tuple(w_in.shape[1:]) != (HV, V, K):
            raise ValueError(f"w_in must be [rows, {HV}, {V}, {K}], got {tuple(w_in.shape)}")
        rank = 1
        # dead pointers (USE_PAYLOAD_INIT=False eliminates the loads)
        pl_args = (q, q, q, q, q, w_rows, 0, 0, 0, 0, 0, 0)

    if out is None:
        out = torch.empty_like(v)
    wi_strides = (
        (w_in.stride(0), w_in.stride(1), w_in.stride(2), w_in.stride(3))
        if w_in is not None
        else (0, 0, 0, 0)
    )
    _launch_common(
        q=q, k=k, v=v, g=g, beta=beta, out=out, w_in=w_in, w_out=w_out,
        w_rows=w_rows, pl_args=pl_args, wi_strides=wi_strides,
        g_strides=(g.stride(0), g.stride(1), g.stride(2), g.stride(3)),
        per_k_decay=True,
        hv=HV, v_dim=V, k_dim=K, t_len=T, h_cnt=H, rank=rank,
        use_payload_init=use_payload_init, scale=scale,
    )
    return out
