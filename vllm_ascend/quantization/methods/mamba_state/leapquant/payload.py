# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Persistent LeapQuant payload storage (RFC vllm-ascend#21097 M3, plan §2.1).

The M1/M3.1 "roundtrip" flush quantizes a window-boundary state row and
immediately rebuilds fp32 in place — the pool stays fp32 and no memory is
saved.  The payload pool instead *stores* the LeapQuant checkpoint per
(slot, head) block, which is the actual memory win::

    codes : int8  [H, V, K]    V*K bytes
    s_k   : fp32  [H, K]       4*K bytes
    s_v   : fp32  [H, V]       4*V bytes
    u     : fp16  [H, R, K]    2*R*K bytes   (key-side compensator factors)
    q     : fp16  [H, R, V]    2*R*V bytes   (value-side compensator factors)
    hcnt  : int32 [H]          4 bytes       (window fill counter)

At GDN/KDA head geometry (V=K=128) and R=4 that is 19460 B per head block
= 1.1877 bytes per state element vs 4 bytes fp32 — a 3.37x smaller pool
(the ~1.19 B/elem figure of the paper).  The transient fp32 workspace the
recurrent kernel updates remains per-active-request only (M3.4 wires
that side).

All views alias one uint8 buffer, laid out head-major inside each slot:

    [ slot ][ head ][ codes | s_k | s_v | u | q | hcnt ]

Field order keeps every offset a multiple of 4 (V/K are powers of two in
every supported model).  ``leapquant_flush_triton(..., payload=pool)`` on
the triton side fills due rows; ``reconstruct_rows`` is the torch-side
read-back used by unit tests and (later) host-side tooling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_ascend.quantization.methods.mamba_state.leapquant.state import (
    reconstruct_state,
)

__all__ = ["PayloadLayout", "LeapQuantPayloadPool"]


@dataclass(frozen=True)
class PayloadLayout:
    """Static byte layout of a payload pool (immutable geometry)."""

    slots: int
    heads: int
    v_dim: int
    k_dim: int
    rank: int

    def __post_init__(self) -> None:
        for name in ("slots", "heads", "v_dim", "k_dim", "rank"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.v_dim & (self.v_dim - 1) or self.k_dim & (self.k_dim - 1):
            raise ValueError(
                f"V/K must be powers of two, got V={self.v_dim} K={self.k_dim}"
            )
        if self.rank > 16:
            raise ValueError(f"rank must be <= 16, got {self.rank}")

    # ---- per-head-block field sizes in bytes --------------------------
    @property
    def codes_bytes(self) -> int:
        return self.v_dim * self.k_dim

    @property
    def s_k_bytes(self) -> int:
        return 4 * self.k_dim

    @property
    def s_v_bytes(self) -> int:
        return 4 * self.v_dim

    @property
    def u_bytes(self) -> int:
        return 2 * self.rank * self.k_dim

    @property
    def q_bytes(self) -> int:
        return 2 * self.rank * self.v_dim

    @property
    def head_block_bytes(self) -> int:
        return (
            self.codes_bytes
            + self.s_k_bytes
            + self.s_v_bytes
            + self.u_bytes
            + self.q_bytes
            + 4  # hcnt
        )

    @property
    def slot_bytes(self) -> int:
        return self.head_block_bytes * self.heads

    @property
    def total_bytes(self) -> int:
        return self.slot_bytes * self.slots

    @property
    def bytes_per_element(self) -> float:
        """Payload bytes per fp32 state element (the paper's ~1.19 B/elem)."""
        return self.head_block_bytes / (self.v_dim * self.k_dim)

    def field_offsets(self) -> dict[str, int]:
        """Byte offset of each field inside one head block."""
        o = 0
        offs: dict[str, int] = {}
        for name, size in (
            ("codes", self.codes_bytes),
            ("s_k", self.s_k_bytes),
            ("s_v", self.s_v_bytes),
            ("u", self.u_bytes),
            ("q", self.q_bytes),
            ("hcnt", 4),
        ):
            offs[name] = o
            o += size
        return offs


class LeapQuantPayloadPool:
    """One uint8 buffer unpacked into the per-field payload views.

    The pool owns nothing but the buffer; the engine-side allocation
    (``get_state_dtype``/``bind_kv_cache`` reinterpret) plugs in at M3.2b.
    """

    layout: PayloadLayout
    buffer: torch.Tensor  # uint8 [total_bytes]

    def __init__(self, buffer: torch.Tensor, layout: PayloadLayout) -> None:
        if buffer.dtype != torch.uint8 or buffer.ndim != 1:
            raise ValueError("buffer must be a flat uint8 tensor")
        if buffer.numel() != layout.total_bytes:
            raise ValueError(
                f"buffer has {buffer.numel()} bytes, layout needs {layout.total_bytes}"
            )
        self.layout = layout
        self.buffer = buffer

    @classmethod
    def allocate(
        cls,
        slots: int,
        heads: int,
        v_dim: int,
        k_dim: int,
        rank: int,
        *,
        device: torch.device,
    ) -> "LeapQuantPayloadPool":
        layout = PayloadLayout(slots, heads, v_dim, k_dim, rank)
        buffer = torch.zeros(layout.total_bytes, dtype=torch.uint8, device=device)
        return cls(buffer, layout)

    # ---- unpacked views (all alias self.buffer) ------------------------
    def _view(self, name: str, dtype: torch.dtype, per_head_shape: tuple[int, ...]):
        lay = self.layout
        off = lay.field_offsets()[name]
        field_numel = 1
        for s in per_head_shape:
            field_numel *= s
        # A flat uint8 -> dtype reinterpret of the whole buffer, then slice
        # head blocks.  head_block_bytes is a multiple of the dtype size, so
        # every field stays aligned regardless of the preceding fields.
        if lay.head_block_bytes % torch.empty((), dtype=dtype).element_size() != 0:
            raise ValueError("head block not aligned for dtype")
        flat = self.buffer.view(dtype)
        per_head_fields = lay.head_block_bytes // torch.empty((), dtype=dtype).element_size()
        # fields of one dtype are contiguous inside the head block only if no
        # differently-sized field interleaves; here each field IS one dtype
        # run, so recompute the field's element offset explicitly.
        elem_off = off // torch.empty((), dtype=dtype).element_size()
        if elem_off * torch.empty((), dtype=dtype).element_size() != off:
            raise ValueError(f"field {name} misaligned for {dtype}")
        field_elems = lay.slot_bytes // torch.empty((), dtype=dtype).element_size()
        field_stride_head = lay.head_block_bytes // torch.empty((), dtype=dtype).element_size()
        field_stride_slot = lay.slot_bytes // torch.empty((), dtype=dtype).element_size()
        # reshape bookkeeping: buffer as [slots, heads, head_block_bytes//esz]
        head_fields = lay.head_block_bytes // torch.empty((), dtype=dtype).element_size()
        base = flat.view(lay.slots, lay.heads, head_fields)
        return base[:, :, elem_off : elem_off + field_numel].reshape(
            lay.slots, lay.heads, *per_head_shape
        )

    @property
    def codes(self) -> torch.Tensor:
        """int8 [slots, H, V, K]."""
        return self._view("codes", torch.int8, (self.layout.v_dim, self.layout.k_dim))

    @property
    def s_k(self) -> torch.Tensor:
        """fp32 [slots, H, K]."""
        return self._view("s_k", torch.float32, (self.layout.k_dim,))

    @property
    def s_v(self) -> torch.Tensor:
        """fp32 [slots, H, V]."""
        return self._view("s_v", torch.float32, (self.layout.v_dim,))

    @property
    def u(self) -> torch.Tensor:
        """fp16 [slots, H, R, K] key-side compensator factors.

        Physically stored ``[K, R]``-row-major (one contiguous ``[K, 16]``
        plane per head, mirroring the kernel scratch layout) and exposed
        transposed; the strided view is fine for the einsum read-back.
        """
        return self._view("u", torch.float16, (self.layout.k_dim, self.layout.rank)).transpose(-2, -1)

    @property
    def q(self) -> torch.Tensor:
        """fp16 [slots, H, R, V] value-side compensator factors ([V, R] plane)."""
        return self._view("q", torch.float16, (self.layout.v_dim, self.layout.rank)).transpose(-2, -1)

    @property
    def hcnt(self) -> torch.Tensor:
        """int32 [slots, H] window fill counters."""
        return self._view("hcnt", torch.int32, (1,)).squeeze(-1)

    # ---- torch-side read-back ------------------------------------------
    def reconstruct_rows(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """Rebuild fp32 state ``[n, H, V, K]`` for the given slots (eq. 8-9)."""
        lay = self.layout
        idx = slot_ids.to(torch.int64)
        rows = torch.stack(
            [
                reconstruct_state(
                    self.codes[idx][:, h].float(),
                    self.s_k[idx][:, h],
                    self.s_v[idx][:, h],
                    self.u[idx][:, h],
                    self.q[idx][:, h],
                )
                for h in range(lay.heads)
            ],
            dim=1,
        )
        return rows
