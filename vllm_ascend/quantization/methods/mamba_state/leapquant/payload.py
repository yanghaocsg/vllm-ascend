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

The buffer may be **strided**: the engine allocation (M3.2b) stores the
payload as the last state component of the mamba page, so the pool view is
``[slots, heads, head_block_bytes]`` uint8 with a slot stride of the whole
(possibly padded) page — conv + ssm + payload (+ padding).  Heads stay
densely packed inside the page and every field offset is 4-byte aligned,
so per-field ``as_strided`` views over ``buffer.view(dtype)`` reach the
same bytes as in the standalone contiguous pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.quantization.methods.mamba_state.leapquant.state import (
    reconstruct_state,
)

if TYPE_CHECKING:
    from vllm_ascend.ascend_config import LeapQuantConfig

__all__ = [
    "PayloadLayout",
    "LeapQuantPayloadPool",
    "leapquant_payload_state_extension",
    "leapquant_payload_pool_for_layer",
]


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

    The pool owns nothing but the buffer.  A standalone pool is
    ``allocate``-d as a contiguous ``[slots, heads, head_block_bytes]``
    uint8 tensor; the engine allocation (M3.2b) instead supplies the
    strided view carved out of the mamba page as the last state component
    (slot stride = whole page, heads densely packed).  Both go through the
    same constructor; ``slot_stride_bytes``/``head_stride_bytes`` report
    the actual byte strides the flush kernel must use.
    """

    layout: PayloadLayout
    buffer: torch.Tensor  # uint8 [slots, heads, head_block_bytes], may be strided

    def __init__(self, buffer: torch.Tensor, layout: PayloadLayout) -> None:
        if buffer.dtype != torch.uint8 or buffer.ndim != 3:
            raise ValueError(
                f"buffer must be a uint8 [slots, heads, head_block_bytes] tensor, "
                f"got dtype={buffer.dtype} ndim={buffer.ndim}"
            )
        if tuple(buffer.shape) != (layout.slots, layout.heads, layout.head_block_bytes):
            raise ValueError(
                f"buffer shape {tuple(buffer.shape)} does not match layout "
                f"({layout.slots}, {layout.heads}, {layout.head_block_bytes})"
            )
        # Heads must be densely packed inside the slot; the slot stride may
        # exceed slot_bytes (engine pages hold conv+ssm+payload+padding).
        if buffer.stride(2) != 1:
            raise ValueError(f"payload buffer inner stride != 1: {buffer.stride()}")
        if buffer.stride(1) != layout.head_block_bytes:
            raise ValueError(
                f"payload head stride {buffer.stride(1)} != head_block_bytes "
                f"{layout.head_block_bytes} (heads must be densely packed)"
            )
        for name, stride in (("slot", buffer.stride(0)), ("storage_offset", buffer.storage_offset())):
            if stride % 4 or stride < 0:
                raise ValueError(
                    f"payload {name} {stride} breaks 4-byte field alignment "
                    "(fp32/fp16/int32 payload fields)"
                )
        if buffer.stride(0) < layout.slot_bytes:
            raise ValueError(
                f"payload slot stride {buffer.stride(0)} < slot_bytes {layout.slot_bytes}"
            )
        self.layout = layout
        self.buffer = buffer

    @property
    def head_stride_bytes(self) -> int:
        """Actual per-head byte stride (== ``head_block_bytes``)."""
        return self.buffer.stride(1)

    @property
    def slot_stride_bytes(self) -> int:
        """Actual per-slot byte stride (engine pages: the whole page)."""
        return self.buffer.stride(0)

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
        buffer = torch.zeros(
            slots, heads, layout.head_block_bytes, dtype=torch.uint8, device=device
        )
        return cls(buffer, layout)

    # ---- unpacked views (all alias self.buffer) ------------------------
    def _view(self, name: str, dtype: torch.dtype, per_head_shape: tuple[int, ...]):
        lay = self.layout
        off = lay.field_offsets()[name]
        esz = torch.empty((), dtype=dtype).element_size()
        if esz != 1 and esz != 2 and esz != 4:
            raise ValueError(f"unsupported payload field dtype {dtype}")
        if off % esz or lay.head_block_bytes % esz or self.head_stride_bytes % esz:
            raise ValueError(f"field {name} misaligned for {dtype}")
        if self.slot_stride_bytes % esz:
            raise ValueError(f"payload slot stride misaligned for {dtype}")
        # dtype reinterpret of the (possibly strided) pool view, then an
        # explicit as_strided: per-head fields are dense inside the head
        # block, slots and heads step by the pool's byte strides.
        typed = self.buffer.view(dtype)
        dense = torch.empty(per_head_shape).stride()
        return torch.as_strided(
            typed,
            size=(lay.slots, lay.heads, *per_head_shape),
            stride=(
                self.slot_stride_bytes // esz,
                self.head_stride_bytes // esz,
                *dense,
            ),
            storage_offset=typed.storage_offset() + off // esz,
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


def leapquant_payload_state_extension(
    state_shapes: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], torch.dtype] | None:
    """Extra mamba state component carrying the payload pool, or ``None``.

    Called from the attention layer's ``get_state_shape``/``get_state_dtype``
    overrides: when ``leapquant_config.payload_pool`` is enabled, the mamba
    spec grows a trailing ``(heads, head_block_bytes)`` uint8 state.  The
    engine allocator (which slices one page into per-state views by
    shape/dtype) then carves the payload pool out of the same page that
    holds conv+ssm, giving it engine-sized slots and the state pool's exact
    lifetime.  ``None`` (component omitted) keeps the page byte-identical
    when LeapQuant is off.

    ``state_shapes`` is the layer's *base* shape tuple; the temporal /
    recurrent state is its last 3-D ``(heads, V, K)`` entry for both GDN
    and KDA.
    """
    from vllm_ascend.quantization.methods.mamba_state.leapquant.runtime import (
        leapquant_config_or_none,
    )

    config = leapquant_config_or_none()
    if config is None or not config.payload_pool:
        return None
    temporal = state_shapes[-1]
    if len(temporal) != 3:
        raise ValueError(
            "leapquant payload_pool: expected the last mamba state to be the "
            f"(heads, V, K) temporal state, got shape {temporal}"
        )
    heads, v_dim, k_dim = temporal
    try:
        layout = PayloadLayout(1, heads, v_dim, k_dim, config.compensator_tokens)
    except ValueError as e:
        raise ValueError(
            "leapquant payload_pool does not support this model's mamba state "
            f"geometry (heads={heads}, V={v_dim}, K={k_dim}): {e}"
        ) from e
    return (heads, layout.head_block_bytes), torch.uint8


def leapquant_extend_model_state_classmethods(model_cls: type) -> None:
    """Extend a model class's config-time mamba state classmethods.

    The platform sizes the shared hybrid page *before* any worker exists:
    ``Platform.update_block_size_for_backend`` ->
    ``_align_hybrid_block_size`` builds ``mamba_page_size`` from
    ``model_cls.get_mamba_state_shape_from_config`` /
    ``get_mamba_state_dtype_from_config`` and then pins
    ``cache_config.mamba_page_size_padded`` to the attention page covering
    it.  Those classmethods know only conv+ssm, so with ``payload_pool`` on
    the real (instance-level) spec grew past the pinned pad and
    ``MambaSpec.page_size_bytes`` asserts ``page_size_padded >= page_size``
    at worker startup.  Wrapping the classmethods to append the same
    trailing ``(heads, head_block_bytes)`` uint8 state makes the config-time
    math see the true page.  Disabled (or no AscendConfig yet) -> the
    original classmethods are kept, byte-identical behavior.

    Idempotent; a per-class marker makes repeated application a no-op (a
    subclass that defines its own classmethods is not skipped, because the
    marker is only read from the class ``__dict__``).
    """
    if "_leapquant_state_extended" in model_cls.__dict__:
        return
    orig_shape = model_cls.get_mamba_state_shape_from_config
    orig_dtype = model_cls.get_mamba_state_dtype_from_config

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):  # type: ignore[no-redef]
        shapes = tuple(orig_shape(vllm_config))
        ext = leapquant_payload_state_extension(shapes)
        return shapes if ext is None else (*shapes, ext[0])

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):  # type: ignore[no-redef]
        shapes = tuple(orig_shape(vllm_config))
        dtypes = tuple(orig_dtype(vllm_config))
        ext = leapquant_payload_state_extension(shapes)
        return dtypes if ext is None else (*dtypes, ext[1])

    model_cls.get_mamba_state_shape_from_config = get_mamba_state_shape_from_config
    model_cls.get_mamba_state_dtype_from_config = get_mamba_state_dtype_from_config
    model_cls._leapquant_state_extended = True


def leapquant_payload_pool_for_layer(
    layer: Any,
    caches: Any,
    state: torch.Tensor,
    config: "LeapQuantConfig",
) -> LeapQuantPayloadPool:
    """Wrap the payload component of a bound mamba kv-cache (cached on the layer).

    ``caches`` is the layer's bound ``kv_cache`` list whose trailing element
    is the engine-carved uint8 payload view; ``state`` is the fp32 temporal
    pool whose geometry the payload must match.  The wrapper is rebuilt only
    when the underlying allocation changes (rebind), not per decode step.
    """
    if len(caches) <= 2:
        raise RuntimeError(
            "leapquant payload_pool is enabled but the bound kv cache has no "
            "payload component; get_state_shape()/get_state_dtype() must append "
            "the payload state when payload_pool is set"
        )
    view = caches[2]
    heads, v_dim, k_dim = state.shape[-3], state.shape[-2], state.shape[-1]
    key = (view.data_ptr(), tuple(view.shape), v_dim, k_dim, config.compensator_tokens)
    cached = getattr(layer, "_leapquant_payload_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    pool = LeapQuantPayloadPool(
        view, PayloadLayout(view.shape[0], heads, v_dim, k_dim, config.compensator_tokens)
    )
    layer._leapquant_payload_cache = (key, pool)
    return pool
