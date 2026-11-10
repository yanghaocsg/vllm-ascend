# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""LeapQuant decode workspace: the shadow fp32 pool W (RFC vllm-ascend#21097 M3.4a).

Under the triton decode backend the recurrent state lives in *two* places
during the migration window:

* ``ssm_state`` — the engine's (possibly page-strided) fp32 pool.  The
  prefill kernels gather/scatter it (three scatter variants in
  ``ops/gdn.py`` §2.3) and it stays the layout-compatible safety net.
* ``W`` — a contiguous ``[slots, HV, V, K]`` fp32 shadow pool with the
  **identity slot mapping** (``W[slot]`` mirrors ``ssm_state[slot]``).  The
  triton decode kernel reads/writes W rows in place and the window flush
  rewrites W rows, so quantization semantics are exactly M3.4's (the flush
  round-trip enters the live state the decode path reads).

Consistency contract (why the identity mapping needs no host-side table):

* every step in which a slot prefills, the post-scatter row is mirrored into
  W (``gather_ssm_states`` → ``scatter_ssm_states_``) right after §2.3 —
  chunked-prefill continuation stays exact because only the prefill path
  writes ``ssm_state`` rows, and it re-mirrors after every chunk;
* a request decodes only after its prefill completes, and decode touches W
  only — so W rows are authoritative across decode streaks;
* slot reuse by a new request is healed unconditionally: the new request's
  first prefill overwrites ``W[slot]`` via the same mirror site;
* rows are bounded by the pool slot count, so no eviction/table is needed.

M3.4a-2 will drop the fp32 page component (keeping payload + W), which is
where the ~3.2x pool-capacity win comes from; this module's allocator is
deliberately the only stateful piece so that step is a page-sizing change.
"""

from __future__ import annotations

import torch

__all__ = ["leapquant_workspace_for"]

# One shadow pool per (device, data_ptr, shape): layers get distinct views
# into the mamba page (distinct data_ptrs), so this keys one W per layer
# pool.  Each entry holds a **strong ref** to the state tensor it was cut
# from — pinning the tensor pins its data_ptr, so the allocator can never
# hand the same address to a *different* pool and alias a stale W onto it
# (exactly what test_workspace_alloc_and_cache reproduces).  A hit
# additionally verifies the underlying storage matches, so callers that
# hand us a fresh view object over the same page every forward still hit.
# The cost: a torn-down engine's pools stay resident until process exit —
# acceptable during the migration window; M3.5 moves this registry to weak
# refs plus a device-side table.
_WORKSPACES: dict[tuple[int, int, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]] = {}


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    sa, sb = a.untyped_storage(), b.untyped_storage()
    return sa.data_ptr() == sb.data_ptr() and sa.nbytes() == sb.nbytes()


def leapquant_workspace_for(state: torch.Tensor) -> torch.Tensor:
    """Return (allocating once) the shadow fp32 pool for ``state``.

    Args:
        state: the layer's ssm state view ``[slots, H, V, K]`` fp32 (may be
            a padded page view; the workspace is allocated contiguous).

    Returns:
        Contiguous fp32 ``[slots, H, V, K]`` tensor, initially a full copy
        of ``state`` (garbage rows included — they are only ever read via
        live slot indices, and the first prefill mirror overwrites them).
    """
    if state.ndim != 4:
        raise ValueError(f"state must be [slots, H, V, K], got ndim={state.ndim}")
    if state.dtype != torch.float32:
        raise TypeError(f"state must be fp32, got {state.dtype}")
    key = (state.device.index, state.data_ptr(), tuple(state.shape))
    entry = _WORKSPACES.get(key)
    if entry is not None and _shares_storage(entry[0], state):
        return entry[1]
    # Contiguous shadow (not clone(): the engine pool may be a padded
    # page view and W has no reason to mirror its strides).  copy_
    # handles arbitrary source strides.
    w = torch.empty(tuple(state.shape), dtype=torch.float32, device=state.device)
    w.copy_(state)
    _WORKSPACES[key] = (state, w)
    return w
