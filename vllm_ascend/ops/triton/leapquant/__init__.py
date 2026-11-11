# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""LeapQuant fused triton-ascend kernels (RFC vllm-ascend#21097 M3)."""

from vllm_ascend.ops.triton.leapquant.flush_kernel import leapquant_flush_triton
from vllm_ascend.ops.triton.leapquant.recurrent_leapquant import (
    kda_expand_gate,
    recurrent_gdn_leapquant,
    recurrent_kda_leapquant,
)

__all__ = [
    "kda_expand_gate",
    "leapquant_flush_triton",
    "recurrent_gdn_leapquant",
    "recurrent_kda_leapquant",
]
