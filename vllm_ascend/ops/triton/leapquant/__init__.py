# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""LeapQuant fused triton-ascend kernels (RFC vllm-ascend#21097 M3)."""

from vllm_ascend.ops.triton.leapquant.flush_kernel import leapquant_flush_triton

__all__ = ["leapquant_flush_triton"]
