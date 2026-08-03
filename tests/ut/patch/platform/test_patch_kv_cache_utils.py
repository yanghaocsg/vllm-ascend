# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _get_kv_cache_config_deepseek_v4,
    _get_kv_cache_groups_uniform_groups,
)


def _uniform_spec(specs):
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(specs)
    assert uniform_spec is not None
    return uniform_spec


def _mla_spec(
    head_size: int,
    compress_ratio: int = 4,
    scale_dim: int = 0,
    dtype: torch.dtype = torch.uint8,
) -> AscendMLAAttentionSpec:
    return AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        compress_ratio=compress_ratio,
        model_version="deepseek_v4",
        scale_dim=scale_dim,
        scale_dtype=torch.float16,
    )


def _swa_spec(block_size: int, head_size: int, dtype: torch.dtype) -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        sliding_window=256,
        model_version="deepseek_v4",
    )


def test_tq4_groups_keep_oversized_compressor_state_page() -> None:
    grouped_specs = [
        _uniform_spec(
            {
                "c4": _mla_spec(320),
                "c4_index": _mla_spec(128, scale_dim=1),
            }
        ),
        _uniform_spec(
            {
                "c128": _mla_spec(
                    512,
                    compress_ratio=128,
                    dtype=torch.bfloat16,
                )
            }
        ),
        _uniform_spec({"swa": _swa_spec(128, 512, torch.bfloat16)}),
        _uniform_spec(
            {
                "c4_index_state": _swa_spec(8, 520, torch.float32),
                "c4_attn_state": _swa_spec(8, 4096, torch.float32),
            }
        ),
        _uniform_spec({"c128_state": _swa_spec(32, 1024, torch.float32)}),
    ]

    groups = _get_kv_cache_groups_uniform_groups(grouped_specs)

    page_sizes = {
        spec.page_size_bytes
        for group in groups
        for spec in group.kv_cache_spec.kv_cache_specs.values()
    }
    assert page_sizes == {16_640, 40_960, 131_072}


def test_tq4_allocator_buckets_all_page_sizes_and_isolates_mtp() -> None:
    groups = [
        KVCacheGroupSpec(
            layer_names=["main.c4", "main.mtp"],
            kv_cache_spec=_uniform_spec(
                {
                    "main.c4": _mla_spec(320),
                    "main.mtp": _mla_spec(320),
                }
            ),
        ),
        KVCacheGroupSpec(
            layer_names=["swa"],
            kv_cache_spec=_uniform_spec({"swa": _swa_spec(128, 512, torch.bfloat16)}),
        ),
        KVCacheGroupSpec(
            layer_names=["state"],
            kv_cache_spec=_uniform_spec({"state": _swa_spec(32, 1024, torch.float32)}),
        ),
    ]
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=None))
    bytes_per_block = 40_960 + 40_960 + 131_072

    num_blocks, tensors = _get_kv_cache_config_deepseek_v4(
        vllm_config, groups, available_memory=bytes_per_block * 10
    )

    assert num_blocks == 10
    assert sorted((tensor.size, tensor.shared_by) for tensor in tensors) == [
        (409_600, ["main.c4"]),
        (409_600, ["main.mtp"]),
        (1_310_720, ["swa", "state"]),
    ]
