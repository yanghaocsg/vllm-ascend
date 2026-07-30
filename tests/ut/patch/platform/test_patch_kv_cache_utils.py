# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendPackedKVCacheTensor,
    AscendSlidingWindowMLASpec,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _get_packed_page_payload_size,
    _get_page_stride_alignment,
    _get_kv_cache_config_deepseek_v4,
    _get_kv_cache_groups_uniform_groups,
    _get_max_concurrency_for_kv_cache_config,
    _is_dsa_tq4_spec,
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
    cache_dtype_str: str | None = None,
    model_version: str = "deepseek_v4",
) -> AscendMLAAttentionSpec:
    return AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        compress_ratio=compress_ratio,
        model_version=model_version,
        scale_dim=scale_dim,
        scale_dtype=torch.float16,
        cache_dtype_str=cache_dtype_str,
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


def test_tq4_packed_layout_is_scoped_to_deepseek_v4() -> None:
    deepseek_spec = _mla_spec(258, cache_dtype_str="turboquant_4bit_nc")
    other_model_spec = _mla_spec(
        258,
        cache_dtype_str="turboquant_4bit_nc",
        model_version="other_model",
    )

    assert _is_dsa_tq4_spec(deepseek_spec)
    assert not _is_dsa_tq4_spec(other_model_spec)


def test_tq4_groups_keep_oversized_compressor_state_page() -> None:
    grouped_specs = [
        _uniform_spec(
            {
                "c4": _mla_spec(258, cache_dtype_str="turboquant_4bit_nc"),
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
    assert page_sizes == {16_640, 33_024, 131_072}


def test_standard_allocator_preserves_existing_layout() -> None:
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
    ]
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=None))
    page_size = groups[0].kv_cache_spec.kv_cache_specs["main.c4"].page_size_bytes

    num_blocks, tensors = _get_kv_cache_config_deepseek_v4(
        vllm_config, groups, available_memory=page_size * 2 * 10
    )

    assert num_blocks == 10
    assert all(not isinstance(tensor, AscendPackedKVCacheTensor) for tensor in tensors)
    assert [tensor.shared_by for tensor in tensors] == [["main.c4"], ["main.mtp"]]


def test_tq4_allocator_packs_logical_pages_into_physical_pages() -> None:
    tq4_group = KVCacheGroupSpec(
        layer_names=["c4.0", "c4.1", "index.0", "index.1"],
        kv_cache_spec=_uniform_spec(
            {
                "c4.0": _mla_spec(258, cache_dtype_str="turboquant_4bit_nc"),
                "c4.1": _mla_spec(258, cache_dtype_str="turboquant_4bit_nc"),
                "index.0": _mla_spec(128, scale_dim=1),
                "index.1": _mla_spec(128, scale_dim=1),
            }
        ),
    )
    c128_group = KVCacheGroupSpec(
        layer_names=["c128.0", "c128.1"],
        kv_cache_spec=_uniform_spec(
            {
                "c128.0": _mla_spec(512, compress_ratio=128, dtype=torch.bfloat16),
                "c128.1": _mla_spec(512, compress_ratio=128, dtype=torch.bfloat16),
            }
        ),
    )
    state_group = KVCacheGroupSpec(
        layer_names=["state.0", "state.1"],
        kv_cache_spec=_uniform_spec(
            {
                "state.0": _swa_spec(32, 1024, torch.float32),
                "state.1": _swa_spec(32, 1024, torch.float32),
            }
        ),
    )
    groups = [tq4_group, c128_group, state_group]
    main_specs = [
        spec
        for group in groups
        for spec in group.kv_cache_spec.kv_cache_specs.values()
    ]
    physical_page_alignment = math.lcm(
        *(_get_page_stride_alignment(spec) for spec in main_specs)
    )
    largest_page_size = max(spec.page_size_bytes for spec in main_specs)
    tq4_physical_page_size = (
        (largest_page_size + physical_page_alignment - 1)
        // physical_page_alignment
        * physical_page_alignment
    )
    packed_bytes_per_block = tq4_physical_page_size + largest_page_size
    exact_bucket_bytes_per_block = sum(
        spec.page_size_bytes
        for spec in tq4_group.kv_cache_spec.kv_cache_specs.values()
    ) + max(
        sum(
            spec.page_size_bytes
            for spec in group.kv_cache_spec.kv_cache_specs.values()
        )
        for group in groups[1:]
    )
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=None))

    num_blocks, tensors = _get_kv_cache_config_deepseek_v4(
        vllm_config,
        groups,
        available_memory=packed_bytes_per_block * 10,
    )

    assert num_blocks == 10
    assert sum(tensor.size for tensor in tensors) == packed_bytes_per_block * num_blocks
    assert packed_bytes_per_block < exact_bucket_bytes_per_block
    assert len(tensors) == 2
    assert all(isinstance(tensor, AscendPackedKVCacheTensor) for tensor in tensors)
    state_tensor = next(tensor for tensor in tensors if "state.0" in tensor.shared_by)
    assert "state.1" not in state_tensor.shared_by
    assert any("state.1" in tensor.shared_by for tensor in tensors)

    tq4_tensor = next(tensor for tensor in tensors if "c4.0" in tensor.shared_by)
    tq4_spec = tq4_group.kv_cache_spec.kv_cache_specs["c4.0"]
    tq4_slot_size = tq4_spec.page_size_bytes // (
        tq4_spec.block_size * tq4_spec.num_kv_heads
    )
    assert tq4_tensor.physical_page_size % tq4_slot_size == 0
    logical_ranges = sorted(
        (
            tq4_tensor.layer_offsets[name],
            tq4_tensor.layer_offsets[name]
            + tq4_group.kv_cache_spec.kv_cache_specs[name].page_size_bytes,
        )
        for name in tq4_group.layer_names
    )
    assert all(
        current_end <= next_start
        for (_, current_end), (next_start, _) in zip(
            logical_ranges, logical_ranges[1:]
        )
    )
    assert logical_ranges[-1][1] <= tq4_tensor.physical_page_size


def test_tq4_allocator_drops_tensor_sharing_only_padding() -> None:
    state_spec = AscendSlidingWindowMLASpec(
        block_size=8,
        num_kv_heads=1,
        head_size=2048,
        dtype=torch.float32,
        sliding_window=8,
        page_size_padded=131_072,
    )

    assert state_spec.real_page_size_bytes == 65_536
    assert state_spec.page_size_bytes == 131_072
    assert _get_packed_page_payload_size(state_spec) == 65_536


def test_deepseek_v4_concurrency_uses_runtime_group_block_budget() -> None:
    groups = [
        KVCacheGroupSpec(
            layer_names=["c4"],
            kv_cache_spec=_uniform_spec(
                {"c4": _mla_spec(258, cache_dtype_str="turboquant_4bit_nc")}
            ),
        ),
        KVCacheGroupSpec(
            layer_names=["c128"],
            kv_cache_spec=_uniform_spec(
                {
                    "c128": _mla_spec(
                        512,
                        compress_ratio=128,
                        dtype=torch.bfloat16,
                    )
                }
            ),
        ),
    ]
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=512),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=20,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )

    assert _get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    ) == 10


def _dsv4_capacity_groups(tq4: bool) -> list[KVCacheGroupSpec]:
    num_c4_layers = 21
    num_c128_layers = 20

    c4_specs = {
        f"c4.attn.{i}": _mla_spec(
            258 if tq4 else 512,
            dtype=torch.uint8 if tq4 else torch.bfloat16,
            cache_dtype_str="turboquant_4bit_nc" if tq4 else None,
        )
        for i in range(num_c4_layers)
    }
    c4_specs.update(
        {
            f"c4.index.{i}": _mla_spec(128, scale_dim=1)
            for i in range(num_c4_layers)
        }
    )
    c4_group = KVCacheGroupSpec(
        layer_names=list(c4_specs),
        kv_cache_spec=_uniform_spec(c4_specs),
    )

    c128_specs = {
        f"c128.attn.{i}": _mla_spec(
            512,
            compress_ratio=128,
            dtype=torch.bfloat16,
        )
        for i in range(num_c128_layers)
    }
    c128_group = KVCacheGroupSpec(
        layer_names=list(c128_specs),
        kv_cache_spec=_uniform_spec(c128_specs),
    )

    swa_specs_0 = {
        f"swa.{i}": _swa_spec(128, 512, torch.bfloat16)
        for i in range(22)
    }
    swa_specs_1 = {
        f"swa.{i}": _swa_spec(128, 512, torch.bfloat16)
        for i in range(22, 43)
    }
    swa_specs_1["model.mtp.swa"] = _swa_spec(128, 512, torch.bfloat16)
    swa_groups = [
        KVCacheGroupSpec(
            layer_names=list(specs),
            kv_cache_spec=_uniform_spec(specs),
        )
        for specs in (swa_specs_0, swa_specs_1)
    ]

    c4_state_specs = {}
    for i in range(num_c4_layers):
        c4_state_specs[f"c4.state.{i}"] = AscendSlidingWindowMLASpec(
            block_size=8,
            num_kv_heads=1,
            head_size=2048,
            dtype=torch.float32,
            sliding_window=8,
            page_size_padded=131_072,
        )
        c4_state_specs[f"c4.index_state.{i}"] = AscendSlidingWindowMLASpec(
            block_size=8,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.float32,
            sliding_window=8,
            page_size_padded=16_640,
        )
    c4_state_group = KVCacheGroupSpec(
        layer_names=list(c4_state_specs),
        kv_cache_spec=_uniform_spec(c4_state_specs),
    )

    c128_state_specs = {
        f"c128.state.{i}": AscendSlidingWindowMLASpec(
            block_size=32,
            num_kv_heads=1,
            head_size=1024,
            dtype=torch.float32,
            sliding_window=128,
            page_size_padded=131_072,
        )
        for i in range(num_c128_layers)
    }
    c128_state_group = KVCacheGroupSpec(
        layer_names=list(c128_state_specs),
        kv_cache_spec=_uniform_spec(c128_state_specs),
    )

    return [
        c4_group,
        c128_group,
        *swa_groups,
        c4_state_group,
        c128_state_group,
    ]


def test_dsv4_tq4_layout_increases_physical_capacity() -> None:
    baseline_groups = _dsv4_capacity_groups(tq4=False)
    tq4_groups = _dsv4_capacity_groups(tq4=True)
    available_memory = 336_409_600
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        model_config=SimpleNamespace(max_model_len=1_048_576),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )

    baseline_num_blocks, baseline_tensors = _get_kv_cache_config_deepseek_v4(
        vllm_config, baseline_groups, available_memory
    )
    tq4_num_blocks, tq4_tensors = _get_kv_cache_config_deepseek_v4(
        vllm_config, tq4_groups, available_memory
    )

    assert tq4_num_blocks > baseline_num_blocks
    assert all(not isinstance(tensor, AscendPackedKVCacheTensor) for tensor in baseline_tensors)
    assert any(isinstance(tensor, AscendPackedKVCacheTensor) for tensor in tq4_tensors)
    assert sum(tensor.size for tensor in baseline_tensors) <= available_memory
    assert sum(tensor.size for tensor in tq4_tensors) <= available_memory
    baseline_group_blocks = [
        group.kv_cache_spec.max_memory_usage_pages(vllm_config)
        for group in baseline_groups
    ]
    assert [
        group.kv_cache_spec.max_memory_usage_pages(vllm_config)
        for group in tq4_groups
    ] == baseline_group_blocks

    baseline_config = KVCacheConfig(
        num_blocks=baseline_num_blocks,
        kv_cache_tensors=baseline_tensors,
        kv_cache_groups=baseline_groups,
    )
    tq4_config = KVCacheConfig(
        num_blocks=tq4_num_blocks,
        kv_cache_tensors=tq4_tensors,
        kv_cache_groups=tq4_groups,
    )
    assert _get_max_concurrency_for_kv_cache_config(
        vllm_config, tq4_config
    ) > _get_max_concurrency_for_kv_cache_config(vllm_config, baseline_config)
