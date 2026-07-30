# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import _approximate_gcd, may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendPackedKVCacheTensor
from vllm_ascend.utils import vllm_version_is

logger = init_logger(__name__)

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes
_orig_get_max_concurrency_for_kv_cache_config = (
    vllm.v1.core.kv_cache_utils.get_max_concurrency_for_kv_cache_config
)

TURBOQUANT_CACHE_DTYPE = "turboquant_4bit_nc"


def _is_dsa_tq4_spec(spec: KVCacheSpec) -> bool:
    return (
        getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE
        and getattr(spec, "model_version", None) == "deepseek_v4"
        and getattr(spec, "sparse_head_dim", None) is None
    )


def _get_page_stride_alignment(spec: KVCacheSpec) -> int:
    """Return the byte alignment required by a logical cache page."""
    alignment = get_dtype_size(spec.dtype)
    scale_dim = getattr(spec, "scale_dim", 0)
    if scale_dim:
        alignment = math.lcm(
            alignment,
            get_dtype_size(spec.scale_dtype),
        )
    if _is_dsa_tq4_spec(spec):
        # The DSA kernel addresses this cache as packed rows and requires the
        # physical block stride to contain an integral number of those rows.
        num_rows = spec.block_size * spec.num_kv_heads
        if spec.page_size_bytes % num_rows != 0:
            raise ValueError(
                "TQ4 KV page size must contain an integral number of "
                f"packed rows, got page_size={spec.page_size_bytes} and "
                f"num_rows={num_rows}."
            )
        alignment = math.lcm(alignment, spec.page_size_bytes // num_rows)
    return alignment


def _get_packed_page_payload_size(spec: KVCacheSpec) -> int:
    """Return bytes that contain logical cache data in a packed page.

    DeepSeek V4 compressor-state pages are padded to another cache's page
    size so the default allocator can share tensors by page-size bucket. The
    stride-aware TQ allocator provides that sharing itself, so retaining this
    historical padding would reserve bytes that no kernel reads or writes.
    """
    if (
        isinstance(spec, SlidingWindowMLASpec)
        and spec.page_size_padded is not None
        and getattr(spec, "alignment", None) is None
    ):
        return spec.real_page_size_bytes
    return spec.page_size_bytes


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support context parallelism (dcp/pcp > 1).
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with CP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp * pcp to maintain alignment, consistent
    with the pre-PR-#40860 behavior of block_size * dcp * pcp.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp * pcp
        return bs, bs

    if dcp != 1 or pcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by the CP factors for proper alignment.
        group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp * pcp
        if not cache_config.enable_prefix_caching:
            return scheduler_block_size, scheduler_block_size
        hash_block_size = math.gcd(*group_block_sizes)
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    ratio_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            ratio_specs[spec.compress_ratio][name] = spec

    mla_uniform_specs = []
    for ratio in sorted(ratio_specs, key=lambda r: (r != 4, r)):
        spec_dict = ratio_specs[ratio]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    full_mla_c128_spec = grouped_specs[1]
    uses_dsa_tq4 = any(
        _is_dsa_tq4_spec(spec)
        for group in grouped_specs
        for spec in group.kv_cache_specs.values()
    )

    assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    full_mla_c128_group = KVCacheGroupSpec(
        layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_c128_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [g_spec.get_num_layer_tuples() for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    # TODO(cmq): this is not general enough
    swa_mla_specs = grouped_specs[2:]

    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # TQ4 shrinks the c4 page below some compressor-state pages. Only that
    # layout may retain an oversized page; standard DeepSeek V4 keeps the
    # existing page-padding contract unchanged.
    all_page_sizes = full_mla_spec.get_page_sizes()
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = sm_spec.get_page_sizes()
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        if not uses_dsa_tq4:
            assert max(sm_page_sizes) <= max(all_page_sizes)
        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            candidates = [x for x in all_page_sizes if x >= ps]
            size_to_candidate[ps] = min(candidates) if candidates else ps
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


def _get_kv_cache_config_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """Use a stride-aware layout only for DeepSeek V4 TQ4 caches."""
    if any(
        _is_dsa_tq4_spec(spec)
        for group in kv_cache_groups
        for spec in group.kv_cache_spec.kv_cache_specs.values()
    ):
        return _get_tq4_kv_cache_config(vllm_config, kv_cache_groups, available_memory)

    # Preserve the existing DeepSeek V4 allocator byte-for-byte for non-TQ
    # models. Its page sharing and MTP isolation are independent of TQ.
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(full_mla_spec.get_page_sizes())
    layer_tuple_page_bytes = sum(page_sizes)

    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        bucket: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            if "mtp" not in name:
                bucket[specs[name].page_size_bytes].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = specs[name].page_size_bytes
        bucketed.append(bucket)

    num_layer_tuples = max(len(layers) for bucket in bucketed for layers in bucket.values()) + len(
        mtp_layer_names
    )
    num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    kv_cache_tensors: list[KVCacheTensor] = []
    for tuple_idx in range(num_layer_tuples - len(mtp_layer_names)):
        for page_size in page_sizes:
            shared_by: list[str] = []
            for bucket in bucketed:
                layers = bucket.get(page_size)
                if layers is not None and tuple_idx < len(layers):
                    shared_by.append(layers[tuple_idx])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )
    for name in mtp_layer_names:
        kv_cache_tensors.append(
            KVCacheTensor(size=mtp_page_size * num_blocks, shared_by=[name])
        )

    return num_blocks, kv_cache_tensors


def _pack_group_layers(
    layer_names: list[str],
    specs: dict[str, KVCacheSpec],
    physical_page_size: int,
) -> list[dict[str, int]]:
    """Pack one cache manager's logical pages into physical page slots."""
    slots: list[dict[str, int]] = []
    slot_sizes: list[int] = []
    ordered_names = sorted(
        enumerate(layer_names),
        key=lambda item: (-_get_packed_page_payload_size(specs[item[1]]), item[0]),
    )
    for _, name in ordered_names:
        page_size = _get_packed_page_payload_size(specs[name])
        page_alignment = _get_page_stride_alignment(specs[name])
        if page_size > physical_page_size:
            raise ValueError(
                f"KV page for {name} ({page_size} bytes) exceeds physical "
                f"page size ({physical_page_size} bytes)."
            )
        for slot_idx, used_bytes in enumerate(slot_sizes):
            offset = round_up(used_bytes, page_alignment)
            if offset + page_size <= physical_page_size:
                slots[slot_idx][name] = offset
                slot_sizes[slot_idx] = offset + page_size
                break
        else:
            slots.append({name: 0})
            slot_sizes.append(page_size)
    return slots


def _get_tq4_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """Plan stride-aware physical pages for DeepSeek V4 TQ4 caches.

    TQ4 compressor and index pages in the same cache-manager group use the
    same block IDs, so they can occupy disjoint offsets of one physical page.
    Different cache-manager groups continue to share physical page slots using
    the existing hybrid KV block-pool contract.
    """
    group_slots: list[list[dict[str, int]]] = []
    mtp_layers: list[tuple[str, int]] = []
    main_specs = {
        name: spec
        for group in kv_cache_groups
        for name, spec in group.kv_cache_spec.kv_cache_specs.items()
        if "mtp" not in name
    }
    if not main_specs:
        raise ValueError(
            "DeepSeek V4 TQ4 KV layout requires at least one main-model cache spec."
        )
    physical_page_alignment = math.lcm(
        *(_get_page_stride_alignment(spec) for spec in main_specs.values())
    )
    packing_page_size = round_up(
        max(_get_packed_page_payload_size(spec) for spec in main_specs.values()),
        physical_page_alignment,
    )

    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        main_layers = []
        for name in group.layer_names:
            if "mtp" in name:
                mtp_layers.append((name, specs[name].page_size_bytes))
            else:
                main_layers.append(name)

        group_slots.append(
            _pack_group_layers(main_layers, specs, packing_page_size)
        )

    num_physical_slots = max(map(len, group_slots))
    physical_page_sizes: list[int] = []
    for slot_idx in range(num_physical_slots):
        required_size = 0
        required_alignment = 1
        for slots in group_slots:
            if slot_idx >= len(slots):
                continue
            for name, offset in slots[slot_idx].items():
                spec = main_specs[name]
                required_size = max(
                    required_size,
                    offset + _get_packed_page_payload_size(spec),
                )
                required_alignment = math.lcm(
                    required_alignment,
                    _get_page_stride_alignment(spec),
                )
        physical_page_sizes.append(round_up(required_size, required_alignment))

    bytes_per_block = sum(physical_page_sizes)
    bytes_per_block += sum(page_size for _, page_size in mtp_layers)
    assert bytes_per_block > 0
    num_blocks = may_override_num_blocks(
        vllm_config, available_memory // bytes_per_block
    )
    logger.info_once(
        "DeepSeek V4 KV cache layout: mode=tq4, available_memory=%d bytes, "
        "bytes_per_block=%d, num_blocks=%d, physical_page_sizes=%s, "
        "dedicated_mtp_bytes=%d",
        available_memory,
        bytes_per_block,
        num_blocks,
        tuple(physical_page_sizes),
        sum(page_size for _, page_size in mtp_layers),
        scope="local",
    )

    kv_cache_tensors: list[KVCacheTensor] = []
    for slot_idx, physical_page_size in enumerate(physical_page_sizes):
        layer_offsets: dict[str, int] = {}
        for slots in group_slots:
            if slot_idx < len(slots):
                layer_offsets.update(slots[slot_idx])
        kv_cache_tensors.append(
            AscendPackedKVCacheTensor(
                size=physical_page_size * num_blocks,
                shared_by=list(layer_offsets),
                physical_page_size=physical_page_size,
                layer_offsets=layer_offsets,
            )
        )
    for name, page_size in mtp_layers:
        kv_cache_tensors.append(
            KVCacheTensor(size=page_size * num_blocks, shared_by=[name])
        )

    return num_blocks, kv_cache_tensors


def _get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> float:
    """Report scheduler capacity from the same packed layout it allocates."""
    uses_dsa_tq4 = any(
        _is_dsa_tq4_spec(spec)
        for group in kv_cache_config.kv_cache_groups
        for spec in group.kv_cache_spec.kv_cache_specs.values()
    )
    if not uses_dsa_tq4:
        return _orig_get_max_concurrency_for_kv_cache_config(
            vllm_config, kv_cache_config
        )
    group_blocks_per_request = [
        group.kv_cache_spec.max_memory_usage_pages(vllm_config)
        for group in kv_cache_config.kv_cache_groups
    ]
    blocks_per_request = sum(group_blocks_per_request)
    assert blocks_per_request > 0
    logger.info_once(
        "DeepSeek V4 KV cache request budget: group_blocks=%s, "
        "blocks_per_request=%d, num_blocks=%d",
        tuple(group_blocks_per_request),
        blocks_per_request,
        kv_cache_config.num_blocks,
        scope="local",
    )
    return kv_cache_config.num_blocks / blocks_per_request


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
vllm.v1.core.kv_cache_utils.get_max_concurrency_for_kv_cache_config = (
    _get_max_concurrency_for_kv_cache_config
)
# vllm v0.24.0 renamed _get_kv_cache_config_deepseek_v4 to _get_kv_cache_config_packed and
# get_kv_cache_config_from_groups now calls _get_kv_cache_config_packed directly, bypassing
# the alias patch above. Patch the canonical name so Ascend's non-packed layout is used.
if vllm_version_is("0.23.0"):
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_deepseek_v4
else:
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed = _get_kv_cache_config_deepseek_v4

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
