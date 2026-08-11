# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

import pytest


@pytest.mark.parametrize("soc_branch", ["ascend910b", "ascend910_93"])
def test_dsv4_compressor_metadata_is_packaged(soc_branch: str) -> None:
    build_script = Path("csrc/build_aclnn.sh").read_text()
    branch = build_script.split(f'elif [[ "$SOC_VERSION" =~ ^{soc_branch} ]]', 1)[1]
    custom_ops = branch.split("CUSTOM_OPS=$(", 1)[0]

    assert '"compressor"' in custom_ops
    assert '"compressor_metadata"' in custom_ops


def test_incremental_build_invalidates_stale_kernel_cache() -> None:
    build_script = Path("csrc/build_aclnn.sh").read_text()

    assert "invalidate_stale_kernel_cache" in build_script
    assert '-newer "${marker}"' in build_script
    assert '-name "${op_name}_${SOC_ARG}_*.done" -delete' in build_script
    assert '"${binary_root}/bin/${op_name}"' in build_script
    assert '"${binary_root}/src/${op_name}"' in build_script
    assert "-path '*/op_impl/ai_core/*'" in build_script
    assert "if [[ -d build ]]" in build_script
    assert "-name binary_info_config.json" in build_script
    assert "-name relocatable_kernel_info_config.json" in build_script


@pytest.mark.parametrize(
    "op_name",
    ["turbo_quant_sparse_flash_attention", "turbo_quant_compress_latent"],
)
def test_turboquant_ops_register_a2_and_a3(op_name: str) -> None:
    op_def = Path(f"csrc/attention/{op_name}/op_host/{op_name}_def.cpp").read_text()

    assert 'AddConfig("ascend910b"' in op_def
    assert 'AddConfig("ascend910_93"' in op_def


def test_turboquant_compress_tiling_uses_platform_core_count() -> None:
    tiling_source = Path(
        "csrc/attention/turbo_quant_compress_latent/op_host/turbo_quant_compress_latent_tiling.cpp"
    ).read_text()

    assert "GetCoreNumAiv()" in tiling_source
    assert "CalcTschBlockDim" in tiling_source
    assert "MAX_AI_CORES" not in tiling_source


def test_turboquant_compress_avoids_unsigned_to_float_device_cast() -> None:
    kernel_source = Path(
        "csrc/attention/turbo_quant_compress_latent/op_kernel/turbo_quant_compress_latent.h"
    ).read_text()

    assert "constexpr float CENTROID_BYTE_STRIDE = 4.0F;" in kernel_source
    assert "Muls(tmp, nib, CENTROID_BYTE_STRIDE, HEAD_DIM);" in kernel_source
    assert "static_cast<float>(sizeof(float))" not in kernel_source


def test_dsv4_tq4_sparse_attention_uses_batched_dequant_fast_paths() -> None:
    kernel_source = Path(
        "csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/"
        "sparse_attn_sharedkv_scfa_block_vector.h"
    ).read_text()

    dequant_source = kernel_source.split("SASVectorBlock<SAST>::DequantTq4Rows", 1)[1]
    dequant_source = dequant_source.split("template <typename SAST>", 1)[0]
    assert "slotU16.GetValue(scaleIndex)" not in dequant_source
    assert "scaleCopyParams.blockCount = static_cast<uint16_t>(currentRows)" in dequant_source
    assert "Gather(scalesFp16, scaleCarriers, scaleGatherIndices" in dequant_source
    assert "Brcb(scalesBrcb, scalesFp32" in dequant_source
    assert "RowMuls(work, work, scalesBrcb" in dequant_source
    assert "TQ4_SCALE_GATHER_INDEX_OFFSET" in kernel_source
    assert "base + static_cast<int64_t>(currentRows) < rowCount" in dequant_source

    copy_out_source = kernel_source.split("SASVectorBlock<SAST>::CopyOutMrgeResult", 1)[1]
    copy_out_source = copy_out_source.split("template <typename SAST>", 1)[0]
    assert "DequantTq4Rows(slots, output, mte2Size - mte3Size)" in copy_out_source
    assert "SetFlag<AscendC::HardEvent::V_MTE3>" in copy_out_source
    assert "WaitFlag<AscendC::HardEvent::V_MTE3>" in copy_out_source

    copy_source = kernel_source.split("SASVectorBlock<SAST>::CopyInKv", 1)[1]
    copy_source = copy_source.split("template <typename SAST>", 1)[0]
    assert "validHitCount != 2" in copy_source
    assert "keyDistance - static_cast<int64_t>(TQ4_SLOT_BYTES)" in copy_source
    assert "TQ4_SLOT_ROW_BYTES - TQ4_SLOT_BYTES" in copy_source
    assert "CopyInSingleKv" in copy_source
    assert "cmpKvGm_" in copy_source


def test_dsv4_tq4_index_prefetch_is_bounded_and_has_gm_fallback() -> None:
    kernel_source = Path(
        "csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/"
        "sparse_attn_sharedkv_scfa_block_vector.h"
    ).read_text()

    assert "TQ4_TOPK_CACHE_CAPACITY" in kernel_source
    assert "TQ4_BLOCK_TABLE_CACHE_CAPACITY" in kernel_source
    assert "requiredTopkCount <= TQ4_TOPK_CACHE_CAPACITY" in kernel_source
    assert "requiredBlockTableCount <= TQ4_BLOCK_TABLE_CACHE_CAPACITY" in kernel_source
    assert "SetFlag<HardEvent::MTE2_S>(0)" in kernel_source
    assert "topkGm_.GetValue(topkGmBaseOffset + topkGmIdx)" in kernel_source
    assert "cmpBlockTableGm_.GetValue" in kernel_source
