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
