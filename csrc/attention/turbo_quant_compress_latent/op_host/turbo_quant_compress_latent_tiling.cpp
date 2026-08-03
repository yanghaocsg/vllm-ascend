#include <algorithm>

#include "turbo_quant_compress_latent_tiling.h"
#include "error/ops_error.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GetPlatformInfo is nullptr."),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    OPS_ERR_IF(socVersion != platform_ascendc::SocVersion::ASCEND910B &&
                   socVersion != platform_ascendc::SocVersion::ASCEND910_93,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Unsupported SoC version: %d.",
                                    static_cast<int32_t>(socVersion)),
        return ge::GRAPH_FAILED);

    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(aivNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "num of AIV cores obtained is 0."),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* latentShape = context->GetInputShape(0);
    uint32_t numTokens = latentShape->GetStorageShape().GetDim(0);
    if (numTokens < 1) numTokens = 1;

    uint32_t usedCoreNum = std::min(numTokens, aivNum);
    uint32_t tokensPerCore = (numTokens + usedCoreNum - 1) / usedCoreNum;
    if (tokensPerCore < 1) tokensPerCore = 1;
    uint32_t taskNum = (numTokens + tokensPerCore - 1) / tokensPerCore;
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(taskNum, 0, aivNum);

    TurboQuantCompressLatentTilingData tilingData;
    tilingData.set_num_tokens(numTokens);
    tilingData.set_tokens_per_core(tokensPerCore);

    context->SetBlockDim(blockDim);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboQuantCompressLatent)
    .Tiling(TilingFunc);

} // namespace optiling
