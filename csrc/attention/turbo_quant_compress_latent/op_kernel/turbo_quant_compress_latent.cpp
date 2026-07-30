#include "turbo_quant_compress_latent.h"

extern "C" __global__ __aicore__ void turbo_quant_compress_latent(
    GM_ADDR latent, GM_ADDR centroids, GM_ADDR slotOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelTurboQuantCompressLatent op;
    op.Init(latent, centroids, slotOut, tilingData.num_tokens, tilingData.tokens_per_core,
            tilingData.tokens_per_batch, tilingData.output_mode);
    op.Process();
}
