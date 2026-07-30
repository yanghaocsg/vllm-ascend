#ifndef TQ_COMPRESS_LATENT_TILING_H_
#define TQ_COMPRESS_LATENT_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

// The codebook scan is a chain of narrow vector instructions whose fixed issue cost dominates the
// arithmetic, so the kernel folds several tokens into every instruction. Measured on 910B4: widening the
// loop 4x costs only 1.82x the time (2.2x better efficiency); the gain flattens past 8 while the UB
// footprint grows linearly, so the batch is capped here.
// UB budget with HEAD_DIM=512 / SLOT_PAD=320: 13696 B per token + 2880 B fixed, so 192 KB fits 14.
constexpr uint32_t TQ_COMPRESS_MAX_TOKENS_PER_BATCH = 12;

BEGIN_TILING_DATA_DEF(TurboQuantCompressLatentTilingData)
TILING_DATA_FIELD_DEF(uint32_t, num_tokens);
TILING_DATA_FIELD_DEF(uint32_t, tokens_per_core);
TILING_DATA_FIELD_DEF(uint32_t, tokens_per_batch);
TILING_DATA_FIELD_DEF(uint32_t, output_mode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboQuantCompressLatent, TurboQuantCompressLatentTilingData)

} // namespace optiling

#endif // TQ_COMPRESS_LATENT_TILING_H_
