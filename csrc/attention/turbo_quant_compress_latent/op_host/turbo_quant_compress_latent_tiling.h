#ifndef TQ_COMPRESS_LATENT_TILING_H_
#define TQ_COMPRESS_LATENT_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TurboQuantCompressLatentTilingData)
TILING_DATA_FIELD_DEF(uint32_t, num_tokens);
TILING_DATA_FIELD_DEF(uint32_t, tokens_per_core);
TILING_DATA_FIELD_DEF(uint32_t, output_mode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboQuantCompressLatent, TurboQuantCompressLatentTilingData)

} // namespace optiling

#endif // TQ_COMPRESS_LATENT_TILING_H_
