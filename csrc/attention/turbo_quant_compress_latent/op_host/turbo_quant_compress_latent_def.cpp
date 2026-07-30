#include "register/op_def_registry.h"

namespace ops {

class TurboQuantCompressLatent : public OpDef {
public:
    explicit TurboQuantCompressLatent(const char* name) : OpDef(name)
    {
        this->Input("latent")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("centroids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Output("slot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        // 0 keeps the GLM-compatible 320-byte layout; 1 emits the compact
        // 258-byte layout consumed by DeepSeek V4 sparse_attn_sharedkv.
        this->Attr("output_mode").AttrType(OPTIONAL).Int(0);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(TurboQuantCompressLatent);

} // namespace ops
