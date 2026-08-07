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
        // 0: legacy 320B raw scale; 1: compact 258B code-norm-corrected scale.
        this->Attr("output_mode").AttrType(OPTIONAL).Int(0);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(TurboQuantCompressLatent);

} // namespace ops
