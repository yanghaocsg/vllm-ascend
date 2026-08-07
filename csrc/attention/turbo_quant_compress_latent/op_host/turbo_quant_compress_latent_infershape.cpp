#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShape4TurboQuantCompressLatent(gert::InferShapeContext* context)
{
    const gert::Shape* latentShape = context->GetInputShape(0);
    int64_t N = latentShape->GetDim(0);
    gert::Shape* slotShape = context->GetOutputShape(0);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* outputModeAttr = attrs->GetAttrPointer<int64_t>(0);
    if (outputModeAttr == nullptr || (*outputModeAttr != 0 && *outputModeAttr != 1)) {
        return ge::GRAPH_FAILED;
    }
    slotShape->SetDimNum(2);
    slotShape->SetDim(0, N);
    slotShape->SetDim(1, *outputModeAttr == 1 ? 258 : 320);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4TurboQuantCompressLatent(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_UINT8);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TurboQuantCompressLatent)
    .InferShape(InferShape4TurboQuantCompressLatent)
    .InferDataType(InferDataType4TurboQuantCompressLatent);

} // namespace ops
