/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file turbo_quant_sparse_flash_attention_service_vector_mla.h
 * \brief
 */
#ifndef TURBOQUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
#define TURBOQUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "turbo_quant_sparse_flash_attention_common.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename QSFAT> class QSFAVectorService {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using KV_T = typename QSFAT::kvType;
    using K_ROPE_T = typename QSFAT::kRopeType;
    using OUT_T = typename QSFAT::outputType;
    using UPDATE_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline QSFAVectorService(){};
    __aicore__ inline void ProcessVec1L(const RunInfo &info);
    __aicore__ inline void ProcessVec2L(const RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo,
                                      const TurboQuantSparseFlashAttentionTilingDataMla *__restrict tilingData);
    __aicore__ inline void InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<int32_t> &kvValidSizeGm,
                                                const GlobalTensor<K_ROPE_T> &kvMergeGm,
                                                const GlobalTensor<K_ROPE_T> &keyRopeGm,
                                                const GlobalTensor<KV_T> &keyGm,
                                                const GlobalTensor<int32_t> &blkTableGm,
                                                const GlobalTensor<half> &sTGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_ROPE_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm,
                                                GlobalTensor<T> softmaxMaxGm, GlobalTensor<T> softmaxSumGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    // ================================Vector0==========================================
    __aicore__ inline void MergeKv(const RunInfo &runInfo);
    __aicore__ inline int64_t GetKeyBNBOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo &runInfo);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                       uint64_t startId, uint64_t endId);
    __aicore__ inline void SetMidInf(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                     uint64_t startId, uint64_t endId);
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo &runInfo);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo);
    // [TQ4] codebook dequant of dealRow combined slots -> antiKvTensorAsB16 [dealRow,512] bf16 (Phase B)
    __aicore__ inline void Tq4DequantRows(LocalTensor<KV_T> &srcTensor, LocalTensor<K_ROPE_T> &dstB16,
                                          int32_t dealRow);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const RunInfo &info, const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                          uint32_t columnCount);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                       LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    __aicore__ inline void CopyFALseToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                         LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                            uint32_t mStartRow, uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                          uint32_t dealRowCount, uint32_t columnCount,
                                          uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t dealRowCount, uint32_t columnCount,
                                             uint32_t actualColumnCount);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);

    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素数
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // repeat stride不能超过256
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    static constexpr bool PAGE_ATTENTION = QSFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = QSFAT::templateMode;
    static constexpr bool FLASH_DECODE = QSFAT::flashDecode;
    static constexpr QSFA_LAYOUT LAYOUT_T = QSFAT::layout;
    static constexpr QSFA_LAYOUT KV_LAYOUT_T = QSFAT::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_512B / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T);  // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                                // 32/4=8
    static constexpr uint32_t LIMIT_DEAL_ROW = 16U;
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr int32_t TQ4_DEQUANT_CHUNK = 4;

    const TurboQuantSparseFlashAttentionTilingDataMla *__restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo constInfo = {};

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<K_ROPE_T> vec1ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blkTableGm_;

    GlobalTensor<K_ROPE_T> kvMergeGm_;
    GlobalTensor<K_ROPE_T> keyRopeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;

    // ================================Local Buffer区====================================
    TBuf<> inputBuff1;  // 32K * 2
    TBuf<> inputBuff2;  // 32K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 4K

    TBuf<> tmpBuff1;         // 32K
    TBuf<> tmpBuff2;         // 8K
    TBuf<> v0ValidSizeBuff;  // 8K

    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxMaxDefaultBuff; // 1K
    TBuf<> softmaxSumDefaultBuff; // 1K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;

    // [TQ4] persistent centSigned codebook (setup once in InitBuffers); int4b_t HW Cast unpack
    // needs no nibble masks / reorder idx.
    TBuf<> tq4CentBuf_;    // 16 float (centSigned[k] = _CENT[(k+8)%16])
    LocalTensor<float> tq4Cent_;

    // [O8/O9 attention-fold] per-token s_t (totalScale) exported by O9 (CopyOutMrgeResult) to sTGm_,
    // consumed per-column by O8 (DealBmm1ResBaseBlock). sTIdxBuf holds Gather byte-offsets i*32
    // (precomputed once @InitBuffers) for the O9 bulk 32B->2B extract.
    GlobalTensor<half> sTGm_;
    TBuf<> sTIdxBuf;
};

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff2, ConstInfo::BUFFER_SIZE_BYTE_4K);

    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, ConstInfo::BUFFER_SIZE_BYTE_8K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::BUFFER_SIZE_BYTE_8K);

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_512B);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_512B);

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff1.Get<KV_T>();

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();

    // [TQ4] one-time setup: centSigned codebook (gather index = int4b signed nibble + 8). Done ONCE.
    pipe->InitBuffer(tq4CentBuf_, ConstInfo::BUFFER_SIZE_BYTE_256B);
    tq4Cent_ = tq4CentBuf_.Get<float>();
    tq4Cent_.SetValue(0,  0.00547294f);  tq4Cent_.SetValue(1,  0.01680406f);   // centSigned[k]=_CENT[(k+8)%16]
    tq4Cent_.SetValue(2,  0.02857605f);  tq4Cent_.SetValue(3,  0.04108622f);
    tq4Cent_.SetValue(4,  0.05492980f);  tq4Cent_.SetValue(5,  0.07101817f);
    tq4Cent_.SetValue(6,  0.09115373f);  tq4Cent_.SetValue(7,  0.12037795f);
    tq4Cent_.SetValue(8, -0.12091285f);  tq4Cent_.SetValue(9, -0.09111122f);
    tq4Cent_.SetValue(10,-0.07112455f);  tq4Cent_.SetValue(11,-0.05513602f);
    tq4Cent_.SetValue(12,-0.04132067f);  tq4Cent_.SetValue(13,-0.02874970f);
    tq4Cent_.SetValue(14,-0.01700489f);  tq4Cent_.SetValue(15,-0.00568677f);
    // [O9] Gather byte-offsets i*32 for bulk s_t extract (first 2B of each 32B row). Max dealRow =
    // INPUT1_BUFFER_OFFSET(32K)/combineDimAlign(~416B) ~= 78, so 128 entries covers it.
    pipe->InitBuffer(sTIdxBuf, 512);
    LocalTensor<uint32_t> sTIdxInit = sTIdxBuf.Get<uint32_t>();
    for (uint32_t i = 0; i < 128; ++i) {
        sTIdxInit.SetValue(i, i * 32);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::InitParams(const struct ConstInfo &constInfo,
                                     const TurboQuantSparseFlashAttentionTilingDataMla *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec0GlobalTensor(
    const GlobalTensor<int32_t> &kvValidSizeGm, const GlobalTensor<K_ROPE_T> &kvMergeGm,
    const GlobalTensor<K_ROPE_T> &keyRopeGm, const GlobalTensor<KV_T> &keyGm, const GlobalTensor<int32_t> &blkTableGm,
    const GlobalTensor<half> &sTGm)
{
    this->kvMergeGm_ = kvMergeGm;
    this->keyRopeGm_ = keyRopeGm;
    this->keyGm_ = keyGm;
    this->blkTableGm_ = blkTableGm;
    this->kvValidSizeGm_ = kvValidSizeGm;
    this->sTGm_ = sTGm;   // [O8/O9] per-token s_t export/consume GM
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_ROPE_T> vec1ResGm,
    GlobalTensor<int32_t> actualSeqLengthsQGm, GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
    GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm,
    GlobalTensor<T> softmaxMaxGm, GlobalTensor<T> softmaxSumGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->topkGm_ = topKGm;
    this->softmaxMaxGm = softmaxMaxGm;
    this->softmaxSumGm = softmaxSumGm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                      GlobalTensor<T> vec2ResGm,
                                                                      GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                      GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::InitSoftmaxDefaultBuffer()
{
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET);
    Duplicate(softmaxSumDefaultUb, ConstInfo::FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ComputeLogSumExpAndCopyToGm(const RunInfo &info,
                                                                                         const MSplitInfo &mSplitInfo,
                                                                                         LocalTensor<T> &softmaxSumUb,
                                                                                         LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t qsfaBaseOffset = mSplitInfo.nBufferStartM / 2;
    size_t qsfaSize = mSplitInfo.vecDealM * FP32_BLOCK_ELEMENT_NUM;
    uint64_t qsfaAccumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t qsfaOffset = (qsfaAccumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize +              // taskoffset
                       info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize + // 份数offset
                       mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                       FP32_BLOCK_ELEMENT_NUM; // m轴offset
    if (info.actualSingleProcessSInnerSize != 0) {
        LocalTensor<T> qsfaTmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(qsfaTmp, softmaxSumUb[qsfaBaseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseSumFdGm[qsfaOffset], qsfaTmp, qsfaSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        qsfaTmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(qsfaTmp, softmaxMaxUb[qsfaBaseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseMaxFdGm[qsfaOffset], qsfaTmp, qsfaSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(lseSumFdGm[qsfaOffset], qsfaSize, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(lseMaxFdGm[qsfaOffset], qsfaSize, SOFTMAX_MIN_NUM);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyFALseToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                        LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM;

    int64_t offset = 0;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t actualSeqQTotal = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(constInfo.batchSize - 1);
        uint64_t actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(info.bIdx - 1);
        offset += info.n2Idx * actualSeqQTotal * constInfo.gSize +
                  (actualSeqQPrefixSum + info.gS1Idx / constInfo.gSize) * constInfo.gSize +
                  mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
    } else {
        offset += info.bIdx * constInfo.kvHeadNum * constInfo.qSeqSize * constInfo.gSize +
                  info.n2Idx * constInfo.qSeqSize * constInfo.gSize +
                  info.gS1Idx / constInfo.gSize * constInfo.gSize +
                  mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
    }

    if (info.actualSingleProcessSInnerSize != 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(T) * size;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        size_t alignedSize = (sizeof(T) * size + 31) / 32 * 32 / sizeof(T);
        LocalTensor<T> tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(tmp, softmaxMaxUb[baseOffset], alignedSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopyPad(softmaxMaxGm[offset], tmp, dataCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(tmp, softmaxSumUb[baseOffset], alignedSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopyPad(softmaxSumGm[offset], tmp, dataCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(softmaxSumGm[offset], size, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(softmaxMaxGm[offset], size, SOFTMAX_MIN_NUM);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ElewiseCompute(const RunInfo &info,
                                                                const LocalTensor<T> &mmResUb,
                                                                uint32_t dealRowCount, uint32_t columnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // v0的无效值判断
        uint64_t qsfaS2ValidSizeFirstPart = v0ValidSizeUb_.GetValue(128 + info.loop % MERGE_CACHE_GM_BUF_NUM);
        uint64_t qsfaS2ValidSizeSecondPart = v0ValidSizeUb_.GetValue(256 + info.loop % MERGE_CACHE_GM_BUF_NUM);

        int64_t qsfaS2ProcessSize = info.actualSingleProcessSInnerSize;
        int64_t qsfaS2Pair = CeilDiv(qsfaS2ProcessSize, 2L * constInfo.sparseBlockSize);
        int64_t qsfaS2Mid = CeilDiv(qsfaS2Pair, 2L) * 2 * constInfo.sparseBlockSize;
        if (qsfaS2Mid > qsfaS2ProcessSize) {
            qsfaS2Mid = qsfaS2ProcessSize;
        }
        if (unlikely(qsfaS2ValidSizeFirstPart < qsfaS2Mid)) {
            int64_t qsfaS2StartCeilAlign = CeilAlign(qsfaS2ValidSizeFirstPart, 8);
            int64_t qsfaS2MidFloorAlign = qsfaS2Mid / 8 * 8;
            // 场景一 s2Mid > s2ValidSizeFirstPart + oneBlk
            // 可以推导出s2StartCeilAlign < s2Mid   第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2MidFloorAlign 第二阶段取到s2MidFloorAlign
            // 场景二 s2Mid <= s2ValidSizeFirstPart + oneBlk
            // 可以推导出 s2StartCeilAlign >= s2Mid 第一阶段取到mid
            // s2StartCeilAlign > s2MidFloorAlign 第二阶段取到s2StartCeilAlign
            SetInfInBlk(mmResUb, dealRowCount, columnCount, qsfaS2ValidSizeFirstPart,
                        qsfaS2StartCeilAlign >= qsfaS2Mid ? qsfaS2Mid : qsfaS2StartCeilAlign);
            SetMidInf(mmResUb, dealRowCount, columnCount, qsfaS2StartCeilAlign, qsfaS2MidFloorAlign);
            SetInfInBlk(mmResUb, dealRowCount, columnCount,
                        qsfaS2StartCeilAlign <= qsfaS2MidFloorAlign ? \
                        qsfaS2MidFloorAlign : qsfaS2StartCeilAlign, qsfaS2Mid);
        }
        if (unlikely(qsfaS2ValidSizeSecondPart < qsfaS2ProcessSize - qsfaS2Mid)) {
            // 场景一 s2Mid + s2ValidSizeSecondPart > s2ProcessSize + oneBlk
            // 可以推导出 s2StartCeilAlign < s2ProcessSize 第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2EndFloorAlign 第二阶段取到s2EndFloorAlign
            // 场景二 s2Mid + s2ValidSizeSecondPart <= s2ProcessSize + oneBlk
            // 可以推导出 s2StartCeilAlign >= s2ProcessSize 第一阶段取到s2ProcessSize
            // s2StartCeilAlign > s2EndFloorAlign 第二阶段取到s2StartCeilAlign
            int64_t qsfaS2StartCeilAlign = CeilAlign(qsfaS2Mid + qsfaS2ValidSizeSecondPart, 8);
            int64_t qsfaS2EndFloorAlign = qsfaS2ProcessSize / 8 * 8;
            SetInfInBlk(mmResUb, dealRowCount, columnCount, qsfaS2Mid + qsfaS2ValidSizeSecondPart,
                        qsfaS2StartCeilAlign >= qsfaS2ProcessSize ? qsfaS2ProcessSize : qsfaS2StartCeilAlign);
            SetMidInf(mmResUb, dealRowCount, columnCount, qsfaS2StartCeilAlign, qsfaS2EndFloorAlign);
            SetInfInBlk(mmResUb, dealRowCount, columnCount,
                        qsfaS2StartCeilAlign <= qsfaS2EndFloorAlign ? qsfaS2EndFloorAlign : qsfaS2StartCeilAlign,
                        qsfaS2ProcessSize);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SetInfInBlk(const LocalTensor<T> &mmResUb,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint64_t startId, uint64_t endId)
{
    //       startId     endId
    // x x x   0      0   0     x x x
    // 从startId到endId部分置-inf, endId、startId为endId一个blk内部的下标
    if (startId >= endId) {
        return;
    }

    uint64_t qsfaStartFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t qsfaNotComputePreMaskOneBlk = (1 << (startId - qsfaStartFloorAlignSize)) - 1;
    uint64_t qsfaNotComputePostMaskOneBlk = ~((1 << (endId - qsfaStartFloorAlignSize)) - 1);
    uint64_t qsfaNotComputeMaskOneBlk = qsfaNotComputePreMaskOneBlk ^ qsfaNotComputePostMaskOneBlk;

    uint64_t qsfaMaskOneBlk = ~qsfaNotComputeMaskOneBlk;
    uint64_t mask[1] = {qsfaMaskOneBlk};
    for (int i = 1; i < 8; i++) {
        mask[0] = mask[0] | (qsfaMaskOneBlk << (i * 8));
    }
    for (uint64_t qsfaRowId = 0; qsfaRowId < dealRowCount; qsfaRowId += 8) {
        Duplicate(mmResUb[qsfaRowId * columnCount + qsfaStartFloorAlignSize], SOFTMAX_MIN_NUM, mask,
                  1, CeilDiv(columnCount, 8), 0);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SetMidInf(const LocalTensor<T> &mmResUb,
                                                           uint32_t dealRowCount, uint32_t columnCount,
                                                           uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }
    // startId        endId
    //    0      ...    0
    // 从startId到endId部分置-inf, startId、endId为32B对齐的下标
    for (uint64_t qsfaRowId = 0; qsfaRowId < dealRowCount; qsfaRowId++) {
        Duplicate(mmResUb[qsfaRowId * columnCount + startId], SOFTMAX_MIN_NUM, endId - startId);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SoftmaxFlashV2Compute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
    }
    if (actualColumnCount !=0) {
        SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, QSFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
        mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
        softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCount);
        PipeBarrier<PIPE_V>();
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCount);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DealBmm1ResBaseBlock(
    const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t loopId)
{
    uint32_t qsfaComputeSize = dealRowCount * columnCount;
    uint64_t qsfaInOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> qsfaMmResUb = inputBuff1.Get<MM1_OUT_T>();
    qsfaMmResUb = qsfaMmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(qsfaMmResUb, mm1ResGm[qsfaInOutGmOffset], qsfaComputeSize);
    // [O8 pre-divide] load per-column s_t [0..columnCount) (fp16) from sTGm_ (written by O9 in
    // CopyOutMrgeResult V0 stage; cross-core staging guarantees visibility, no extra flag). One load
    // feeds BOTH per-column Muls (pre-softmax F*s_t, post-softmax P*s_t). tmpBuff2 free here (softmax
    // uses tmpBuff1); sTUbF32 survives across SoftmaxFlashV2.
    LocalTensor<half> sTUbHalf = tmpBuff2.Get<half>();
    LocalTensor<float> sTUbF32 = tmpBuff2.Get<float>()[256];
    DataCopyExtParams sTLoadParams;
    sTLoadParams.blockCount = 1;
    sTLoadParams.blockLen = columnCount * sizeof(half);
    sTLoadParams.srcStride = 0;
    sTLoadParams.dstStride = 0;
    DataCopyPadExtParams<half> sTPadParams{false, 0, 0, 0};
    DataCopyPad(sTUbHalf, sTGm_[info.loop % MERGE_CACHE_GM_BUF_NUM * constInfo.s2BaseSize], sTLoadParams, sTPadParams);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        if (loopId == 0) {
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(info, qsfaMmResUb, dealRowCount, columnCount);

    // [O8 pre-divide pre-softmax] cube produced F = true_score/s_t (nope unscaled + rope/s_t);
    // ElewiseCompute applied scaleValue. Per-column Mul(F, s_t) -> true_score. -inf cols stay -inf.
    PipeBarrier<PIPE_V>();
    Cast(sTUbF32, sTUbHalf, AscendC::RoundMode::CAST_NONE, columnCount);
    PipeBarrier<PIPE_V>();
    for (uint32_t r = 0; r < dealRowCount; ++r) {
        Mul(qsfaMmResUb[r * columnCount], qsfaMmResUb[r * columnCount], sTUbF32[0], columnCount);
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<T> qsfaTmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> qsfaSoftmaxTmpUb = qsfaTmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(info, mSplitInfo, qsfaMmResUb, qsfaSoftmaxTmpUb, startRow, dealRowCount, columnCount,
                            info.actualSingleProcessSInnerSize);

    // [O8 pre-divide post-softmax] per-column Mul(P, s_t) so MMAD#2 (P*s_t)@y_hat_unscaled = P@V_true
    // (V=unscaled y_hat). Masked cols P=0 -> 0*s_t=0. sTUbF32 survived softmax in tmpBuff2.
    PipeBarrier<PIPE_V>();
    for (uint32_t r = 0; r < dealRowCount; ++r) {
        Mul(qsfaMmResUb[r * columnCount], qsfaMmResUb[r * columnCount], sTUbF32[0], columnCount);
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<K_ROPE_T> tmpMMResCastTensor = outputBuff1.Get<K_ROPE_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, qsfaMmResUb, AscendC::RoundMode::CAST_ROUND, qsfaComputeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[qsfaInOutGmOffset], tmpMMResCastTensor, qsfaComputeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec1SingleBuf(const RunInfo &info,
                                                                                  const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t qsfaMSplitSize = info.actualSingleProcessSInnerSize == 0 ?
        16 : (BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign);
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    qsfaMSplitSize = qsfaMSplitSize >> 3U << 3U;

    if (qsfaMSplitSize > mSplitInfo.vecDealM) {
        qsfaMSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t qsfaLoopCount = (mSplitInfo.vecDealM + qsfaMSplitSize - 1) / qsfaMSplitSize;
    uint32_t qsfaTailSplitSize = mSplitInfo.vecDealM - (qsfaLoopCount - 1) * qsfaMSplitSize;

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 256 * sizeof(int32_t);
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = 0;
        DataCopyPadExtParams<int32_t> padParams;
        // 额外偏移128个元素，避免不同loop下v0和v1互相影响
        DataCopyPad(v0ValidSizeUb_[128], kvValidSizeGm_[info.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2)],
                    dataCopyParams, padParams);
        SetFlag<HardEvent::MTE2_S>(0);
        if (unlikely(qsfaLoopCount == 0)) {
            // scalar同步影响较大，挪到循环内部进行
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    for (uint32_t qsfaI = 0, dealSize = qsfaMSplitSize; qsfaI < qsfaLoopCount; qsfaI++) {
        if (qsfaI == (qsfaLoopCount - 1)) {
            dealSize = qsfaTailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, qsfaI * qsfaMSplitSize, dealSize,
            info.actualSingleProcessSInnerSizeAlign, qsfaI);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                              int64_t topkGmBaseOffset, const RunInfo &runInfo)
{
    int64_t qsfaTopkGmIdx = (s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) / constInfo.sparseBlockSize;
    if (unlikely(qsfaTopkGmIdx >= constInfo.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + qsfaTopkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) % constInfo.sparseBlockSize);
}

template <typename QSFAT>
__aicore__ inline int64_t QSFAVectorService<QSFAT>::GetKeyBNBOffset(int64_t realS2Idx,
                                                                    const RunInfo &runInfo, int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyBNBOffset = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx = realS2Idx / constInfo.kvCacheBlockSize;
        int64_t blkTableOffset = realS2Idx % constInfo.kvCacheBlockSize;
        realKeyBNBOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo.maxBlockNumPerBatch + blkTableIdx) *
                                static_cast<int64_t>(constInfo.kvCacheBlockSize) *
                                static_cast<int64_t>(constInfo.kvHeadNum) +
                                blkTableOffset;
    } else {
        realKeyBNBOffset = (runInfo.tensorBOffset +
                           realS2Idx * constInfo.kvHeadNum * constInfo.combineHeadDim) /
                           constInfo.combineHeadDim;
    }
    return realKeyBNBOffset;
}

// [TQ4] Phase B: codebook dequant of `dealRow` combined slots (int4 nope + rope + 2B scale)
// -> dstB16 [dealRow,headDim] bf16 (Hadamard-space K=V). Read each aligned slot row
// directly as int4b_t, then batch index/Gather/scale/output Cast for each 4-row chunk.
// The 2B slot scale is expected to be vecNorm/sqrt(Sum c^2) on the fused-SFA write path.
template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Tq4DequantRows(LocalTensor<KV_T> &srcTensor, LocalTensor<K_ROPE_T> &dstB16, int32_t dealRow)
{
    uint32_t HD = constInfo.headDim;
    uint32_t ROW_BYTES = QSFAAlign(static_cast<uint32_t>(tilingData->baseParams.dSizeVInput),
        static_cast<uint32_t>(BYTE_BLOCK));
    uint32_t SCALE_BYTE = constInfo.headDim / 2 + constInfo.headDimRope * sizeof(K_ROPE_T);
    constexpr int32_t CHUNK = TQ4_DEQUANT_CHUNK;
    constexpr uint32_t CHUNK_ELEMS = CHUNK * 512;
    constexpr uint32_t WORK_BYTE_OFF = 0;
    constexpr uint32_t WORK_BYTES = CHUNK_ELEMS * sizeof(float);
    constexpr uint32_t SHALF_BYTE_OFF = WORK_BYTE_OFF + WORK_BYTES;
    constexpr uint32_t SHALF_BYTES = CHUNK_ELEMS * sizeof(half);
    constexpr uint32_t IDX_BYTE_OFF = SHALF_BYTE_OFF + SHALF_BYTES;
    constexpr uint32_t IDX_BYTES = CHUNK_ELEMS * sizeof(int32_t);
    static_assert(IDX_BYTE_OFF + IDX_BYTES <= ConstInfo::BUFFER_SIZE_BYTE_32K,
                  "TQ4 dequant scratch exceeds inputBuff2");

    LocalTensor<float> workBase = inputBuff2.Get<float>()[WORK_BYTE_OFF / sizeof(float)];
    LocalTensor<half> sHalfBase = inputBuff2.Get<half>()[SHALF_BYTE_OFF / sizeof(half)];
    LocalTensor<int32_t> idxI = inputBuff2.Get<int32_t>()[IDX_BYTE_OFF / sizeof(int32_t)];
    LocalTensor<uint32_t> idxU = inputBuff2.Get<uint32_t>()[IDX_BYTE_OFF / sizeof(uint32_t)];
    LocalTensor<float> centBuf = tq4Cent_;
    LocalTensor<uint16_t> slotU16 = srcTensor.template ReinterpretCast<uint16_t>();
    LocalTensor<int4b_t> srcI4 = srcTensor.template ReinterpretCast<int4b_t>();

    PipeBarrier<PIPE_ALL>();
    if (unlikely(dealRow <= 0)) {
        return;
    }

    for (int32_t base = 0; base < dealRow; base += CHUNK) {
        int32_t cur = (base + CHUNK <= dealRow) ? CHUNK : (dealRow - base);
        uint32_t cnt = static_cast<uint32_t>(cur) * HD;

        for (int32_t rr = 0; rr < cur; ++rr) {
            int32_t r = base + rr;
            Cast(sHalfBase[rr * HD], srcI4[r * ROW_BYTES * 2], RoundMode::CAST_NONE, HD);  // int4b -> half, -8..7
        }
        PipeBarrier<PIPE_V>();
        Adds(sHalfBase, sHalfBase, static_cast<half>(8.0f), cnt);   // signed nibble +8 -> [0,15]
        PipeBarrier<PIPE_V>();
        Muls(sHalfBase, sHalfBase, static_cast<half>(4.0f), cnt);   // *4 -> byte offset [0,60]
        PipeBarrier<PIPE_V>();
        Cast(idxI, sHalfBase, RoundMode::CAST_ROUND, cnt);
        PipeBarrier<PIPE_V>();
        Gather(workBase, centBuf, idxU, 0, cnt);                    // non-negative offsets only (base 0)
        PipeBarrier<PIPE_V>();

        // [O8 pre-divide] per-row Muls(s_t) DELETED: output UNSCALED y_hat (centroids only). s_t is
        // exported to sTGm_ by O9 below and re-applied per-column on the attention side
        // (DealBmm1ResBaseBlock pre/post-softmax Mul). V=unscaled y_hat.
        if constexpr (IsSameType<K_ROPE_T, bfloat16_t>::value) {
            Cast(dstB16[base * HD], workBase, RoundMode::CAST_RINT, cnt);
        } else {
            Cast(dstB16[base * HD], workBase, RoundMode::CAST_ROUND, cnt);
        }
        PipeBarrier<PIPE_V>();
    }
    SetFlag<AscendC::HardEvent::V_MTE3>(0);
    WaitFlag<AscendC::HardEvent::V_MTE3>(0);
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                         int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        ((realS2Idx + constInfo.sparseBlockSize > s2IdLimit) ? (s2IdLimit - realS2Idx) : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams;

    intriParams.blockCount = validS2Count;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // 当前仅支持COMBINE模式
    if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
        // [TQ4] slot = headDim/2 int4 nope + headDimRope*sizeof(K_ROPE_T) + 2B vecNorm(fp16)
        uint32_t combineBytes = constInfo.headDim / 2 + constInfo.headDimRope * sizeof(K_ROPE_T) + sizeof(half);
        intriParams.blockLen = combineBytes;
        uint32_t combineDim = combineBytes / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)  + (mte2Size - mte3Size) *
                combineDimAlign], keyGm_[keyBNBOffset * combineDim], intriParams, padParams);
    }
    mte2Size += validS2Count;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                          int64_t realS2Idx1, int64_t realS2Idx2,
                                                          const RunInfo &runInfo)
{
    int64_t s2IdLimit = runInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        s2IdLimit = runInfo.curActualSeqLenOri - runInfo.actS1Size + runInfo.gS1Idx / constInfo.gSize + 1;
    }

    int64_t keyBNBOffset1 = GetKeyBNBOffset(realS2Idx1, runInfo, s2IdLimit);
    int64_t keyBNBOffset2 = GetKeyBNBOffset(realS2Idx2, runInfo, s2IdLimit);
    if (unlikely(keyBNBOffset1 < 0 && keyBNBOffset2 < 0)) {
        return;
    }

    int64_t sparseBlockSrcStride =
        ((keyBNBOffset1 > keyBNBOffset2 ? (keyBNBOffset1 - keyBNBOffset2) :
        (keyBNBOffset2 - keyBNBOffset1)) - constInfo.sparseBlockSize);
    // [TQ4] slot = headDim/2 int4 nope + headDimRope*sizeof(K_ROPE_T) + 2B vecNorm(fp16)
    uint32_t combineBytes = constInfo.headDim / 2 + constInfo.headDimRope * sizeof(K_ROPE_T) + sizeof(half);
    int64_t keySrcStride = sparseBlockSrcStride * combineBytes;
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 ||
        realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
        realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit) ||
        constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyBNBOffset1, s2IdLimit, runInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyBNBOffset2, s2IdLimit, runInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyBNBOffset1 >= 0) + (keyBNBOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyBNBOffset1 > -1 ? keyBNBOffset1 : keyBNBOffset2;
        if (keyBNBOffset2 > -1 && keyBNBOffset2 < keyBNBOffset1) {
            startGmOffset = keyBNBOffset2;
        }

        // 当前仅支持COMBINE模式
        if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
            intriParams.blockLen = constInfo.sparseBlockSize * combineBytes;
            uint32_t combineDim = combineBytes / sizeof(KV_T);
            uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = combineDimAlign - combineDim;
            padParams.paddingValue = 0;
            DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) *
                        combineDimAlign], keyGm_[startGmOffset * combineDim], intriParams, padParams);
        }
        mte2Size += ((keyBNBOffset1 > -1) + (keyBNBOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                   int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                   const RunInfo &runInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    int32_t dealRow = mte2Size - mte3Size;
    SetFlag<AscendC::HardEvent::MTE2_V>(0);
    WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    LocalTensor<KV_T> srcTensor = kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)];
    LocalTensor<K_ROPE_T> antiKvTensorAsB16 = tmpBuff1.Get<K_ROPE_T>();
    uint64_t mask = ConstInfo::BUFFER_SIZE_BYTE_256B / sizeof(half);
    uint32_t qsfaRopeByteOff = constInfo.headDim * sizeof(KV_T);
    uint32_t slotBytes = static_cast<uint32_t>(tilingData->baseParams.dSizeVInput);
    uint8_t qsfaRopeRowStrideBlk = static_cast<uint8_t>(
        QSFAAlign(slotBytes, static_cast<uint32_t>(BYTE_BLOCK)) / BYTE_BLOCK);
    uint64_t mergeGmStride = 512 * constInfo.combineHeadDim;
    if (constInfo.keyQuantMode == QUANT_MODE::TQ4) {
        // [TQ4] codebook dequant -> antiKvTensorAsB16 [dealRow,headDim] bf16
        Tq4DequantRows(srcTensor, antiKvTensorAsB16, dealRow);
        // [O9] bulk s_t export -> sTGm_ (consumed per-column by O8 DealBmm1ResBaseBlock). Carrier scale is
        // fp16 @ byte SCALE_BYTE=headDim/2+headDimRope*2=384 (half idx 192), token stride ROW_BYTES=
        // CeilAlign(dSizeVInput,32)=416B=13 blocks. NBURST: read 1 block(32B)/token, srcStride=12 blocks
        // (640B) -> [dealRow,32B] in sTUb32; Gather first 2B of each 32B row -> contiguous sTUb[dealRow].
        {
            uint32_t sTScaleByteOff = constInfo.headDim / 2 + constInfo.headDimRope * sizeof(K_ROPE_T);
            uint32_t sTRowStrideBlk = QSFAAlign(slotBytes, static_cast<uint32_t>(BYTE_BLOCK)) / BYTE_BLOCK;
            LocalTensor<half> sTUb = tmpBuff2.Get<half>();
            LocalTensor<half> sTUb32 = tmpBuff2.Get<half>()[512];
            LocalTensor<half> srcHalf = srcTensor.template ReinterpretCast<half>()[sTScaleByteOff / 2];
            DataCopyParams sTGathParams;
            sTGathParams.blockCount = static_cast<uint16_t>(dealRow);
            sTGathParams.blockLen = 1;
            sTGathParams.srcStride = static_cast<uint16_t>(sTRowStrideBlk - 1);
            sTGathParams.dstStride = 0;
            DataCopy(sTUb32, srcHalf, sTGathParams);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint32_t> sTIdx = sTIdxBuf.Get<uint32_t>();
            Gather(sTUb, sTUb32, sTIdx, 0, static_cast<uint32_t>(dealRow));
            SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
            WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
            DataCopyExtParams sTParams;
            sTParams.blockCount = 1;
            sTParams.blockLen = static_cast<uint32_t>(dealRow) * sizeof(half);
            sTParams.srcStride = 0;
            sTParams.dstStride = 0;
            DataCopyPad(sTGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * constInfo.s2BaseSize + s2GmStartOffset + mte3Size],
                        sTUb, sTParams);
        }
        qsfaRopeByteOff = constInfo.headDim / 2;

        DataCopyExtParams tq4DataCopyParams;
        tq4DataCopyParams.blockCount = static_cast<uint16_t>(dealRow);
        tq4DataCopyParams.blockLen = constInfo.headDim * sizeof(K_ROPE_T);
        tq4DataCopyParams.srcStride = 0;
        tq4DataCopyParams.dstStride = (constInfo.combineHeadDim - constInfo.headDim) * sizeof(K_ROPE_T);
        uint64_t tq4GmBase = runInfo.loop % MERGE_CACHE_GM_BUF_NUM * mergeGmStride +
            (s2GmStartOffset + mte3Size) * constInfo.combineHeadDim;
        DataCopyPad(kvMergeGm_[tq4GmBase], antiKvTensorAsB16, tq4DataCopyParams);

        LocalTensor<K_ROPE_T> tq4KRopeUb = srcTensor[qsfaRopeByteOff].template ReinterpretCast<K_ROPE_T>();
        tq4DataCopyParams.blockLen = constInfo.headDimRope * sizeof(K_ROPE_T);
        // DataCopyPad UB-side srcStride is in 32B datablocks, not bytes (cf. sparse_flash_attention
        // CopyOutMrgeResult). The rope row pitch is qsfaRopeRowStrideBlk blocks; subtract blockLen's blocks.
        tq4DataCopyParams.srcStride = static_cast<uint32_t>(qsfaRopeRowStrideBlk) -
            constInfo.headDimRope * sizeof(K_ROPE_T) / BYTE_BLOCK;
        tq4DataCopyParams.dstStride = (constInfo.combineHeadDim - constInfo.headDimRope) * sizeof(K_ROPE_T);
        DataCopyPad(kvMergeGm_[tq4GmBase + constInfo.headDim], tq4KRopeUb, tq4DataCopyParams);
        return;
    }
}

// b s1 k
template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::MergeKv(const RunInfo &runInfo)
{
    int64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = 0;

    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t qsfaActualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(runInfo.bIdx - 1);
        topkGmBaseOffset += (qsfaActualSeqQPrefixSum + runInfo.gS1Idx / constInfo.gSize) * constInfo.kvHeadNum *
                            constInfo.sparseBlockCount + runInfo.n2Idx * constInfo.sparseBlockCount;
    } else {
        topkGmBaseOffset += runInfo.bIdx * constInfo.qSeqSize * constInfo.sparseBlockCount +
                            runInfo.gS1Idx / constInfo.gSize * constInfo.sparseBlockCount;
    }
    int64_t qsfaMergeMte3Idx = 0;
    int64_t qsfaMte2Size = 0;
    int64_t qsfaMte3Size = 0;
    int64_t qsfaS2IdxArray0 = -1;
    int64_t qsfaS2IdxArray1 = -1;
    bool qsfaNeedWaitMte3ToMte2 = true;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    int64_t qsfaS2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t qsfaS2GmLimit = GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize: s2ProcessSize;
    if (qsfaS2GmLimit > s2ProcessSize) {
        qsfaS2GmLimit = s2ProcessSize;
    }
    for (int64_t s2GmOffsetArray = qsfaS2GmStartOffset; s2GmOffsetArray < qsfaS2GmLimit; s2GmOffsetArray += 2 *
        constInfo.sparseBlockSize) {
        if (qsfaNeedWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % 2);
            qsfaNeedWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, qsfaS2IdxArray0, topkGmBaseOffset, runInfo);
        if (unlikely(qsfaS2IdxArray0 < 0)) {
            CopyOutMrgeResult(qsfaMte2Size, qsfaMte3Size, qsfaS2GmStartOffset, qsfaMergeMte3Idx, runInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % 2);
            qsfaMergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, qsfaS2IdxArray1, topkGmBaseOffset, runInfo);
        CopyInKv(qsfaMte2Size, qsfaMte3Size, qsfaMergeMte3Idx, qsfaS2IdxArray0, qsfaS2IdxArray1, runInfo);
        if ((qsfaMte2Size - qsfaMte3Size + 2 * constInfo.sparseBlockSize > 32) ||
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= qsfaS2GmLimit) {
            CopyOutMrgeResult(qsfaMte2Size, qsfaMte3Size, qsfaS2GmStartOffset, qsfaMergeMte3Idx, runInfo);
            qsfaMte3Size = qsfaMte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % 2);
            qsfaMergeMte3Idx++;
            qsfaNeedWaitMte3ToMte2 = true;
        }
    }

    if (unlikely(qsfaS2GmStartOffset + qsfaMte2Size < qsfaS2GmLimit)) {
        uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx & 1);
        LocalTensor<K_ROPE_T> mergeUb = kvMergUb_.template ReinterpretCast<K_ROPE_T>();
        Duplicate(mergeUb, static_cast<K_ROPE_T>(0.0), constInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = constInfo.headDim / blockElementNum;
        dataCopyParams.blockLen = blockElementNum * sizeof(K_ROPE_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.s2BaseSize - 1) * blockElementNum * sizeof(K_ROPE_T);
        uint64_t mergeGmStride = 512 * constInfo.combineHeadDim;
        for (int64_t s2GmOffset = qsfaS2GmStartOffset + qsfaMte2Size; s2GmOffset < qsfaS2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * mergeGmStride + s2GmOffset * blockElementNum],
                        mergeUb, dataCopyParams);
        }
        dataCopyParams.blockCount = constInfo.headDimRope / blockElementNum;
        for (int64_t s2GmOffset = qsfaS2GmStartOffset + qsfaMte2Size; s2GmOffset < qsfaS2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * mergeGmStride + 512 * constInfo.headDim +
                                   s2GmOffset * blockElementNum],
                        mergeUb, dataCopyParams);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx & 1);
        qsfaMergeMte3Idx++;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM, qsfaMte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 128 * sizeof(int32_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(kvValidSizeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) + GetSubBlockIdx() * 128],
                v0ValidSizeUb_, dataCopyParams);
    return;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec1L(const RunInfo &info)
{
    uint32_t qsfaNBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t qsfaNBufferTail = info.actMBaseSize - (qsfaNBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t qsfaI = 0; qsfaI < qsfaNBufferLoopTimes; qsfaI++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = qsfaI;
        mSplitInfo.nBufferStartM = qsfaI * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (qsfaI + 1 != qsfaNBufferLoopTimes) ? constInfo.nBufferMBaseSize : qsfaNBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
                                                                (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(constInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
        // move lse for flash decode
        if (info.s2Idx == info.curSInnerLoopTimes - 1 && (constInfo.returnSoftmaxLse || info.tndIsS2SplitCore)) {
            uint32_t outIdx = info.loop % (constInfo.preLoadNum);
            auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
            auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
            if (constInfo.returnSoftmaxLse) {
                CopyFALseToGm(info, mSplitInfo, sumTensor, maxTensor);
            }
            if (info.tndIsS2SplitCore) {
                if constexpr (FLASH_DECODE) {
                    ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
                }
            }
        }
    }
}

template <typename QSFAT>
__aicore__ inline uint64_t QSFAVectorService<QSFAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    return 0;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2SingleBuf(const RunInfo &info,
                                                                      const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    uint32_t gPreSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (gPreSplitSize > mSplitInfo.vecDealM) {
        gPreSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + gPreSplitSize - 1) / gPreSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * gPreSplitSize;

    for (uint32_t i = 0, dealSize = gPreSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * gPreSplitSize, dealSize, constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                                      uint32_t startRow, uint32_t dealRowCount,
                                                                      uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    size_t batchBase = 0;
    uint64_t inOutBaseOffset = (mSplitInfo.vecStartM + startRow) * columnCount;
    uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;

    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset + batchBase], vec2ComputeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<T> bmm2ResPreUb = inputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
        uint64_t vecPre2ResGmOffset = ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
            inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vecPre2ResGmOffset + batchBase], vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        LocalTensor<T> softmaxExpBrcb =  tmpBuff2.Get<T>();
        Brcb(softmaxExpBrcb, softmaxExpUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
            (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpBrcb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    }
    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        LocalTensor<T> softmaxSumBrcb =  tmpBuff2.Get<T>();
        Brcb(softmaxSumBrcb, softmaxSumUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
            (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumBrcb, dealRowCount, columnCount, actualColumnCount);

        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUb, mSplitInfo.vecStartM + startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> tmpBmm2Res = outputBuff1.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t vecPre2ResGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vecPre2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
}

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2L(const RunInfo &info)
{
    uint32_t qsfaNBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t qsfaNBufferTail = info.actMBaseSize - (qsfaNBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t qsfaI = 0; qsfaI < qsfaNBufferLoopTimes; qsfaI++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = qsfaI;
        mSplitInfo.nBufferDealM = (qsfaI + 1 != qsfaNBufferLoopTimes) ? constInfo.nBufferMBaseSize : qsfaNBufferTail;
        mSplitInfo.nBufferStartM = qsfaI * constInfo.nBufferMBaseSize;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
            (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(constInfo.syncC2V2);
        ProcessVec2SingleBuf(info, mSplitInfo);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2Inner(const RunInfo &info,
                                                                  const MSplitInfo &mSplitInfo,
                                                                  uint32_t mStartRow, uint32_t mDealSize)
{
    uint32_t qsfaMSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (qsfaMSplitSize > mDealSize) {
        qsfaMSplitSize = mDealSize;
    }

    uint32_t qsfaLoopCount = (mDealSize + qsfaMSplitSize - 1) / qsfaMSplitSize;
    uint32_t qsfaTailSplitSize = mDealSize - (qsfaLoopCount - 1) * qsfaMSplitSize;
    for (uint32_t qsfaI = 0, dealSize = qsfaMSplitSize; qsfaI < qsfaLoopCount; qsfaI++) {
        if (qsfaI == (qsfaLoopCount - 1)) {
            dealSize = qsfaTailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, qsfaI * qsfaMSplitSize + mStartRow, dealSize,
                             constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}


template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                            uint32_t wsMStart, uint32_t dealRowCount,
                                            uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = accumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim + // taskoffset
        info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim + // 份数offset
        wsMStart * actualColumnCount; // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    if (info.actualSingleProcessSInnerSize == 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, dealRowCount * actualColumnCount, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb,
                                               uint32_t wsMStart, uint32_t dealRowCount,
                                               uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[info.attenOutOffset + wsMStart * actualColumnCount], attenOutUb, dataCopyParams);
    return;
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                             uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                             uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> qsfaTmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(qsfaTmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(qsfaTmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, qsfaTmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                         uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    if constexpr (!FLASH_DECODE) {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    } else {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t qsfaDtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t qsfaDLoop = actualColumnCount / qsfaDtypeMask;
    uint32_t qsfaDRemain = actualColumnCount % qsfaDtypeMask;

    BinaryRepeatParams qsfaRepeatParamsDiv;
    qsfaRepeatParamsDiv.src0BlkStride = 1;
    qsfaRepeatParamsDiv.src1BlkStride = 0;
    qsfaRepeatParamsDiv.dstBlkStride = 1;
    qsfaRepeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    qsfaRepeatParamsDiv.src1RepStride = 1;
    qsfaRepeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t qsfaColumnRepeatCount = qsfaDLoop;
    if (qsfaColumnRepeatCount <= dealRowCount) {
        uint32_t qsfaOffset = 0;
        for (uint32_t qsfaI = 0; qsfaI < qsfaDLoop; qsfaI++) {
            Div(dstUb[qsfaOffset], src0Ub[qsfaOffset], src1Ub, qsfaDtypeMask, dealRowCount, qsfaRepeatParamsDiv);
            qsfaOffset += qsfaDtypeMask;
        }
    } else {
        BinaryRepeatParams qsfaColumnRepeatParams;
        qsfaColumnRepeatParams.src0BlkStride = 1;
        qsfaColumnRepeatParams.src1BlkStride = 0;
        qsfaColumnRepeatParams.dstBlkStride = 1;
        qsfaColumnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        qsfaColumnRepeatParams.src1RepStride = 0;
        qsfaColumnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t qsfaOffset = 0;
        for (uint32_t qsfaI = 0; qsfaI < dealRowCount; qsfaI++) {
            Div(dstUb[qsfaOffset], src0Ub[qsfaOffset], src1Ub[qsfaI * FP32_BLOCK_ELEMENT_NUM], qsfaDtypeMask,
                qsfaColumnRepeatCount, qsfaColumnRepeatParams);
            qsfaOffset += columnCount;
        }
    }
    if (qsfaDRemain > 0) {
        Div(dstUb[qsfaDLoop * qsfaDtypeMask], src0Ub[qsfaDLoop * qsfaDtypeMask], src1Ub, qsfaDRemain,
            dealRowCount, qsfaRepeatParamsDiv);
    }
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t qsfaRepeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t qsfaBlockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        qsfaRepeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        qsfaBlockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t qsfaDLoop = actualColumnCount / qsfaRepeatElementNum;
    uint32_t qsfaDRemain = actualColumnCount % qsfaRepeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * qsfaBlockElementNum) {
        BinaryRepeatParams qsfaRepeatParams;
        qsfaRepeatParams.src0BlkStride = 1;
        qsfaRepeatParams.src1BlkStride = 0;
        qsfaRepeatParams.dstBlkStride = 1;
        qsfaRepeatParams.src0RepStride = columnCount / qsfaBlockElementNum;
        qsfaRepeatParams.src1RepStride = 1;
        qsfaRepeatParams.dstRepStride = columnCount / qsfaBlockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (qsfaDLoop <= dealRowCount) {
            uint32_t qsfaOffset = 0;
            for (uint32_t qsfaI = 0; qsfaI < qsfaDLoop; qsfaI++) {
                Mul(dstUb[qsfaOffset], src0Ub[qsfaOffset], src1Ub, qsfaRepeatElementNum, dealRowCount,
                    qsfaRepeatParams);
                qsfaOffset += qsfaRepeatElementNum;
            }
        } else {
            BinaryRepeatParams qsfaColumnRepeatParams;
            qsfaColumnRepeatParams.src0BlkStride = 1;
            qsfaColumnRepeatParams.src1BlkStride = 0;
            qsfaColumnRepeatParams.dstBlkStride = 1;
            qsfaColumnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            qsfaColumnRepeatParams.src1RepStride = 0;
            qsfaColumnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t qsfaI = 0; qsfaI < dealRowCount; qsfaI++) {
                Mul(dstUb[qsfaI * columnCount], src0Ub[qsfaI * columnCount], src1Ub[qsfaI * qsfaBlockElementNum],
                    qsfaRepeatElementNum, qsfaDLoop, qsfaColumnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (qsfaDRemain > 0) {
            Mul(dstUb[qsfaDLoop * qsfaRepeatElementNum], src0Ub[qsfaDLoop * qsfaRepeatElementNum], src1Ub,
                qsfaDRemain, dealRowCount, qsfaRepeatParams);
        }
    } else {
        BinaryRepeatParams qsfaRepeatParams;
        qsfaRepeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        qsfaRepeatParams.src0BlkStride = 1;
        qsfaRepeatParams.src1RepStride = 0;
        qsfaRepeatParams.src1BlkStride = 0;
        qsfaRepeatParams.dstRepStride = 8;
        qsfaRepeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t qsfaI = 0; qsfaI < dealRowCount; qsfaI++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[qsfaI * columnCount], src0Ub[qsfaI * columnCount], src1Ub[qsfaI * qsfaBlockElementNum],
                qsfaRepeatElementNum, qsfaDLoop, qsfaRepeatParams);
            //  计算一行中的尾块
            if (qsfaDRemain > 0) {
                Mul(dstUb[qsfaI * columnCount + qsfaDLoop * qsfaRepeatElementNum],
                    src0Ub[qsfaI * columnCount + qsfaDLoop * qsfaRepeatElementNum],
                    src1Ub[qsfaI * qsfaBlockElementNum], qsfaDRemain, 1, qsfaRepeatParams);
            }
        }
    }
}

#endif // TURBOQUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
