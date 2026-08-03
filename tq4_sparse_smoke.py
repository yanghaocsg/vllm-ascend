import os
import sys
import time


def report(message: str) -> None:
    os.write(2, f"{message}\n".encode())


report("import:torch:start")
import torch
report("import:torch:done")
import torch_npu
report("import:torch_npu:done")

device_id = int(os.environ.get("SMOKE_DEVICE", "0"))
torch.npu.set_device(device_id)
report(f"device:set:{device_id}")

import vllm_ascend.vllm_ascend_C  # noqa: F401
report("import:vllm_ascend:done")

query_tokens = int(os.environ.get("SMOKE_TOKENS", "192"))
num_query_heads = 16
head_dim = 512
block_size = 32
num_blocks = (query_tokens + block_size - 1) // block_size
compressed_tokens = max(1, (query_tokens + 3) // 4)
num_compressed_blocks = (compressed_tokens + block_size - 1) // block_size
kv_quant_mode = int(sys.argv[1]) if len(sys.argv) > 1 else 3

query = torch.zeros(
    (query_tokens, num_query_heads, head_dim),
    dtype=torch.bfloat16,
    device="npu",
)
ori_kv = torch.zeros((num_blocks, block_size, 1, head_dim), dtype=torch.bfloat16, device="npu")
cmp_slot_size = 320 if kv_quant_mode == 3 else head_dim
cmp_dtype = torch.uint8 if kv_quant_mode == 3 else torch.bfloat16
cmp_kv = torch.zeros(
    (num_compressed_blocks, block_size, 1, cmp_slot_size),
    dtype=cmp_dtype,
    device="npu",
)
ori_block_table = torch.arange(num_blocks, dtype=torch.int32, device="npu").view(1, -1)
cmp_block_table = torch.arange(num_compressed_blocks, dtype=torch.int32, device="npu").view(1, -1)
cmp_sparse_indices = torch.full((query_tokens, 1, 512), -1, dtype=torch.int32, device="npu")
valid_sparse_tokens = min(compressed_tokens, 512)
cmp_sparse_indices[:, 0, :valid_sparse_tokens] = torch.arange(
    valid_sparse_tokens,
    dtype=torch.int32,
    device="npu",
)
cu_seqlens_q = torch.tensor([0, query_tokens], dtype=torch.int32, device="npu")
cu_seqlens_cmp_kv = torch.tensor([0, compressed_tokens], dtype=torch.int32, device="npu")
seqused_kv = torch.tensor([query_tokens], dtype=torch.int32, device="npu")
seqused_q = torch.tensor([query_tokens], dtype=torch.int32, device="npu")
sinks = torch.zeros((num_query_heads,), dtype=torch.float32, device="npu")

report("metadata:start")
metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
    num_heads_q=num_query_heads,
    num_heads_kv=1,
    head_dim=head_dim,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_ori_kv=cu_seqlens_q,
    seqused_q=seqused_q,
    seqused_kv=seqused_kv,
    batch_size=1,
    max_seqlen_q=query_tokens,
    max_seqlen_kv=query_tokens,
    cmp_ratio=4,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_ND",
    has_ori_kv=True,
    has_cmp_kv=True,
    device="npu",
)
torch.npu.synchronize()
report("metadata:done")

start = time.perf_counter()
report("attention:start")
output, _ = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
    query,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    cmp_sparse_indices=cmp_sparse_indices,
    ori_block_table=ori_block_table,
    cmp_block_table=cmp_block_table,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
    seqused_kv=seqused_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=head_dim**-0.5,
    cmp_ratio=4,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_ND",
    kv_quant_mode=kv_quant_mode,
)
report("attention:submitted")
torch.npu.synchronize()
report(f"attention:done elapsed={time.perf_counter() - start:.6f}s shape={tuple(output.shape)}")
