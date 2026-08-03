# DSV4 TQ4 功能验证报告

日期：2026-08-03

## 1. 结论

DeepSeek V4 Attention 下的 TQ4 功能流程已经跑通。

- 4 层裁剪模型 TP4 权重加载成功。
- OpenAI 兼容服务启动成功，`/v1/models` 返回 HTTP 200。
- DSV4 c4 sparse compressor 层实际使用 UINT8 TQ4 KV；SWA/raw window 和 c128 CFA 保持 BF16。
- 真实 TQ4 融合算子在 1-token、192-token 单算子输入下均正常返回。
- 端到端完成 192 prompt tokens + 32 output tokens，未出现 worker 超时或内核挂死。
- 不验证精度，本报告不包含输出质量结论。

本次验证使用最小混合 KV 方案。DSV4 与 GLM 的共同原则是只搬运并解码当前 sparse attention 命中的 TQ4 slot，不把整个 KV cache 反量化到全局 workspace。DSV4 的 SWA cache 仍需保留 BF16，并在 c4 层写入前转换到 Hadamard 空间；attention 输出再执行 inverse Hadamard。

## 2. 验证环境与模型

- 原始权重：`/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp`
- 裁层模型：`/tmp/dsv4-4layer`
- 模型层数覆盖：`{"num_hidden_layers": 4}`
- 层类型：layer 0/1 为 SWA，layer 2 为 c4 sparse compressor，layer 3 为 c128 CFA
- 并行方式：TP4 + EP
- 最终使用物理 Die：2、3、4、5
- 模式：eager，`max_model_len=1024`
- TQ4 开关：`--kv-cache-dtype turboquant_4bit_nc`
- 安装命令：`pip install -e . --no-build-isolation --no-deps`

每次启动单算子或服务前均通过 `npu-smi info` 检查 Die 占用。

## 3. 实测结果

### 3.1 编译与回归测试

- editable 安装成功，未安装额外依赖。
- 修复增量构建后，`SparseAttnSharedkv` 两个内核 `.o` 的时间、大小和 SHA256 均发生变化，安装目录与构建目录哈希一致。
- 定向 UT：`15 passed, 14 warnings`。
- `git diff --check`：通过。

### 3.2 单算子功能

| 场景 | 结果 | 耗时 |
| --- | --- | ---: |
| TQ4 mixed SCFA，1 token，真实逐 slot 解码 | 通过 | 0.166878 s |
| TQ4 mixed SCFA，192 tokens，真实逐 slot 解码 | 通过 | 0.162451 s |
| BF16 SCFA 基线，1 token | 通过 | 0.166787 s |

1-token 单算子下 TQ4 与 BF16 基本持平，差异约 0.05%，处于短任务测量波动范围。单算子结论用于证明路径可执行，不作为正式性能结论。

### 3.3 TP4 服务与端到端推理

- 31.91 GiB、8 个 safetensors shards 加载完成，rank 0 权重加载耗时 3.07 s。
- 每个 TP rank 模型权重约 7.8943 GB。
- engine 初始化完成，服务监听 `http://0.0.0.0:9999`。
- `/v1/models` 返回 `dsv4-functional`。
- 首个 192+32 请求耗时 5.264103 s，包含算子冷启动。
- 热态 3 轮：0.770669 s、0.756157 s、0.755428 s，平均 0.760751 s。
- 已有相同 token 规模 BF16 功能基线：0.7726 s。
- 小样本热态端到端时延相对 BF16 改善约 1.5%。该数字仅表示当前 4 层 eager 功能模型，不可外推为完整 DSV4 收益。

### 3.4 KV 存储收益

c4 sparse compressor 的单 token latent slot：

- BF16：512 x 2 = 1024 bytes
- TQ4：256 bytes int4 codes + 2 bytes FP16 scale + padding = 320 bytes
- 单 slot 减少：704 bytes，即 68.75%

本次方案只量化 c4 compressor cache。SWA/raw window 和 c128 CFA 仍为 BF16，因此 68.75% 是 c4 compressor 局部收益，不是整个模型 KV cache 的总收益。

## 4. 方案设计与 GLM 逻辑对应关系

### 4.1 为什么不能直接照搬 GLM 的全 UINT8 输入

GLM 已跑通的关键逻辑是“根据 sparse index 只读取当前命中的 320-byte TQ4 slot，在当前 attention tile 内解码”。本次沿用了这个数据流，但没有照搬其输入 dtype 组合。

DSV4 的 SCFA 一次计算同时消费两类 KV：

- `ori_kv`：SWA/raw window，逻辑 head dim 为 512，attention 直接使用。
- `cmp_kv`：c4 compressor latent，通过 sparse index 选择 slot 后使用。

只有 `cmp_kv` 具备与 GLM 相同的 sparse slot-selection 入口。如果把 `ori_kv` 也定义成 UINT8，SCFA 会失去当前窗口所需的 BF16/FP16 原始 KV；如果把 c128 CFA 也量化，则没有对应的逐 slot 稀疏解码入口。因此最终采用 mixed KV，而不是全 UINT8：

```text
q             BF16/FP16, logical head dim = 512
ori_kv        BF16/FP16, logical head dim = 512
cmp_kv        UINT8,     physical slot width = 320 bytes
attn_out      与 q/ori_kv 相同
kv_quant_mode 3
layout        PA_ND
template      SCFA
```

### 4.2 启用范围

TQ4 的 Python 侧总开关仍来自已有 `enable_dsa_tq_latent` 配置，但增加了 `compress_ratio == 4` 的层级约束：

```python
self.use_tq_latent = (
    ascend_config.enable_dsa_tq_latent
    and self.compress_ratio == 4
)
```

4 层功能模型中的实际策略为：

| 层 | Attention 类型 | KV dtype | 是否 TQ4 |
| --- | --- | --- | --- |
| layer 0/1 | SWA | BF16 | 否 |
| layer 2 | c4 SCFA | ori BF16 + cmp UINT8 | 是 |
| layer 3 | c128 CFA | BF16 | 否 |

### 4.3 最终算子内数据流

```text
sparse index
    -> 计算 cmp_kv 中命中 slot 的 byte offset
    -> GM 搬运一个 320B slot 到 UB
    -> 解包 512 个 int4 code
    -> code + 8 映射到旋转后的 centroid 表
    -> Gather 512 个 FP32 centroid
    -> 读取 slot[256:258] 的 FP16 scale 并向量广播
    -> centroid * scale
    -> 转为 BF16/FP16，直接写当前 kvMergUb_ row
    -> 进入原 SCFA Cube/Vector pipeline
```

该路径不会生成“整个 KV cache 的 BF16 副本”，也不需要所有核等待完整 cache 解码完成。这一点与 GLM 的已验证原则一致。

## 5. 代码级修改明细

### 5.1 `vllm_ascend/attention/dsa_v1.py`

修改对象：`AscendDSAImpl.__init__`、`_write_dsa_kv_cache`，并复用已有的 `_transform_dsa_query`、`_restore_dsa_output`。

1. `AscendDSAImpl.__init__` 将 TQ4 限定在 c4 compressor：

   ```python
   self.use_tq_latent = (
       ascend_config.enable_dsa_tq_latent
       and self.compress_ratio == 4
   )
   ```

   修改前只检查全局开关，会让 c128 CFA 也进入 TQ4 路径；修改后 c128 保持原 BF16 路径。已有的 DSA context parallel 和 A2/A3 设备校验继续生效，没有扩大支持范围。

2. `_write_dsa_kv_cache` 按目标 cache dtype 分流：

   ```python
   if cache.dtype != torch.uint8:
       if self.use_tq_latent:
           kv = _tq_store().had_fwd(
               kv.reshape(-1, DSA_TQ_HEAD_DIM),
               head_dim=DSA_TQ_HEAD_DIM,
           ).view_as(kv)
       DeviceOperator.dsa_kv_compress_scatter(cache, kv, slot_mapping)
       return
   ```

   c4 层的 SWA cache 仍是 BF16，但必须与 TQ4 compressor/query 位于相同 Hadamard 空间，因此在 scatter 前执行 `had_fwd`。非 TQ4 层不会执行该转换。

3. UINT8 cache 增加显式契约检查，并调用已有 TurboQuant compressor：

   ```python
   if not self.use_tq_latent:
       raise RuntimeError("DeepSeek V4 uint8 KV cache requires TurboQuant ...")
   if cache.dtype != torch.uint8 or cache.shape[-1] != DSA_TQ_SLOT_SIZE:
       raise RuntimeError(...)
   packed_kv, workspace = _tq_store().compress_kernel(
       kv.reshape(-1, DSA_TQ_HEAD_DIM),
       head_dim=DSA_TQ_HEAD_DIM,
   )
   DeviceOperator.dsa_kv_compress_scatter(
       cache,
       packed_kv.view(-1, 1, DSA_TQ_SLOT_SIZE),
       slot_mapping,
   )
   ```

   即逻辑输入固定 512 维，物理输出固定 UINT8/320B。`workspace` 保持存活到 scatter 入队后再释放，避免异步执行时源数据生命周期提前结束。

4. prefill 和 decode 均在 `self.use_tq_latent` 时把已有常量 `DSA_TQ_KV_QUANT_MODE` 作为 `kv_quant_mode` 传给 sparse attention op。`_transform_dsa_query` 使用已有 `had_fwd`，`_restore_dsa_output` 使用已有 `had_inv`。这两个变换函数和 `kv_quant_mode` 传递框架不是本轮新增，本轮仅让它们在正确的 DSV4 c4 层生效。

影响范围：仅 DSV4 c4 TQ4；普通 DSA、c128 CFA、未开启 TQ4 的路径保持原行为。

### 5.2 `vllm_ascend/models/layer/attention/layer.py`

修改对象：`DSAAttention.get_kv_cache_spec`。

```python
if (
    get_ascend_config().enable_dsa_tq_latent
    and self.compress_ratio == 4
):
    ...
```

只有 c4 compressor 申请 `dtype=torch.uint8`、`head_size=320` 的 TQ4 cache spec。c128 不再因为全局开关被错误分配 UINT8 cache。A2/A3、DSA CP 和 cache dtype 的既有校验没有放宽。

### 5.3 `vllm_ascend/models/deepseek_v4.py`

修改对象：`AscendDeepseekV4SWACache.get_kv_cache_spec`、`AscendDeepseekV4ForCausalLM.load_weights`，新增 `_is_extra_main_model_layer_weight`。

1. 删除 `AscendDeepseekV4SWACache` 的 TQ4 UINT8 spec 分支。SWA 始终沿用其 BF16/FP16 cache dtype，解决 DSV4 mixed KV 中 `ori_kv` 必须是浮点 KV 的问题。

2. 为裁层验证新增严格的主模型层识别：

   ```python
   def _is_extra_main_model_layer_weight(config, weight_name):
       layer_prefix = "model.layers."
       if not weight_name.startswith(layer_prefix):
           return False
       layer_index, separator, _ = (
           weight_name.removeprefix(layer_prefix).partition(".")
       )
       if not separator or not layer_index.isdigit():
           return False
       return int(layer_index) >= config.num_hidden_layers
   ```

3. `load_weights` 完成名称规范化后、参数映射前跳过超范围权重：

   ```python
   if _is_extra_main_model_layer_weight(self.config, name):
       continue
   ```

该逻辑只匹配 `model.layers.<数字>.`，不会误跳过 embedding、`lm_head` 或 MTP 权重。完整层数模型中所有合法 layer index 均小于 `num_hidden_layers`，行为不变。

### 5.4 `vllm_ascend/device/device_op.py`

修改对象：`BaseDeviceAdaptor.dsa_kv_compress_scatter`。

```python
scatter_cache = cache.view(torch.int8) if cache.dtype == torch.uint8 else cache
scatter_x = x.view(torch.int8) if x.dtype == torch.uint8 else x
torch.ops._C_ascend.npu_scatter_nd_update_v2(
    scatter_cache, slot_mapping, scatter_x
)
```

`aclnnScatterNdUpdateV2` 支持 INT8 但不支持 UINT8。TQ4 slot 是不参与有符号数值计算的 opaque bytes，因此 `.view(torch.int8)` 只重解释 dtype，不复制、不改变 bit pattern，也不改变 `data_ptr()`。BF16 等已支持 dtype 仍原样传入。

### 5.5 `vllm_ascend/patch/platform/patch_kv_cache_utils.py`

修改对象：`_get_kv_cache_groups_uniform_groups`、`_get_kv_cache_config_deepseek_v4`。

1. 删除旧的强假设：

   ```python
   assert max(sm_page_sizes) <= max(all_page_sizes)
   ```

   对每个 SWA page size 查找不小于它的 canonical page；若不存在则保留自身大小：

   ```python
   size_to_candidate[ps] = next(
       (x for x in all_page_sizes if x >= ps), ps
   )
   ```

2. allocator 不再只使用 full-MLA group 的 page size 集合，而是遍历所有 group 的真实 `page_size_bytes`。每个 group 内按 `(page_size, slot_idx)` 放置 layer：

   ```python
   slot_idx = slot_count[page_size]
   slot_count[page_size] += 1
   buckets[page_size][slot_idx].append(name)
   ```

3. 每 block 内存按所有 bucket 和 MTP tensor 求和：

   ```python
   bytes_per_block = sum(
       page_size * len(slots)
       for page_size, slots in buckets.items()
   )
   bytes_per_block += sum(page_size for _, page_size in mtp_layers)
   ```

4. MTP layer 单独生成 `KVCacheTensor`，不与主模型共享 slot。这样可防止独立 block-table namespace 的 MTP 映射污染主模型 cache。

这部分修改是 TQ4 cache、BF16 SWA 和 FP32 compressor state 形成异构 page size 后的必要修复；相同 page size、相同 slot index 的普通 layer 仍可共享 tensor。

### 5.6 `csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_def.cpp`

`SparseAttnSharedkv` 的 dtype tuple 从普通 FP16/BF16 两组扩展为四组：

| tuple | q | ori_kv | cmp_kv | attn_out |
| --- | --- | --- | --- | --- |
| 普通 FP16 | FP16 | FP16 | FP16 | FP16 |
| 普通 BF16 | BF16 | BF16 | BF16 | BF16 |
| TQ4 FP16 | FP16 | FP16 | UINT8 | FP16 |
| TQ4 BF16 | BF16 | BF16 | UINT8 | BF16 |

所有伴随输入的 dtype/format 列表同步扩为四项，保证 OpDef 的同一 tuple index 表示完整一致的输入组合。与早期方案不同，TQ4 tuple 不再把 `ori_kv` 声明成 UINT8。

### 5.7 `csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp`

修改对象：dtype 支持表、`GenerateInfo`、`CheckFeatureShape`、`CheckOriAndCmpKv`、`DoOpTiling`。

- `ori_kv` 只允许 FP16/BF16；`cmp_kv` 允许 FP16/BF16/UINT8。
- TQ4 下 `ori_kv` head dim 固定 512，`cmp_kv` physical head dim 固定 320。
- `q` dtype 始终必须等于 `ori_kv` dtype。
- 普通模式 `kv_quant_mode == 0` 才要求 `cmp_kv` 与 `ori_kv` dtype 一致；TQ4 要求 mixed dtype。
- `kv_quant_mode` 仅支持 0 或 3。
- mode 3 仅允许 `PA_ND` layout 和 `SCFA_TEMPLATE_MODE`，明确拒绝 SWA/CFA 等未实现组合。
- 删除 `oriKvSlotCount` 计算，因为 ori cache 不再量化。
- 删除按所有 slot 分配 BF16 解码 workspace 的逻辑。
- `oriKvStride` 保留真实 BF16/FP16 stride，不再按 320B slot 换算为 512 维 stride。

这些校验把 unsupported combination 阻断在 host tiling 阶段，避免错误 dtype/layout 进入 AICore 后产生难定位的内存访问问题。

### 5.8 `csrc/attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv.cpp`

kernel dispatch 原本只联合判断 `q/ori_kv/attn_out`，现在显式加入 `ORIG_DTYPE_CMP_KV`：

```cpp
ORIG_DTYPE_Q == DT_BF16 &&
ORIG_DTYPE_ORI_KV == DT_BF16 &&
ORIG_DTYPE_CMP_KV == DT_UINT8 &&
ORIG_DTYPE_ATTN_OUT == DT_BF16
```

FP16 mixed tuple 同样增加。普通 FP16/BF16 dispatch 也显式要求 cmp dtype 与其一致，防止一个 dtype tuple 被错误分派到另一个模板实例。

### 5.9 `csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_kernel.h`

修改对象：global tensor 绑定、workspace 初始化和 `Process`。
- 删除 `oriTqGm`，mode 3 时 `oriKvGm` 直接绑定 BF16/FP16 `oriKV`，仅 `cmpTqGm` 绑定 UINT8 `cmpKV`。
- 删除在 workspace 中为完整 ori cache 创建 BF16 副本的逻辑。
- 删除 kernel 入口的 `DequantTqCache` 全量循环。
- 删除 `SYNC_TQ_CACHE_READY_FLAG`、`SyncAll` 和 AIV/AIC 跨核等待。

普通 SCFA 的计算入口未改变；TQ4 解码被下沉到当前 sparse tile 的搬运阶段。

### 5.10 `csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_block_vector.h`

这是按 GLM 逐命中 slot 解码原则完成的核心调整。

1. 删除旧的 `DequantTqCache`、`CopyTqKv`、`ProcessVec0Tq` 三条旁路。旧路径在正常 ping-pong pipeline 之外单独构造 TQ4 流程，既需要全量 workspace，也容易破坏同步关系。

2. 在已有 `CopyInSingleKv` 中增加 mode 3 分支：

   ```cpp
   DataCopy(slot, cmpTqGm_[keyBNBOffset], TQ4_SLOT_BYTES);
   SetFlag<HardEvent::MTE2_V>(0);
   WaitFlag<HardEvent::MTE2_V>(0);
   DequantTq4Slot(
       slot, output, work, signedCodes, indices, tqCentBuff.Get<float>()
   );
   ++mte2Size;
   ```

   `keyBNBOffset` 在 mode 3 下按 block table 和 `TQ4_SLOT_BYTES` 计算 byte offset；一次只搬 320B。解码输出直接指向 `kvMergUb_` 当前 ping-pong buffer 的当前 row。

3. `CopyInKv` 原本一次处理两个 sparse 命中位置；TQ4 分支分别调用两次 `CopyInSingleKv`，无效 index 仍由 `keyBNBOffset < 0` 直接过滤。

4. 普通 BF16 路径的数据来自 MTE2，而 TQ4 row 在 `DequantTq4Slot` 中经过 Vector 指令生成。因此 `CopyOutMrgeResult` 对 mode 3 使用 `V_MTE3` 同步，普通路径继续使用 `MTE2_MTE3`。这是保证 CopyOut 读取到完整解码结果的必要依赖关系。

### 5.11 `csrc/attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_common.h`

修改对象：`DequantTq4Slot`。

- 新增 `TQ4_CENTROID_BYTE_STRIDE = 4.0F`，替代 `sizeof(float)` 参与 AICore 浮点转换的写法，并消除 magic number。
- 将 512 个 int4 signed code cast 为 half，执行 `+8` 后得到 0..15 的索引；乘 4 转为 FP32 centroid table 的 byte offset。
- `Gather` 得到 512 个 FP32 centroid。
- 从 slot 的 `[256:258]` 读取 FP16 scale；通过 `Cast`、`Brcb` 和 `Mul` 在 Vector 侧广播并应用 scale。
- 输出为 BF16 时使用 `CAST_RINT`，输出为 FP16 时使用 `CAST_ROUND`。

关键实现：

```cpp
Cast(scale, slotHalf[TQ4_PACKED_BYTES / sizeof(half)],
     RoundMode::CAST_NONE, 8);
Brcb(scaleBrcb, scale, 1, {1, 8});
Mul(work, work, scaleBrcb, 64, 8, {1, 1, 0, 8, 8, 0});
```

该写法避免在 AICore 函数中执行 host C++ 风格的标量 bit reinterpret，也避免 device tensor 标量回读。

### 5.12 `csrc/build_aclnn.sh`

修改对象：A2/A3 `CUSTOM_OPS_ARRAY` 和增量构建缓存失效逻辑。

1. `ascend910b`、`ascend910_93` 两个分支均加入 `compressor` 和 `compressor_metadata`，保证 DSV4 TQ4 写 cache 所需算子进入 editable wheel/自定义 OPP 包。

2. 新增 `invalidate_stale_kernel_cache`。对每个已选算子检查 `op_kernel` 源文件是否比对应 `gen/<op>_<soc>_*.done` 新；仅在 stale 时删除：

   - `binary/<soc>/gen/<op>*.done`
   - `binary/<soc>/bin/<op>`
   - `binary/<soc>/src/<op>`
   - CPack 暂存树中对应的 `op_impl/ai_core/<soc>/<op>`

3. 删除旧的 `binary_info_config.json` 和 `relocatable_kernel_info_config.json`。这两个 manifest 的原构建依赖只跟目录相关，增量加入算子时不会自动重建。

该修改不是全量 clean：只使源码确实变新的选中算子失效，保留其他算子的编译缓存，符合快速迭代和最小影响原则。

### 5.13 测试与复现代码

- `tests/ut/device/test_device_op.py`
  - `test_dsa_kv_compress_scatter_reinterprets_uint8_without_copy`：检查 UINT8 变为 INT8 view，并断言 view 与原 tensor 的 `data_ptr()` 相同。
  - `test_dsa_kv_compress_scatter_preserves_supported_dtype`：检查 BF16 tensor 原样传入 ACLNN。
- `tests/ut/models/test_deepseek_v4.py`
  - 验证 layer 0/3 不跳过、layer 4/42 跳过。
  - 验证 embedding、`lm_head`、MTP 不会被识别为超范围主模型层。
- `tests/ut/patch/platform/test_patch_kv_cache_utils.py`
  - 使用最终 mixed-KV 布局（c4 UINT8/320B，c128/SWA BF16/512）验证超出
    full-MLA 最大 page 的 compressor state page 保持自身大小。
  - 验证所有 page size 分桶、`bytes_per_block`、跨 group tensor 复用和 MTP
    tensor 隔离。
- `tests/ut/test_build_aclnn.py`
  - 验证 A2/A3 分支均打包 compressor/metadata。
  - 验证 `.done/bin/src/CPack/manifest` 的 stale 清理逻辑存在。
- `tests/ut/attention/test_deepseek_v4_turboquant_math.py`
  - 验证 Q/K/V 同时执行正交 Hadamard 变换并逆变换输出时 attention 数学不变。
  - 验证 512 维 TQ4 slot 的 320B 布局及 c4 局部 68.75% 存储缩减。
- `tq4_sparse_smoke.py`
  - 构造 mixed BF16/UINT8 SCFA 最小输入，分别用于 mode 3 TQ4 和 mode 0 BF16 单算子探针。

## 6. 完整问题排查与解决记录

### 6.1 裁层模型加载失败：`model.layers.4` KeyError

**现象**：使用 `--hf-overrides '{"num_hidden_layers":4}'` 后，模型实例只有 layer 0..3，但加载第 6 个 shard 附近 worker 全部退出。

**原始错误**（`/tmp/dsv4_tq4_baseline_trimmed.log:218`）：

```text
param = params_dict[name]
KeyError: 'model.layers.4.self_attn.attn_sink'
```

**根因**：HF override 只缩短模型配置和实例，原始 8 个 safetensors shard 仍包含完整模型的 layer 4+ 权重。原 loader 直接执行 `params_dict[name]`，裁掉的层没有对应参数。

**排查过程**：确认错误发生在 `AscendDeepseekV4ForCausalLM.load_weights`，且首个缺失 key 的 layer index 正好等于 override 后的 `num_hidden_layers=4`，不是权重名称转换错误。

**修复**：新增 `_is_extra_main_model_layer_weight`，只在名称严格匹配 `model.layers.<数字>.` 且 index 超范围时跳过；在参数字典查找前调用。

**验证**：8 个 shard 全部加载完成，最终日志显示 checkpoint 31.91 GiB、rank 0 加载 3.07 s；UT 同时验证 layer、非 layer 和 MTP 名称边界。

### 6.2 UINT8 cache scatter 失败

**现象**：服务可以启动，但第一个 192-token 请求在写 KV cache 时 worker 退出。

**原始错误**（`/tmp/dsv4_tq4_run.log:784`）：

```text
AclNN_Parameter_Error(EZ1001):
Tensor varRef not implemented for DT_UINT8,
should be in dtype support list [..., DT_INT8, ...]
```

**根因**：`aclnnScatterNdUpdateV2` 的 varRef 不支持 UINT8，但 TQ4 compressor 输出和 cache spec 都是 UINT8。错误发生在 cache 写入接口，而不是 TQ4 压缩算法。

**修复**：在 `BaseDeviceAdaptor.dsa_kv_compress_scatter` 中只对 UINT8 source/cache 建立 INT8 view，再调用原 ACLNN op。slot 是 opaque bytes，因此有符号解释不影响 bit pattern。

**验证**：UT 断言 dtype 变为 INT8、`data_ptr()` 不变；BF16 path 仍使用原 tensor。修复后请求越过 cache scatter。

### 6.3 完整 cache 反量化导致 22.08 GiB workspace OOM

**现象**：早期 mixed dtype 已能启动服务，但首个请求在执行 attention 前申请巨量内存并 OOM。

**原始错误**（`/tmp/dsv4_tq4_run2.log:288`、`:364`）：

```text
torch.OutOfMemoryError: NPU out of memory.
Tried to allocate 22.08 GiB
```

**根因**：初版 tiling 按全部 ori/cmp slot 数量分配 `slot_count * 512 * sizeof(BF16)` workspace，kernel 入口再把整个 cache 解码为 BF16。即使是 4 层功能模型，block 数和 slot 数也会把临时空间放大到约 22 GiB。该方案也偏离 GLM 的逐命中 slot 解码逻辑。

**排查过程**：根据 OOM 分配大小反查 `DoOpTiling` 中 mode 3 独有的 workspace 增量，再对照 kernel 入口 `DequantTqCache` 的 slot 全遍历，确认内存申请与完整 cache BF16 副本一一对应。

**修复**：删除 tiling 的完整 cache workspace，删除 `oriKvSlotCount`、`DequantTqCache` 和跨核 ready barrier；在 `CopyInSingleKv` 内只搬运当前 sparse index 命中的 320B slot并解码到当前 tile。

**验证**：后续服务初始化不再申请 22.08 GiB 临时空间；1-token 和 192-token 单算子均正常返回，端到端 192+32 请求完成。

### 6.4 请求持续卡死，最终表现为 RPC timeout

**现象**：去掉巨量 workspace 后，请求仍长期无返回；1-token 探针也会挂起。服务侧最终只看到聚合层超时。

**原始错误**：

```text
TimeoutError: RPC call to sample_tokens timed out.
```

日志位置：`/tmp/dsv4_tq4_run3.log:341`、`/tmp/dsv4_tq4_run4.log:345`。

**初步排查**：曾围绕新 pipeline 的 barrier、CopyOut 依赖和解码输出做零填充探针。仅从 RPC timeout 无法判断是 kernel 死锁、非法访问还是运行了旧 binary。

**关键发现**：修改后的 header 时间已经更新，但 `csrc/build/binary/.../sparse_attn_sharedkv/*.o` 的时间仍停留在旧版本，大小和 SHA256 也未变化；`gen/` 下已有 `.done` marker。多次执行指定的 editable 安装命令只重新打包了旧 `.o`。因此此前“新 kernel 仍挂死”的结论无效，实际请求仍在运行旧 kernel。

**修复**：在 `build_aclnn.sh` 增加按 source mtime 检查的 `invalidate_stale_kernel_cache`，清除该算子的 `.done/bin/src/CPack` 缓存和旧 manifest，再按原命令重编。

**验证**：修复后 `.o` 的修改时间更新，大小由约 514 KiB 变为约 505 KiB，SHA256 改变；build 目录与安装目录的 hash 一致。零填充探针先正常返回，恢复真实 `DequantTq4Slot` 后再次重编，最终真实 TQ4 也正常返回。旧 kernel 缓存是本次耗时较长和多次误判的主要原因。

### 6.5 AICore 编译失败及 FATBIN 连锁报错

**现象**：执行用户指定命令：

```bash
pip install -e . --no-build-isolation --no-deps
```

editable wheel 构建失败。日志末尾是：

```text
ninja: build stopped: subcommand failed.
ERROR: Failed building editable for vllm_ascend
```

**真实首错**（`/tmp/dsv4_tq4_pip_install.log:3377`）：

```text
sparse_attn_sharedkv_common.h:358:54: error:
cast between floating and unsigned integer variable
is not allowed in aicore function
Muls(..., static_cast<half>(sizeof(float)), ...)
```

**后续表象**（同日志 `:3392` 起）：

```text
ld.lld: error: cannot open ..._mix_aic_0.o: No such file or directory
ld.lld: error: cannot open ..._mix_aiv_0.o: No such file or directory
CompileStage.FATBIN
```

这些缺 `.o` 的 linker/FATBIN 错误不是独立根因。AICore/AIV 源码编译先失败，对象文件没有生成，链接阶段才批量报告文件不存在。排查时必须抓第一条 compiler error，不能从日志末尾反推成 linker 配置问题。

**根因**：AICore 对 host C++ 风格的浮点/无符号转换限制更严格。旧实现既用 `sizeof(float)` 参与 half 标量转换，又从 UINT16 bits 做标量 half reinterpret；即使只修复首个 `sizeof` 报错，后者仍不是稳妥的 device vector 实现。

**修复**：

- 用明确的浮点常量 `TQ4_CENTROID_BYTE_STRIDE = 4.0F` 表达 centroid FP32 byte stride。
- scale 通过 `LocalTensor<half>` 读取，使用 `Cast + Brcb + Mul` 完成 FP32 向量广播和乘法，不再做标量 bit reinterpret。
- 根据目标 dtype 使用明确的 BF16/FP16 rounding mode。

**验证**：修复并确保 stale kernel cache 失效后，editable build 成功，最终安装没有添加依赖；新 `.o` hash 与安装目录一致，真实单算子和 TP4 服务都使用新 kernel 完成请求。

### 6.6 DSV4 mixed KV 契约修正

**错误方向**：早期曾把 `ori_kv` 和 `cmp_kv` 都声明为 UINT8，并尝试让 c128 CFA 进入 TQ4。

**根因**：这只复用了 GLM 的 dtype 外形，没有复用其正确的数据选择前提。DSV4 SCFA 的 `ori_kv` 是 SWA/raw window，必须直接供 attention 使用；只有 c4 `cmp_kv` 经过 sparse index 逐 slot 选择。c128 CFA 没有同构入口。

**修复**：将最终契约固定为 `q/ori_kv=BF16或FP16`、`cmp_kv=UINT8`；Python cache spec、OpDef tuple、tiling 校验和 kernel dispatch 四层同时收紧，且只允许 `compress_ratio == 4`、SCFA、PA_ND。

**验证**：4 层模型中 layer 2 实际走 TQ4，layer 0/1 SWA 和 layer 3 c128 保持 BF16；单算子 mixed input 与服务端路径均通过。

### 6.7 KV allocator 的 page size 假设失效

**现象**：启用 mixed cache 后，同时存在 UINT8 latent、BF16 SWA、FP32 compressor state，多组 page size 无法落入原 full-MLA canonical bucket。

**旧假设**：

```python
assert max(sm_page_sizes) <= max(all_page_sizes)
```

该断言在 compressor state page 大于所有 full-MLA page 时不成立；旧 `_get_kv_cache_config_deepseek_v4` 也只遍历 full-MLA page size，可能漏算或错误共享异构 tensor。

**修复**：无法向上对齐时保留原 page size；allocator 采集所有真实 page size，按每组的 slot index 分桶，`bytes_per_block` 对全部 bucket 求和，并让 MTP 独占 tensor。

**验证**：UT 覆盖 16,640/40,960/131,072-byte page 集合、10 blocks 的内存计算、主模型同 size 共享以及 MTP 独立分配。最终服务 cache 初始化完成。

### 6.8 单算子启动时覆盖 `LD_LIBRARY_PATH`

**现象**：第一次运行临时单算子探针时 import 失败：

```text
ImportError: libascend_hal.so: cannot open shared object file
```

**根因**：临时命令把 `LD_LIBRARY_PATH` 整体替换为 custom op API 路径，丢失原有 Ascend driver/toolkit 库路径。这不是源码或 wheel 依赖缺失。

**修复**：只前置追加 custom op 路径：

```bash
export LD_LIBRARY_PATH=<custom-op-api>:$LD_LIBRARY_PATH
```

**验证**：torch_npu 和 custom op 均可加载，之后单算子探针进入真实 kernel。

### 6.9 非阻断告警

最终服务日志仍包含以下告警，但均未阻断加载、服务或推理，本次没有为消除告警扩大修改范围：

- `rope_parameters` 中整数类型提示。
- `VLLM_ASCEND_ENABLE_FLASHCOMM1`、`VLLM_ASCEND_ENABLE_FUSED_MC2` 旧环境变量提示。
- TurboQuant 强制 flash attention version 2 的提示。

## 7. 复现命令

安装：

```bash
pip install -e . --no-build-isolation --no-deps
```

定向 UT：

```bash
pytest -q \
  tests/ut/device/test_device_op.py \
  tests/ut/models/test_deepseek_v4.py \
  tests/ut/patch/platform/test_patch_kv_cache_utils.py \
  tests/ut/test_build_aclnn.py \
  tests/ut/attention/test_deepseek_v4_turboquant_math.py
```

单算子真实 TQ4：

```bash
ASCEND_RT_VISIBLE_DEVICES=2 \
SMOKE_DEVICE=0 \
SMOKE_TOKENS=192 \
ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer \
python tq4_sparse_smoke.py 3
```

TP4 服务的关键参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3,4,5 vllm serve /tmp/dsv4-4layer \
  --port 9999 \
  --tokenizer-mode deepseek_v4 \
  --max-model-len 1024 \
  --quantization ascend \
  --enforce-eager \
  --served-model-name dsv4-functional \
  --hf-overrides '{"num_hidden_layers":4}' \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --block-size 128 \
  --gpu-memory-utilization 0.3 \
  --kv-cache-dtype turboquant_4bit_nc \
  --num-gpu-blocks-override 128 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 4
```

## 8. 验证边界

- 未验证生成精度，随机文本输出不用于判断正确性。
- 使用 4 层裁剪模型，未验证完整层数 DSV4 的容量和性能。
- 使用 eager 模式，未验证 ACL graph/torch.compile。
- c128 CFA 当前保持 BF16；本次结论是 DSV4 c4 sparse Attention 下 TQ4 流程可用。
- 性能数据为单并发、小样本功能对比，不替代正式 benchmark。

模型配置的理论最大长度为 1,048,576 tokens；本次服务仅以
`max_model_len=1024` 完成功能验证，因此 1024 是本报告的实测配置，不代表运行时
容量上限。skill 建议的 128K + bs16 容量基线未执行，不能据此声明长上下文容量。

### 8.1 特性状态

| 特性 | 状态 | 证据或限制 |
| --- | --- | --- |
| c4 SCFA TQ4 | 已验证 | 真实 TQ4 单算子 1/192 tokens 与 TP4 端到端 192+32 均完成 |
| Expert Parallel | 已验证 | 服务以 TP4 + EP 启动并完成真实权重请求 |
| flashcomm1 | 已开启但未单独验证 | 日志确认配置为开启；未采集算子级证据，不能单独归因 |
| ACLGraph | 未验证 | 最终服务使用 `--enforce-eager` |
| MTP | checkpoint 存在但未验证 | checkpoint/config 含 1 个 next-token 层，服务未启用 speculative decoding |
| 多模态 | 不适用 | 本模型和本次请求为文本路径 |
| c128 CFA TQ4 | 不支持 | 当前无逐 sparse slot 选择入口，保持 BF16 |
| DSA Context Parallel | 不支持 | TQ4 初始化阶段显式拒绝 |

### 8.2 Dummy 与真实权重

| 阶段 | 状态 | 能证明的范围 |
| --- | --- | --- |
| Dummy | 未执行 | 无 dummy-only 结论 |
| 真实权重 | 已执行 | 4 层实例加载 31.91 GiB shards，HTTP 200 且输出非空，覆盖真实压缩、scatter 和 attention kernel |

## 9. 日志与证据索引

| 文件 | 内容 |
| --- | --- |
| `/tmp/dsv4_tq4_baseline_trimmed.log` | 4 层裁剪首次加载失败，layer 4 KeyError |
| `/tmp/dsv4_tq4_pip_install.log` | AICore 首错、FATBIN 缺 `.o` 连锁错误、editable build 失败 |
| `/tmp/dsv4_tq4_pip_install_clean.log` | 修复后的 editable wheel 构建和安装成功 |
| `/tmp/dsv4_tq4_run.log` | ACLNN UINT8 scatter 失败 |
| `/tmp/dsv4_tq4_run2.log` | 完整 cache 解码 workspace 导致 22.08 GiB OOM |
| `/tmp/dsv4_tq4_run3.log` | 请求卡死后的 `sample_tokens` RPC timeout |
| `/tmp/dsv4_tq4_run4.log` | 再次复现 kernel 卡死及 RPC timeout |
| `/tmp/dsv4_tq4_final.log` | 最终 TP4 加载、服务启动、HTTP 200 和端到端请求成功 |

最终结论以 `/tmp/dsv4_tq4_final.log` 和新 kernel hash 一致性检查为准。
