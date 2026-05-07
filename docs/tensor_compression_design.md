# Tensor 压缩传输设计方案

## 概述

为 TransferQueue 的 SimpleStorage 后端引入 zstd 等 tensor 压缩算法，目标是：

1. **降低 Manager ↔ StorageUnit 之间的网络传输带宽消耗**；
2. **避免 SU 在每次 GET 时重复执行压缩**——压缩工作摊销到一次 PUT，SU 之后所有 GET 都直接发送已压缩的字节。

第二点是本方案的核心约束：SU 不调用任何压缩/解压算法。压缩在 Manager PUT 编码时完成、解压在 Manager GET 解码时完成。SU 在两个方向上都只做"opaque bytes 透传 + 按 sample 索引"。

### 设计原则

- **按行压缩**：以"batched tensor 的行（dim 0 切片）"为压缩粒度，单行独立压缩为一段 bytes，并作为一个 ZMQ frame 发送。这样 SU 仍按 `global_index` 逐行索引存储，与现有 SU 存储模型完全契合。
- **SU 真零 CPU**：SU 在 PUT 接收和 GET 发送两个方向上都不解压、不压缩，只做按行存取与零拷贝 frame 转发。
- **零拷贝保留**：未启用压缩的小 tensor 完全走现有零拷贝路径；启用压缩后的 compressed bytes 同样作为 memoryview frame 零拷贝发送，不经过 pickle 序列化数据内容。
- **可插拔**：算法通过字符串 dispatch（zstd → lz4 等）。
- **向后兼容**：未启用压缩时行为完全不变；同字段允许整字段命中或跳过压缩（不混合，简化布局）。
- **滚动升级不保证**：启用压缩功能后，Manager 与 SU 必须同时升级到支持 `Ext(6)` 的版本。

---

## 当前架构分析

### 数据流关键路径

```
SimpleStorage (ZMQ-based, metadata-backed):

Put: TensorDict → _select_by_positions() → ZMQMessage.serialize() → encode()
       → MsgpackEncoder (zero-copy buffers → multipart frames) → ZMQ send
       → SimpleStorageUnit._worker_routine() → ZMQMessage.deserialize() → StorageUnitData
       → StorageUnitData.put_data() 按 global_index 逐行写入 dict

Get: SimpleStorageUnit._handle_get() → StorageUnitData.get_data()
       → ZMQMessage.serialize() → encode() → ZMQ send
       → AsyncSimpleStorageManager → decode() → _pack_field_values() → TensorDict
```

**关键约束**：`StorageUnitData.put_data()` 内部用 `zip(global_indexes, values, strict=True)` 按 dim 0 把 batched 数据拆成逐 sample 写入 `field_data[f][gi]`。这意味着 SU 接收到的 `values` 必须是**按行可迭代**的容器（torch.Tensor 沿 dim 0 迭代得到 row view，list 自然按元素迭代）。

### 现有零拷贝序列化机制 (serial_utils.py)

- `MsgpackEncoder._encode_regular_tensor()`：
  1. `tensor.flatten().view(torch.uint8).numpy()` → 零拷贝的 uint8 numpy 视图
  2. `memoryview(arr)` → 包装为 memoryview
  3. `aux_buffers.append(buf)` → 仅引用，不复制
  4. 返回 `msgpack.Ext(CUSTOM_TYPE_TENSOR, pickle.dumps((dtype, shape, idx)))`
     — 仅元数据 (~50 bytes) 经过 pickle，数据本身零拷贝
- `encode()` 返回 `[msgpack_meta_frame, tensor_buffer_frame, ...]`
- ZMQ 将 memoryview 直接作为 Frame 发送，不额外拷贝
- `MsgpackDecoder._decode_tensor()`：`torch.frombuffer(buffer)` → 零拷贝视图重建 tensor

### 压缩切入点选择

| 层级 | 优点 | 缺点 |
|------|------|------|
| **A. 序列化层（按行压缩，选用）** | 一处改动覆盖所有 ZMQ 传输；与 SU 按行存储模型天然对齐；压缩与零拷贝 frame 机制并存 | 牺牲了整批共享 dict 的压缩比（可后续用预训练 dict 弥补） |
| B. 序列化层（整批压缩） | 压缩比最佳 | 整批 opaque blob 无法在 SU 按 `global_index` 切分；与 SU 存储模型冲突 |
| C. ZMQMessage 层 | 简单 | 元数据也被压缩（浪费）；无法区分大小 tensor |
| D. SU 内部 PUT-时压缩 | Manager 侧零代价 | SU 反而增加 CPU；PUT 网络无收益 |

**选择 A（按行压缩）**：在 `MsgpackEncoder._encode_regular_tensor()` 内部，将 batched tensor 沿 dim 0 拆为 N 行、每行独立压缩，emit 为 N 个 `Ext(6)` 各带一段 compressed buffer frame。SU 解码后得到 `list[CompressedTensor]`（长度 = N），交由 `zip(global_indexes, values)` 按行存储——存储路径完全复用现有逻辑。

---

## 性能分析

### 压缩 vs 零拷贝的定量对比

假设 10 Gbps 网络（实际带宽 ~1 GB/s），zstd level 3，**按行压缩比**（保守估计，未启用 dict 训练）：

| 单行大小 | 原始网络耗时 | 压缩比（行级估算）| 压缩 CPU 耗时 | 压缩后网络耗时 | **总耗时** | **节省** |
|---------|-------------|------------------|--------------|--------------|----------|---------|
| < 1 KB   | < 0.001 ms | — | — | — | < 0.001 ms | 跳过(min_bytes) |
| 4 KB     | 0.004 ms   | 1.5x | 0.01 ms | 0.003 ms | **0.013 ms** | 接近持平 |
| 64 KB    | 0.06 ms    | 2x   | 0.05 ms | 0.03 ms  | **0.08 ms**  | 与原网络耗时持平 |
| 1 MB     | 0.8 ms     | 2.5x | 0.5 ms  | 0.32 ms  | **0.82 ms**  | **PUT 持平，但 GET 摊销后净赚** |
| 16 MB    | 13 ms      | 3x   | 7 ms    | 4.3 ms   | **11.3 ms**  | 13% ↓ |

> **注**：以上是**单次 PUT** 的对比。本方案的核心收益不在单次 PUT，而是 **N 次 GET 摊销**——SU 一次也不做 zstd，每次 GET 节省单 GET 的全部压缩 CPU。下表给出摊销视角。

### 摊销视角：典型 PPO/GRPO 内 epoch 复读

设单 sample 行 GET 次数 = `R`（PPO inner epoch、GRPO group 多次抽样）：

| 场景 | 当前方案 | 本方案 | 节省 |
|---|---|---|---|
| Manager 端 PUT 压缩 CPU | 0 | C | -C |
| SU 端 GET 压缩 CPU | R × C_su | 0 | **+R × C_su** |
| 网络（PUT + R × GET）| (R+1) × T_raw | (R+1) × T_compressed | **+ (R+1) × ΔT** |
| 净收益（R ≥ 1） | — | — | 单调递增于 R |

其中 `C_su` 是 SU 一侧的压缩耗时（与 Manager 压缩耗时数量级一致，但 SU 实例数通常少、是潜在瓶颈），`ΔT` 是网络节省。**只要 R ≥ 1 且 Manager 数 ≫ SU 数，本方案净收益为正**。

### 为什么按行压缩仍然零拷贝

- 小行（per-row bytes < min_bytes）：跳过压缩，走现有零拷贝路径，**完全无开销**
- 大行：以行的 byte view（与 tensor storage 共享）作为 zstd 输入，无源端拷贝；`zstd.compress()` 输出的新 bytes 经 `memoryview` 包装为 ZMQ frame 同样零拷贝发送
- 元数据 pickle 不变：每个 Ext(6) 只 pickle `(dtype, row_shape, idx, algorithm, level)` 这个 ~80-byte tuple
- **数据本身不经过 pickle**：压缩后的 bytes 直接作为 ZMQ Frame 发送

### Frame 数量估算

| 维度 | 当前 | 本方案 | 评估 |
|---|---|---|---|
| 单 PUT 单 SU 的 frame 数 | F + 1 | N × F + 1 | 例：F=8, N=64 → 513 frames |
| 单 frame ZMQ 头开销 | 1-9 B | 1-9 B | N=64 时 +5 KB header |
| pickled meta 总开销 | F × 50 B | N × F × 80 B | N=64, F=8 → 41 KB |
| 占典型 5 MB 压缩 payload 比 | < 0.01% | **< 1%** | 可接受 |

> ⚠️ 与 `_select_by_positions` 中 case 3 注释提到的 "excessive multipart frames" 不冲突——后者是按**元素**切的极端情况（frame 数 ~ numel），本方案按行切（frame 数 ~ batch_size），量级完全不同。

---

## 详细设计

### 涉及文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `transfer_queue/utils/compression.py` | 新建 | `CompressedTensor` + `TensorCompressor` |
| `transfer_queue/utils/serial_utils.py` | 修改 | 新增 `Ext(6)` + 按行 encode/decode + `CompressedTensor` enc_hook 分支 |
| `transfer_queue/storage/managers/simple_storage_manager.py` | 修改 | `__init__` 末尾调用 `configure_serialization()` |
| `transfer_queue/storage/simple_storage.py` | **不改** | SU 透明 pass-through |
| `transfer_queue/config.yaml` | 修改 | 增加 compression 配置块 |
| `requirements.txt` | 修改 | 增加 zstandard 为 optional dep |
| `pyproject.toml` | 修改 | 增加 `[project.optional-dependencies]` 下的 `compression` 组 |
| `tests/test_compression.py` | 新建 | `TensorCompressor` 单元测试 |
| `tests/test_serial_utils_on_cpu.py` | 修改 | 增加按行压缩 tensor 的 roundtrip 测试 |
| `tests/e2e/test_e2e_lifecycle_consistency.py` | 修改 | 参数化加 `compression=zstd` 用例 |

---

### Step 1 — 新建 `transfer_queue/utils/compression.py`

**`CompressedTensor`**（frozen dataclass）：承载 batched tensor 的**单行**压缩态，SU 按 `global_index` 存储其实例。关键字段：

- `data: bytes` — 单行压缩字节流
- `dtype: str`、`shape: tuple` — 原 torch dtype 与**单行 shape**（不含 batch 维）
- `algorithm: str`、`level: int` — SU GET 重新打 Ext(6) 时无需查询 compressor 配置即可填回 meta

**`TensorCompressor`**：只读配置 + 算法分发对象，挂在 encoder/decoder 上。公开 API：

- `enabled` / `algorithm` / `level` / `min_bytes` 属性
- `should_compress_field(tensor) -> bool`：字段级决策。仅在 `enabled`、非 nested/sparse、dim ≥ 1、`per_row_bytes >= min_bytes` 时返回 True
- `compress_bytes(raw) -> bytes` / `decompress_bytes(compressed) -> bytes` — 字节流接口，避免 encoder 调用时构造临时 dataclass

zstandard 等后端 lazy-import；不启用时 import 不触发。

---

### Step 2 — 修改 `serial_utils.py`

新增 Ext 类型 `CUSTOM_TYPE_COMPRESSED_TENSOR = 6`，承载**单行**压缩态。其 meta tuple 结构为 `(dtype, row_shape, buffer_idx, algorithm, level)`。

#### MsgpackEncoder

- `__init__` 增加 `compressor: TensorCompressor | None` 参数
- `enc_hook` 增加 `CompressedTensor` 分支 → `_encode_compressed_tensor()`
- `_encode_regular_tensor()` 在 `compressor.should_compress_field(obj)` 命中时改走新增的 `_encode_compressed_rows()`，否则保持现有 `Ext(3)` 零拷贝路径
- `_encode_compressed_rows(tensor [N, ...]) -> list[Ext]`：沿 dim 0 逐行调用 `compressor.compress_bytes()`，每行 `memoryview(compressed)` 追加到 `aux_buffers`，产出一个 `Ext(6, meta_i)`；返回 N 元素 Python list，msgpack 自然序列化为 native array
- `_encode_compressed_tensor(ct) -> Ext`：仅 `memoryview(ct.data) → aux_buffers → Ext(6, meta)`。**SU GET 路径上的核心保证：零 zstd 调用**

SU 一侧 `enc_hook` 永远收到单个 `CompressedTensor`（来自 `dict[gi]` 的逐元素查询），由 msgspec 对 `list[CompressedTensor]` 的原生迭代分发，整个 GET 路径上 SU 不触发 `compress_bytes`。

#### MsgpackDecoder

- `__init__` 增加 `compressor` 参数
- `ext_hook` 增加 `Ext(6) → _decode_compressed_row(meta)`
- `_decode_compressed_row(meta)` 行为按是否有 compressor 分叉：
  - 无 compressor（SU 侧）：返回 `CompressedTensor(data=bytes(buffer), ...)`。`bytes()` 拷贝必要，因 ZMQ frame 在 message 处理后释放
  - 有 compressor（Manager 侧）：解压 → `torch.frombuffer(...).view(dtype).view(row_shape)` → 单行 tensor

#### 全局入口

`configure_serialization(compressor)` 一次性重建模块级 `_encoder` / `_decoder`。重复调用且配置不一致时 emit warning（V1 不支持单进程内多 backend 并存不同压缩配置）。`ContextVar` 隔离 `aux_buffers` 的机制不变；`compressor` 是 encoder/decoder 实例的只读属性，无竞争。

#### Wire 格式

| 字段类型 | Encoder 产出 | Decoder（无 compressor，SU）| Decoder（有 compressor，Manager）|
|---------|-------------|---------------------------|-------------------------------|
| 小 tensor / 不压缩 | `Ext(3, meta)` + 1 buffer frame | `torch.Tensor [N, ...]` | 同左 |
| 大 tensor / 压缩 | msgpack array `[Ext(6, meta_i)] × N` + N buffer frames | `list[CompressedTensor]` 长度 N | `list[torch.Tensor]` 长度 N |
| 非 tensor 字段 | 原有路径 | 原有路径 | 原有路径 |

向后兼容：未启用压缩或未命中 `should_compress_field()` 时不产生 `Ext(6)`，所有路径同今。

---

### Step 3 — 修改 `AsyncSimpleStorageManager.__init__()`

`__init__` 末尾按 `config.compression` / 环境变量决定压缩配置：

- `algorithm == "none"`（默认）：不调用 `configure_serialization()`，行为完全不变
- 否则构造 `TensorCompressor(algorithm, level, min_bytes)` 并 `configure_serialization()` 注册到全局 encoder/decoder
- 检测到 `data_parser` 与 compression 同时启用时 raise（详见风险 #4）

`put_data` / `get_data` / `_put_to_single_storage_unit` / `_get_from_single_storage_unit` 完全不变。压缩由全局 encoder/decoder 透明执行。

---

### Step 4 — SimpleStorageUnit 零改动验证

`SimpleStorageUnit` 进程是独立的 Ray actor，**不继承 manager 进程的全局 encoder/decoder 配置**。其全局 `_encoder` / `_decoder` 始终保持 `compressor=None`。

**SU 作为接收方（Put 路径）：**

```
ZMQMessage.deserialize() → _decoder.decode(frames):
  字段类型分发：
    - msgpack array of Ext(6) (压缩字段):
        每个 Ext(6) → _decode_compressed_row(meta, compressor=None):
          → 返回 CompressedTensor (单行)
        最终 field 值 = list[CompressedTensor] 长度 N
    - 单 Ext(3) (未压缩字段):
        → _decode_tensor(meta) → torch.Tensor [N, ...]
        （沿 dim 0 自然可迭代）

→ _handle_put 取 body["data"] → field_data
→ StorageUnitData.put_data(field_data, global_indexes):
    for f, values in field_data.items():
        for key, val in zip(global_indexes, values, strict=True):
            self.field_data[f][key] = val   # val 是 CompressedTensor 或 row tensor view
```

**SU 作为发送方（Get 路径）：**

```
StorageUnitData.get_data(fields, global_indexes):
  result[field] = [self.field_data[field][k] for k in global_indexes]
  返回 dict[field, list[CompressedTensor 或 row tensor]]

→ _handle_get → ZMQMessage.create(body={"data": result_data, ...})
→ ZMQMessage.serialize() → encode():
    msgspec 遍历 result_data 中的每个 list:
      - list[CompressedTensor]:
          每个 CompressedTensor → enc_hook → _encode_compressed_tensor():
            memoryview(ct.data) → aux_buffers → Ext(6)
          msgpack array of Ext(6) (零 zstd 调用！)
      - list[torch.Tensor (row view)]:
          每个 row → enc_hook → _encode_tensor() → 各 row 独立 Ext(3)
          msgpack array of Ext(3) (零拷贝)
→ ZMQ 发送
```

**关键不变量**：SU 的 `_worker_routine`、`_handle_put`、`_handle_get`、`StorageUnitData.put_data/get_data/clear` 完全不变。SU 仅把 `CompressedTensor` 视为普通 Python 对象逐 sample 存取。

---

### Step 5 — 配置扩展

#### config.yaml

```yaml
backend:
  SimpleStorage:
    total_storage_size: 100000
    num_data_storage_units: 2
    zmq_info: null
    # Tensor compression config (SimpleStorage only)
    compression:
      algorithm: none       # none | zstd (extensible: lz4 in future)
      level: 3              # zstd compression level (1-22)
      min_bytes: 1024       # skip rows smaller than this (per-row bytes)
```

#### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TQ_COMPRESSION_ALGORITHM` | `"none"` | 压缩算法：`none` / `zstd` |
| `TQ_COMPRESSION_LEVEL` | `3` | 压缩级别（zstd: 1-22） |
| `TQ_COMPRESSION_MIN_BYTES` | `1024` | 跳过单行小于此值的字段 |

Env 变量在 config.yaml 未指定对应字段时作为 fallback。

#### requirements.txt

```
# Optional: tensor compression
# zstandard>=0.22
```

#### pyproject.toml

```toml
[project.optional-dependencies]
compression = [
    "zstandard>=0.22",
]
```

---

### Step 6 — 测试计划

#### tests/test_compression.py（新建）

| 测试用例 | 覆盖点 |
|----------|--------|
| `test_compressor_disabled` | algorithm="none" → enabled=False, should_compress_field always False |
| `test_should_compress_field_per_row_threshold` | 单行 nbytes 跨 min_bytes 阈值的边界判定 |
| `test_should_compress_field_skip_nested_sparse_0d` | nested / sparse / 0-dim tensor 一律跳过 |
| `test_zstd_compress_decompress_float32` | float32 row roundtrip |
| `test_zstd_compress_decompress_float16/int64/int8/bool` | 各 dtype roundtrip |
| `test_zstd_compress_decompress_high_dim_row` | 行 shape 多维（如 [seqlen, hidden]）|
| `test_compressed_tensor_dataclass_fields` | CompressedTensor 字段正确性、frozen 性 |

#### tests/test_serial_utils_on_cpu.py（追加）

| 测试用例 | 覆盖点 |
|----------|--------|
| `test_compressed_field_enc_dec_roundtrip` | Encoder(zstd) + Decoder(zstd) → batched tensor 端到端 roundtrip |
| `test_compressed_field_su_passthrough` | Encoder(zstd) → Decoder(None) [SU PUT] → list[CompressedTensor] → Encoder(None) [SU GET] → Decoder(zstd) → batched tensor，验证全程 SU 不 zstd |
| `test_mixed_compressed_uncompressed_dict` | 同 dict 中混合大小 tensor 字段，分别走 Ext(6) 和 Ext(3) |
| `test_compressed_field_with_n_equals_1` | batch_size=1 边界 |
| `test_compressed_field_preserves_dtype_shape` | 各 dtype/shape 经压缩往返一致 |
| `test_compressed_field_special_values` | inf / nan / 0 tensor 经压缩往返 |
| `test_compressed_empty_field_skips` | 空 batch 不进入压缩路径 |
| `test_below_min_bytes_skips_compression` | 单行小于 min_bytes 时落到 Ext(3) |
| `test_configure_serialization_warns_on_conflicting_reconfig` | 重复调用不同配置时 warning |

#### tests/e2e/test_e2e_lifecycle_consistency.py（追加参数化）

| 测试用例 | 覆盖点 |
|----------|--------|
| `test_e2e_lifecycle_consistency[compression=zstd]` | 在端到端 lifecycle 测试中开启压缩，验证 PUT → GET → clear 全流程数据一致性，并断言 SU 进程未发生 zstd 调用（通过 patch / counter 验证）|

---

## 完整数据流总结

```
PUT (Manager → SU):
  ============ Manager 侧 (Encoder has compressor=zstd) ============
  TensorDict → ZMQMessage.serialize → encode → enc_hook:
    torch.Tensor [N, ...] (per-row ≥ min_bytes):
      逐行 byte view (ZERO COPY) → zstd.compress (CPU) → memoryview(compressed)
        × N → msgpack array of N × Ext(6) + N 个 compressed buffer frames
    torch.Tensor [N, ...] (per-row < min_bytes):
      整批 byte view (ZERO COPY) → Ext(3) + 1 buffer frame

  ============ ZMQ multipart ============
  [identity | msgpack_frame | row_0_frame | ... | row_{N-1}_frame | other_field_frames...]

  ============ SU 侧 (Decoder has compressor=None) ============
  ZMQMessage.deserialize → decode:
    msgpack array of N × Ext(6) → list[CompressedTensor] (长度 N)
    Ext(3) → torch.Tensor [N, ...]
  → _handle_put → StorageUnitData.put_data:
    按行写入 field_data[f][gi]（CompressedTensor 或 row tensor view）


GET (SU → Manager):
  ============ SU 侧 (Encoder has compressor=None) ============
  StorageUnitData.get_data → list[CompressedTensor] 或 list[row tensor]

  ZMQMessage.serialize → encode → msgspec iterates list:
    CompressedTensor → _encode_compressed_tensor → Ext(6)
      仅引用已压缩字节为 frame，**ZERO zstd CALLS ON SU**
    row torch.Tensor → _encode_tensor → Ext(3) (零拷贝)

  ============ ZMQ multipart ============
  (与 PUT 对称的 row-level frame layout)

  ============ Manager 侧 (Decoder has compressor=zstd) ============
  ZMQMessage.deserialize → decode:
    msgpack array of N × Ext(6) → zstd.decompress × N → list[torch.Tensor]
    msgpack array of Ext(3) → list of row tensors

  → AsyncSimpleStorageManager.get_data 装配 ordered_data
  → _pack_field_values → torch.stack → batched torch.Tensor [N, ...]
```

**SU 进程内 zstd 调用次数：恒为 0**。这是本设计的核心保证。

---

## 风险与注意事项

1. **zstd 依赖**：`zstandard` 是含 C 扩展的 Python 包，需用户手动安装 `pip install zstandard` 或 `pip install -e ".[compression]"`。未安装时不配置压缩即可正常运行。

2. **行级压缩比 vs 整批**：典型 RLHF 数据（fp32/int64）按行压缩比通常比整批低 10-30%。如果实测发现压缩比损失过大，V2 可引入**预训练 zstd dictionary**（首次 PUT 时采样训练，之后所有 compress/decompress 复用同一 dict），将行级压缩比拉回到接近整批水平。dict 大小通常 ~64 KB，随 manager 启动时下发到 SU，存储成本可忽略。

3. **multipart frame 数量增长**：从 F+1 增长到 N×F+1。在典型 RLHF 规模（N=64, F=8）下经测算 metadata + frame header 总开销 < 1%。但若 batch 极大（N > 1024）或字段极多（F > 50），需重新评估。建议在 `tests/e2e/` 中加入 N=512 的压力测试。

4. **`data_parser` 不兼容**：当前 `_handle_put` 中 `data_parser` 假设字段值是单一对象（tensor / NonTensorStack），无法处理 `list[CompressedTensor]`。V1 在 manager 初始化时检测 `data_parser` 与 `compression` 共启即报错。V2 可让 `data_parser` 在 SU 一侧操作 row tensor 之前接入（需要 SU 显式解压 → 调用 → 重压，违背"SU 零 zstd"目标，因此 V2 也未必采纳）。

5. **滚动升级不兼容**：未升级的 SU 收到 `Ext(6)` 会抛 `NotImplementedError`。启用压缩前必须保证所有 SU 进程已升级。建议在 `configure_serialization()` 中 emit 一条 INFO 日志记录启用版本号，便于运维确认一致。

6. **嵌套 tensor 不压缩**：`torch.nested` 类型的 tensor 跳过压缩（通过 `should_compress_field` 中的 `is_nested` 判断）。嵌套 tensor 的零拷贝序列化通过 unbind 子 tensor 实现，每个子 tensor 是独立 row，本可逐 row 压缩——V2 可扩展。V1 保持现状。

7. **GPU tensor 处理**：`_encode_regular_tensor()` 中 GPU tensor 先 `.cpu()` 再判断是否压缩。.cpu() 拷贝是已有开销，与压缩无关。

8. **与其他后端的关系**：Mooncake / Yuanrong / RayStore 后端不经过此压缩路径——它们的 `KVStorageManager` 直接调用 `storage_client.put/get`，绕过 `MsgpackEncoder` 的 tensor 编码路径。`compression` 配置块仅对 SimpleStorage 生效，其他 backend 配置文件不暴露此选项。

9. **多 backend 共存**：单进程只支持一组 compression 配置（全局 encoder/decoder 单例）。若未来扩展为多 backend 同进程，需将 compressor 从全局态迁移为 per-encoder ContextVar 或随 `ZMQMessage.serialize()` 调用栈传入。当前 V1 不解决。

10. **`bytes(buffer)` 拷贝开销**：SU 解码时为每行 CompressedTensor 调用一次 `bytes(buffer)`，从 ZMQ frame 拷出独占数据。对单 PUT 一个 SU 的 N 行字段，开销为 `N × per_row_compressed_bytes` 的 memcpy。这与现有未压缩路径中 `torch.frombuffer(buffer)` 的 view 开销不同——后者是零拷贝。但 SU 必须持有独立 bytes 副本（ZMQ frame 在 message 处理后释放），无法规避。这部分开销已纳入"摊销视角"中的 PUT 端 cost。
