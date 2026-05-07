# 子计划 02: 修改 serial_utils.py

> 父计划: `00_overview.md`
> 前置: 子计划 01 (compression.py)

## 目标

扩展 `serial_utils.py`，新增：
- `CUSTOM_TYPE_COMPRESSED_TENSOR = 6` Ext 类型
- 按行压缩的 encode/decode 逻辑
- `configure_serialization(compressor)` 全局入口

## 实施步骤

### 2.1 新增常量与导入

```python
from transfer_queue.utils.compression import CompressedTensor, TensorCompressor

CUSTOM_TYPE_COMPRESSED_TENSOR = 6
```

### 2.2 MsgpackEncoder 修改

`__init__` 增加可选参数：

```python
def __init__(self, compressor: TensorCompressor | None = None):
    self.compressor = compressor
    self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
```

`enc_hook` 增加分支（插在最前，在 `isinstance(obj, torch.Tensor)` 之前）：

```python
if isinstance(obj, CompressedTensor):
    return self._encode_compressed_tensor(obj)
```

`_encode_regular_tensor` 开头增加压缩判断：

```python
def _encode_regular_tensor(self, obj):
    if self.compressor is not None and self.compressor.should_compress_field(obj):
        return self._encode_compressed_rows(obj)
    # ... 现有逻辑不变
```

新增 `_encode_compressed_rows`：

```python
def _encode_compressed_rows(self, tensor: torch.Tensor) -> list[msgpack.Ext]:
    results = []
    for row in tensor:  # dim 0 iteration
        if row.device.type != "cpu":
            row = row.cpu()
        if not row.is_contiguous():
            row = row.contiguous()
        arr = row.flatten().view(torch.uint8).numpy()
        raw = arr.tobytes()  # need bytes for zstd input
        compressed = self.compressor.compress_bytes(raw)
        buf = memoryview(compressed)
        idx = len(self.aux_buffers)
        self.aux_buffers.append(buf)
        dtype = str(row.dtype).removeprefix("torch.")
        meta = (dtype, tuple(row.shape), idx, self.compressor.algorithm, self.compressor.level)
        results.append(msgpack.Ext(CUSTOM_TYPE_COMPRESSED_TENSOR, pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)))
    return results
```

**⚠️ 注意**: `row` 是 dim 0 的 slice，`tensor[0]` 的 `flatten+view` 本质上和 tensor storage 共享底层——但 `view(torch.uint8)` 已经在 `_encode_regular_tensor` 中有 `.numpy()` 这一步。压缩时如果我们先 `.numpy().tobytes()` 这会拷贝。但我们需要压缩，拷贝一次是可以接受的（本来就是压缩的对象）。关键是 `arr = row.flatten().view(torch.uint8).numpy()` → `raw = arr.tobytes()` — 这个拷贝是必需的，因为 zstd 需要 `bytes` 输入。GPU tensor 需先迁移到 CPU（`row.cpu()`），否则 `numpy()` 无法调用。

新增 `_encode_compressed_tensor`（SU GET 路径所用）：

```python
def _encode_compressed_tensor(self, ct: CompressedTensor) -> msgpack.Ext:
    buf = memoryview(ct.data)  # 已压缩的 bytes，零拷贝引用
    idx = len(self.aux_buffers)
    self.aux_buffers.append(buf)
    meta = (ct.dtype, ct.shape, idx, ct.algorithm, ct.level)
    return msgpack.Ext(CUSTOM_TYPE_COMPRESSED_TENSOR, pickle.dumps(meta))
```

### 2.3 MsgpackDecoder 修改

`__init__` 增加可选参数：

```python
def __init__(self, compressor: TensorCompressor | None = None):
    self.compressor = compressor
    self.decoder = msgpack.Decoder(ext_hook=self.ext_hook)
```

`_reconstruct_special_types` 需要处理 `list[CompressedTensor]` 包装——现有逻辑已递归处理 list，但 `CompressedTensor` 不需要特殊处理，它本身已是最终值。实际上没有问题，因为 `CompressedTensor` 已经由 `ext_hook` 产出并放入 list，`_reconstruct_special_types` 只是递归遍历。

`ext_hook` 增加 `Ext(6)` 分支：

```python
if code == CUSTOM_TYPE_COMPRESSED_TENSOR:
    meta = pickle.loads(data)
    return self._decode_compressed_row(meta)
```

新增 `_decode_compressed_row`：

```python
def _decode_compressed_row(self, meta: tuple) -> CompressedTensor | torch.Tensor:
    dtype_str, shape, idx, algorithm, level = meta
    buffer = self.aux_buffers[idx]

    if self.compressor is None:
        # SU 侧: 不解压，返回 CompressedTensor
        return CompressedTensor(
            data=bytes(buffer),  # ZMQ frame 生命周期外必须拷贝
            dtype=dtype_str,
            shape=shape,
            algorithm=algorithm,
            level=level,
        )
    else:
        # Manager 侧: 解压 → tensor
        raw = self.compressor.decompress_bytes(bytes(buffer))
        torch_dtype = getattr(torch, dtype_str)
        arr = torch.frombuffer(raw, dtype=torch.uint8)
        return arr.view(torch_dtype).view(shape)
```

### 2.4 全局 encoder/decoder 重建

```python
_encoder = MsgpackEncoder()  # 默认，compressor=None
_decoder = MsgpackDecoder()

def configure_serialization(compressor: TensorCompressor | None):
    global _encoder, _decoder
    _encoder = MsgpackEncoder(compressor=compressor)
    _decoder = MsgpackDecoder(compressor=compressor)
```

### 2.5 Wire 格式验证

| 场景 | Encoder 产出 | Decoder(compressor=None) | Decoder(compressor=zstd) |
|------|-------------|--------------------------|-------|
| 小 tensor | Ext(3) + 1 frame | batched torch.Tensor | 同左 |
| 大 tensor 压缩 | `list[Ext(6)]×N` + N frames | `list[CompressedTensor]` | `list[torch.Tensor]` |
| CompressedTensor 单行 | Ext(6) + 1 frame | CompressedTensor | 同左（compressor=None 不走解压） |

## 验证标准

- `test_serial_utils_on_cpu.py` 中所有新增压缩 roundtrip 测试通过
- 现有 Ext(1-5) 测试全部通过（向后兼容）
- SU passthrough 路径验证：Encoder(zstd) → Decoder(None) → Encoder(None) → Decoder(zstd) ≈ 原始 tensor
