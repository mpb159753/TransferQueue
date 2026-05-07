# 子计划 05: 测试

> 父计划: `00_overview.md`
> 前置: 子计划 01, 02, 03, 04 (全部代码就绪)

## 覆盖范围

### 5.1 `tests/test_compression.py`（新建）

**TensorCompressor 测试**：

| 测试 | 覆盖点 |
|------|--------|
| `test_compressor_disabled` | `TensorCompressor(algorithm="none")` → `enabled=False`, `should_compress_field` 始终 False |
| `test_should_compress_field_min_bytes` | 行 nbytes 跨 threshold 判定 |
| `test_should_compress_field_skip_nested` | `tensor.is_nested` → False |
| `test_should_compress_field_skip_sparse` | `tensor.is_sparse` → False |
| `test_should_compress_field_skip_0d` | 0-dim tensor → False |
| `test_zstd_compress_decompress_float32` | float32 single-row roundtrip |
| `test_zstd_compress_decompress_float16` / `int64` / `int8` / `bool` | 各 dtype roundtrip |
| `test_zstd_compress_decompress_high_dim_row` | 行 shape 多维（如 `[128, 768]`） |
| `test_zstd_level_range` | level=1..22 均可压缩/解压 |
| `test_compressed_tensor_fields` | CompressedTensor 字段正确性、frozen 不可变 |

### 5.2 `tests/test_serial_utils_on_cpu.py`（追加）

| 测试 | 覆盖点 |
|------|--------|
| `test_compressed_field_enc_dec_roundtrip` | Encoder(zstd) → Encoder.encode → Decoder(zstd).decode ≈ 原始 batched tensor |
| `test_compressed_field_su_passthrough` | Encoder(zstd) encode → Decoder(None) decode (SU PUT侧) → 取出 list[CompressedTensor] → Encoder(None) encode (SU GET侧) → Decoder(zstd) decode → **验证全程 SU 不调用 compress/decompress** |
| `test_mixed_compressed_uncompressed` | 同 dict 混合大小 tensor 字段，验证 Ext(6)+Ext(3) 共存 |
| `test_compressed_field_batch_size_1` | N=1 边界 |
| `test_compressed_field_dtype_shape_preserved` | 各 dtype/shape roundtrip 一致 |
| `test_compressed_special_values` | inf / nan / 0 tensor roundtrip |
| `test_compressed_empty_field_skips` | 空 batch 不进入压缩路径 |
| `test_below_min_bytes_skips` | 整批单行 < min_bytes → 落到 Ext(3) |
| `test_configure_serialization_warns_reconfig` | 重复调用不同配置 emit warning |

### 5.3 `tests/e2e/test_e2e_lifecycle_consistency.py`（追加参数化）

关键 e2e 检查：**SU 进程内 zstd 调用次数 = 0**。

实现思路：
1. 在 `TensorCompressor.compress_bytes` / `decompress_bytes` 上通过 `pytest-mock` 的 `spy` 或 `wraps` 记录 Manager 侧调用次数
2. 在 SimpleStorageUnit worker 线程内，monkey-patch 检查 zstd 函数是否被调用
3. 参数化 `test_e2e_lifecycle_consistency[compression=zstd]`：验证 PUT→GET→clear 全流程数据一致性

## 验证标准

```bash
# 不启用压缩（默认）
pytest tests

# 仅压缩单元测试
pytest tests/test_compression.py -v

# 序列化 roundtrip
pytest tests/test_serial_utils_on_cpu.py -v

# E2E（需要 Ray）
TQ_COMPRESSION_ALGORITHM=zstd pytest tests/e2e/test_e2e_lifecycle_consistency.py -v
```
