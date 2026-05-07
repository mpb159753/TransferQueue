# 子计划 03: 修改 AsyncSimpleStorageManager

> 父计划: `00_overview.md`
> 前置: 子计划 02 (serial_utils.py)

## 目标

在 `AsyncSimpleStorageManager.__init__()` 末尾按配置注册压缩 encoder/decoder，让 PUT/GET 路径透明获得压缩能力。

## 实施步骤

### 3.1 导入

```python
from transfer_queue.utils.serial_utils import configure_serialization
from transfer_queue.utils.compression import TensorCompressor
```

### 3.2 `__init__` 末尾新增

```python
# === Tensor compression configuration ===
compression_cfg = config.get("compression", {})
algorithm = os.environ.get("TQ_COMPRESSION_ALGORITHM", compression_cfg.get("algorithm", "none"))
if algorithm != "none":
    level = int(os.environ.get("TQ_COMPRESSION_LEVEL", compression_cfg.get("level", 3)))
    min_bytes = int(os.environ.get("TQ_COMPRESSION_MIN_BYTES", compression_cfg.get("min_bytes", 1024)))

    compressor = TensorCompressor(
        algorithm=algorithm,
        level=level,
        min_bytes=min_bytes,
    )
    configure_serialization(compressor)

    logger.info(
        "Tensor compression enabled: algorithm=%s, level=%d, min_bytes=%d",
        algorithm, level, min_bytes,
    )
```

### 3.3 不兼容检测

若 `put_data` / `get_data` 的非 None `data_parser` 与 compression 同时启用，需要在 `__init__` 末尾或 `put_data` 入口检测并报错：

最简单的策略：在 compression 初始化之后加一段注释说明风险 #4（data_parser 不兼容），但 V1 的实现是在 `put_data` 入口检测 `data_parser is not None and compression enabled` → raise。不过更简洁的是在 `__init__` 处 logging.warning。

### 3.4 `_pack_field_values` 兼容性

当前 `_pack_field_values`（line 325-361）见 `simple_storage_manager.py:325`：
- 输入 `list[torch.Tensor]`（每行一个 tensor）→ `torch.stack()` 
- Manager GET 解码后产出的是 `list[torch.Tensor]`（单行），正好被此函数 stack 为 batched tensor → 无需修改

### 3.5 put_data / get_data 路径不变

- `put_data` → `_put_to_single_storage_unit` → `ZMQMessage.serialize()` → 全局 `_encoder.encode()` — 压缩透明执行
- `get_data` → ... → `ZMQMessage.deserialize()` → 全局 `_decoder.decode()` — 解压透明执行
- Manager 侧所有 ZMQ 调用经过的是同一组全局 encoder/decoder（会由 SU 返回的 CompressedTensor 在 Manager decoder 中解压）

## 验证标准

- 不启用压缩时现有所有测试通过，行为不变
- 启用压缩后 `pytest tests/test_client.py` 等 SimpleStorage 相关测试通过
