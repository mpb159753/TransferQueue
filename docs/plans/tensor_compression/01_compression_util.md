# 子计划 01: 新建 compression.py

> 父计划: `00_overview.md`
> 并行约束: 可与 04 并行，无其他依赖

## 目标

新建 `transfer_queue/utils/compression.py`，提供：
- `CompressedTensor` — 单行压缩态的 frozen dataclass
- `TensorCompressor` — 压缩/解压调度器

## 实施步骤

### 1.1 CompressedTensor

Frozen dataclass，字段：

```
data: bytes       — 单行压缩字节流
dtype: str        — 原 torch dtype 字符串
shape: tuple      — 单行 shape（不含 batch 维）
algorithm: str    — 压缩算法名
level: int        — 压缩级别
```

- 使用 `@dataclass(frozen=True)` + `slots=True`
- 实现 `__hash__`（若 `frozen + eq=True` 自动生成，但需验证 `bytes` 字段的 hash 行为）

### 1.2 TensorCompressor

构造参数：`algorithm: str`, `level: int`, `min_bytes: int | None`

属性：
- `enabled -> bool`: `algorithm != "none"`
- `min_bytes -> int`: 给定值或默认 1024

公开方法：
- `should_compress_field(tensor: torch.Tensor) -> bool`:
  1. `not self.enabled` → False
  2. `tensor.is_nested` → False
  3. `tensor.is_sparse` → False
  4. `tensor.ndim < 1`（0-dim）→ False
  5. `per_row_bytes = tensor[0].nbytes`，若 `< self.min_bytes` → False
  6. 否则 True

- `compress_bytes(raw: bytes) -> bytes`: `zstd.compress(raw, self.level)`
- `decompress_bytes(compressed: bytes) -> bytes`: `zstd.decompress(compressed)`

**实现细节**：
- zstandard 使用 lazy import（仅在启用且调用时 import）
- `_get_backend()` 私有方法：检查 algorithm 是否支持，返回压缩/解压函数
- 若 algorithm 为未知值，在首次调用时 raise `ValueError`

### 1.3 文件模板

```python
# Copyright ... (标准 license header)

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CompressedTensor:
    data: bytes
    dtype: str
    shape: tuple
    algorithm: str
    level: int


class TensorCompressor:
    def __init__(self, algorithm: str = "none", level: int = 3, min_bytes: int | None = None):
        ...

    @property
    def enabled(self) -> bool:
        ...

    @property
    def min_bytes(self) -> int:
        ...

    def should_compress_field(self, tensor: torch.Tensor) -> bool:
        ...

    def compress_bytes(self, raw: bytes) -> bytes:
        ...

    def decompress_bytes(self, compressed: bytes) -> bytes:
        ...
```

## 验证标准

- `tests/test_compression.py`（见子计划 05）中的基础单元测试全部通过
- `algorithm="none"` 时 `enabled=False`，`should_compress_field` 恒返回 False
- `algorithm="zstd"` 时各 dtype roundtrip 一致
