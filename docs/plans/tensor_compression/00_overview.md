# Tensor 压缩传输 — 实施总览

> 设计文档: `docs/design/tensor_compression_design.md`

## 子计划依赖关系

```
01_compression_util.md ──┬──▶ 02_serial_utils.md ──▶ 03_storage_manager.md
                          │                                │
04_config_and_deps.md ────┴────────────────────────────────┤
                                                            │
                                           05_tests.md ◀───┘
```

- **01** (compression.py) 无依赖，可率先启动
- **04** (config/deps) 与 01 并行启动
- **02** (serial_utils.py) 依赖 01 提供的类型与 compressor
- **03** (simple_storage_manager.py) 依赖 02 提供的 `configure_serialization`
- **05** (tests) 依赖全部代码就绪后执行

## 涉及文件

| 文件 | 操作 | 子计划 |
|------|------|--------|
| `transfer_queue/utils/compression.py` | 新建 | 01 |
| `transfer_queue/utils/serial_utils.py` | 修改 | 02 |
| `transfer_queue/storage/managers/simple_storage_manager.py` | 修改 | 03 |
| `transfer_queue/config.yaml` | 修改 | 04 |
| `requirements.txt` | 修改 | 04 |
| `pyproject.toml` | 修改 | 04 |
| `tests/test_compression.py` | 新建 | 05 |
| `tests/test_serial_utils_on_cpu.py` | 修改 | 05 |
| `tests/e2e/test_e2e_lifecycle_consistency.py` | 修改 | 05 |

## 核心不变量

1. **SU 零 zstd 调用** — SimpleStorageUnit 在任何路径上不解压、不压缩
2. **向后兼容** — `algorithm=none`（默认）时行为完全不变
3. **零拷贝保留** — 未压缩 tensor 走现有 Ext(3) 零拷贝路径
4. **SU 不改代码** — `simple_storage.py` 无需任何修改

## 执行顺序建议

1. 并行: 子计划 01 + 子计划 04
2. 顺序: 子计划 02（依赖 01）
3. 顺序: 子计划 03（依赖 02）
4. 顺序: 子计划 05（依赖 01-04）
