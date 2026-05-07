# 子计划 04: 配置与依赖扩展

> 父计划: `00_overview.md`
> 并行约束: 可与 01 并行，无代码依赖

## 目标

配置层面支持压缩功能：config.yaml、环境变量、requirements.txt、pyproject.toml。

## 实施步骤

### 4.1 config.yaml

在 `backend.SimpleStorage` 下新增 `compression` 配置块：

```yaml
  # For SimpleStorage:
  SimpleStorage:
    # ... 现有字段不变 ...

    # Tensor compression config (SimpleStorage only)
    compression:
      algorithm: none       # none | zstd (extensible: lz4 in future)
      level: 3              # zstd compression level (1-22)
      min_bytes: 1024       # skip rows smaller than this (per-row bytes)
```

### 4.2 requirements.txt

追加（注释掉，按需启用）：

```
# Optional: tensor compression
# zstandard>=0.22
```

### 4.3 pyproject.toml

在 `[project.optional-dependencies]` 下新增：

```toml
compression = [
    "zstandard>=0.22",
]
```

### 4.4 环境变量（文档级，代码在子计划 03 中处理）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TQ_COMPRESSION_ALGORITHM` | `"none"` | `none` / `zstd` |
| `TQ_COMPRESSION_LEVEL` | `3` | zstd level (1-22) |
| `TQ_COMPRESSION_MIN_BYTES` | `1024` | 跳过单行小于此值的字段 |

环境变量读取逻辑在子计划 03 的 `__init__` 中实现——与 config.yaml 合并，env var 优先。

## 验证标准

- config.yaml 合法的 YAML，可被 OmegaConf 解析
- `pip install -e ".[compression]"` 成功安装 zstandard
- 不改 config.yaml 时（algorithm=none 默认），行为完全不变
