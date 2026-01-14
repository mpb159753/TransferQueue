# TQ Benchmark Package

TransferQueue 性能基准对比测试包。

## 目录结构

```
tq_benchmark_package/
├── run_benchmark.py      # 主运行脚本 (Docker)
├── run_profile.sh        # py-spy Profiling 脚本
├── run_profiling.py      # 容器内 Profiling 辅助
├── src_main/             # main 分支代码 (基线)
└── src_optimized/        # feature/optimize-serialization-v0.15 分支代码 (优化版)
```

## 版本信息

| 目录 | 分支 | 提交 |
|------|------|------|
| `src_main/` | main | 7f70edd |
| `src_optimized/` | feature/optimize-serialization-v0.15 | b8cfd5d |

**更新时间**: 2026-01-14

## 使用方法

### 1. 准备 Docker 镜像

```bash
# 构建测试镜像 (需要包含 torch, ray, tensordict 等依赖)
docker build -t run_test .
```

### 2. 运行基准测试

```bash
cd tq_benchmark_package

# 运行所有配置
python run_benchmark.py --cpus 30 --rounds 10

# 运行指定配置
python run_benchmark.py --filter_config small --rounds 10

# 定时运行
python run_benchmark.py --start_time 02:00 --cpus 30 --rounds 20
```

### 3. 双节点测试 (自动部署)

使用 `--worker-ip` 参数指定远程节点 IP，脚本将自动完成代码部署和 Worker 启动。

**前提条件**:
- 本机和远程节点已安装 Docker
- 本机可通过 SSH 免密登录远程节点 (默认使用 `~/.ssh/id_rsa`)
- 远程节点有权限运行 docker run

```bash
# 从 Head 节点发起测试 (自动部署到 Worker)
python run_benchmark.py \
  --worker-ip 192.168.1.101 \
  --cpus 30 \
  --rounds 20 \
  --ssh-user ubuntu \
  --deploy-path ~/tq_benchmark
```

### 4. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--filter_config` | - | 指定配置: debug/tiny/small/medium/large/xlarge/huge |
| `--rounds` | 20 | 每个配置的测试轮数 |
| `--cpus` | 30 | Docker 容器分配的 CPU 数量 |
| `--shards` | 8 | 存储分片数量 |
| `--output` | benchmark_summary.json | 结果输出文件 |
| `--worker-ip` | - | **[双节点]** 远程 Worker 节点 IP (启用双节点模式) |
| `--ssh-user` | 当前用户 | **[双节点]** SSH 用户名 |
| `--deploy-path` | ~/tq_benchmark | **[双节点]** 远程代码部署路径 |
| `--start_time` | - | 定时启动 (格式: HH:MM) |

## 测试配置

| 配置 | 数据量 |
|------|--------|
| debug | ~32KB |
| tiny | ~1MB |
| small | ~100MB |
| medium | ~1GB |
| large | ~5GB |
| xlarge | ~10GB |
| huge | ~20GB |

## 对比版本

- **optimized-v0.15**: 优化版序列化实现
- **main-no-zerocopy**: 主分支 (禁用 ZeroCopy)
- **main-zerocopy**: 主分支 (启用 ZeroCopy)
