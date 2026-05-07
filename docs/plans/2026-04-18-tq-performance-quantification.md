# TransferQueue Performance Quantification Plan

> For Codex: execute this plan task by task. Keep the plan at the design level; do not copy it verbatim into code.

**Goal:** Quantify where `tq_new/recipe/async_flow/utils/transfer_queue` loses performance relative to the main-repo `SimpleStorage` implementation, and attribute the gap to specific control-plane, serialization, storage, and concurrency costs.

**Architecture:** Use a three-layer methodology: fair end-to-end baselines, stage-level tracing, and isolated microbench/ablation experiments. Reuse the existing benchmark entrypoints where possible, then add a thin comparison harness and common JSON outputs so the two implementations can be compared under the same workload matrix.

**Tech Stack:** Python, Ray, ZMQ, PyTorch, NumPy, VizTracer, repository-local benchmark scripts, `./venv/bin/python`

---

## Background

The comparison target is:

- Main repo `SimpleStorage`
  - `transfer_queue/storage/simple_backend.py`
  - `transfer_queue/storage/managers/simple_backend_manager.py`
  - `transfer_queue/controller.py`
  - `transfer_queue/client.py`
  - `transfer_queue/metadata.py`
  - `transfer_queue/utils/zmq_utils.py`
  - `transfer_queue/utils/serial_utils.py`

- `tq_new`
  - `tq_new/recipe/async_flow/utils/transfer_queue/tq_mgr.py`
  - `tq_new/recipe/async_flow/utils/transfer_queue/tq_data.py`
  - `tq_new/recipe/async_flow/utils/transfer_queue/tq_client.py`
  - `tq_new/recipe/async_flow/utils/transfer_queue/tq_structures.py`
  - `tq_new/recipe/async_flow/utils/transfer_queue/tq_utils.py`

Current static analysis suggests the largest likely gap contributors are:

- extra control-plane RPC and bookkeeping in `tq_new`
- heavier Python `set/dict/sort/group` paths in `tq_new` manager sampling/status logic
- extra flatten/concat/copy work before `tq_new` network writes
- `pickle` use in `tq_new` shard request/response headers and non-tensor paths
- lock-held slab reconstruction in `tq_new` shard table reads

Current reusable benchmark/profiling entrypoints are:

- `tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py`
- `tq_new/recipe/async_flow/utils/transfer_queue/1.md`
- `scripts/performance_test.py`

The resulting work should create a repeatable workflow, not a one-off measurement.

## Success Criteria

- We can reproduce a fair end-to-end performance comparison between `tq_new` and `SimpleStorage`.
- We can measure `put`, `get`, and round-trip performance under the same workload matrix.
- We can attribute the gap to a small number of named contributors with numeric deltas.
- All benchmark outputs are written in machine-readable JSON so later sessions can continue analysis without re-reading raw logs.

## Measurement Model

Use three layers of evidence:

1. End-to-end benchmarks
   - Compare total `put` throughput, `get` throughput, and per-round latency.
2. Stage breakdown
   - Measure time spent in index allocation, routing, serialization, socket send/recv, status update, storage put/get, and aggregation.
3. Isolated microbench/ablation
   - Disable or isolate one suspected cost at a time to measure local and end-to-end deltas.

## Canonical Metrics

All benchmark and microbench outputs should use a stable JSON schema with at least:

- `implementation`
- `workload_name`
- `data_mode`
- `tensor_only`
- `shared_columns_enabled`
- `uuid_mode_enabled`
- `batch_size`
- `seq_length`
- `field_count`
- `non_tensor_field_count`
- `num_shards_or_storage_units`
- `num_clients`
- `rounds`
- `total_bytes`
- `put_seconds`
- `get_seconds`
- `round_trip_seconds`
- `put_gbps`
- `get_gbps`
- `round_trip_gbps`
- `latency_p50_ms`
- `latency_p95_ms`
- `client_cpu_seconds`
- `manager_or_controller_cpu_seconds`
- `shard_or_storage_cpu_seconds`
- `rss_peak_mb`
- `rpc_count_total`
- `rpc_count_by_stage`
- `notes`

For trace-derived outputs, add:

- `stage_breakdown_ms`
- `stage_breakdown_pct`
- `trace_source`

## Workload Matrix

Start from the smallest matrix that still separates root causes.

### Baseline Family: Common-Core

Purpose: compare only shared semantics.

- `n_samples_per_prompt=1`
- tensor-only
- no shared columns
- no UUID allocation
- no version filtering
- one partition/topic

Suggested sizes:

- `debug`
- `tiny`
- `small`

Suggested fan-out:

- `1`, `4`, `8` shards/storage units
- `1`, `4` clients

### Mixed Family: Non-Tensor Overhead

Purpose: quantify non-tensor serialization and `pickle` cost.

- tensor + non-tensor mixed
- same total bytes as Common-Core where possible

Suggested sweep:

- non-tensor field ratio `0%`, `25%`, `50%`

### Group Family: Feature Tax

Purpose: quantify `tq_new` semantic overhead that the main path does not need.

- `n_samples_per_prompt > 1`
- shared columns on
- UUID/group allocation path on
- optional version recording on

Suggested sweep:

- `n_samples_per_prompt = 4, 8`
- shared-column count `0`, `1`, `2`

## Output Conventions

Use these output locations unless a later session finds a better existing convention:

- Benchmark JSON:
  - `docs/perf/results/tq_new/*.json`
  - `docs/perf/results/simple_storage/*.json`
- Trace files:
  - `docs/perf/traces/tq_new/...`
  - `docs/perf/traces/simple_storage/...`
- Summary tables and attribution report:
  - `docs/perf/reports/tq_vs_simple_storage_summary.md`
  - `docs/perf/reports/tq_vs_simple_storage_attribution.json`

## Task Plan

### Task 1: Build a Fair Comparison Harness

**Files:**
- Modify: `tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py`
- Modify: `scripts/performance_test.py`
- Create: `docs/perf/results/.gitkeep`
- Create: `docs/perf/traces/.gitkeep`
- Create: `docs/perf/reports/.gitkeep`
- Optional Create: `scripts/compare_tq_backends.py`

**Design Points:**
- Do not compare the two implementations with different data semantics by default.
- Add a "common-core" mode that forces the two paths into the closest possible shared behavior.
- Ensure both benchmark entrypoints can emit the canonical JSON schema.
- Keep existing benchmark behavior backward compatible.

**Interface:**

```python
def run_comparison_suite(
    *,
    implementation: str,
    workload_name: str,
    output_path: str,
    common_core: bool = True,
) -> dict[str, Any]:
    ...
```

**Test Scenarios:**
- Happy path: one small run produces valid JSON.
- Edge case: common-core mode suppresses `tq_new` shared/UUID/version extras.
- Regression guard: existing benchmark CLI still works.

**Verification:**
- Run: `./venv/bin/python tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py --config debug --mode tensor --rounds 1 --output /tmp/tq_new_debug.json`
- Run: `./venv/bin/python scripts/performance_test.py tq-normal`

### Task 2: Add Comparable Stage-Level Tracing

**Files:**
- Modify: `tq_new/recipe/async_flow/utils/transfer_queue/tq_client.py`
- Modify: `tq_new/recipe/async_flow/utils/transfer_queue/tq_mgr.py`
- Modify: `tq_new/recipe/async_flow/utils/transfer_queue/tq_data.py`
- Modify: `transfer_queue/client.py`
- Modify: `transfer_queue/storage/managers/simple_backend_manager.py`
- Modify: `transfer_queue/storage/simple_backend.py`
- Modify: `transfer_queue/controller.py`
- Optional Create: `transfer_queue/utils/trace_markers.py`

**Design Points:**
- Use the same stage names across both implementations whenever possible.
- Prefer long-lived lightweight markers over ad hoc logging spam.
- Trace output must be usable both for timeline inspection and structured summary extraction.

**Stage Names to Standardize:**
- `allocate_indexes`
- `route_targets`
- `serialize_request`
- `send_request`
- `storage_put`
- `status_update`
- `fetch_targets`
- `storage_get`
- `deserialize_response`
- `aggregate_results`

**Test Scenarios:**
- Happy path: a profiled round emits a trace file.
- Edge case: profiling disabled remains near-no-op.
- Consistency: same stage names appear in both traces.

**Verification:**
- Run: `./venv/bin/python tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py --config debug --rounds 1 --profile --profile-output-dir /tmp/tq_new_trace`
- Run the equivalent profiled SimpleStorage benchmark command added in this task.

### Task 3: Add Isolated Microbenchmarks

**Files:**
- Create: `tests/perf/test_tq_microbench.py`
- Optional Create: `scripts/perf_microbench.py`

**Design Points:**
- Isolate local hot paths from Ray/network noise.
- Cover serialization, storage write, storage read, manager/controller selection logic, and client aggregation separately.
- Emit both wall time and bytes processed so results can be normalized.

**Microbench Targets:**
- `tq_utils.serialize_batch` vs `transfer_queue.utils.serial_utils.encode`
- `ExperienceTable.put_batch/get_batch` vs `StorageUnitData.put_data/get_data`
- `tq_mgr` ready/usable selection vs `controller.scan_data_status`
- `tq_client` result aggregation vs `SimpleStorageManager.get_data` post-processing

**Test Scenarios:**
- Happy path: each microbench runs locally with `./venv/bin/python`.
- Edge case: tensor-only and mixed-object variants both work.
- Stability: repeated runs show low enough variance for comparison.

**Verification:**
- Run: `./venv/bin/python scripts/perf_microbench.py --output /tmp/tq_microbench.json`
- Run: `./venv/bin/python -m pytest tests/perf/test_tq_microbench.py -q`

### Task 4: Add Ablation Switches

**Files:**
- Modify: `tq_new/recipe/async_flow/utils/transfer_queue/tq_client.py`
- Modify: `tq_new/recipe/async_flow/utils/transfer_queue/tq_mgr.py`
- Optional Modify: `tq_new/recipe/async_flow/utils/transfer_queue/tq_config.py`

**Design Points:**
- Each switch should disable exactly one suspected overhead source.
- Switches should be easy to set by env var or CLI flag.
- Ablations must be clearly labeled in JSON outputs.

**Initial Ablations:**
- disable shared-column written checks
- disable shared-column mark writes
- disable version recording
- bypass UUID allocation
- pre-allocate indexes up front
- force tensor-only raw path

**Test Scenarios:**
- Happy path: each ablation changes results but still completes.
- Safety: disabling a feature in common-core mode does not change semantics.
- Observability: output clearly records active ablations.

**Verification:**
- Run one baseline and one ablated run and compare JSON fields.

### Task 5: Run the Comparison Matrix and Write the Attribution Report

**Files:**
- Create: `docs/perf/reports/tq_vs_simple_storage_summary.md`
- Create: `docs/perf/reports/tq_vs_simple_storage_attribution.json`
- Optional Create: `scripts/summarize_perf_results.py`

**Design Points:**
- The report must separate evidence by layer: end-to-end, stage-level, microbench, ablation.
- Every claimed bottleneck needs a number and a source file or experiment.
- Prefer a ranked attribution list over a changelog-style dump.

**Report Shape:**
- workload summary table
- top contributors ranked by impact
- confidence level per contributor
- open questions and next experiments

**Test Scenarios:**
- Happy path: report can be rebuilt from saved JSON and traces.
- Edge case: incomplete runs still produce partial summaries.

**Verification:**
- Run the summary script over previously generated JSON.

## Recommended Execution Order

1. Build fair comparison harness and JSON output.
2. Add tracing markers and confirm trace generation.
3. Add isolated microbench script.
4. Add ablation switches.
5. Run baseline workloads.
6. Run ablations.
7. Write attribution report.

## Prompt Library

The prompts below are intentionally self-contained so they can be pasted into a fresh Codex session with minimal context loss.

### Prompt 1: Build the Comparison Harness

```text
仓库路径是 `/Users/mpb/WorkSpace/TransferQueue`。

背景：
- 我们要量化比较两个实现的性能差异：
  1. 主仓库 `SimpleStorage`，重点文件：
     - `transfer_queue/storage/simple_backend.py`
     - `transfer_queue/storage/managers/simple_backend_manager.py`
     - `transfer_queue/controller.py`
     - `transfer_queue/client.py`
     - `transfer_queue/metadata.py`
     - `transfer_queue/utils/zmq_utils.py`
     - `transfer_queue/utils/serial_utils.py`
  2. `tq_new`，重点文件：
     - `tq_new/recipe/async_flow/utils/transfer_queue/tq_mgr.py`
     - `tq_new/recipe/async_flow/utils/transfer_queue/tq_data.py`
     - `tq_new/recipe/async_flow/utils/transfer_queue/tq_client.py`
     - `tq_new/recipe/async_flow/utils/transfer_queue/tq_structures.py`
     - `tq_new/recipe/async_flow/utils/transfer_queue/tq_utils.py`
- 当前判断 `tq_new` 性能较差的可疑点包括：
  - 更多 control-plane RPC
  - manager 中大量 Python `set/dict/sort/group` 路径
  - 写前 flatten/concat/copy 成本
  - shard request/response 中的 `pickle`
  - shard 内 `ExperienceTable` 的锁与 slab 重组成本
- 仓库里已有 benchmark 入口：
  - `tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py`
  - `scripts/performance_test.py`

目标：
- 在不破坏现有 benchmark 用法的前提下，建立一个公平的 comparison harness。
- 增加一个 “common-core” 比较模式，让两边尽量只比较共有语义：
  - `n_samples_per_prompt=1`
  - tensor-only
  - 不启用 shared columns
  - 不启用 UUID/group/version 相关额外能力
- 两边 benchmark 输出统一 JSON schema，至少包含：
  - implementation
  - workload_name
  - batch_size
  - seq_length
  - field_count
  - non_tensor_field_count
  - num_shards_or_storage_units
  - num_clients
  - rounds
  - total_bytes
  - put_seconds
  - get_seconds
  - round_trip_seconds
  - put_gbps
  - get_gbps
  - round_trip_gbps
  - notes
- 输出目录建议：
  - `docs/perf/results/tq_new/`
  - `docs/perf/results/simple_storage/`

要求：
- 先阅读相关 benchmark 代码，再修改。
- 用 `apply_patch` 改文件。
- 不要删除或回滚当前工作区里与本任务无关的改动。
- 优先复用现有 benchmark 逻辑，不要重写整套。
- 最后运行最小验证命令，至少验证每边各有一个小规模 JSON 输出。

交付：
- 直接完成代码修改。
- 在最终回复里说明改了哪些文件、如何运行、以及最小验证结果。
```

### Prompt 2: Add Stage-Level Tracing to Both Implementations

```text
仓库路径是 `/Users/mpb/WorkSpace/TransferQueue`。

背景：
- 我们正在量化比较主仓库 `SimpleStorage` 和 `tq_new` 的性能差异。
- 重点实现位置：
  - 主仓库：
    - `transfer_queue/client.py`
    - `transfer_queue/storage/managers/simple_backend_manager.py`
    - `transfer_queue/storage/simple_backend.py`
    - `transfer_queue/controller.py`
  - `tq_new`：
    - `tq_new/recipe/async_flow/utils/transfer_queue/tq_client.py`
    - `tq_new/recipe/async_flow/utils/transfer_queue/tq_mgr.py`
    - `tq_new/recipe/async_flow/utils/transfer_queue/tq_data.py`
- `tq_new` 已经有 VizTracer 相关说明：
  - `tq_new/recipe/async_flow/utils/transfer_queue/1.md`
- 我们需要两边都有可比的 stage-level tracing，而不是只有总吞吐。

目标：
- 给两边都加轻量 tracing/marker，使得一次 put/get benchmark 可以拆成统一阶段：
  - `allocate_indexes`
  - `route_targets`
  - `serialize_request`
  - `send_request`
  - `storage_put`
  - `status_update`
  - `fetch_targets`
  - `storage_get`
  - `deserialize_response`
  - `aggregate_results`
- 尽量复用现有 `VizTracerProfileSession` / `TraceMarker` 风格；如果主仓库没有对应工具，可以补一个轻量公共工具，但要尽量小。
- profiling 关闭时要接近 no-op。
- profiling 打开时，能稳定产出 trace 文件，供后续做 breakdown。

要求：
- 先阅读相关 tracing 和 benchmark 代码。
- 用 `apply_patch` 改文件。
- 不要删除或回滚当前工作区里与本任务无关的改动。
- 如有必要，可新增一个轻量 trace helper 文件，但不要做大规模重构。

验证：
- 至少跑一个小规模 `tq_new` profile。
- 至少跑一个小规模主仓库 `SimpleStorage` profile。
- 确认 trace 文件存在，且能看见统一 stage 名称。

交付：
- 直接完成修改和最小验证。
- 最终回复里列出修改文件、profile 命令、trace 输出路径。
```

### Prompt 3: Add Isolated Microbenchmarks

```text
仓库路径是 `/Users/mpb/WorkSpace/TransferQueue`。

背景：
- 我们已经有一个性能比较课题：量化 `tq_new` 相对于主仓库 `SimpleStorage` 的性能差异来源。
- 可疑热点包括：
  - `tq_new` 的 `serialize_batch` / `torch_to_numpy`
  - `tq_new` 的 `ExperienceTable.put_batch/get_batch`
  - `tq_new` manager 里 `set/dict/sort/group` 的 ready/usable 选择逻辑
  - `tq_new` client 聚合 shard 返回结果的逻辑
  - 主仓库对应热点：
    - `transfer_queue/utils/serial_utils.py`
    - `transfer_queue/storage/simple_backend.py`
    - `transfer_queue/controller.py`
    - `transfer_queue/storage/managers/simple_backend_manager.py`

目标：
- 增加一套 isolated microbench，不依赖完整 Ray/ZMQ 分布式链路，也尽量减少网络噪声。
- 至少覆盖下面几类对比：
  1. `tq_new` `serialize_batch` vs 主仓库 `serial_utils.encode`
  2. `ExperienceTable.put_batch/get_batch` vs `StorageUnitData.put_data/get_data`
  3. `tq_mgr` 的 ready/usable 选择逻辑 vs `controller.scan_data_status`
  4. `tq_client` shard 结果聚合 vs 主仓库 get 后处理
- 输出 machine-readable JSON，建议路径：
  - `docs/perf/results/microbench/`

要求：
- 新增一个可直接运行的脚本，例如 `scripts/perf_microbench.py`。
- 如有必要，补一个最小 pytest 测试文件来保护核心 microbench helper。
- 用 `apply_patch` 修改。
- 不要删除或回滚当前工作区里与本任务无关的改动。
- 优先让脚本能在当前仓库和 `./venv/bin/python` 下直接运行。

验证：
- 运行 microbench 脚本，得到 JSON 输出。
- 如新增 pytest，跑对应最小测试。

交付：
- 直接完成实现。
- 最终回复给出脚本路径、运行命令、输出路径和关键结果摘要。
```

### Prompt 4: Add Ablation Switches for `tq_new`

```text
仓库路径是 `/Users/mpb/WorkSpace/TransferQueue`。

背景：
- 我们要量化 `tq_new` 相对主仓库 `SimpleStorage` 的性能差异来源。
- 当前怀疑 `tq_new` 的性能损失来自几个可独立关闭的点：
  - shared-column written checks
  - shared-column mark writes
  - version recording
  - UUID allocation path
  - 动态分配 index / group 的路径
  - 非 tensor / pickle 路径
- 重点文件：
  - `tq_new/recipe/async_flow/utils/transfer_queue/tq_client.py`
  - `tq_new/recipe/async_flow/utils/transfer_queue/tq_mgr.py`
  - `tq_new/recipe/async_flow/utils/transfer_queue/tq_config.py`

目标：
- 给 `tq_new` 增加一组可控的 ablation 开关，方便跑 “只关闭一个点” 的实验。
- 开关可以用 env var、CLI flag 或配置项，但要简单可发现。
- 每次 benchmark 输出必须记录当前开启了哪些 ablation。
- 推荐优先支持：
  - disable shared-column written checks
  - disable shared-column mark writes
  - disable version recording
  - bypass UUID allocation
  - pre-allocate indexes
  - force tensor-only raw path

要求：
- 先阅读现有 `tq_new` 路径与 benchmark 代码。
- 用 `apply_patch` 修改。
- 不要删除或回滚当前工作区里与本任务无关的改动。
- 默认行为必须保持兼容。

验证：
- 跑一个 baseline 和一个 ablation 小规模 benchmark。
- 确认输出里能看见 ablation 配置，且程序正常完成。

交付：
- 直接完成实现与验证。
- 最终回复里说明每个开关的入口、影响范围和运行方式。
```

### Prompt 5: Run the Full Comparison and Write the Attribution Report

```text
仓库路径是 `/Users/mpb/WorkSpace/TransferQueue`。

背景：
- 当前课题是量化比较：
  1. 主仓库 `SimpleStorage`
  2. `tq_new/recipe/async_flow/utils/transfer_queue`
- 比较目标不是只看总吞吐，而是要做 attribution，给出“性能差距由哪些点引入、每个点大约贡献多少”。
- 理想情况下，仓库里已经具备：
  - 公平 comparison harness
  - 统一 JSON benchmark 输出
  - stage-level traces
  - isolated microbench
  - `tq_new` ablation 开关
- 如果有缺失，请先检查并在不大改架构的前提下补齐最小缺口，再继续实验。

目标：
- 跑一轮完整的性能量化实验，并生成报告：
  - end-to-end baseline
  - stage breakdown
  - microbench
  - ablation
- 结果输出建议：
  - 原始结果：`docs/perf/results/...`
  - traces：`docs/perf/traces/...`
  - 总结报告：`docs/perf/reports/tq_vs_simple_storage_summary.md`
  - attribution JSON：`docs/perf/reports/tq_vs_simple_storage_attribution.json`

建议实验顺序：
1. Common-core baseline
   - `n_samples_per_prompt=1`
   - tensor-only
   - no shared columns
   - no UUID/version extras
2. Mixed workload
   - tensor + non-tensor
3. Group/shared workload
   - shared columns + group path
4. `tq_new` ablations

建议至少覆盖：
- workload: `debug`, `tiny`, `small`
- shards/storage units: `1`, `4`, `8`
- clients: `1`, `4`

报告要求：
- 按贡献度排序列出 top bottlenecks
- 每个结论都要给证据来源：
  - baseline / trace / microbench / ablation
- 最终给出类似：
  - control-plane extra RPC: X%
  - Python set/dict sampling bookkeeping: Y%
  - write-side flatten/concat/copy: Z%
  - pickle / mixed-object path: W%
- 如果数据不足，明确写 open questions

要求：
- 优先复用已有脚本和输出，不要重复造轮子。
- 用 `./venv/bin/python` 运行 Python。
- 不要删除或回滚当前工作区里与本任务无关的改动。

交付：
- 直接完成实验与报告。
- 最终回复中给出：
  - 实际跑过的命令
  - 结果文件路径
  - attribution 结论摘要
  - 仍需继续验证的点
```

## Verification Commands For the Plan Itself

- Ensure the plan file exists:
  - `ls docs/plans`
- Optionally preview:
  - `sed -n '1,260p' docs/plans/2026-04-18-tq-performance-quantification.md`

## Notes For Future Sessions

- Prefer new sessions for Prompt 2 through Prompt 5 if the current thread context gets too large.
- Each prompt above includes enough background to be pasted into a fresh Codex session.
- If the workspace is still dirty, future sessions must avoid deleting unrelated untracked files such as existing local benchmark work.
