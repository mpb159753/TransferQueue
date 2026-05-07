# TQ New 字典存储后端改造方案

> For Codex: 按任务顺序执行这份方案。保持在设计层，不要把方案原文直接抄进代码。

**目标：** 将 `tq_new` 中 shard 侧 `ExperienceTable` 的 `zmq.Frame/ref_count` 存储后端改成类似 `transfer_queue/storage/simple_backend.py` 的 dict 存储模型，消除 `put_batch` 热路径里对 `zmq.Frame` 生命周期和大粒度锁的依赖。

**架构：** 保持现有 client/shard 通信协议不变，只重写 `ExperienceTable` 的 shard 本地内存存储：在锁外准备 Python 自主管理的 payload 条目，在锁内完成 dict 提交；`GET` 时在锁内只做条目快照，真正的响应拼装放到锁外。共享列按 `group_id` 存一份，非共享列按 `global_idx` 存一份。

**技术栈：** Python、Ray、ZMQ、NumPy、PyTorch、仓库本地 `./venv/bin/python`

---

## 背景

当前需要重点关注的热路径集中在：

- `tq_new/recipe/async_flow/utils/transfer_queue/tq_client.py`
- `tq_new/recipe/async_flow/utils/transfer_queue/tq_data.py`
- `tq_new/recipe/async_flow/utils/transfer_queue/tq_structures.py`

目前静态分析得到的结论如下：

- `ExperienceTable` 通过 `MemorySegment` 持有 `zmq.Frame`，并用手工 `ref_count` 管理引用和释放。
- `put_batch()` 虽然把一部分校验放在锁外，但创建 `MemorySegment`、写入索引映射、更新 shape、释放旧 segment 这些动作仍然在同一个全局锁里完成。
- `get_batch()` 也持有同一把锁，在锁内解析条目并把所有请求数据拷贝到新的响应 buffer 中。
- client 已经通过 `ref_multiplier` 和 `group_ids` 把共享列所需的信息带到了 shard，所以存储层完全可以切换成 `group_id` 键控的 dict 存储，而不用修改网络协议。
- 主仓 `transfer_queue/storage/simple_backend.py` 里的 `StorageUnitData` 已经给出了更接近目标的模型：用多层 dict 以逻辑 ID 为 key 持有 Python 自己管理的数据，不再依赖额外的引用计数生命周期。

## 核心设计决策

### 1. 保持 wire protocol 不变

不改这些内容：

- `tq_client.py` 里 `PUT` 请求 header 的结构
- `tq_client.py` 读取 `GET` 响应时依赖的 meta 格式
- `tq_data.py` / `tq_mgr.py` 中 manager 侧的数据状态更新和 group 归属流程

这样改造可以严格收敛在 shard 本地存储层，降低联动风险。

### 2. 用 dict 替代 frame-centric 存储

把当前 `MemorySegment + (segment, offset, length)` 的存储表示替换成 dict 结构。

建议的存储形态：

- 列级静态元信息：
  - `dtype`
  - `encoding`
  - `is_shared`
- 列级 payload 存储：
  - 非共享列：`column_entries[col][global_idx] = StoredBlob(...)`
  - 共享列：`column_entries[col][group_id] = StoredBlob(...)`

其中每个 `StoredBlob` 都应该持有 Python 自己管理的 payload，而不是活着的 `zmq.Frame`。

建议第一版采用的 payload 方案：

- raw 列：先把整块 frame 拷贝成 Python 自主管理的不可变 blob，再为每个 item 建立 `memoryview` 切片
- pickle 列：同样处理，只是 `lengths` 本身就是字节长度

这样虽然不是绝对零拷贝，但能显著降低实现复杂度，同时避免 shard 端做“完整反序列化成 tensor/object，再重新序列化回 bytes”的来回折腾。

### 3. 共享语义通过 storage key 表达，不再靠引用计数表达

对于共享列：

- 只按 `group_id` 存一份
- 读取时通过 `global_idx // n_samples_per_prompt` 推导存储 key
- 完全删除当前自定义 `release()` / `ref_count` 逻辑

对于非共享列：

- 仍然按 `global_idx` 一一存储

这样能保留共享列“同组只存一份”的语义，但不再需要手工维护底层 `zmq.Frame` 的生命周期。

### 4. 大幅缩短锁范围

目标锁模型如下：

- `put_batch()`
  - 锁外：校验 header、计算 storage key、把 frame 拷贝为 Python-owned blob、构造待写入的 prepared dict
  - 锁内：初始化列元信息，把 prepared 条目写进 dict
- `get_batch()`
  - 锁内：只解析需要返回的 `StoredBlob` 引用和 shape 快照
  - 锁外：真正拼接 payload bytes 并构造响应 buffer
- `prune()/clear()`
  - 只做 dict 删 key，不再显式遍历 segment 做 release

锁应该只负责保护表结构一致性，不再包住大量 payload 复制工作。

### 5. 必须保持现有语义不变

这次改造必须保住这些不变量：

- 列的 `dtype` 和 `encoding` 一旦第一次写入成功，后续必须保持一致
- 共享列依然允许 `PUT` 时只按 `group_id` 发送一份数据
- `GET` 返回的 `meta['columns'][col]` 字段保持不变：
  - `dtype`
  - `encoding`
  - `lengths`
  - `shapes`
- `prune()` 对共享列仍然能做到“删一个 prompt group 时整组失效”
- `owned_groups` 语义保持不变，因为 shard 归属仍然是 group 级别

## 任务拆解

### 任务 1：把 `ExperienceTable` 改成 dict 存储模型

**文件：**
- 修改：`tq_new/recipe/async_flow/utils/transfer_queue/tq_structures.py`
- 新增：`tq_new/recipe/async_flow/utils/transfer_queue/test_tq_structures.py`

**设计要点：**
- 删除 `MemorySegment` 以及当前 `(segment, offset, byte_length)` 的存储路径。
- 用更直接的 dict 结构替换 `self.indices` 和 `self.col_shapes`，让每个已存储条目本身就携带 payload slice、逻辑长度和 shape。
- 共享列按 `group_id` 存储，而不是像现在一样把同一个 entry 展开写到组内每个 index 上。
- `self.col_metas` 可以保留，但缩减成仅保存校验和响应拼装真正需要的元数据。
- 第一版优先选择“Python-owned immutable blob + memoryview 切片”，不要一开始就把 shard 存储层做成 tensor/object 级重建。

**接口：**

```python
@dataclass(slots=True)
class StoredBlob:
    payload: memoryview
    logical_length: int
    shape: tuple[int, ...] | None

def _storage_key(self, global_idx: int, *, is_shared: bool) -> int:
    ...

def _prepare_column_write(
    self,
    *,
    col_name: str,
    frame: zmq.Frame,
    col_info: dict[str, Any],
    global_ids: list[int],
) -> PreparedColumnWrite:
    ...
```

**测试场景：**
- 正常路径：非共享 raw tensor 列可以经过 `put_batch()` / `get_batch()` 正常往返。
- 正常路径：共享列只按 group 存一份，但组内每个 index 读取到的数据一致。
- 混合路径：同一个表里同时存在 raw 列和 pickle 列。
- 覆盖路径：重复写已有 index/group 时，新值能覆盖旧值，不产生脏读。
- 异常路径：`lengths`、`group_ids`、dtype/encoding 不匹配时仍然要报错。

**验证：**
- 运行：`./venv/bin/python -m pytest tq_new/recipe/async_flow/utils/transfer_queue/test_tq_structures.py -q`

### 任务 2：把 put/get/prune/clear 的临界区收缩到最小

**文件：**
- 修改：`tq_new/recipe/async_flow/utils/transfer_queue/tq_structures.py`
- 修改：`tq_new/recipe/async_flow/utils/transfer_queue/tq_data.py`

**设计要点：**
- 尽量不改 `ExperienceTable` 对外方法签名，保证 `tq_data.py` 的 shard RPC 层不需要协议调整。
- 把昂贵的 byte slicing / payload 拼接从 table 锁里挪出去。
- `get_batch()` 中，在锁内只拿 `StoredBlob` 引用快照；释放锁后再构造 `result_frames`。
- `prune()` 对共享列按 `group_id` 去重删除，避免同组多个 index 反复删同一条存储记录。
- `clear()` 直接清空 dict，交给 Python GC 回收，不再执行整表 release 遍历。
- 只有在 helper 重命名或共享列语义表达需要更清晰时，才调整 `tq_data.py`；避免修改 `PUT/GET` 请求结构。

**接口：**

```python
def get_batch(
    self,
    target_global_idxs: list[int],
    target_cols: list[str],
) -> tuple[dict[str, Any], list[bytes]]:
    ...

def prune(self, global_idxs_to_remove: list[int]) -> None:
    ...
```

**测试场景：**
- 正常路径：`get_batch()` 返回的 meta 格式与 `_fetch_one_shard()` 当前消费方式完全兼容。
- 边界场景：prune 一个完整 prompt group 时，共享数据只删除一次，但组内所有 index 后续都不可读。
- 边界场景：`clear()` 后再次写入，行为和第一次写入一致。
- 回归保护：manager 侧仍然把共享列视为 group 级 ready 数据，不受影响。

**验证：**
- 运行：`./venv/bin/python -m pytest tq_new/recipe/async_flow/utils/transfer_queue/test_transferqueue.py -q`

### 任务 3：用 benchmark 验证改造是否真带来收益

**文件：**
- 如有必要，小改：`tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py`
- 否则复用现有 benchmark 入口，不额外改功能

**设计要点：**
- 重点确认 `put_batch` 耗时是否下降，同时不要把 `GET` 性能拖垮。
- 先跑最小配置，再跑一个开启共享列的中等配置。
- 如果第一版 dict backend 明显缩短了锁时间，但总体吞吐因为 payload 复制过多而回退，优先继续优化存储条目表示，而不是重新引入 `ref_count` 或 ZMQ 特化抽象。

**测试场景：**
- tensor-only workload：对比改造前后的 `put` 吞吐。
- shared-column workload：对比改造前后的 `put` 吞吐，同时确认无正确性回归。
- raw/pickle 混合 workload：确认不会崩，并观察 `get` 延迟是否基本稳定。

**验证：**
- 运行：`./venv/bin/python tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py --config debug --mode tensor --rounds 3`
- 运行：`./venv/bin/python tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py --config debug --mode uuid --rounds 3 --n-samples-per-prompt 4`

## 落地建议

- 如果测试面和 benchmark 结果都比较明确，建议一次性切换，不必长期双轨维护。
- 如果你想降低实现风险，可以临时保留旧 backend 开关，做一次 A/B benchmark 后再在同一分支里删除旧路径，但这个开关应该是短生命周期的。
- 在当前分析下，不建议去改 `tq_client.py` 的请求构造逻辑；现有协议已经足够支撑 shard 侧存储改造。

## 预期结果

改完之后，整体效果应该是：

- shard 侧存储从 `zmq.Frame` 生命周期管理收敛成普通 dict 管理
- `put_batch()` 的锁持有时间明显下降
- `clear()`、覆盖写和 prune 逻辑都更容易理解和维护
- 共享列语义不变，但复杂度从“手工引用计数”转移成“确定性的 storage key 规则”
