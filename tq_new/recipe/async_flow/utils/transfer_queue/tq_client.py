# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import asyncio
import logging
import pickle
import threading
from typing import Any, Optional

import numpy as np
import ray
import torch
import zmq
import zmq.asyncio

from recipe.async_flow.utils.transfer_queue.tq_config import (
    DEFAULT_TOPIC,
    GROUP_SHARED_COLUMNS,
    MAX_CONCURRENT_GETS,
    NUM_SAMPLE_PER_SEGMENT,
    ZMQ_HWM,
)
from recipe.async_flow.utils.transfer_queue.tq_sampler import BaseSampler
from recipe.async_flow.utils.transfer_queue.tq_utils import (
    ZMQClient,
    deserialize_column_from_frame,
    deserialize_column_pickle_from_frame,
    serialize_batch,
    serialize_batch_pickle,
    torch_to_numpy,
)

# ---------------------------------------------------------------------------
# Sync → Async bridge: replace asyncio.run() with a safe persistent-loop
# approach that does NOT create/destroy event loops per call.
# ---------------------------------------------------------------------------
_persistent_loop = None
_persistent_loop_thread = None
_persistent_loop_lock = threading.Lock()


def _get_persistent_loop():
    """Get or create a singleton daemon-thread event loop for sync wrappers."""
    global _persistent_loop, _persistent_loop_thread
    if _persistent_loop is not None and _persistent_loop.is_running():
        return _persistent_loop
    with _persistent_loop_lock:
        if _persistent_loop is not None and _persistent_loop.is_running():
            return _persistent_loop
        _persistent_loop = asyncio.new_event_loop()

        def _run_loop():
            asyncio.set_event_loop(_persistent_loop)
            _persistent_loop.run_forever()

        _persistent_loop_thread = threading.Thread(target=_run_loop, daemon=True, name="tq-sync-wrapper-loop")
        _persistent_loop_thread.start()
        return _persistent_loop


def _run_coroutine_from_sync(coro):
    """
    Run an async coroutine from synchronous code safely.

    - No running event loop  → submit to the persistent daemon loop.
    - Already inside a loop  → use run_coroutine_threadsafe (caller thread blocks).
    This avoids asyncio.run() which creates/destroys a loop each call and
    crashes when a loop is already running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — use the persistent background loop.
        loop = _get_persistent_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    # Already inside a running loop — submit to it and block the calling thread.
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


class TransferQueueClient:
    def __init__(
        self,
        manager_handle,
        logger=None,
        micro_batch_size=NUM_SAMPLE_PER_SEGMENT,
        max_concurrent_gets=MAX_CONCURRENT_GETS,
    ):
        """
        Initialize the client interface.
        - manager_handle: A Ray actor handle for Manager.
        """
        self.manager = manager_handle
        self.zmq_context = zmq.Context.instance()
        self.async_ctx = zmq.asyncio.Context()
        self._local = threading.local()
        # self.socket_cache = {}
        self.num_sample_per_segment = micro_batch_size
        self.max_concurrent_gets = max_concurrent_gets
        self.get_semaphore = asyncio.Semaphore(self.max_concurrent_gets)
        self.topic_configs = {}  # 存储每个topic的配置信息
        self._endpoint_locks = {}

        self.zmq_client = ZMQClient(
            self.async_ctx,
            endpoint=ray.get(self.manager.get_manager_endpoint.remote()),
        )

        if logger is None:
            self.logger = logging.getLogger("TransferQueueClient")
        else:
            self.logger = logger

    def add_topic(
        self,
        prompts_num: int,
        n_samples_per_prompt: int,
        experience_columns: list[str],
        experience_consumers: list[str],
        topic: str = DEFAULT_TOPIC,
    ) -> None:
        """
        Register a topic on the manager and provision per-shard storage.
        - Pure schema setup; no data is inserted here.
        - Raises if the topic already exists.
        """
        if not topic:
            raise ValueError("Topic must be non-empty")
        self.topic_configs[topic] = {"n_samples_per_prompt": n_samples_per_prompt}
        ray.get(
            self.manager.add_topic.remote(
                topic,
                prompts_num,
                n_samples_per_prompt,
                experience_columns,
                experience_consumers,
            )
        )
        self.logger.info(
            f"Client: Created topic '{topic}' with n_samples_per_prompt={self.topic_configs[topic]['n_samples_per_prompt']}, "
            f"experience_columns={experience_columns}, experience_consumers={experience_consumers}"
        )

    def delete_topic(self, topic: str = DEFAULT_TOPIC) -> None:
        """
        Delete the specified topic, including tables on the manager and all shards.
        """
        ray.get(self.manager.delete_topic.remote(topic))
        # 清理本地缓存的配置
        if topic in self.topic_configs:
            del self.topic_configs[topic]

    def reset_all(self):
        """Fully reset back to post-__init__ state: drop all topics/tables and reset manager state."""
        # NOTE: this removes schemas entirely; add_topic must be called again afterwards.
        ray.get(self.manager.reset_all.remote())
        self.topic_configs.clear()  # 清空所有缓存的topic配置
        self.logger.info("Client: Fully reset TransferQueue to post-__init__ state (ALL topics dropped).")

    async def _get_n_samples_per_prompt_async(self, topic: str) -> int:
        """
        获取指定topic的n_samples_per_prompt配置值（异步版本）

        采用懒加载模式：
        1. 优先从本地缓存获取
        2. 如果本地没有，从manager远程查询并缓存
        3. 如果manager也没有，抛出错误

        Args:
            topic: topic名称

        Returns:
            该topic配置的n_samples_per_prompt值

        Raises:
            ValueError: 如果topic在本地和manager都不存在
        """
        # 如果本地缓存中有，直接返回
        if topic in self.topic_configs:
            return self.topic_configs[topic]["n_samples_per_prompt"]

        # 本地没有，尝试从manager获取（懒加载）
        try:
            n_samples = await self.manager.get_n_samples_per_prompt.remote(topic)
            # 缓存到本地，避免后续重复查询
            self.topic_configs[topic] = {"n_samples_per_prompt": n_samples}
            return n_samples
        except Exception as e:
            raise ValueError(
                f"Topic '{topic}' not found in both client cache and manager. "
                f"Error from manager: {e}. "
                f"Please ensure the topic exists (call add_topic() to create it)."
            )

    def _get_n_samples_per_prompt(self, topic: str) -> int:
        """同步版本，委托到 _get_n_samples_per_prompt_async"""
        return _run_coroutine_from_sync(self._get_n_samples_per_prompt_async(topic))

    def _get_async_socket(self, endpoint: str) -> zmq.asyncio.Socket:
        """Get or create a ZMQ DEALER socket for the specific endpoint."""
        if not hasattr(self._local, "socket_cache"):
            self._local.socket_cache = {}

        if endpoint not in self._local.socket_cache:
            sock = self.async_ctx.socket(zmq.DEALER)

            sock.setsockopt(zmq.SNDHWM, ZMQ_HWM)
            sock.setsockopt(zmq.RCVHWM, ZMQ_HWM)

            # Set Linger to 0 to ensure non-blocking close
            sock.setsockopt(zmq.LINGER, 0)

            sock.connect(endpoint)
            self._local.socket_cache[endpoint] = sock

        return self._local.socket_cache[endpoint]

    def _invalidate_socket(self, endpoint: str):
        if not hasattr(self._local, "socket_cache"):
            return

        if endpoint in self._local.socket_cache:
            sock = self._local.socket_cache[endpoint]
            try:
                sock.close(linger=0)
            except Exception:
                pass
            del self._local.socket_cache[endpoint]

    def put_experience(
        self,
        data_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        indexes: Optional[list[int]] = None,
        version: int = None,
        topic: str = DEFAULT_TOPIC,
        save_dtype: str = None,
        data_batch_size: int = None,
    ) -> list[int]:
        return _run_coroutine_from_sync(
            self.put_experience_async(data_dict, indexes, version, topic, save_dtype, data_batch_size)
        )

    async def put_experience_async(
        self,
        data_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        indexes: Optional[list[int]] = None,
        version: int = None,
        topic: str = DEFAULT_TOPIC,
        save_dtype: str = None,
        data_batch_size: int = None,
    ) -> list[int]:
        """
        Write data to TransferQueue.

        All columns in data_dict must be per-index (len = len(indexes)).
        For shared columns, callers should use reference repetition (zero-copy)
        to build per-index lists where same-group entries point to the same object.
        TQ internally deduplicates shared columns before transmission.

        Args:
            data_dict: Data dictionary {column_name: data}. Each value is a list
                       where data[i] corresponds to indexes[i].
            indexes: Pre-allocated indexes or None for automatic allocation
            version: Version number
            topic: Topic name
            save_dtype: Save data type for storage optimization
            data_batch_size: Override inferred batch size
        """
        # Step 1: 推断 batch size
        n_samples_per_prompt = await self._get_n_samples_per_prompt_async(topic)
        if not data_batch_size:
            data_batch_size = self._infer_data_batch_size(data_dict)

        # Step 2: 索引分配
        if indexes is None:
            uuids = self._extract_uuids(data_dict)
            if uuids:
                indexes = await self.zmq_client.call(
                    "allocate_by_uuid",
                    topic=topic,
                    uuids=uuids,
                )
            else:
                indexes = await self.zmq_client.call("allocate_batches", topic=topic, data_batch_size=data_batch_size)

        if data_batch_size != len(indexes):
            raise ValueError(
                f"[TQ-Put] Putting {data_batch_size} samples. It doesn't match with length of indexes:{len(indexes)}"
            )

        # Step 3: 共享列去重检查
        shared_col_names = {col for col in data_dict if col in GROUP_SHARED_COLUMNS}
        unrecorded_groups = {}
        if shared_col_names:
            unique_groups = sorted({idx // n_samples_per_prompt for idx in indexes})
            unrecorded_groups = await self.manager.get_unrecorded_groups.remote(topic, shared_col_names, unique_groups)

        # Step 4: 路由映射（1 次 RPC）
        route_map = await self.zmq_client.call(
            "get_targets_for_put",
            topic=topic,
            indexes=indexes,
        )

        # Step 5: 逐 endpoint 构建 + 发送
        global_idx_to_local = {idx: i for i, idx in enumerate(indexes)}
        tasks = []
        for endpoint, ep_indexes in route_map.items():
            tasks.append(
                self._process_node_data(
                    topic=topic,
                    data_dict=data_dict,
                    endpoint=endpoint,
                    ep_indexes=ep_indexes,
                    global_idx_to_local=global_idx_to_local,
                    n_samples_per_prompt=n_samples_per_prompt,
                    save_dtype=save_dtype,
                    unrecorded_groups=unrecorded_groups,
                )
            )

        if not tasks:
            self.logger.warning(f"No data to send for indexes {indexes}")
            return []

        results = await asyncio.gather(*tasks)

        # Step 6: 标记共享列写入状态 + 版本记录
        has_marks = bool(unrecorded_groups)  # 判断 dict 是否非空
        has_version = version is not None and bool(indexes)  # 判断是否有新版本索引

        if has_marks or has_version:
            # 只有当其中之一有数据时，才发起 RPC
            await self.manager.record_groups_and_versions.remote(
                topic=topic, marks=unrecorded_groups, version=version, indexes=indexes
            )

            # # 单次批量 RPC 调用
            # if marks:
            #     await self.zmq_client.call("batch_mark_columns_written", topic=topic, marks=marks)
            #
            # # Step 6: 记录版本
            # if version is not None:
            #     await self.zmq_client.call(
            #         "record_versions", topic=topic, version=version, indexes=indexes,
            #     )

        # if version is not None:
        #     await self.manager.record_versions.remote(topic, version, indexes)

        # Logging
        total_bytes = sum(r["bytes"] for r in results)
        total_mb = total_bytes / (1024 * 1024)
        self.logger.info(
            f"[TransferQueue Put] | Total Volume: {total_mb:.2f} MB | "
            f"Data_size: {len(indexes)} | Shared_cols: {shared_col_names or 'none'}"
        )

        if self.logger.isEnabledFor(logging.DEBUG):
            for res in results:
                self.logger.debug(
                    f"  -> Node: {res['endpoint']} | Cnt: {res['count']} | Sz: {res['bytes'] / 1024:.2f} KB"
                )
        return indexes

    async def _process_node_data(
        self,
        topic: str,
        data_dict: dict[str, Any],
        endpoint: str,
        ep_indexes: list[int],
        global_idx_to_local: dict[int, int],
        n_samples_per_prompt: int = 1,
        save_dtype: str = None,
        unrecorded_groups: dict[str, list[int]] = None,
    ) -> dict[str, Any]:
        """
        Build and send ZMQ messages to one endpoint.

        Two scenarios (needs_expansion eliminated):
        1. Shared col: raw_data[i] = indexes[i]. Deduplicate by taking any
        available index per group from the current segment.
           Transmit 1 data item per group with ref_multiplier=nspp.
        2. Non-shared col: raw_data[i] = indexes[i]. 1:1 transmission.

        Args:
            endpoint: Target shard endpoint
            ep_indexes: Global indexes assigned to this endpoint (subset of full indexes)
            data_dict: Data dictionary. data_dict[col][local_idx] corresponds to
                       the local position in global_idx_to_local.
            global_idx_to_local: Mapping from global index to local position in data_dict.
                                 Built once in put_experience_async and shared across endpoints.
            topic: Topic name
            n_samples_per_prompt: Samples per group
            save_dtype: Save data type
            unrecorded_groups: Dict of {col_name: [group_ids not yet written]}.
                               Keys implicitly define which columns are shared.
        """
        # 共享列 = unrecorded_groups 的 keys
        if unrecorded_groups is None:
            unrecorded_groups = {}
        shared_col_names = set(unrecorded_groups.keys())

        # 预计算：每个共享列的 missing_set（仅构建一次，避免 segment 循环内重复）
        missing_sets = {col: set(gids) for col, gids in unrecorded_groups.items()}

        # 预计算：ep_indexes 的 group_id 列表（避免 segment 循环内重复 idx // n）
        ep_group_ids = [idx // n_samples_per_prompt for idx in ep_indexes]

        total_sent_bytes = 0
        async with self.get_semaphore:
            lock = self._get_endpoint_lock(endpoint)
            async with lock:
                try:
                    sock = self._get_async_socket(endpoint)
                    recv_futures = []

                    for seg_start in range(0, len(ep_indexes), self.num_sample_per_segment):
                        seg_indexes = ep_indexes[seg_start : seg_start + self.num_sample_per_segment]
                        seg_group_ids = ep_group_ids[seg_start : seg_start + self.num_sample_per_segment]

                        header = {"topic": topic, "indexes": seg_indexes, "columns": {}, "order": []}
                        payload_frames = []

                        for col_name, raw_data in data_dict.items():
                            is_shared = col_name in shared_col_names

                            if is_shared:
                                # --- 场景1: 共享列 ---
                                missing_set = missing_sets.get(col_name)
                                if not missing_set:
                                    continue  # 该列所有 group 已写入，跳过

                                # 从预计算的 ep_group_ids 切片获取该 segment 的 unique groups
                                sorted_seg_group_ids = sorted(set(seg_group_ids))
                                send_group_ids = [g for g in sorted_seg_group_ids if g in missing_set]

                                if not send_group_ids:
                                    continue  # 该 segment 无需发送此列

                                # 去重：每组取 segment 内任一可用 index（同组引用同一对象，取谁都行）
                                seg_gid_to_idx = {}
                                for idx, gid in zip(seg_indexes, seg_group_ids, strict=False):
                                    if gid not in seg_gid_to_idx:
                                        seg_gid_to_idx[gid] = idx
                                representative_indices = [seg_gid_to_idx[gid] for gid in send_group_ids]
                                col_batch = [raw_data[global_idx_to_local[ri]] for ri in representative_indices]

                                # 序列化
                                final_buffer, lengths, dtype_str, shapes, encoding = self._serialize_column(
                                    col_batch, save_dtype
                                )

                                header["columns"][col_name] = {
                                    "dtype": dtype_str,
                                    "lengths": lengths,
                                    "shapes": shapes,
                                    "ref_multiplier": n_samples_per_prompt,
                                    "encoding": encoding,
                                    "group_ids": send_group_ids,
                                }
                            else:
                                # --- 场景2: 非共享列 ---
                                col_batch = [raw_data[global_idx_to_local[gidx]] for gidx in seg_indexes]

                                final_buffer, lengths, dtype_str, shapes, encoding = self._serialize_column(
                                    col_batch, save_dtype
                                )

                                header["columns"][col_name] = {
                                    "dtype": dtype_str,
                                    "lengths": lengths,
                                    "shapes": shapes,
                                    "ref_multiplier": 1,
                                    "encoding": encoding,
                                }

                            header["order"].append(col_name)
                            payload_frames.append(final_buffer)

                        # 发送
                        if header["order"]:
                            header_bytes = pickle.dumps(header)
                            await sock.send_multipart([b"PUT", header_bytes] + payload_frames, copy=False)
                            total_sent_bytes += len(header_bytes) + sum(
                                b.nbytes if hasattr(b, "nbytes") else len(b) for b in payload_frames
                            )
                            recv_futures.append(sock.recv())

                    if recv_futures:
                        responses = await asyncio.gather(*recv_futures)
                        for idx, resp in enumerate(responses):
                            if resp != b"ACK":
                                error_msg = f"Batch {idx} on {endpoint} failed. Got: {resp}"
                                self.logger.error(error_msg)
                                raise RuntimeError(error_msg)

                except Exception as e:
                    self.logger.error(f"Put error on {endpoint}: {e}")
                    self._invalidate_socket(endpoint)
                    raise e

        return {"endpoint": endpoint, "count": len(ep_indexes), "bytes": total_sent_bytes}

    # def _serialize_column(self, col_batch, save_dtype=None):
    #     """
    #     Serialize a column's batch data.

    #     Returns:
    #         (final_buffer, lengths, dtype_str, shapes, encoding)
    #     """
    #     if not col_batch:
    #         return b'', [], "raw", [], "raw"

    #     sample = col_batch[0]
    #     if isinstance(sample, (torch.Tensor, np.ndarray)):
    #         # raw encoding
    #         if isinstance(sample, torch.Tensor):
    #             col_batch = [torch_to_numpy(t) for t in col_batch]

    #         # Check if 2D contiguous ndarray for efficient serialization
    #         if (isinstance(col_batch, np.ndarray) and col_batch.ndim == 2) or \
    #            (isinstance(col_batch, list) and len(col_batch) > 0 and
    #             isinstance(col_batch[0], np.ndarray) and col_batch[0].ndim >= 1 and
    #             all(isinstance(a, np.ndarray) and a.shape == col_batch[0].shape and
    #                 a.dtype == col_batch[0].dtype for a in col_batch)):
    #             try:
    #                 stacked = np.stack(col_batch) if isinstance(col_batch, list) else col_batch
    #                 if stacked.ndim == 2:
    #                     col_batch = stacked
    #             except Exception:
    #                 pass

    #         final_buffer, lengths, dtype_str, shapes = serialize_batch(
    #             col_batch, None, None, save_dtype
    #         )
    #         return final_buffer, lengths, dtype_str, shapes, "raw"
    #     else:
    #         # pickle encoding
    #         final_buffer, lengths, dtype_str = serialize_batch_pickle(col_batch)
    #         return final_buffer, lengths, dtype_str, None, "pickle"

    def _serialize_column(self, col_batch, save_dtype=None):
        if not col_batch:
            return b"", [], "raw", [], "raw"

        sample = col_batch[0]

        # 分流 1: 数值型数据 (Tensor/Numpy)
        if isinstance(sample, (torch.Tensor, np.ndarray)):
            # 统一转为 Numpy
            forced_dtype_str = None
            if isinstance(sample, torch.Tensor):
                if save_dtype is None and sample.dtype == torch.bfloat16:
                    col_batch = [t.detach().contiguous().view(torch.int16).cpu().numpy() for t in col_batch]
                    forced_dtype_str = "bfloat16"
                elif save_dtype is None and hasattr(torch, "float8_e4m3fn") and sample.dtype == torch.float8_e4m3fn:
                    col_batch = [t.detach().contiguous().view(torch.int8).cpu().numpy() for t in col_batch]
                    forced_dtype_str = "float8_e4m3fn"
                elif save_dtype is None and hasattr(torch, "float8_e5m2") and sample.dtype == torch.float8_e5m2:
                    col_batch = [t.detach().contiguous().view(torch.int8).cpu().numpy() for t in col_batch]
                    forced_dtype_str = "float8_e5m2"
                else:
                    col_batch = [torch_to_numpy(t) for t in col_batch]

            # 【核心优化】：尝试 stack 以触发 serialize_batch 的零拷贝快链
            # 只要 stack 成功且是 2D，serialize_batch 内部的 ravel() 就能起飞
            try:
                stacked = np.stack(col_batch)
                if stacked.ndim == 2:
                    col_batch = stacked
            except (ValueError, TypeError):
                pass

            # 调用真正的打包机
            final_buffer, lengths, dtype_str, shapes = serialize_batch(col_batch, None, None, save_dtype)
            if forced_dtype_str is not None:
                dtype_str = forced_dtype_str
            return final_buffer, lengths, dtype_str, shapes, "raw"

        # 分流 2: 非数值对象 (String/Dict 等)
        else:
            final_buffer, lengths, dtype_str = serialize_batch_pickle(col_batch)
            return final_buffer, lengths, dtype_str, None, "pickle"

    def _extract_uuids(self, data_dict: dict[str, Any]) -> list[str]:
        """Extract UUID column values (if any)"""
        uuid_col = [k for k in data_dict if k.endswith(("_uuid", "_uid"))]
        if len(uuid_col) > 1:
            raise ValueError(f"Expected at most 1 UUID column, found {len(uuid_col)}: {uuid_col}")
        if not uuid_col:
            return []
        raw_list = data_dict[uuid_col[0]]
        uuids = [str(x) for x in raw_list] if raw_list else []
        return uuids

    def _infer_data_batch_size(self, data_dict: dict[str, Any]) -> int:
        """推断数据批量大小"""
        # 1. 优先 List 的长度（最明确）
        for v in data_dict.values():
            if isinstance(v, list) and len(v) > 0:
                return len(v)

        # 2. 统计推断：所有多维 Tensor 出现频率最高的第 0 维
        shapes = []
        for v in data_dict.values():
            if torch.is_tensor(v) and v.ndim >= 2:
                shapes.append(v.shape[0])

        # 3. 兜底策略：默认为 1
        if not shapes:
            return 1

        most_common = max(set(shapes), key=shapes.count)
        return most_common

    # def _normalize_single_item_data(
    #     self,
    #     data_dict: Dict[str, Any]
    #     ) -> Dict[str, Any]:
    #     """
    #     Normalize single-item input to batch format.

    #     当 batch_size==1 时，将非 list 的值封装成 [value]

    #     Args:
    #         data_dict: 输入数据字典

    #     Returns:
    #         Normalized data_dict
    #     """
    #     normalized = {}
    #     for key, value in data_dict.items():
    #         if isinstance(value, list):
    #             normalized[key] = value
    #         else:
    #             if isinstance(value, torch.Tensor) and value.ndim == 0:
    #                 # 0D 标量 → 转换为 1D 再包装
    #                 normalized[key] = [value.unsqueeze(0)]
    #             else:
    #                 normalized[key] = [value]
    #     return normalized

    def get_experience(
        self,
        consumer: str,
        experience_columns: list[str],
        experience_count: int = None,
        indexes: list[int] = None,
        get_n_samples: bool = False,
        allowed_staleness: int = None,
        latest_version: int = None,
        topic: str = DEFAULT_TOPIC,
        sampler_func: Optional[BaseSampler] = None,
        copy: bool = False,
    ) -> tuple[dict[str, torch.Tensor | list[torch.Tensor] | list[Any]], list[int]]:
        """
        Get experience interface, pass consumer, experience_columns, experience_count to get specified amount of experience
        Wraps get_experience_async, asynchronous fetch with synchronous return
        Args:
            consumer (str):
            experience_columns (List[str]):
            experience_count (int) = None:
            indexes (List[int]): = None,
            get_n_samples (bool): = False,
            topic (str): Defaults to DEFAULT_TOPIC if not provided,
            copy (bool): Memory strategy control.
                - False(default): Zero-copy. Tensor shares underlying memory with Frame.
                  Note: Generated Tensor may be read-only; modifying Tensor will directly affect underlying Buffer
                - True: Deep copy. Tensor has independent memory, safe to write, unaffected by Frame lifecycle.
        Returns:
            Dict[str, Union[torch.Tensor, List[torch.Tensor]]]: key is columns, value is list of tensors for that column.
            List[int]: List of acquired indexes
        """
        return _run_coroutine_from_sync(
            self.get_experience_async(
                consumer,
                experience_columns,
                experience_count,
                indexes,
                get_n_samples,
                allowed_staleness,
                latest_version,
                topic,
                sampler_func,
                copy,
            )
        )

    async def get_experience_async(
        self,
        consumer: str,
        experience_columns: list[str],
        experience_count: int = None,
        indexes: list[int] = None,
        get_n_samples: bool = False,
        allowed_staleness: int = None,
        latest_version: int = None,
        topic: str = DEFAULT_TOPIC,
        sampler_func: Optional[BaseSampler] = None,
        copy: bool = False,
    ) -> tuple[dict[str, torch.Tensor | list[torch.Tensor] | list[Any]], list[int]]:
        # 1. Validation & Defaulting
        if experience_count is None and indexes is None:
            raise ValueError("Either experience_count or indexes must be provided")
        if experience_count is not None and experience_count <= 0:
            raise ValueError("experience_count must be greater than 0")

        # 2. Ask Manager for Shard Allocation
        # shard_map structure returned by Manager: {endpoint_url: [global_index_1, global_index_2, ...]}
        # sampler_func (if any) is a Python object; ManagerRpcClient auto-pickles it.
        try:
            if indexes is not None:
                shard_map = await self.zmq_client.call(
                    "allocate_shard_for_indexes",
                    topic=topic,
                    consumer=consumer,
                    experience_columns=experience_columns,
                    indexes=indexes,
                    allowed_staleness=allowed_staleness,
                    latest_version=latest_version,
                )
            else:
                shard_map = await self.zmq_client.call(
                    "allocate_shard_and_indexes",
                    topic=topic,
                    consumer=consumer,
                    experience_columns=experience_columns,
                    experience_count=experience_count,
                    get_n_samples=get_n_samples,
                    allowed_staleness=allowed_staleness,
                    latest_version=latest_version,
                    sampler_func=sampler_func,
                )
        except Exception as e:
            self.logger.error(f"Failed to allocate shards: {e}")
            return None, None

        if not shard_map:
            self.logger.warning("Get experience returned no data from manager.")
            return None, None

        # Create asynchronous fetch tasks
        tasks = []
        for endpoint, target_indexes in shard_map.items():
            tasks.append(self._fetch_one_shard(endpoint, topic, experience_columns, target_indexes, copy))

        # Wait for all shards to return
        # results type: List[Optional[Dict]]
        shard_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Summarize byte count
        total_bytes = 0
        for result in shard_results:
            if result is not None and not isinstance(result, Exception):
                total_bytes += result.get("bytes", 0)
        total_mb = total_bytes / (1024 * 1024)

        # Aggregate — 直接 extend，消除 id_to_data_map 和 sample_pack 中间层
        final_indexes = []
        final_columns = {col: [] for col in experience_columns}
        total_items = 0
        successful_shards = 0
        failed_shards = 0

        for res in shard_results:
            if isinstance(res, Exception):
                self.logger.warning(f"Shard fetch failed with exception: {res}")
                failed_shards += 1
                continue
            if res is None:
                self.logger.warning("Shard fetch returned None (connection error)")
                failed_shards += 1
                continue
            successful_shards += 1
            returned_ids = res["indexes"]
            shard_data = res["data"]

            count = len(returned_ids)
            total_items += count
            final_indexes.extend(returned_ids)
            for col in experience_columns:
                if col in shard_data:
                    final_columns[col].extend(shard_data[col])

        if total_items == 0:
            return None, None

        # 按请求顺序或 global_idx 排序重排
        if indexes is not None:
            idx_to_pos = {idx: i for i, idx in enumerate(final_indexes)}
            reordered_indexes = []
            reordered_columns = {col: [] for col in experience_columns}
            for target_id in indexes:
                pos = idx_to_pos.get(target_id)
                if pos is not None:
                    reordered_indexes.append(target_id)
                    for col in experience_columns:
                        reordered_columns[col].append(final_columns[col][pos])
                else:
                    self.logger.warning(f"Index {target_id} not found in shard results")
            final_indexes = reordered_indexes
            final_columns = reordered_columns
        else:
            sort_order = sorted(range(len(final_indexes)), key=lambda i: final_indexes[i])
            final_indexes = [final_indexes[i] for i in sort_order]
            for col in experience_columns:
                final_columns[col] = [final_columns[col][i] for i in sort_order]

        # 6. Logging
        self.logger.info(
            f"[TransferQueue Get] | Total Volume: {total_mb:.2f} MB | Shards: {successful_shards} | "
            f"Routed_experts exists : {'routed_experts' in experience_columns} | Data_size: {len(final_indexes)} | Total_items: {total_items}"
        )

        return final_columns, final_indexes

    async def _fetch_one_shard(
        self,
        endpoint: str,
        topic: str,
        columns: list[str],
        indexes: list[int],
        copy=False,
    ) -> dict[str, Any]:
        # Global concurrency control
        async with self.get_semaphore:
            # Get exclusive lock for this Endpoint to prevent concurrent writes causing ZMQ state machine confusion or packet mixing
            lock = self._get_endpoint_lock(endpoint)
            async with lock:
                try:
                    sock = self._get_async_socket(endpoint)
                    req_header = {"topic": topic, "experience_columns": columns, "indexes": indexes}

                    await sock.send_multipart([b"GET", pickle.dumps(req_header)])

                    reply = await sock.recv_multipart(copy=False)

                    # Calculate total bytes received (actual network transfer bytes)
                    received_bytes = sum(
                        frame.nbytes if hasattr(frame, "nbytes") else len(frame.bytes) for frame in reply
                    )

                    meta_frame = reply[0]
                    if meta_frame.bytes.startswith(b"ERROR:"):
                        raise RuntimeError(f"Remote Error: {meta_frame.bytes.decode()}")

                    meta = pickle.loads(meta_frame.bytes)
                    returned_indexes = meta["indexes"]
                    col_meta_map = meta["columns"]

                    # Deserialize
                    result_data = {}
                    if len(reply) - 1 != len(columns):
                        raise RuntimeError(f"Column mismatch from {endpoint}. Exp {len(columns)}, Got {len(reply) - 1}")

                    for i, col_name in enumerate(meta["order"]):
                        if i + 1 >= len(reply):
                            break

                        frame = reply[i + 1]
                        col_info = col_meta_map.get(col_name)

                        if col_info:
                            encoding = col_info.get("encoding", "raw")  # Default to raw for backward compatibility

                            if encoding == "pickle":
                                objects = deserialize_column_pickle_from_frame(
                                    frame,
                                    col_info["lengths"],  # byte lengths for pickle
                                )
                                result_data[col_name] = objects
                            else:  # 'raw' encoding
                                tensors = deserialize_column_from_frame(
                                    frame, col_info["dtype"], col_info["lengths"], copy, col_info["shapes"]
                                )
                                result_data[col_name] = tensors
                        else:
                            result_data[col_name] = []

                    return {"data": result_data, "indexes": returned_indexes, "bytes": received_bytes}

                except (zmq.ZMQError, zmq.Again, ConnectionError) as e:
                    # ZMQ 连接问题：socket 已被 invalidate，返回 None 让上层跳过
                    self.logger.warning(f"ZMQ connection error fetching shard {endpoint}: {e}")
                    self._invalidate_socket(endpoint)
                    return None
                except RuntimeError as e:
                    # shard 端错误（如数据不存在、列不匹配）：向上传播，不应静默
                    self.logger.error(f"Shard {endpoint} returned error: {e}")
                    self._invalidate_socket(endpoint)
                    raise
                except Exception as e:
                    # 其他未知错误：记录并返回 None
                    self.logger.error(f"Unexpected error fetching shard {endpoint}: {e}")
                    self._invalidate_socket(endpoint)
                    return None

    def _get_endpoint_lock(self, endpoint: str) -> asyncio.Lock:
        if endpoint not in self._endpoint_locks:
            self._endpoint_locks[endpoint] = asyncio.Lock()
        return self._endpoint_locks[endpoint]
        # existing_lock = self._endpoint_locks.get(endpoint)
        # if existing_lock is not None:
        #     try:
        #         # 尝试获取当前循环
        #         current_loop = asyncio.get_running_loop()
        #         # 重点：不要直接调用 lock._get_loop()，
        #         # 而是直接尝试 acquire。如果 loop 不对，这里会报错。
        #         # 或者我们直接访问内部属性（虽然不推荐但安全）：
        #         if existing_lock._loop is current_loop:
        #             return existing_lock
        #     except (RuntimeError, AttributeError):
        #         # 如果报错了，说明 loop 变了，我们需要创建一个新的
        #         pass

        # # 走到这里说明：要么没锁，要么锁不可用
        # new_lock = asyncio.Lock()
        # self._endpoint_locks[endpoint] = new_lock
        # return new_lock

    # def _get_socket(self, endpoint: str) -> zmq.Socket:
    #     """Get or create a Dealer socket connected to the given endpoint."""
    #     # Use thread-local sockets because ZeroMQ sockets are not thread-safe.
    #     if not hasattr(self._local, "sockets"):
    #         self._local.sockets = {}
    #     tl_sockets: Dict[str, zmq.Socket] = self._local.sockets
    #     if endpoint in tl_sockets:
    #         return tl_sockets[endpoint]
    #     sock = self.zmq_context.socket(zmq.DEALER)
    #     sock.connect(endpoint)
    #     tl_sockets[endpoint] = sock
    #     return sock

    async def delete_experience_async(
        self,
        topic: str = DEFAULT_TOPIC,
        indexes: list[int] = None,
        versions: list[int] = None,
        latest_version: int = None,
        allowed_staleness: int = None,
        delete_all: bool = False,
    ):
        res = await self.manager.delete_experience.remote(
            topic, indexes, versions, latest_version, allowed_staleness, delete_all
        )
        return res

    def delete_experience(
        self,
        topic: str = DEFAULT_TOPIC,
        indexes: list[int] = None,
        versions: list[int] = None,
        latest_version: int = None,
        allowed_staleness: int = None,
        delete_all: bool = False,
    ):
        return _run_coroutine_from_sync(
            self.delete_experience_async(topic, indexes, versions, latest_version, allowed_staleness, delete_all)
        )

    def get_data_ready_set(
        self,
        topic: str = DEFAULT_TOPIC,
        experience_columns: list[str] = None,
        indexes: list[int] = None,
        get_n_samples: bool = False,
    ):
        return ray.get(self.manager.get_data_ready_set.remote(topic, experience_columns, indexes, get_n_samples))

    def get_data_consumed_set(
        self, topic: str = DEFAULT_TOPIC, consumer: str = None, indexes: list[int] = None, get_n_samples: bool = False
    ):
        return ray.get(self.manager.get_data_consumed_set.remote(topic, consumer, indexes, get_n_samples))

    def get_data_usable_set(
        self,
        topic: str = DEFAULT_TOPIC,
        consumer: str = None,
        experience_columns: list[str] = None,
        allowed_staleness: int = None,
        latest_version: int = None,
        indexes: list[int] = None,
        get_n_samples: bool = False,
    ):
        return ray.get(
            self.manager.get_data_usable_set.remote(
                topic, consumer, experience_columns, allowed_staleness, latest_version, indexes, get_n_samples
            )
        )


def get_transferqueue_client(name: str = "TransferQueueManager") -> TransferQueueClient:
    """
    Get a new Client instance connected to the named TransferQueueManager actor.
    This uses Ray's global name registry to locate the manager.
    """
    if name is None:
        name = "TransferQueueManager"
    mgr = ray.get_actor(name)
    return TransferQueueClient(mgr)
