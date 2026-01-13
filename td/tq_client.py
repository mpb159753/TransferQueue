# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import asyncio
import logging
import pickle
import threading
import time
from collections import defaultdict
from typing import List, Dict, Union, Tuple, Any, Optional

import numpy as np
import ray
import torch
import zmq
import zmq.asyncio
import zmq.asyncio
import zmq.asyncio
import zmq.asyncio
from torch import Tensor

from metrics import Metric
from tq_sampler import BaseSampler, RandomSampler, VersionSampler, SeqLenBalSampler
from tq_utils import get_no_pad_length, serialize_batch, deserialize_column_from_frame, \
    get_numpy_dtype

DEFAULT_TOPIC = "Trainer"
NUM_SAMPLE_PER_SEGMENT = 1024
MAX_CONCURRENT_GETS = 10
ZMQ_HWM = 2000


class TransferQueueClient:
    def __init__(self, manager_handle,
                 logger=None,
                 micro_batch_size=NUM_SAMPLE_PER_SEGMENT,
                 max_concurrent_gets=MAX_CONCURRENT_GETS):
        """
        Initialize the client interface.
        - manager_handle: A Ray actor handle for Manager.
        """
        self.manager = manager_handle
        self.zmq_context = zmq.Context.instance()
        self.async_ctx = zmq.asyncio.Context()
        self._local = threading.local()
        self.socket_cache = {}
        self.num_sample_per_segment = micro_batch_size
        self.max_concurrent_gets = max_concurrent_gets
        self.get_semaphore = asyncio.Semaphore(self.max_concurrent_gets)

        if logger is None:
            self.logger = logging.getLogger("TransferQueueClient")
        else:
            self.logger = logger

    def _get_async_socket(self, endpoint: str) -> zmq.asyncio.Socket:
        """Get or create a ZMQ DEALER socket for the specific endpoint."""
        if not hasattr(self._local, "socket_cache"):
            self._local.socket_cache = {}

        if endpoint not in self._local.socket_cache:
            sock = self.async_ctx.socket(zmq.DEALER)

            sock.setsockopt(zmq.SNDHWM, ZMQ_HWM)
            sock.setsockopt(zmq.RCVHWM, ZMQ_HWM)

            # 设置 Linger 为 0，确保关闭时不阻塞
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

    def put_experience(self,
                       data_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
                       indexes: List[int],
                       unpad_pairs: List[Tuple[str, str, str]] = None,
                       topic: str = None, save_dtype: str = None,
                       is_prompt: bool = False):
        return asyncio.run(self.put_experience_async(data_dict, indexes, unpad_pairs, topic, save_dtype, is_prompt))

    async def put_experience_async(self,
                                   data_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
                                   indexes: List[int],
                                   unpad_pairs: List[Tuple[str, str, str]] = None,
                                   topic: str = None,
                                   save_dtype: str = None,
                                   is_prompt: bool = False
                                   ):
        start_time = time.perf_counter()

        # 处理 unpad 映射关系: col_name -> (len_col_name, mode)
        unpad_map = {}
        if unpad_pairs:
            for col, len_col, mode in unpad_pairs:
                unpad_map[col] = (len_col, mode)
        if topic is None:
            topic = DEFAULT_TOPIC
        # 获取存储位置
        try:
            route_map = await self.manager.get_targets_for_put.remote(topic, indexes, is_prompt)
        except Exception as e:
            self.logger.error(f"Routing failed: {e}")
            return

        gid_to_local_map = {gid: idx for idx, gid in enumerate(indexes)}
        endpoint_groups = defaultdict(list)
        routed_gids = set()  # 用于记录已成功路由的ID

        for endpoint, gids in route_map.items():
            for gid in gids:
                local_idx = gid_to_local_map[gid]
                endpoint_groups[endpoint].append((local_idx, gid))
                routed_gids.add(gid)

        # 检查是否有 ID 未找到路由
        if len(routed_gids) < len(indexes):
            missing_gids = set(indexes) - routed_gids
            for missing_gid in missing_gids:
                self.logger.warning(f"No route found for index {missing_gid}")

        # 并行发送数据
        tasks = []
        for endpoint, items in endpoint_groups.items():
            tasks.append(
                self._process_node_data(endpoint, items, data_dict, unpad_map, topic, save_dtype, is_prompt)
            )

        results = await asyncio.gather(*tasks)

        # 记录日志
        total_latency = (time.perf_counter() - start_time) * 1000  # ms
        total_bytes = sum(r['bytes'] for r in results)
        total_mb = total_bytes / (1024 * 1024)

        self.logger.info(
            f"[TransferQueue Put] Topic: {topic} | Total Items: {len(indexes)} | "
            f"Total Latency: {total_latency:.2f} ms | Total Volume: {total_mb:.2f} MB")

        if self.logger.isEnabledFor(logging.DEBUG):
            for res in results:
                self.logger.debug(
                    f"  -> Node: {res['endpoint']} | Cnt: {res['count']} | "
                    f"Sz: {res['bytes'] / 1024:.2f} KB | Lat: {res['latency']:.2f} ms"
                )

    async def _process_node_data(
            self,
            endpoint: str,
            items: List[Tuple[int, int]],
            data_dict: Dict[str, Any],
            unpad_map: Dict[str, Tuple[str, str]],
            topic: str,
            save_dtype: str = None,
            is_prompt: bool = False
    ) -> Dict[str, Any]:

        node_start = time.perf_counter()
        total_sent_bytes = 0
        async with self.get_semaphore:
            lock = self._get_endpoint_lock(endpoint)
            async with lock:
                try:
                    sock = self._get_async_socket(endpoint)
                    recv_futures = []

                    # Micro-batch 循环
                    for i in range(0, len(items), self.num_sample_per_segment):
                        batch_items = items[i: i + self.num_sample_per_segment]

                        # 通过 start, end 获取到切片范围
                        start_local_idx = batch_items[0][0]
                        batch_len = len(batch_items)
                        end_local_idx = start_local_idx + batch_len
                        slice_obj = slice(start_local_idx, end_local_idx)

                        global_ids_batch = [x[1] for x in batch_items]

                        col_order = []
                        header = {"topic": topic, "indexes": global_ids_batch, "columns": {},
                                  "order": [], "is_prompt": is_prompt}
                        payload_frames = []

                        for col_name, raw_data in data_dict.items():
                            # Tensor 路径使用切片
                            if isinstance(raw_data, torch.Tensor):
                                batch_data = raw_data[slice_obj].detach().cpu().numpy()
                            elif isinstance(raw_data, list):
                                batch_data = [raw_data[idx].detach().cpu().numpy() for idx in
                                              range(start_local_idx, end_local_idx)]
                            else:
                                raise TypeError(f"Col {col_name}: Must be Tensor or List[Tensor]")

                            # 准备 Unpad 长度信息
                            batch_lens = None
                            pad_side = None

                            if col_name in unpad_map:
                                len_col_name, pad_side = unpad_map[col_name]
                                full_len_col = data_dict[len_col_name]

                                # 长度列如果是 Tensor，同样使用切片优化
                                if isinstance(full_len_col, torch.Tensor):
                                    batch_lens = full_len_col[slice_obj].detach().cpu().numpy()
                                else:
                                    batch_lens = np.array(full_len_col[slice_obj])

                                # 根据要求更更改存储类型
                                if save_dtype:
                                    batch_lens = batch_lens.astype(get_numpy_dtype(save_dtype))

                            final_buffer, lengths, dtype_str = serialize_batch(batch_data, batch_lens,
                                                                               pad_side, save_dtype)

                            col_order.append(col_name)
                            header["columns"][col_name] = {"dtype": dtype_str, "lengths": lengths}
                            payload_frames.append(final_buffer)

                        # 发送
                        header["order"] = col_order
                        header_bytes = pickle.dumps(header)
                        await sock.send_multipart([b'PUT', header_bytes] + payload_frames, copy=False)
                        total_sent_bytes += len(header_bytes) + sum(b.nbytes for b in payload_frames)
                        # 异步等待传输结果，并行处理后续 batch
                        recv_futures.append(sock.recv())

                    if recv_futures:
                        responses = await asyncio.gather(*recv_futures)
                        for idx, resp in enumerate(responses):
                            if not resp == b"ACK":
                                error_msg = f"Batch {idx} on {endpoint} failed. Expected b'ACK', got: {resp}"
                                self.logger.error(error_msg)
                                raise RuntimeError(error_msg)

                except Exception as e:
                    self.logger.error(f"Put error on {endpoint}: {e}")
                    self._invalidate_socket(endpoint)
                    raise e
        return {
            "endpoint": endpoint, "count": len(items),
            "bytes": total_sent_bytes, "latency": (time.perf_counter() - node_start) * 1000
        }

    def get_experience(
            self,
            consumer: str,
            experience_columns: List[str],
            experience_count: int,
            indexes: List[int] = None,
            get_n_samples: bool = True,
            reward_tags: str = None,
            allowed_staleness: int = None,
            latest_version: int = None,
            topic: str = DEFAULT_TOPIC,
            use_batch_seqlen_balance: bool = False,
            sampler_func: Optional[BaseSampler] = None,
            copy: bool = False,
    ) -> Tuple[Dict[str, Union[torch.Tensor, List[torch.Tensor]]], List[int]]:
        """
        获取 experience 接口，传入 consumer、experience_columns、experience_count 获取指定数量的 experience
        封装 get_experience_async，异步获取，同步返回
        Args:
            consumer (str):
            experience_columns (List[str]):
            experience_count (int):
            indexes (List[int]): = None,
            get_n_samples (bool): = True,
            reward_tags (str): = None,
            topic (str): 不传入则默认为 DEFAULT_TOPIC,
            use_batch_seqlen_balance (bool): = False,
            copy (bool): 内存策略控制。
                - False(默认): 执行 Zero-copy。Tensor 与 Frame 共享底层内存。
                  注意：生成的 Tensor 可能为只读；修改 Tensor 会直接影响底层 Buffer
                - True: 执行深拷贝。Tensor 拥有独立内存，可安全写入，不受 Frame 生命周期影响。
        Returns:
            Dict[str, Union[torch.Tensor, List[torch.Tensor]]]: key 为 columns, value 为该 columns 的 tensor 构成的 list。
            List[int]: 获取到的 index 列表
        """
        return asyncio.run(self.get_experience_async(
            consumer, experience_columns, experience_count, indexes,
            get_n_samples, reward_tags, allowed_staleness, latest_version,
            topic, use_batch_seqlen_balance, sampler_func, copy
        ))

    async def get_experience_async(
            self,
            consumer: str,
            experience_columns: List[str],
            experience_count: int,
            indexes: List[int] = None,
            get_n_samples: bool = True,
            reward_tags: str = None,
            allowed_staleness: int = None,
            latest_version: int = None,
            topic: str = DEFAULT_TOPIC,
            use_batch_seqlen_balance: bool = False,
            sampler_func: Optional[BaseSampler] = None,
            copy: bool = False,
    ) -> Tuple[Dict[str, Union[torch.Tensor, List[torch.Tensor]]], List[int]]:

        start_time = time.perf_counter()

        # 1. Validation & Defaulting
        if topic is None:
            topic = DEFAULT_TOPIC
        if experience_count is None and indexes is None:
            raise ValueError("Either experience_count or indexes must be provided")

        # 2. RPC: Ask Manager for Shard Allocation
        # Manager 返回的 shard_map 结构: {endpoint_url: [global_index_1, global_index_2, ...]}
        try:
            if indexes is not None:
                shard_map = await self.manager.allocate_shard_for_indexes.remote(
                    topic, consumer, experience_columns, indexes,
                    reward_tags, allowed_staleness, latest_version
                )
            else:
                if use_batch_seqlen_balance:
                    sampler_func = SeqLenBalSampler
                shard_map = await self.manager.allocate_shard_and_indexes.remote(
                    topic, consumer, experience_columns, experience_count,
                    get_n_samples, reward_tags, allowed_staleness, latest_version, sampler_func,
                )
        except Exception as e:
            self.logger.error(f"Failed to allocate shards: {e}")
            return None, None

        if not shard_map:
            self.logger.info("Get experience returned no data from manager.")
            return None, None

        # 创建异步抓取任务
        tasks = []
        for endpoint, target_indexes in shard_map.items():
            tasks.append(self._fetch_one_shard(endpoint, topic, experience_columns, target_indexes, copy))

        # 等待所有分片返回
        # results 类型: List[Optional[Dict]]
        shard_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 聚合
        id_to_data_map = {}
        total_items = 0
        successful_shards = 0

        for res in shard_results:
            if isinstance(res, Exception) or res is None:
                continue
            successful_shards += 1
            returned_ids = res['indexes']
            shard_data = res['data']

            count = len(returned_ids)
            total_items += count
            # 遍历存入字典, 不产生搬运
            for i in range(count):
                gid = returned_ids[i]
                sample_pack = {
                    col: shard_data[col][i]
                    for col in experience_columns
                    if col in shard_data
                }
                id_to_data_map[gid] = sample_pack

        if total_items == 0:
            return None, None

        # 根据输入的 indexes 顺序生成最终列表
        final_indexes = []
        final_columns = {col: [] for col in experience_columns}

        # 未提供 index 时, 根据返回结果排序构建最终列表
        if indexes is None:
            target_iterator = sorted(id_to_data_map.keys())
        else:
            target_iterator = indexes

        for target_id in target_iterator:
            if target_id in id_to_data_map:
                final_indexes.append(target_id)
                sample = id_to_data_map[target_id]
                for col in experience_columns:
                    final_columns[col].append(sample[col])
            else:
                continue

        # 6. Logging
        end_time = time.perf_counter()
        total_latency = (end_time - start_time) * 1000

        # 异步发送监控埋点，不阻塞主流程
        self.manager.accumulate_timing.remote("get", float(total_latency / 1000.0))

        self.logger.info(
            f"[TransferQueue Get] Topic: {topic} | Items: {total_items} | "
            f"Latency: {total_latency:.2f} ms | Shards: {successful_shards}"
        )

        return final_columns, final_indexes

    async def _fetch_one_shard(
            self,
            endpoint: str,
            topic: str,
            columns: List[str],
            indexes: List[int],
            copy=False,
    ) -> Dict[str, Any]:

        # 全局并发控制
        async with self.get_semaphore:
            # 获取该 Endpoint 的专属锁，防止并发写入导致 ZMQ 状态机混乱或串包
            lock = self._get_endpoint_lock(endpoint)
            async with lock:
                try:
                    sock = self._get_async_socket(endpoint)
                    req_header = {
                        "topic": topic,
                        "experience_columns": columns,
                        "indexes": indexes
                    }

                    await sock.send_multipart([b"GET", pickle.dumps(req_header)])

                    reply = await sock.recv_multipart(copy=False)

                    if not reply:
                        raise RuntimeError("Empty reply received")

                    meta_frame = reply[0]
                    if meta_frame.bytes.startswith(b"ERROR:"):
                        raise RuntimeError(f"Remote Error: {meta_frame.bytes.decode()}")

                    meta = pickle.loads(meta_frame.bytes)
                    returned_indexes = meta['indexes']
                    col_meta_map = meta['columns']

                    # 反序列化
                    result_data = {}
                    if len(reply) - 1 != len(columns):
                        raise RuntimeError(f"Column mismatch from {endpoint}. Exp {len(columns)}, Got {len(reply) - 1}")

                    for i, col_name in enumerate(meta['order']):
                        if i + 1 >= len(reply): break

                        frame = reply[i + 1]
                        col_info = col_meta_map.get(col_name)

                        if col_info:
                            tensors = deserialize_column_from_frame(
                                frame,
                                col_info['dtype'],
                                col_info['lengths'],
                                copy
                            )
                            result_data[col_name] = tensors
                        else:
                            result_data[col_name] = []

                    return {
                        "data": result_data,
                        "indexes": returned_indexes
                    }

                except Exception as e:
                    self.logger.error(f"Error fetching shard {endpoint}: {e}")
                    self._invalidate_socket(endpoint)
                    return None

    def _get_endpoint_lock(self, endpoint: str) -> asyncio.Lock:
        """
        获取针对特定 Endpoint 的协程锁，确保复用 Socket 时的 Send-Recv 原子性。
        """
        if not hasattr(self, "_endpoint_locks"):
            self._endpoint_locks = {}

        if endpoint not in self._endpoint_locks:
            self._endpoint_locks[endpoint] = asyncio.Lock()
        return self._endpoint_locks[endpoint]

    def _get_socket(self, endpoint: str) -> zmq.Socket:
        """Get or create a Dealer socket connected to the given endpoint."""
        # Use thread-local sockets because ZeroMQ sockets are not thread-safe.
        if not hasattr(self._local, "sockets"):
            self._local.sockets = {}
        tl_sockets: Dict[str, zmq.Socket] = self._local.sockets
        if endpoint in tl_sockets:
            return tl_sockets[endpoint]
        sock = self.zmq_context.socket(zmq.DEALER)
        sock.connect(endpoint)
        tl_sockets[endpoint] = sock
        return sock

    def add_topic(
            self,
            prompts_num: int,
            n_samples_per_prompt: int,
            experience_columns: List[str],
            experience_consumers: List[str],
            metrics: Metric,
            topic: str = DEFAULT_TOPIC,
    ) -> None:
        """
        Register a topic on the manager and provision per-shard storage.
        - Pure schema setup; no data is inserted here.
        - Raises if the topic already exists.
        """
        if not topic:
            raise ValueError("Topic must be non-empty")
        ray.get(self.manager.add_topic.remote(
            topic, prompts_num, n_samples_per_prompt, experience_columns, experience_consumers, metrics
        ))
        self.logger.info(
            f"Client: Created topic '{topic}' with n_samples_per_prompt={n_samples_per_prompt}, "
            f"experience_columns={experience_columns}, experience_consumers={experience_consumers}"
        )

    def delete_topic(self, topic: str = DEFAULT_TOPIC) -> None:
        """
        Delete the specified topic, including tables on the manager and all shards.
        """
        if topic is None:
            topic = DEFAULT_TOPIC
        ray.get(self.manager.delete_topic.remote(topic))
        self.logger.info(f"Client: Deleted topic '{topic}'")

    def clear_topic(self, topic: str = DEFAULT_TOPIC):
        """Clear data the specified topic across all shards, and reset its manager-side states."""
        if topic is None:
            topic = DEFAULT_TOPIC
        ray.get(self.manager.clear_topic.remote(topic))
        self.logger.info(f"Client: Cleared data for topic '{topic}'.")

    def prune_topic_by_indexes(self, topic: str = DEFAULT_TOPIC, indexes: List[int] = None):
        """Prune the specified topic index across all shards and reset its tracking state."""
        if topic is None:
            topic = DEFAULT_TOPIC
        if indexes is None:
            raise ValueError("Indexes must be provided for prune_topic_by_indexes")
        ray.get(self.manager.prune_topic_by_indexes.remote(topic, indexes))
        self.logger.info(f"Client: Pruned data for indexes {indexes} on topic '{topic}'.")

    def clear_and_resize_topic(self, prompts_num: int, n_samples_per_prompt: int, topic: str = DEFAULT_TOPIC) -> None:
        """
        Clear and resize the specified topic. Other parameters (column names, consumers, metrics) are preserved.
        """
        if topic is None:
            topic = DEFAULT_TOPIC
        ray.get(self.manager.clear_and_resize_topic.remote(
            topic, prompts_num, n_samples_per_prompt
        ))
        self.logger.info(f"Client: Resized topic '{topic}' to n_samples_per_prompt={n_samples_per_prompt}")

    def put_padded_prompts_experience(
            self,
            batch: Dict[str, Union[Tensor, List[Tensor]]],
            dataset_additional_keys: List[str] = None,
            pad_id: int = None,
            topic: str = DEFAULT_TOPIC,
    ):
        """
        Distribute padded prompts data to DATA shards.

        Args:
            topic: Name of the topic to which this data belongs.
            batch: A dict of padded tensors, must include 'prompts'.
            dataset_additional_keys: Other column names excepts 'prompts' and 'prompts_length' to stored in td.
            pad_id: The padding token ID used in 'prompts'.

        Procedure:
        1. Compute true lengths of each prompt by removing padding via get_no_pad_length.
        2. Build a data_dict of 'prompts' and 'prompt_length', and any additional keys.
        3. Call unpad_data_dict to truncate padded sequences back to their original lengths.
        4. Convert the unpadded data_dict to experience_columns and experience entries.
        5. Fetch shard endpoints and counts, then split the experience list into slices per shard.
        6. Send each shard its corresponding slice over ZMQ using PUT_PROMPT messages.
        """
        if topic is None:
            topic = DEFAULT_TOPIC
        start_time = time.perf_counter()
        # 1. Compute unpadded prompt lengths
        prompts_tensor = batch["prompts"]
        prompt_lengths = get_no_pad_length(prompts_tensor, pad_id)

        n_prompts = len(prompts_tensor)
        data_dict = {
            "prompts": prompts_tensor,
            "prompt_length": prompt_lengths,
        }

        if dataset_additional_keys:
            for key in dataset_additional_keys:
                if key in batch:
                    data_dict[key] = batch[key]

        next_group_id = ray.get(self.manager.get_next_group_id.remote(topic))

        indexes = list(range(next_group_id, next_group_id + n_prompts))

        self.put_experience(
            data_dict=data_dict,
            indexes=indexes,
            unpad_pairs=[("prompts", "prompt_length", "right_pad")],
            topic=topic,
            is_prompt=True
        )

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        ray.get(self.manager.accumulate_timing.remote("put_prompt", float(elapsed)))

        n_samples_per_prompt = ray.get(self.manager.get_n_samples_per_prompt.remote(topic))
        tot_samples = n_prompts * n_samples_per_prompt
        return list(range(next_group_id, next_group_id + tot_samples))

    def all_consumed(self, consumer: str, topic: str = DEFAULT_TOPIC, indexes: List[int] = None) -> bool:
        """
        Check if the given consumer has consumed all data.
        """
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.all_consumed.remote(topic=topic, consumer=consumer, indexes=indexes))

    def all_updated(self, column: str, topic: str = DEFAULT_TOPIC, indexes: List[int] = None) -> bool:
        """
        Check if all data in the specified column has been updated (filled).
        """
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.all_updated.remote(topic=topic, column=column, indexes=indexes))

    def reset_all(self):
        """Fully reset back to post-__init__ state: drop all topics/tables and reset manager state."""
        # NOTE: this removes schemas entirely; add_topic must be called again afterwards.
        ray.get(self.manager.reset_all.remote())
        self.logger.info("Client: Fully reset TransferQueue to post-__init__ state (ALL topics dropped).")

    def get_metrics(self, topic: str = DEFAULT_TOPIC):
        """Fetch metrics object for a topic."""
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.get_metrics.remote(topic))

    def update_metrics(self, key: str = "", value=None, cumulate: bool = False, topic: str = DEFAULT_TOPIC):
        """Update metrics for a topic."""
        if topic is None:
            topic = DEFAULT_TOPIC
        ray.get(self.manager.update_metrics.remote(
            topic, key, value, cumulate=cumulate
        ))

    def create_timing_item(self, name: str) -> None:
        """Create (ensure) a timing item on the manager."""
        ray.get(self.manager.create_timing_item.remote(name))

    def accumulate_timing(self, name: str, seconds: float) -> None:
        """Accumulate elapsed seconds into a timing item on the manager."""
        ray.get(self.manager.accumulate_timing.remote(name, float(seconds)))

    def get_timing(self, name: str) -> float:
        """Return a single timing item (seconds)."""
        return ray.get(self.manager.get_timing.remote(name))

    def get_timings(self) -> Dict[str, float]:
        """Return all timing items (a dict of name -> seconds)."""
        return ray.get(self.manager.get_timings.remote())

    def reset_timings(self) -> None:
        """Reset all timing items to zero."""
        ray.get(self.manager.reset_timings.remote())

    # def get_max_len(self, topic: str = DEFAULT_TOPIC) -> int:
    #     if topic is None:
    #         topic = DEFAULT_TOPIC
    #     return ray.get(self.manager.get_max_len.remote(topic))

    def set_reward_tags_to_indexes(self, reward_tags: Union[Tensor, List[int]], indexes: List[int],
                                   topic: str = DEFAULT_TOPIC) -> None:
        """
        Set the reward_tags_to_indexes on the manager for the given topic.
        """
        if topic is None:
            topic = DEFAULT_TOPIC
        ray.get(self.manager.set_reward_tags_to_indexes.remote(topic, reward_tags, indexes))

    def get_allocation_for_new_groups(self, topic: str, num_new_groups: int) -> List[List[int]]:
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.get_allocation_for_new_groups.remote(topic, num_new_groups))

    def record_versions(self, topic: str, version: int, indexes: List[int]):
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.record_versions.remote(topic, version, indexes))

    def get_versions_by_index(self, topic: str, index: int) -> int:
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.get_versions_by_index.remote(topic, index))

    def reset_versions(self, topic: str, version: int, indexes: List[int]):
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.reset_versions.remote(topic, version, indexes))

    def clear_data_by_version(self, topic: str, version: int):
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.clear_data_by_version.remote(topic, version))

    def clear_data_by_staleness(self, topic: str, allowed_staleness: int, latest_version: int):
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.clear_data_by_staleness.remote(topic, allowed_staleness, latest_version))


def get_transferqueue_client(name: str = "TransferQueueManager") -> TransferQueueClient:
    """
    Get a new Client instance connected to the named TransferQueueManager actor.
    This uses Ray's global name registry to locate the manager.
    """
    if name is None:
        name = "TransferQueueManager"
    mgr = ray.get_actor(name)
    return TransferQueueClient(mgr)
