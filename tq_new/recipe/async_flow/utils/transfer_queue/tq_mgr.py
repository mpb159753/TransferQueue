# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import asyncio
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import ray
import zmq
import zmq.asyncio

from recipe.async_flow.utils.transfer_queue.tq_config import (
    GROUP_SHARED_COLUMNS,
    TransferQueueAblationConfig,
    normalize_ablation_config,
)
from recipe.async_flow.utils.transfer_queue.tq_data import TransferQueueShard
from recipe.async_flow.utils.transfer_queue.tq_sampler import BaseSampler
from recipe.async_flow.utils.transfer_queue.tq_utils import (
    decode_rpc_request,
    encode_rpc_reply_err,
    encode_rpc_reply_ok,
    setup_logger,
)
from recipe.async_flow.utils.transfer_queue.viztracer_tools import TraceMarker, VizTracerProfileSession


@dataclass()
class TopicMeta:
    nums_tq_data: int
    prompts_num: int
    n_samples_per_prompt: int
    num_samples_gbs: int = field(init=False)

    experience_columns: list[str]
    experience_consumers: list[str]
    experience_ready: dict[str, set[int]] = field(init=False)
    experience_consumed: dict[str, set[int]] = field(init=False)
    consumer_sampling_lock: dict[str, threading.Lock] = field(init=False)
    # assign_new_index_lock: threading.Lock = threading.Lock()
    # assign_new_index_lock = asyncio.Lock()
    # assign_new_index_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    assign_new_index_lock: asyncio.Lock = field(init=False)
    # assign_new_group_lock: threading.Lock = threading.Lock()
    versions_lock: threading.Lock = threading.Lock()
    max_cur_index: int = -1  # The maximum index value currently allocated, -1 means no index has been allocated yet

    global_idx_to_shard: dict[int, int] = field(default_factory=dict)
    group_ids_per_shard: list[set[int]] = field(init=False)
    group_shared_col_written: dict[str, set[int]] = field(default_factory=dict)
    group_index_sets: dict[int, set[int]] = field(default_factory=dict)
    uuid_to_group: dict[str, int] = field(default_factory=dict)
    # 【新增】反向映射：group_id -> Set[uuid]
    # 约束：每个 uuid 对应一个 group_id，但一个 group_id 可能有多个 uuid
    group_to_uuids: dict[int, set[str]] = field(default_factory=dict)
    versions_to_indexes: dict[int, set[int]] = field(default_factory=dict)

    def __post_init__(self):
        if self.nums_tq_data <= 0:
            raise ValueError("nums_tq_data must be > 0")
        self.experience_ready = {col: set() for col in self.experience_columns}
        self.experience_consumed = {col: set() for col in self.experience_consumers}
        self.group_ids_per_shard = [set() for _ in range(self.nums_tq_data)]
        self.num_samples_gbs = self.prompts_num * self.n_samples_per_prompt
        self.consumer_sampling_lock = {key: threading.Lock() for key in self.experience_consumers}
        self.assign_new_index_lock = asyncio.Lock()


@ray.remote(max_concurrency=100, num_cpus=10, name="TransferQueueManager")
class TransferQueueManager:
    def __init__(
        self,
        nums_tq_data: int = 1,
        base_port: Optional[int] = None,
        ablation: TransferQueueAblationConfig | Mapping[str, Any] | None = None,
    ) -> None:
        """
        Manager initialization.
        - nums_tq_data: Number of DATA shards to create.
        - base_port: Starting port number for DATA servers. If None, shards bind to a random available port.
        """
        self.logger = setup_logger("TransferQueueManager")
        self.ablation = normalize_ablation_config(ablation)

        self.nums_tq_data = nums_tq_data
        # if(self.nums_tq_data>8):
        #     raise ValueError("Not support DATA Shard nums larger than 8.")

        self.topics: dict[str, TopicMeta] = {}

        self.data_actors = []  # Ray actor handles for DATA shards
        self.data_endpoints = []  # ZMQ endpoints (addresses) for each shard

        # TODO: 以下是检测所有Ray结点，并依此尽量分散放置DATA Shard来避免DATA Shard都在同一个物理节点上，该方案并不优雅
        all_nodes = [node for node in ray.nodes() if node["Alive"]]
        target_node_ips = [node["NodeManagerAddress"] for node in all_nodes if node["Resources"].get("CPU", 0) >= 1]
        if not target_node_ips:
            raise RuntimeError("No available Ray nodes with CPU resources.")

        self.zmq_context = zmq.asyncio.Context()
        self.router = self.zmq_context.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        rpc_port = self.router.bind_to_random_port("tcp://0.0.0.0")
        self.endpoint = f"tcp://{ray.util.get_node_ip_address()}:{rpc_port}"
        self.zmq_pool = ThreadPoolExecutor(max_workers=100, thread_name_prefix="tq_mgr_rpc")

        self._rpc_methods = {
            "allocate_batches": self.allocate_batches,
            "allocate_by_uuid": self.allocate_by_uuid,
            "get_targets_for_put": self.get_targets_for_put,
            "batch_mark_columns_written": self.batch_mark_columns_written,
            "record_versions": self.record_versions,
            "allocate_shard_for_indexes": self.allocate_shard_for_indexes,
            "allocate_shard_and_indexes": self.allocate_shard_and_indexes,
            "update_data_status": self.update_data_status,
            "allocate_sequentially": self.allocate_sequentially,
        }
        self._running = True
        self._zmq_task = asyncio.get_event_loop().create_task(self._serve_loop())
        self.logger.info(f"Manager: RPC server at {self.endpoint}")
        if self.ablation.active_flags():
            self.logger.info("Manager ablations enabled: %s", self.ablation.label())

        for i in range(nums_tq_data):
            # Round-robin allocate DATA Shard
            node_ip = target_node_ips[i % len(target_node_ips)]
            data_actor = TransferQueueShard.options(
                num_cpus=1, resources={f"node:{node_ip}": 0.01}, name=f"TransferQueueShard_{i}"
            ).remote(
                i,
                self.endpoint,
            )
            self.data_actors.append(data_actor)

        # Fetch endpoints concurrently to avoid serial waits.
        endpoint_refs = [a.get_endpoint.remote() for a in self.data_actors]
        self.data_endpoints = ray.get(endpoint_refs)
        self._profile_session: VizTracerProfileSession | None = None

        self.logger.info(f"Manager: Initialized {nums_tq_data} data shards with endpoints: {self.data_endpoints}")

    def init_ready(self):
        return True

    def configure_profile(
        self,
        output_dir: str,
        enabled: bool = False,
        tracer_entries: int = 1_000_000,
        min_duration_us: int = 0,
    ) -> dict[str, list[str] | str]:
        self._profile_session = VizTracerProfileSession(
            output_dir=output_dir,
            component_name="manager",
            enabled=enabled,
            tracer_entries=tracer_entries,
            min_duration_us=min_duration_us,
            log_async=True,
        )
        shard_component_names = ray.get(
            [
                actor.configure_profile.remote(
                    output_dir=output_dir,
                    component_name=f"shard_{shard_id:02d}",
                    enabled=enabled,
                    tracer_entries=tracer_entries,
                    min_duration_us=min_duration_us,
                )
                for shard_id, actor in enumerate(self.data_actors)
            ]
        )
        return {
            "manager": "manager",
            "shards": shard_component_names,
        }

    def start_profile(self, round_number: int) -> dict[str, list[str | None] | str | None]:
        manager_output = None
        if self._profile_session is not None:
            manager_path = self._profile_session.start(round_number)
            manager_output = str(manager_path) if manager_path is not None else None
        shard_outputs = ray.get([actor.start_profile.remote(round_number) for actor in self.data_actors])
        return {
            "manager": manager_output,
            "shards": shard_outputs,
        }

    def stop_profile(self) -> dict[str, list[str | None] | str | None]:
        manager_output = None
        if self._profile_session is not None:
            manager_path = self._profile_session.stop()
            manager_output = str(manager_path) if manager_path is not None else None
        shard_outputs = ray.get([actor.stop_profile.remote() for actor in self.data_actors])
        return {
            "manager": manager_output,
            "shards": shard_outputs,
        }

    def get_manager_endpoint(self) -> str:
        return self.endpoint

    async def _serve_loop(self):
        """
        核心事件循环：持续监听并分发任务。
        """
        while self._running:
            try:
                frames = await self.router.recv_multipart(copy=False)
                asyncio.create_task(self._process_message(frames))
            except zmq.ZMQError as e:
                self.logger.error(f"Manager RPC: ZMQ error in serve loop: {e}")
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.exception(f"Manager RPC: unexpected error in serve loop: {e}")
                await asyncio.sleep(0.1)

    async def _process_message(self, frames):
        """
        协议解析与路由分发:
          REQ : [identity, method_name, msgpack_payload, *pickle_side_frames]
          RESP: [identity, msgpack({"ok": True/False, "result"/"error": ...})]
        """
        if len(frames) < 3:
            self.logger.error(f"Manager RPC: malformed request, frames={len(frames)}")
            return
        identity = frames[0]
        method_name = frames[1].bytes.decode("utf-8", errors="replace")
        try:
            method = self._rpc_methods.get(method_name)

            loop = asyncio.get_event_loop()
            if method is None:
                raise ValueError(f"Unknown method {method_name!r}")
            kwargs = decode_rpc_request(frames[2].bytes, frames[3:])
            if inspect.iscoroutinefunction(method):
                result = await method(**kwargs)
            else:
                result = await loop.run_in_executor(self.zmq_pool, lambda: method(**kwargs))
            # self.logger.info(f"Manager RPC: received request: {method_name}")
            # result = await loop.run_in_executor(
            #     self.zmq_pool, lambda: method(**kwargs)
            # )
            # self.logger.info(f"Manager Debug: {type(result)}")
            reply = encode_rpc_reply_ok(result)
        except Exception as e:
            self.logger.exception(f"Manager RPC: error processing {method_name!r}")
            reply = encode_rpc_reply_err(e)
        try:
            await self.router.send_multipart([identity, reply])
        except Exception as send_err:
            self.logger.error(f"Manager RPC: failed to send reply: {send_err}")

    def add_topic(
        self,
        topic: str,
        prompts_num: int,
        n_samples_per_prompt: int,
        experience_columns: list[str],
        experience_consumers: list[str],
    ) -> None:
        """
        Register a topic and create storage tables on all shards.
        - Pure schema setup; no data is inserted here.
        - Will raise if the topic already exists.
        """
        if topic in self.topics:
            raise ValueError(f"Topic '{topic}' already exists")
        meta = TopicMeta(
            prompts_num=prompts_num,
            n_samples_per_prompt=n_samples_per_prompt,
            nums_tq_data=self.nums_tq_data,
            experience_columns=experience_columns,
            experience_consumers=experience_consumers,
        )
        self.topics[topic] = meta
        # for actor in self.data_actors:
        #     ray.get(actor.add_experience_table.remote(
        #         topic=topic,
        #         n_samples_per_prompt=meta.n_samples_per_prompt,
        #         experience_columns=meta.experience_columns,
        #     ))

        # 修改后：并行执行 RPC
        # 1. 批量提交异步任务，获取所有引用 (ObjectRefs)
        refs = [
            actor.add_experience_table.remote(
                topic=topic,
                n_samples_per_prompt=meta.n_samples_per_prompt,
                experience_columns=meta.experience_columns,
            )
            for actor in self.data_actors
        ]

        # 2. 一次性阻塞等待所有任务完成（并行等待）
        ray.get(refs)

        # self.logger.info(f"Manager: Added topic '{topic}' across {self.nums_tq_data} shards ")

    def delete_topic(self, topic: str) -> None:
        """
        Delete the specified topic by removing its tables from all shards and clearing metadata.
        """
        if topic not in self.topics:
            self.logger.info(f"Manager: Unknown topic '{topic}'")
            return
        # Remove tables from each data shard
        # for actor in self.data_actors:
        #     ray.get(actor.remove_experience_table.remote(topic))

        # 修改后：并行执行删除
        # 1. 立即触发所有 Actor 的删除任务
        delete_refs = [actor.remove_experience_table.remote(topic) for actor in self.data_actors]

        # 2. 批量等待所有节点删除完成
        ray.get(delete_refs)

        # Remove metadata
        del self.topics[topic]
        self.logger.info(f"Manager: Deleted topic '{topic}'")

    def reset_all(self):
        """
        Fully reset to post-__init__ state:
        - Instruct all shards to drop ALL topic tables.
        - Remove ALL topics metadata.
        """
        # Drop all per-topic tables in every shard (after this, shards have no topics).
        for actor in self.data_actors:
            ray.get(actor.reset_all.remote())
        # Remove manager-side topics.
        self.topics = {}
        self.logger.info("Manager: Fully reset to post-__init__ state (no topics present).")

    def shutdown(self):
        """
        Terminate all DATA Shard actors (optional cleanup).
        """
        for actor in self.data_actors:
            ray.kill(actor)
        self.data_actors = []
        self.data_endpoints = []
        self.logger.info("Manager: All data shards have been shut down.")

    def get_targets_for_put(
        self,
        topic: str,
        indexes: list[int],
    ) -> dict[str, list[int]]:
        """
        Determine which DATA shard(s) should handle the given global indexes
        for a put_experience operation, and group indexes by shard endpoint.
        """
        with TraceMarker.scope("route_targets", topic=topic, indexes=len(indexes)):
            # Validate indexes and topic
            if not indexes:
                raise ValueError("No indexes provided for get_targets_for_put.")
            if topic not in self.topics:
                raise ValueError(f"Unknown topic '{topic}'")

            # Prepare mapping from endpoint -> list of indexes
            meta = self.topics[topic]
            invalid_global_idxs = set(indexes) - meta.global_idx_to_shard.keys()
            if invalid_global_idxs:
                raise ValueError(f"Global indexes {invalid_global_idxs} does not belong to any shard.")

            # Prefetch references to reduce attribute lookups inside the loop.
            endpoint_map = {ep: [] for ep in self.data_endpoints}
            global_idx_to_shard = meta.global_idx_to_shard
            endpoints = self.data_endpoints
            for idx in indexes:
                shard_id = global_idx_to_shard[idx]
                endpoint_map[endpoints[shard_id]].append(idx)

            # Prune empty entries
            return {ep: idxs for ep, idxs in endpoint_map.items() if idxs}

    def allocate_shard_for_indexes(
        self,
        topic: str,
        consumer: str,
        experience_columns: list[str],
        indexes: list[int],
        allowed_staleness: int = None,
        latest_version: int = None,
    ) -> Optional[dict[str, list[int]]]:
        """
        Return ready indexes of the given indexes, and group them by endpoint.
        """
        with TraceMarker.scope("fetch_targets", topic=topic, indexes=len(indexes or []), mode="indexes"):
            # 1. Validate topic, consumer, indexes
            if topic not in self.topics:
                raise ValueError(f"Unknown topic '{topic}'")
            meta = self.topics[topic]
            if consumer not in meta.experience_consumers:
                raise ValueError(f"Unregistered consumers: '{consumer}'.")
            if not indexes:
                raise ValueError("indexes must be provided when experience_count are not specified.")
            if (latest_version is None) != (allowed_staleness is None):
                raise ValueError(
                    "tq_mgr/allocate_shard_for_indexes: Parameters latest_version and allowed_staleness must be provided together or not at all."
                )

            # 2. Sample ready indexes
            with meta.consumer_sampling_lock[consumer]:
                ready_indexes = set(indexes).intersection(*(meta.experience_ready[col] for col in experience_columns))
                usable_indexes = ready_indexes - meta.experience_consumed[consumer]

                self.logger.debug(f"[Sample Debug] Step 1 (Ready Intersection): count={len(ready_indexes)}")
                self.logger.debug(
                    f"[Sample Debug] Step 2 (Consumed Filter): consumed_total={len(meta.experience_consumed[consumer])}, remaining={len(usable_indexes)}"
                )

                if latest_version and allowed_staleness:
                    with meta.versions_lock:
                        fresh_indexes = set().union(
                            *(
                                meta.versions_to_indexes[v]
                                for v in range(latest_version, latest_version - allowed_staleness - 1, -1)
                                if v in meta.versions_to_indexes
                            )
                        )
                        usable_indexes &= fresh_indexes
                        for v in meta.versions_to_indexes:
                            self.logger.debug(f"[Sample Debug] Step 3a version{v}: {v}")
                        self.logger.debug(
                            f"[Sample Debug] Step 3b (Version Pool): range=[{latest_version}-{latest_version - allowed_staleness - 1}], fresh_pool_size={len(fresh_indexes)}"
                        )
                        self.logger.debug(
                            f"[Sample Debug] Step 3c (Version Intersect): remaining={len(usable_indexes)}"
                        )

                self.logger.debug(
                    f"[Sample Debug] Step 4 (Indexes-provided): raw_indexes={len(usable_indexes)}, req_indexes={indexes}"
                )

                if len(usable_indexes) < len(indexes):
                    return None

                usable_indexes = sorted(usable_indexes)
                meta.experience_consumed[consumer].update(usable_indexes)

            # 3. Group by shard: prepare mapping from endpoint -> list of indexes
            unmatched = [idx for idx in usable_indexes if idx not in meta.global_idx_to_shard]
            if unmatched:
                raise ValueError(f"Global indexes {unmatched} does not belong to any shard.")
            endpoint_map: dict[str, list[int]] = {ep: [] for ep in self.data_endpoints}
            for idx in usable_indexes:
                shard_id = meta.global_idx_to_shard[idx]
                endpoint_map[self.data_endpoints[shard_id]].append(idx)

            # Prune empty entries
            return {ep: idxs for ep, idxs in endpoint_map.items() if idxs}

    def allocate_shard_and_indexes(
        self,
        topic: str,
        consumer: str,
        experience_columns: list[str],
        experience_count: int,
        get_n_samples: bool,
        allowed_staleness: int = None,
        latest_version: int = None,
        sampler_func: Optional[BaseSampler] = None,
    ) -> Optional[dict[str, list[int]]]:
        """
        Allocate a set of global indexes for a consumer and group them by shard endpoint.

        Steps:
        1. Validate the consumer and sampling parameters.
        2. Sample ready global indexes (either in multiples of n_samples_per_prompt or freely).
        3. Map each chosen index to its owning shard via shard_sample_offsets.
        4. Group indexes by the shard's ZMQ endpoint.
        5. Return a dict: { endpoint_str: [global_idx, ...], ... }.

        Returns:
            A dict mapping each shard endpoint to the list of global indexes
            that consumer should fetch from that shard. Returns None if no
            indexes are available.
        """
        with TraceMarker.scope("fetch_targets", topic=topic, count=experience_count or 0, mode="sample"):
            # 1. Validation
            if topic not in self.topics:
                raise ValueError(f"Unknown topic '{topic}'")
            meta = self.topics[topic]
            if consumer not in meta.experience_consumers:
                raise ValueError(f"Consumer '{consumer}' not recognized.")
            if experience_count is None or experience_count <= 0:
                raise ValueError("experience_count must be a positive integer.")
            if get_n_samples and (experience_count % meta.n_samples_per_prompt != 0):
                raise ValueError(
                    f"get_n_samples=True requires experience_count ({experience_count}) "
                    f"to be divisible by n_samples_per_prompt ({meta.n_samples_per_prompt})."
                )
            if (latest_version is None) != (allowed_staleness is None):
                raise ValueError(
                    "tq_mgr/allocate_shard_and_indexes: Parameters latest_version and allowed_staleness must be provided together or not at all."
                )

            # 2. Sample ready indexes
            if get_n_samples:
                chosen = self._sample_ready_index_n_samples(
                    topic,
                    consumer,
                    experience_count,
                    experience_columns,
                    allowed_staleness=allowed_staleness,
                    latest_version=latest_version,
                    sampler_func=sampler_func,
                )
            else:
                chosen = self._sample_ready_index(
                    topic,
                    consumer,
                    experience_count,
                    experience_columns,
                    allowed_staleness=allowed_staleness,
                    latest_version=latest_version,
                    sampler_func=sampler_func,
                )
            if not chosen:
                return None

            # 3. Group by shard: prepare mapping from endpoint -> list of indexes
            unmatched = [idx for idx in chosen if idx not in meta.global_idx_to_shard]
            if unmatched:
                raise ValueError(f"Global indexes {unmatched} does not belong to any shard.")
            endpoint_map: dict[str, list[int]] = {ep: [] for ep in self.data_endpoints}
            for idx in chosen:
                shard_id = meta.global_idx_to_shard[idx]
                endpoint_map[self.data_endpoints[shard_id]].append(idx)

            # Prune empty entries
            return {ep: idxs for ep, idxs in endpoint_map.items() if idxs}

    def _sample_ready_index(
        self,
        topic,
        consumer: str,
        experience_count: int,
        experience_columns: list[str],
        allowed_staleness: int = None,
        latest_version: int = None,
        sampler_func: Optional[BaseSampler] = None,
    ) -> Optional[list[int]]:
        meta = self.topics[topic]

        with meta.consumer_sampling_lock[consumer]:
            ready_indexes = set.intersection(*(meta.experience_ready[col] for col in experience_columns))
            usable_indexes = ready_indexes - meta.experience_consumed[consumer]

            self.logger.debug(f"[Sample Debug] Step 1 (Ready Intersection): count={len(ready_indexes)}")
            self.logger.debug(
                f"[Sample Debug] Step 2 (Consumed Filter): consumed_total={len(meta.experience_consumed[consumer])}, remaining={len(usable_indexes)}"
            )

            if latest_version and allowed_staleness:
                with meta.versions_lock:
                    fresh_indexes = set().union(
                        *(
                            meta.versions_to_indexes[v]
                            for v in range(latest_version, latest_version - allowed_staleness - 1, -1)
                            if v in meta.versions_to_indexes
                        )
                    )
                    usable_indexes &= fresh_indexes
                    for v in meta.versions_to_indexes:
                        self.logger.debug(f"[Sample Debug] Step 3a version{v}: {v}")
                    self.logger.debug(
                        f"[Sample Debug] Step 3b (Version Pool): range=[{latest_version}-{latest_version - allowed_staleness - 1}], fresh_pool_size={len(fresh_indexes)}"
                    )
                    self.logger.debug(f"[Sample Debug] Step 3c (Version Intersect): remaining={len(usable_indexes)}")

            self.logger.debug(
                f"[Sample Debug] Step 4 (Non-grouping): raw_indexes={len(usable_indexes)}, req_count={experience_count}"
            )

            usable_indexes = sorted(usable_indexes)
            if len(usable_indexes) < experience_count:
                self.logger.debug(
                    f"[Sample Debug] Step 5 FAILED: Insufficient indexes. Need {experience_count}, have {len(usable_indexes)}"
                )
                return None

            if sampler_func is None:
                sampled_indexes = usable_indexes[:experience_count]
            else:
                sample_args = {"indexes": usable_indexes, "count": experience_count}
                if "versions" in inspect.signature(sampler_func.sample).parameters:
                    idx_to_version = {idx: v for v, idxs in meta.versions_to_indexes.items() for idx in idxs}
                    versions = [idx_to_version.get(idx) for idx in usable_indexes]
                    sample_args["versions"] = versions
                sampled_indexes = sampler_func.sample(**sample_args)

            meta.experience_consumed[consumer].update(sampled_indexes)

        return sampled_indexes

    def _sample_ready_index_n_samples(
        self,
        topic,
        consumer: str,
        experience_count: int,
        experience_columns: list[str],
        allowed_staleness: int = None,
        latest_version: int = None,
        sampler_func: Optional[BaseSampler] = None,
    ) -> Optional[list[int]]:
        meta = self.topics[topic]
        experience_count_n_samples = experience_count // meta.n_samples_per_prompt

        with meta.consumer_sampling_lock[consumer]:
            ready_indexes = set.intersection(*(meta.experience_ready[col] for col in experience_columns))
            usable_indexes = ready_indexes - meta.experience_consumed[consumer]

            self.logger.debug(f"[Sample Debug] Step 1 (Ready Intersection): count={len(ready_indexes)}")
            self.logger.debug(
                f"[Sample Debug] Step 2 (Consumed Filter): consumed_total={len(meta.experience_consumed[consumer])}, remaining={len(usable_indexes)}"
            )

            if latest_version and allowed_staleness:
                with meta.versions_lock:
                    fresh_indexes = set().union(
                        *(
                            meta.versions_to_indexes[v]
                            for v in range(latest_version, latest_version - allowed_staleness - 1, -1)
                            if v in meta.versions_to_indexes
                        )
                    )
                    usable_indexes &= fresh_indexes
                    for v in meta.versions_to_indexes:
                        self.logger.debug(f"[Sample Debug] Step 3a version{v}: {v}")
                    self.logger.debug(
                        f"[Sample Debug] Step 3b (Version Pool): range=[{latest_version}-{latest_version - allowed_staleness - 1}], fresh_pool_size={len(fresh_indexes)}"
                    )
                    self.logger.debug(f"[Sample Debug] Step 3c (Version Intersect): remaining={len(usable_indexes)}")

            usable_indexes = sorted(usable_indexes)
            groups = {}
            for idx in usable_indexes:
                groups.setdefault(idx // meta.n_samples_per_prompt, []).append(idx)
            usable_groups = []
            for group_id, sub_indexes in groups.items():
                if len(set(sub_indexes)) == meta.n_samples_per_prompt:
                    usable_groups.append(group_id)
            self.logger.debug(
                f"[Sample Debug] Step 4 (Grouping): raw_indexes={len(usable_indexes)}, valid_groups={len(usable_groups)}, req_groups={experience_count_n_samples}"
            )

            if len(usable_groups) < experience_count_n_samples:
                self.logger.debug(
                    f"[Sample Debug] Step 5 FAILED: Insufficient groups. Need {experience_count_n_samples}, have {len(usable_groups)}"
                )
                return None

            if sampler_func is None:
                sampled_indexes_n_sample = usable_groups[:experience_count_n_samples]
            else:
                sample_args = {"indexes": usable_groups, "count": experience_count_n_samples}
                if "versions" in inspect.signature(sampler_func.sample).parameters:
                    idx_to_version = {idx: v for v, idxs in meta.versions_to_indexes.items() for idx in idxs}
                    versions = [idx_to_version.get(idx * meta.n_samples_per_prompt) for idx in usable_groups]
                    sample_args["versions"] = versions
                sampled_indexes_n_sample = sampler_func.sample(**sample_args)

            sampled_indexes = []
            for n_sample_index in sampled_indexes_n_sample:
                index_list = []
                for index in range(
                    n_sample_index * meta.n_samples_per_prompt, (n_sample_index + 1) * meta.n_samples_per_prompt
                ):
                    index_list.append(index)

                sampled_indexes += index_list

            meta.experience_consumed[consumer].update(sampled_indexes)
        return sampled_indexes

    def delete_experience(
        self,
        topic: str,
        indexes: list[int] = None,
        versions: list[int] = None,
        latest_version: int = None,
        allowed_staleness: int = None,
        delete_all: bool = False,
    ):
        """
        Clear specified topic indexes across all shards and reset thier tracking status.

        Args:
            indexes: List of indexes to delete
            versions: List of versions to delete
            latest_version: Latest version number
            allowed_staleness: Allowed staleness
            delete_all: Whether to delete all data (requires indexes=None)
            topic: Topic name
        """
        # 1. Validation
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        # Check that exactly one mode is selected
        if (latest_version is None) != (allowed_staleness is None):
            raise ValueError(
                "tq_mgr/delete_experience: Parameters latest_version and allowed_staleness must be provided together or not at all."
            )
        modes = [indexes is not None, versions is not None, latest_version is not None, delete_all]
        if sum(modes) != 1:
            raise ValueError(
                "Must provide exactly one: 'indexes', 'versions', 'delete_all', or 'latest_version & allowed_staleness'."
            )

        # 2. Map versions to indexes
        meta = self.topics[topic]
        max_stored_index = max(meta.global_idx_to_shard.keys(), default=0)

        if delete_all:
            if max_stored_index > 0:
                indexes = list(range(max_stored_index + 1))
            else:
                self.logger.info(f"tq_mgr/delete_experience: No data found in topic '{topic}'")
                return

        if latest_version is not None:
            threshold = latest_version - allowed_staleness
            with meta.versions_lock:
                versions = [v for v in meta.versions_to_indexes.keys() if v < threshold]

        if versions is not None:
            with meta.versions_lock:
                indexes = [
                    idx for v in versions if v in meta.versions_to_indexes for idx in meta.versions_to_indexes[v]
                ]

        # 3. Delete data
        invalid_indexes = [idx for idx in indexes if idx > max_stored_index]
        if invalid_indexes:
            raise ValueError(
                f"Invalid indexes {invalid_indexes} exceed max stored index {max_stored_index}. "
                f"To delete all data, use delete_all=True with indexes=None."
            )

        self._prune_topic_by_indexes(topic, indexes)
        self.logger.debug(f"Cleared data from topic '{topic}' at indexes: {indexes}.")

    def _prune_topic_by_indexes(self, topic: str, indexes: list[int]):
        """
        Prune the specified topic index across all shards and reset its tracking state.
        """
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if not indexes:
            raise ValueError("No indexes provided for prune_topic_by_indexes.")

        meta = self.topics[topic]
        unmatched = [idx for idx in indexes if idx not in meta.global_idx_to_shard]
        if unmatched:
            raise ValueError(f"Global indexes {unmatched} does not belong to any shard.")

        for col in meta.experience_columns:
            meta.experience_ready[col] -= set(indexes)
        for col in meta.experience_consumers:
            meta.experience_consumed[col] -= set(indexes)

        # Classify the indexes by shard
        index_to_prune_by_shard = {}
        groups_may_be_removed = set()
        for idx in indexes:
            # Get shard_id and clean meta.global_idx_to_shard by 'pop'
            shard_id = meta.global_idx_to_shard.pop(idx)
            index_to_prune_by_shard.setdefault(shard_id, []).append(idx)
            groups_may_be_removed.add((idx // meta.n_samples_per_prompt, shard_id))

        # for shard_id, sub_indexes in index_to_prune_by_shard.items():
        #     # Prune the table by shard
        #     ray.get(self.data_actors[shard_id].prune_experience_table.remote(topic, sub_indexes))
        # 1. 异步发射：收集所有的远程调用句柄，而不直接 get
        futures = [
            self.data_actors[shard_id].prune_experience_table.remote(topic, sub_indexes)
            for shard_id, sub_indexes in index_to_prune_by_shard.items()
        ]
        # 2. 阻塞等待：一次性等待所有任务完成
        ray.get(futures)

        for group_id, shard_id in groups_may_be_removed:
            start = group_id * meta.n_samples_per_prompt
            end = start + meta.n_samples_per_prompt
            if all(idx not in meta.global_idx_to_shard for idx in range(start, end)):
                # Remove group_id if every sample of this group is already removed
                meta.group_ids_per_shard[shard_id].discard(group_id)
                # 清理 group_index_sets 中的 group
                if group_id in meta.group_index_sets:
                    del meta.group_index_sets[group_id]
                # 【优化】使用反向映射 O(1) 查找，而非遍历整个 uuid_to_group
                # 清理 uuid_to_group 中的映射
                if group_id in meta.group_to_uuids:
                    uuids_to_remove = meta.group_to_uuids[group_id]
                    for uuid in uuids_to_remove:
                        del meta.uuid_to_group[uuid]
                    del meta.group_to_uuids[group_id]  # 清理反向映射本身
                # 清理共享列写入状态
                for col_name in list(meta.group_shared_col_written.keys()):
                    meta.group_shared_col_written[col_name].discard(group_id)

                # 清理 uuid_to_group 中的反向映射
                # uuids_to_remove = [uuid for uuid, group_id in meta.uuid_to_group.items() if group_id == group_id]

                # 【修正/增量】这里原代码有 bug: group_id == group_id 永远为真，应修正为局部变量比较
                # 但按要求我不删除你的代码，这里仅做逻辑保留
                # TODO 检查这里的逻辑
                # uuids_to_remove = [uuid for uuid, g_id in meta.uuid_to_group.items() if g_id == group_id]
                # for uuid in uuids_to_remove:
                #     del meta.uuid_to_group[uuid]

        # 清理 group_index_sets 中的索引（对于部分删除的 group）
        for idx in indexes:
            group_id = idx // meta.n_samples_per_prompt
            if group_id in meta.group_index_sets:
                meta.group_index_sets[group_id].discard(idx)

        with meta.versions_lock:
            for v in list(meta.versions_to_indexes.keys()):
                meta.versions_to_indexes[v] -= set(indexes)
                if not meta.versions_to_indexes[v]:
                    del meta.versions_to_indexes[v]

        self.logger.debug(f"Manager: Indexes {indexes} on topic '{topic}' has been cleared.")

    def update_data_status(self, topic: str, indexes: list[int], columns: list[str]) -> None:
        """
        Update the data readiness status for given columns at specified indexes.
        This should be called by DATA shards after they finish storing new data.
        """
        if not indexes or not columns:
            return
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")

        with TraceMarker.scope("status_update", topic=topic, indexes=len(indexes), columns=len(columns)):
            # Update all given indexes in this column
            self.logger.debug(f"Updating {topic} - [{columns}] status at indexes {indexes}.")
            meta = self.topics[topic]
            for col in columns:
                if col in meta.experience_ready:
                    self.logger.debug(
                        f"Updating {topic} - [{col}], ready indexes before update: {meta.experience_ready[col]}."
                    )
                    meta.experience_ready[col].update(indexes)
                else:
                    self.logger.warning(f"update_data_status - [{col}] not in meta.experience_ready.")
        # Note: We do not alter consumer statuses here. Consumers will be marked upon get_experience.

    def update_consumer_status(self, topic: str, indexes: list[int], consumer: str) -> None:
        """
        Update the data consumption status for given consumer at specified indexes.
        This should be called by DATA shards after they finish fetching data.
        """
        if not indexes or not consumer:
            return
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")

        # Update all given indexes in this consumer
        meta = self.topics[topic]
        meta.experience_consumed[consumer].update(indexes)

    def record_versions(self, topic: str, version: int, indexes: list[int]) -> None:
        """为指定索引记录版本号。当索引被分配新版本时，会从其之前的版本中移除。"""
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if not indexes:
            raise ValueError("No indexes provided for record_versions.")

        with TraceMarker.scope("status_update", topic=topic, indexes=len(indexes), version=version):
            meta = self.topics[topic]
            indexes_set = set(indexes)
            with meta.versions_lock:
                # 从之前的版本中移除索引，并清理空的版本条目
                for v in list(meta.versions_to_indexes.keys()):
                    overlap = meta.versions_to_indexes[v] & indexes_set
                    if overlap:
                        meta.versions_to_indexes[v] -= overlap
                        if not meta.versions_to_indexes[v]:
                            del meta.versions_to_indexes[v]
                # 添加到新版本
                meta.versions_to_indexes.setdefault(version, set()).update(indexes_set)

    def get_data_ready_set(
        self,
        topic: str,
        experience_columns: list[str],
        indexes: list[int] = None,
        get_n_samples: bool = False,
    ) -> tuple[int, list[int]]:
        # Validate topic and column
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if not experience_columns:
            raise ValueError("No experience_columns provided.")
        for col in experience_columns:
            if col not in meta.experience_columns:
                raise ValueError(f"Unregistered column '{col}'.")

        ready_indexes = set.intersection(*(meta.experience_ready[col] for col in experience_columns))
        if indexes:
            ready_indexes &= set(indexes)
        ready_indexes = sorted(ready_indexes)

        if get_n_samples:
            groups = {}
            for idx in ready_indexes:
                groups.setdefault(idx // meta.n_samples_per_prompt, []).append(idx)

            ready_groups = []
            for group_id, sub_indexes in groups.items():
                if len(set(sub_indexes)) == meta.n_samples_per_prompt:
                    ready_groups.append(group_id)

            ready_indexes = []
            for group_id in ready_groups:
                for index in range(group_id * meta.n_samples_per_prompt, (group_id + 1) * meta.n_samples_per_prompt):
                    ready_indexes.append(index)

        return len(ready_indexes), ready_indexes

    def get_data_consumed_set(
        self,
        topic: str,
        consumer: str,
        indexes: list[int] = None,
        get_n_samples: bool = False,
    ) -> tuple[int, list[int]]:
        # Validate topic and consumer
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if consumer not in meta.experience_consumers:
            raise ValueError(f"Unregistered consumer '{consumer}'.")

        with meta.consumer_sampling_lock[consumer]:
            consumed_indexes = meta.experience_consumed[consumer]
            if indexes:
                consumed_indexes &= set(indexes)
            consumed_indexes = sorted(consumed_indexes)

            if get_n_samples:
                groups = {}
                for idx in consumed_indexes:
                    groups.setdefault(idx // meta.n_samples_per_prompt, []).append(idx)

                consumed_groups = []
                for group_id, sub_indexes in groups.items():
                    if len(set(sub_indexes)) == meta.n_samples_per_prompt:
                        consumed_groups.append(group_id)

                consumed_indexes = []
                for group_id in consumed_groups:
                    for index in range(
                        group_id * meta.n_samples_per_prompt, (group_id + 1) * meta.n_samples_per_prompt
                    ):
                        consumed_indexes.append(index)

            return len(consumed_indexes), consumed_indexes

    def get_data_usable_set(
        self,
        topic: str,
        consumer: str,
        experience_columns: list[str],
        allowed_staleness: int = None,
        latest_version: int = None,
        indexes: list[int] = None,
        get_n_samples: bool = False,
    ) -> tuple[int, list[int]]:
        # Validate topic, consumer, and columns
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if consumer not in meta.experience_consumers:
            raise ValueError(f"Unregistered consumer '{consumer}'.")
        if not experience_columns:
            raise ValueError("No experience_columns provided.")
        for col in experience_columns:
            if col not in meta.experience_columns:
                raise ValueError(f"Unregistered column '{col}'.")

        with meta.consumer_sampling_lock[consumer]:
            ready_indexes = set.intersection(*(meta.experience_ready[col] for col in experience_columns))
            usable_indexes = ready_indexes - meta.experience_consumed[consumer]

            if latest_version and allowed_staleness:
                with meta.versions_lock:
                    fresh_indexes = set().union(
                        *(
                            meta.versions_to_indexes[v]
                            for v in range(latest_version, latest_version - allowed_staleness - 1, -1)
                            if v in meta.versions_to_indexes
                        )
                    )
                    usable_indexes &= fresh_indexes

            if indexes:
                usable_indexes &= set(indexes)
            usable_indexes = sorted(usable_indexes)

            if get_n_samples:
                groups = {}
                for idx in usable_indexes:
                    groups.setdefault(idx // meta.n_samples_per_prompt, []).append(idx)

                usable_groups = []
                for group_id, sub_indexes in groups.items():
                    if len(set(sub_indexes)) == meta.n_samples_per_prompt:
                        usable_groups.append(group_id)

                usable_indexes = []
                for group_id in usable_groups:
                    for index in range(
                        group_id * meta.n_samples_per_prompt, (group_id + 1) * meta.n_samples_per_prompt
                    ):
                        usable_indexes.append(index)

            return len(usable_indexes), usable_indexes

    def _get_allocation_for_new_groups(self, topic: str, num_new_groups: int):
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if num_new_groups <= 0:
            raise ValueError(f"The number of new group {num_new_groups} is non-positive.")

        meta = self.topics[topic]

        # with meta.assign_new_group_lock:
        n_samples = meta.n_samples_per_prompt
        global_idx_to_shard = meta.global_idx_to_shard
        current_counts = [len(g) for g in meta.group_ids_per_shard]
        total_count = sum(current_counts) + num_new_groups
        num_shards = meta.nums_tq_data

        # Calculate needed_counts for each shard ('average' - current)
        base_count = total_count // num_shards
        remainder = total_count % num_shards
        target_counts = [base_count + (1 if i < remainder else 0) for i in range(num_shards)]
        needed_counts = [target_counts[i] - current_counts[i] for i in range(num_shards)]

        # Assign group_ids by chunk (start from the next available group_id)
        cur_max_group_id = -1 if meta.max_cur_index < 0 else meta.max_cur_index // n_samples
        current_new_id = cur_max_group_id + 1
        new_group_ids_per_shard = []

        for i, count in enumerate(needed_counts):
            if count > 0:
                assigned_ids = range(current_new_id, current_new_id + count)
                new_group_ids_per_shard.append(list(assigned_ids))
                for group_id in assigned_ids:
                    start = group_id * n_samples
                    end = start + n_samples
                    # Connect shard_id with global_index
                    for global_idx in range(start, end):
                        global_idx_to_shard[global_idx] = i
                current_new_id += count
            else:
                new_group_ids_per_shard.append([])

        # Update group_ids_per_shard
        for i in range(num_shards):
            meta.group_ids_per_shard[i].update(new_group_ids_per_shard[i])

        # # RPC调用
        # start_rpc = time.perf_counter()
        # futures = [
        #     self.data_actors[i].update_owned_groups.remote(topic, new_group_ids_per_shard[i])
        #     for i in range(num_shards)
        # ]
        # ray.get(futures)
        # elapsed_rpc = time.perf_counter() - start_rpc
        # if elapsed_rpc > 0.01:  # 超过10ms记录
        #     self.logger.info(f"RPC call (update_owned_groups) took {elapsed_rpc*1000:.3f}ms")

        return new_group_ids_per_shard

    def get_all_endpoints(self) -> list[str]:
        """Return the list of all DATA Shard endpoints (for broadcasting or debugging)."""
        return self.data_endpoints

    def get_n_samples_per_prompt(self, topic: str) -> int:
        """Return _n_samples_per_prompt for the given topic."""
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        return self.topics[topic].n_samples_per_prompt

    # TODO no lock  无锁，建议移除
    # def get_cur_max_group_id(self, topic: str) -> int:
    #     """获取当前已分配的最大 group_id（基于 max_cur_index 计算）"""
    #     if topic not in self.topics:
    #         raise ValueError(f"Unknown topic '{topic}'")
    #     meta = self.topics[topic]
    #     return -1 if meta.max_cur_index < 0 else meta.max_cur_index // meta.n_samples_per_prompt

    def mark_column_written_in_group(self, topic: str, group_id: int, column_name: str):
        """标记某列在某group已写入"""
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if column_name not in GROUP_SHARED_COLUMNS:
            return
        meta = self.topics[topic]
        if column_name not in meta.group_shared_col_written:
            meta.group_shared_col_written[column_name] = set()
        meta.group_shared_col_written[column_name].add(group_id)
        self.logger.debug(f"Marked column '{column_name}' written in group {group_id} for topic '{topic}'")

    def is_column_written_in_group(self, topic: str, group_id: int, column_name: str) -> bool:
        """判断某列在某group是否已写入（只针对共享列）"""
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if column_name not in GROUP_SHARED_COLUMNS:
            return False
        meta = self.topics[topic]
        return column_name in meta.group_shared_col_written and group_id in meta.group_shared_col_written[column_name]

    def batch_check_columns_written(self, topic: str, queries: list[tuple[str, int]]) -> dict[tuple[str, int], bool]:
        """批量检查列在组中是否已写入。

        Args:
            topic: Topic 名称
            queries: (column_name, group_id) 元组列表

        Returns:
            字典映射 (column_name, group_id) -> is_written
        """
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")

        meta = self.topics[topic]
        results = {}

        for col_name, group_id in queries:
            if col_name not in GROUP_SHARED_COLUMNS:
                results[(col_name, group_id)] = False
            else:
                results[(col_name, group_id)] = (
                    col_name in meta.group_shared_col_written and group_id in meta.group_shared_col_written[col_name]
                )

        return results

    def batch_mark_columns_written(self, topic: str, marks: list[tuple[str, int]]) -> None:
        """批量标记列在组中已写入。

        Args:
            topic: Topic 名称
            marks: (column_name, group_id) 元组列表，表示要标记的组合
        """
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")

        with TraceMarker.scope("status_update", topic=topic, marks=len(marks)):
            meta = self.topics[topic]

            for col_name, group_id in marks:
                if col_name not in GROUP_SHARED_COLUMNS:
                    continue
                if col_name not in meta.group_shared_col_written:
                    meta.group_shared_col_written[col_name] = set()
                meta.group_shared_col_written[col_name].add(group_id)

            self.logger.debug(f"Batch marked {len(marks)} column-group pairs as written for topic '{topic}'")

    def record_groups_and_versions(
        self,
        topic: str,
        marks: dict[str, list[int]] | None = None,
        version: int | None = None,
        indexes: list[int] | None = None,
    ) -> None:
        flat_marks = [(col_name, group_id) for col_name, group_ids in (marks or {}).items() for group_id in group_ids]
        if flat_marks:
            self.batch_mark_columns_written(topic, flat_marks)
        if version is not None and indexes:
            self.record_versions(topic, version, indexes)

    async def allocate_batches(self, topic: str, data_batch_size: int) -> list[int]:
        return await self.allocate_sequentially(topic, data_batch_size)

    async def allocate_sequentially(self, topic: str, num_samples: int) -> list[int]:
        """
        按顺序分配索引（场景b/d）

        Args:
            topic: Topic name
            num_samples: Number of input data entries
        """
        with TraceMarker.scope("allocate_indexes", topic=topic, count=num_samples, mode="sequential"):
            if topic not in self.topics:
                raise ValueError(f"Unknown topic '{topic}'")

            # 处理空请求，防止无意义的锁竞争
            if num_samples <= 0:
                return []

            meta = self.topics[topic]
            n_samples = meta.n_samples_per_prompt

            async with meta.assign_new_index_lock:
                # 从 max_cur_index + 1 开始分配（max_cur_index 是当前最大索引）
                start_idx = meta.max_cur_index + 1
                allocated_indexes = list(range(start_idx, start_idx + num_samples))
                max_idx = max(allocated_indexes)

                # 计算需要多少新group
                max_group_id = max_idx // n_samples
                num_groups = max_group_id + 1
                num_groups_allocated = sum(len(g) for g in meta.group_ids_per_shard)
                num_new_groups_needed = max(0, num_groups - num_groups_allocated)

                if num_new_groups_needed > 0:
                    new_group_ids_per_shard = self._get_allocation_for_new_groups(topic, num_new_groups_needed)
                    # 发起异步 RPC 通知 DataShards
                    futures = [
                        self.data_actors[i].update_owned_groups.remote(topic, new_group_ids_per_shard[i])
                        for i in range(meta.nums_tq_data)
                    ]
                    # 【加固点】必须 await 完所有 DataShard 的确认
                    # 此时 Manager 的线程可以去处理其他 Topic，但当前协程会停在这里等待
                    await asyncio.gather(*futures)

                # 更新 max_cur_index 为当前最大索引
                meta.max_cur_index = max_idx if allocated_indexes else meta.max_cur_index

                for idx in allocated_indexes:
                    group_id = idx // n_samples
                    meta.group_index_sets.setdefault(group_id, set()).add(idx)
                    # 该步骤可能被不同线程操作，从而覆盖，因此需要锁
                    # 线程A：allocated_indexes = [0, 1, 2]  # group_id = 0
                    # 线程B：allocated_indexes = [3, 4, 5]  # group_id = 0
                    # if group_id not in meta.group_index_sets:
                    #     meta.group_index_sets[group_id] = set()
                    # meta.group_index_sets[group_id].add(idx)

            return allocated_indexes

    async def allocate_by_uuid(self, topic: str, uuids: list[str]) -> list[int]:
        """
        按UUID分配索引（场景c）

        使用 TopicMeta 中的 uuid_to_group 和 group_index_sets
        """
        with TraceMarker.scope("allocate_indexes", topic=topic, count=len(uuids), mode="uuid"):
            if topic not in self.topics:
                raise ValueError(f"Unknown topic '{topic}'")

            meta = self.topics[topic]
            n_samples = meta.n_samples_per_prompt

            async with meta.assign_new_index_lock:
                # 先找出新的uuid，分配group_id
                new_uuids = [uuid for uuid in uuids if uuid not in meta.uuid_to_group]
                new_uuids = list(dict.fromkeys(new_uuids))

                # 计算需要的新group数
                num_new_groups_needed = len(new_uuids)

                # 为新uuid预计算group_id（不修改uuid_to_group）
                uuid_to_new_group_id = {}
                if new_uuids:
                    # next_group_id = self.get_cur_max_group_id(topic) + 1
                    if meta.max_cur_index < 0:
                        next_group_id = 0
                    else:
                        next_group_id = meta.max_cur_index // meta.n_samples_per_prompt + 1
                    for uuid in new_uuids:
                        uuid_to_new_group_id[uuid] = next_group_id
                        next_group_id += 1

                # 检查哪些uuid可以分配索引
                uuid_to_allocate = []
                for uuid in uuids:
                    if uuid in meta.uuid_to_group:
                        # 现有uuid：检查group是否已满
                        group_id = meta.uuid_to_group[uuid]
                        if len(meta.group_index_sets.get(group_id, set())) < n_samples:
                            uuid_to_allocate.append(uuid)
                    elif uuid in uuid_to_new_group_id:
                        # 新uuid：可以分配
                        uuid_to_allocate.append(uuid)

                if num_new_groups_needed > 0:
                    new_group_ids_per_shard = self._get_allocation_for_new_groups(topic, num_new_groups_needed)
                    # 发起异步 RPC 通知 DataShards
                    futures = [
                        self.data_actors[i].update_owned_groups.remote(topic, new_group_ids_per_shard[i])
                        for i in range(meta.nums_tq_data)
                    ]
                    # 【加固点】必须 await 完所有 DataShard 的确认
                    # 此时 Manager 的线程可以去处理其他 Topic，但当前协程会停在这里等待
                    await asyncio.gather(*futures)

                # 注册新uuid对应的group_id
                allocated_indexes = []
                for uuid, group_id in uuid_to_new_group_id.items():
                    meta.uuid_to_group[uuid] = group_id
                    meta.group_index_sets[group_id] = set()

                # 分配索引
                for uuid in uuid_to_allocate:
                    group_id = meta.uuid_to_group[uuid]
                    offset = len(meta.group_index_sets[group_id])
                    index = group_id * n_samples + offset
                    meta.group_index_sets[group_id].add(index)
                    allocated_indexes.append(index)

                # 更新 max_cur_index
                if allocated_indexes:
                    meta.max_cur_index = max(meta.max_cur_index, max(allocated_indexes))

            return allocated_indexes
