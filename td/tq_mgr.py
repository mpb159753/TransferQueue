# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Union

import ray
import torch

from metrics import Metric
from tq_sampler import BaseSampler, RandomSampler, VersionSampler, SeqLenBalSampler 
from seqlen_balancing import get_seqlen_balanced_partitions
from tq_data import TransferQueueShard
from tq_utils import setup_logger


@dataclass()
class TopicMeta:
    nums_tq_data: int
    prompts_num: int
    n_samples_per_prompt: int
    num_samples_gbs: int = field(init=False)

    metrics: Metric
    experience_columns: List[str]
    experience_consumers: List[str]
    experience_ready: Dict[str, Set[int]] = field(init=False)
    experience_consumed: Dict[str, Set[int]] = field(init=False)
    consumer_sampling_lock: Dict[str, threading.Lock] = field(init=False)
    assign_new_index_lock: threading.Lock = threading.Lock()

    reward_tags_to_indexes: Dict[int, Set[int]] = field(default_factory=dict)
    gid_to_shard: Dict[int, int] = field(default_factory=dict)
    group_ids_per_shard: List[Set[int]] = field(init=False)
    next_group_id: int = 0

    versions_to_indexes: Dict[int, Set[int]] = field(default_factory=dict)

    def __post_init__(self):
        if self.nums_tq_data <= 0:
            raise ValueError("nums_tq_data must be > 0")
        self.experience_ready = {col: set() for col in self.experience_columns}
        self.experience_consumed = {col: set() for col in self.experience_consumers}
        self.group_ids_per_shard = [set() for _ in range(self.nums_tq_data)]
        self.num_samples_gbs = self.prompts_num * self.n_samples_per_prompt
        self.consumer_sampling_lock = {
            key: threading.Lock() for key in self.experience_consumers
        }


@ray.remote(max_concurrency=100, num_cpus=10, name="TransferQueueManager")
class TransferQueueManager:
    def __init__(self, nums_tq_data: int = 1, base_port: Optional[int] = None) -> None:
        """
        Manager initialization.
        - nums_tq_data: Number of DATA shards to create.
        - base_port: Starting port number for DATA servers. If None, shards bind to a random available port.
        """
        self.logger = setup_logger("TransferQueueManager")
        self.logger.info("Manager initialized successfully")

        self.nums_tq_data = nums_tq_data
        # if(self.nums_tq_data>8):
        #     raise ValueError("Not support DATA Shard nums larger than 8.")

        self.topics: Dict[str, TopicMeta] = {}

        self.data_actors = []  # Ray actor handles for DATA shards
        self.data_endpoints = []  # ZMQ endpoints (addresses) for each shard

        # TODO: 以下是检测所有Ray结点，并依此尽量分散放置DATA Shard来避免DATA Shard都在同一个物理节点上，该方案并不优雅
        all_nodes = [node for node in ray.nodes() if node["Alive"]]
        target_node_ips = [
            node["NodeManagerAddress"]
            for node in all_nodes
            if node["Resources"].get("CPU", 0) >= 1
        ]
        if not target_node_ips:
            raise RuntimeError("No available Ray nodes with CPU resources.")

        for i in range(nums_tq_data):
            port = None if base_port is None else base_port + i

            # Round-robin allocate DATA Shard
            node_ip = target_node_ips[i % len(target_node_ips)]
            data_actor = TransferQueueShard.options(
                resources={f"node:{node_ip}": 0.01},
                name=f"TransferQueueShard_{i}"
            ).remote(
                i,
                port,
            )
            self.data_actors.append(data_actor)

        # Fetch endpoints concurrently to avoid serial waits.
        endpoint_refs = [a.get_endpoint.remote() for a in self.data_actors]
        self.data_endpoints = ray.get(endpoint_refs)

        self.logger.info(f"Manager: Initialized {nums_tq_data} data shards with endpoints: {self.data_endpoints}")

        # Timing accumulators (seconds) stored in a single dict (name -> seconds).
        # Custom items can be added at runtime. Values use 6-decimal precision.
        self._timing_lock = threading.Lock()
        self._timings: Dict[str, float] = {
            "put": 0.0,
            "get": 0.0,
            "put_prompt": 0.0,
            "dispatch": 0.0,
        }

    def add_topic(
            self,
            topic: str,
            prompts_num: int,
            n_samples_per_prompt: int,
            experience_columns: List[str],
            experience_consumers: List[str],
            metrics=None,
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
            metrics=metrics,
            experience_columns=experience_columns,
            experience_consumers=experience_consumers,
        )
        self.topics[topic] = meta
        for actor in self.data_actors:
            ray.get(actor.add_experience_table.remote(
                topic=topic,
                n_samples_per_prompt=meta.n_samples_per_prompt,
                experience_columns=meta.experience_columns,
            ))
        # self.logger.info(f"Manager: Added topic '{topic}' across {self.nums_tq_data} shards ")

    def delete_topic(self, topic: str) -> None:
        """
        Delete the specified topic by removing its tables from all shards and clearing metadata.
        """
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        # Remove tables from each data shard
        for actor in self.data_actors:
            actor.remove_experience_table.remote(topic)
        # Remove metadata
        del self.topics[topic]
        # self.logger.info(f"Manager: Deleted topic '{topic}'")

    def clear_topic(self, topic: str):
        """
        Clear ONLY the specified topic across all shards and reset its status tracking.
        """
        # Instruct all DATA shards to clear this topic
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")

        # Reset per table
        for actor in self.data_actors:
            ray.get(actor.clear_experience_table.remote(topic))
        # Reset per-topic statuses, version dict, reward tag mask, and metrics
        meta = self.topics[topic]
        meta.experience_ready = {col: set() for col in meta.experience_columns}
        meta.experience_consumed = {col: set() for col in meta.experience_consumers}
        meta.reward_tags_to_indexes = {}
        meta.gid_to_shard = {}
        meta.group_ids_per_shard = [set() for _ in range(self.nums_tq_data)]
        meta.next_group_id = 0
        if meta.metrics is not None:
            meta.metrics.reset()
        # self.logger.info(f"Manager: Topic '{topic}' has been cleared.")

    def prune_topic_by_indexes(self, topic: str, indexes: List[int]):
        """
        Prune the specified topic index across all shards and reset its tracking state.
        """
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if not indexes:
            raise ValueError("No indexes provided for prune_topic_by_indexes.")

        meta = self.topics[topic]
        unmatched = [idx for idx in indexes if idx not in meta.gid_to_shard]
        if unmatched:
            raise ValueError(f"Global indexes {unmatched} does not belong to any shard.")

        for col in meta.experience_columns:
            meta.experience_ready[col] -= set(indexes)
        for col in meta.experience_consumers:
            meta.experience_consumed[col] -= set(indexes)
        for tag in meta.reward_tags_to_indexes.keys():
            meta.reward_tags_to_indexes[tag] -= set(indexes)

        # Classify the indexes by shard
        index_to_prune_by_shard = {}
        groups_may_be_removed = set()
        for idx in indexes:
            # Get shard_id and clean meta.gid_to_shard by 'pop'
            shard_id = meta.gid_to_shard.pop(idx)
            index_to_prune_by_shard.setdefault(shard_id, []).append(idx)
            groups_may_be_removed.add((idx // meta.n_samples_per_prompt, shard_id))

        for shard_id, sub_indexes in index_to_prune_by_shard.items():
            # Prune the table by shard
            ray.get(self.data_actors[shard_id].prune_experience_table.remote(topic, sub_indexes))

        for (group_id, shard_id) in groups_may_be_removed:
            start = group_id * meta.n_samples_per_prompt
            end = start + meta.n_samples_per_prompt
            if all(idx not in meta.gid_to_shard for idx in range(start, end)):
                # Remove group_id if every sample of this group is already removed
                meta.group_ids_per_shard[shard_id].discard(group_id)

        for v in meta.versions_to_indexes.keys():
            meta.versions_to_indexes[v] -= set(indexes)

        if meta.metrics is not None:
            meta.metrics.reset()
        # self.logger.info(f"Manager: Indexes {indexes} on topic '{topic}' has been cleared.")

    def clear_and_resize_topic(self, topic: str, prompts_num: int, n_samples_per_prompt: int) -> None:
        """
        Resize the specified topic by deleting it and creating a new one with updated sizes.
        Other parameters (columns, consumers, metrics) are preserved.
        """
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        # Preserve existing parameters
        old_meta = self.topics[topic]
        experience_columns = old_meta.experience_columns
        experience_consumers = old_meta.experience_consumers
        metrics = old_meta.metrics
        # Delete existing topic
        self.delete_topic(topic)
        # Add new topic with updated sizes
        self.add_topic(topic, prompts_num, n_samples_per_prompt,
                       experience_columns, experience_consumers, metrics)
        # self.logger.info(
        #    f"Manager: Resized topic '{topic}' to prompts_num={prompts_num}, "
        #    f"n_samples_per_prompt={n_samples_per_prompt}"
        # )

    def get_targets_for_put(
            self,
            topic: str,
            indexes: List[int],
            is_prompt: bool = False,
    ) -> Dict[str, List[int]]:
        """
        Determine which DATA shard(s) should handle the given global indexes
        for a put_experience operation, and group indexes by shard endpoint.
        """
        # Validate indexes and topic
        if not indexes:
            raise ValueError("No indexes provided for get_targets_for_put.")
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")

        # Prepare mapping from endpoint -> list of indexes
        meta = self.topics[topic]
        if is_prompt:
            new_group_ids_per_shard = self.get_allocation_for_new_groups(topic, num_new_groups=len(indexes))
            endpoint_map = dict(zip(self.data_endpoints, new_group_ids_per_shard))
        else:
            invalid_gids = set(indexes) - meta.gid_to_shard.keys()
            if invalid_gids:
                raise ValueError(f"Global indexes {invalid_gids} does not belong to any shard.")
            # Prefetch references to reduce attribute lookups inside the loop.
            endpoint_map = {ep: [] for ep in self.data_endpoints}
            gid_to_shard = meta.gid_to_shard
            endpoints = self.data_endpoints
            for idx in indexes:
                shard_id = gid_to_shard[idx]
                endpoint_map[endpoints[shard_id]].append(idx)

        # Prune empty entries
        return {ep: idxs for ep, idxs in endpoint_map.items() if idxs}

    def allocate_shard_and_indexes(
            self,
            topic: str,
            consumer: str,
            experience_columns: List[str],
            experience_count: int,
            get_n_samples: bool,
            reward_tags: str = None,
            allowed_staleness: int = None,
            latest_version: int = None,
            sampler_func: Optional[BaseSampler] = None,
    ) -> Optional[Dict[str, List[int]]]:
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
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]

        # 1. Validate consumer
        if consumer not in meta.experience_consumers:
            raise ValueError(f"Consumer '{consumer}' not recognized.")

        if experience_count is None:
            raise ValueError("experience_count must be specified when indexes are not provided.")

        if get_n_samples and (experience_count % meta.n_samples_per_prompt != 0):
            raise ValueError(
                f"get_n_samples=True requires experience_count ({experience_count}) "
                f"to be divisible by n_samples_per_prompt ({meta.n_samples_per_prompt})."
            )

        # 2. Sample ready indexes
        if get_n_samples:
            chosen = self._sample_ready_index_n_samples(
                topic,
                consumer,
                experience_count,
                experience_columns,
                reward_tags=reward_tags,
                allowed_staleness=allowed_staleness,
                latest_version=latest_version,
                sampler_func=sampler_func
            )
        else:
            chosen = self._sample_ready_index(
                topic,
                consumer,
                experience_count,
                experience_columns,
                reward_tags=reward_tags,
                allowed_staleness=allowed_staleness,
                latest_version=latest_version,
                sampler_func=sampler_func
            )

        if not chosen:
            # No data available
            return None

        # 3. Group by shard: prepare mapping from endpoint -> list of indexes
        unmatched = [idx for idx in chosen if idx not in meta.gid_to_shard]
        if unmatched:
            raise ValueError(f"Global indexes {unmatched} does not belong to any shard.")
        endpoint_map: Dict[str, List[int]] = {ep: [] for ep in self.data_endpoints}
        for idx in chosen:
            shard_id = meta.gid_to_shard[idx]
            endpoint_map[self.data_endpoints[shard_id]].append(idx)

        # Prune empty entries
        return {ep: idxs for ep, idxs in endpoint_map.items() if idxs}

    def allocate_shard_for_indexes(
            self,
            topic: str,
            consumer: str,
            experience_columns: List[str],
            indexes: int,
            reward_tags: str = None,
            allowed_staleness: int = None,
            latest_version: int = None,
    ) -> Optional[Dict[str, List[int]]]:
        """
        Return ready indexes of the given indexes, and group them by endpoint.
        """
        # 1. Validate topic, consumer, indexes, and reward_tags
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if consumer not in meta.experience_consumers:
            raise ValueError(f"Uregistered consumers: '{consumer}'.")
        if not indexes:
            raise ValueError("indexes must be provided when experience_count are not specified.")
        if isinstance(reward_tags, str):
            if reward_tags == 'reward':
                reward_tags = 1
            elif reward_tags == 'neural_checker':
                reward_tags = 2
            else:
                raise ValueError(f"Unknown reward_tags '{reward_tags}'")

        # 2. Sample ready indexes
        with meta.consumer_sampling_lock[consumer]:
            ready_indexes = set(indexes).intersection(
                *(meta.experience_ready[col] for col in experience_columns)
            )

            if reward_tags is not None:
                usable_indexes = ready_indexes & meta.reward_tags_to_indexes[reward_tags]
            else:
                usable_indexes = ready_indexes

            usable_indexes -= meta.experience_consumed[consumer]

            if latest_version and allowed_staleness:
                fresh_indexes = set().union(*(
                    meta.versions_to_indexes[v]
                    for v in range(latest_version, latest_version - allowed_staleness - 1, -1)
                    if v in meta.versions_to_indexes
                ))
                usable_indexes &= fresh_indexes
            else:
                self.logger.debug("Manager: Both 'latest_version' and 'allowed_staleness' must be provided.")

            if not usable_indexes:
                return None

            usable_indexes = list(usable_indexes)
            meta.experience_consumed[consumer].update(usable_indexes)

        # 3. Group by shard: prepare mapping from endpoint -> list of indexes
        unmatched = [idx for idx in usable_indexes if idx not in meta.gid_to_shard]
        if unmatched:
            raise ValueError(f"Global indexes {unmatched} does not belong to any shard.")
        endpoint_map: Dict[str, List[int]] = {ep: [] for ep in self.data_endpoints}
        for idx in usable_indexes:
            shard_id = meta.gid_to_shard[idx]
            endpoint_map[self.data_endpoints[shard_id]].append(idx)

        # Prune empty entries
        return {ep: idxs for ep, idxs in endpoint_map.items() if idxs}

    def _sample_ready_index(
            self,
            topic,
            consumer: str,
            experience_count: int,
            experience_columns: List[str],
            reward_tags: str = None,
            allowed_staleness: int = None,
            latest_version: int = None,
            sampler_func: Optional[BaseSampler] = None,
    ) -> Optional[List[int]]:
        meta = self.topics[topic]

        if isinstance(reward_tags, str):
            if reward_tags == 'reward':
                reward_tags = 1
            elif reward_tags == 'neural_checker':
                reward_tags = 2
            else:
                raise ValueError(f"Unknown reward_tags '{reward_tags}'")

        with meta.consumer_sampling_lock[consumer]:
            ready_indexes = set.intersection(*(meta.experience_ready[col] for col in experience_columns))
            if reward_tags is not None:
                usable_indexes = ready_indexes & meta.reward_tags_to_indexes[reward_tags]
            else:
                usable_indexes = ready_indexes

            usable_indexes -= meta.experience_consumed[consumer]

            if latest_version and allowed_staleness:
                fresh_indexes = set().union(*(
                    meta.versions_to_indexes[v]
                    for v in range(latest_version, latest_version - allowed_staleness - 1, -1)
                    if v in meta.versions_to_indexes
                ))
                usable_indexes &= fresh_indexes
            else:
                self.logger.debug("Manager: Both 'latest_version' and 'allowed_staleness' must be provided.")

            if len(usable_indexes) < experience_count:
                if reward_tags is None:
                    return None
                else:
                    experience_count = len(usable_indexes)

            if experience_count > 0:
                if sampler_func is None:
                    sampler_func = RandomSampler(seed=42)
                    sampled_indexes = sampler_func.sample(usable_indexes, experience_count)
                else:
                    idx_to_version = {idx: v for v, idxs in meta.versions_to_indexes.items() for idx in idxs}
                    versions = [idx_to_version[idx] for idx in usable_indexes]
                    sampled_indexes = sampler_func.sample(usable_indexes, experience_count, versions=versions)
                meta.experience_consumed[consumer].update(sampled_indexes)
            else:
                sampled_indexes = None

        return sampled_indexes

    def _sample_ready_index_n_samples(
            self,
            topic,
            consumer: str,
            experience_count: int,
            experience_columns: List[str],
            reward_tags: str = None,
            allowed_staleness: int = None,
            latest_version: int = None,
            sampler_func: Optional[BaseSampler] = None,
    ) -> Optional[List[int]]:
        meta = self.topics[topic]
        if isinstance(reward_tags, str):
            if reward_tags == 'reward':
                reward_tags = 1
            elif reward_tags == 'neural_checker':
                reward_tags = 2
            else:
                raise ValueError(f"Unknown reward_tags '{reward_tags}'")

        experience_count_n_samples = experience_count // meta.n_samples_per_prompt
        with meta.consumer_sampling_lock[consumer]:
            ready_indexes = set.intersection(*(meta.experience_ready[col] for col in experience_columns))
            if reward_tags is not None:
                usable_indexes = ready_indexes & meta.reward_tags_to_indexes[reward_tags]
            else:
                usable_indexes = ready_indexes

            usable_indexes -= meta.experience_consumed[consumer]

            if latest_version and allowed_staleness:
                fresh_indexes = set().union(*(
                    meta.versions_to_indexes[v]
                    for v in range(latest_version, latest_version - allowed_staleness - 1, -1)
                    if v in meta.versions_to_indexes
                ))
                usable_indexes &= fresh_indexes
            else:
                self.logger.debug("Manager: Both 'latest_version' and 'allowed_staleness' must be provided.")

            groups = {}
            for idx in usable_indexes:
                groups.setdefault(idx // meta.n_samples_per_prompt, []).append(idx)
            usable_groups = []
            for group_id, sub_indexes in groups.items():
                if len(set(sub_indexes)) == meta.n_samples_per_prompt:
                    usable_groups.append(group_id)

            ####### important notice, if we have reward models and reward tags, for each tag, the sample number is smaller than experience_count ######
            if len(usable_groups) < experience_count_n_samples:
                if reward_tags is None:
                    return None
                else:
                    experience_count_n_samples = len(usable_groups)

            if experience_count_n_samples > 0:
                if sampler_func is None:
                    sampler_func = RandomSampler(seed=42)
                    sampled_indexes_n_sample = sampler_func.sample(usable_groups, experience_count_n_samples)
                else:
                    idx_to_version = {idx: v for v, idxs in meta.versions_to_indexes.items() for idx in idxs}
                    versions = [idx_to_version[idx] for idx in usable_groups]
                    sampled_indexes_n_sample = sampler_func.sample(usable_groups, experience_count_n_samples, versions=versions)
            else:
                sampled_indexes_n_sample = []

            sampled_indexes = []
            for n_sample_index in sampled_indexes_n_sample:
                index_list = []
                for index in range(
                        n_sample_index * meta.n_samples_per_prompt,
                        (n_sample_index + 1) * meta.n_samples_per_prompt
                ):
                    index_list.append(index)

                sampled_indexes += index_list

            if not sampled_indexes:
                return None
            meta.experience_consumed[consumer].update(sampled_indexes)
        return sampled_indexes

    def _fetch_column_values(self, topic: str, column: str, global_indexes: List[int]):
        """
        Fetch column values for arbitrary global indexes across shards via Ray.
        """
        if not global_indexes:
            return []
        meta = self.topics[topic]
        # group by shard id while keeping original positions
        shard_groups = {}  # sid -> list[(pos, gi)]
        for pos, gi in enumerate(global_indexes):
            sid = self._find_shard_id(meta, gi)
            if sid is None:
                raise RuntimeError(f"Global index {gi} does not belong to any shard.")
            if sid not in shard_groups:
                shard_groups[sid] = []
            shard_groups[sid].append((pos, gi))

        # fire RPCs
        pending = {}
        for sid, pairs in shard_groups.items():
            idxs = [gi for _, gi in pairs]
            pending[sid] = self.data_actors[sid].get_values.remote(topic, column, idxs)

        per_shard_results = {sid: ray.get(obj) for sid, obj in pending.items()}

        # stitch back to original order
        out = [None] * len(global_indexes)
        for sid, pairs in shard_groups.items():
            vals = per_shard_results[sid]
            for local_i, (pos, _) in enumerate(pairs):
                out[pos] = vals[local_i]
        return out

    def all_consumed(self, topic: str, consumer: str, indexes: List[int]) -> bool:
        """
        Check if the given consumer has consumed all data.
        Returns True if all indices have been consumed by this consumer.
        """
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if consumer not in meta.experience_consumers:
            raise ValueError(f"Uregistered consumers: '{consumer}'.")

        if not indexes:
            return len(meta.experience_consumed[consumer]) == meta.num_samples_gbs
        else:
            return all(idx in meta.experience_consumed[consumer] for idx in indexes)

    def all_updated(self, topic: str, column: str, indexes: List[int]) -> bool:
        """
        Check if all data in the specified column has been updated (filled).
        Returns True if all entries in that column are marked as updated.
        """
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if column not in meta.experience_columns:
            raise ValueError(f"Uregistered columns '{column}'.")

        if not indexes:
            return len(meta.experience_ready[column]) == meta.num_samples_gbs
        else:
            return all(idx in meta.experience_ready[column] for idx in indexes)

    def reset_all(self):
        """
        Fully reset to post-__init__ state:
        - Instruct all shards to drop ALL topic tables.
        - Remove ALL topics metadata.
        - Reset timings to initial values.
        """
        # Drop all per-topic tables in every shard (after this, shards have no topics).
        for actor in self.data_actors:
            ray.get(actor.reset_all.remote())
        # Remove manager-side topics and associated runtime state.
        self.topics = {}
        # Reset timings to the same initial keys/values as in __init__.
        with self._timing_lock:
            self._timings = {
                "put": 0.0,
                "get": 0.0,
                "put_prompt": 0.0,
                "dispatch": 0.0,
            }
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

    # def get_shard_sample_counts(self, topic: str) -> List[int]:
    #     """Return the number of samples assigned to each shard for a topic."""
    #     if topic not in self.topics:
    #         raise ValueError(f"Unknown topic '{topic}'")
    #     meta = self.topics[topic]
    #     return list(meta.per_shard_max_len_list)

    def get_all_endpoints(self) -> List[str]:
        """Return the list of all DATA Shard endpoints (for broadcasting or debugging)."""
        return self.data_endpoints

    def get_n_samples_per_prompt(self, topic: str) -> int:
        """Return _n_samples_per_prompt for the given topic."""
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        return self.topics[topic].n_samples_per_prompt

    def update_data_status(self, topic: str, indexes: List[int], columns: List[str]) -> None:
        """
        Update the data readiness status for given columns at specified indexes.
        This should be called by DATA shards after they finish storing new data.
        """
        if not indexes or not columns:
            return
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")

        # Update all given indexes in this column
        meta = self.topics[topic]
        for col in columns:
            if col in meta.experience_ready:
                meta.experience_ready[col].update(indexes)
        # Note: We do not alter consumer statuses here. Consumers will be marked upon get_experience.

    def update_consumer_status(self, topic: str, indexes: List[int], consumer: str) -> None:
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

    # def set_reward_tag_mask(self, topic: str, prompt_reward_tags: Tensor) -> None:
    #     """
    #     Obtain the list of reward tags corresponding to each prompt and expand (replicate) it for each sample.
    #     """
    #     if topic not in self.topics:
    #         raise ValueError(f"Unknown topic '{topic}'")
    #     meta = self.topics[topic]
    #     tag_tensor = torch.tensor(prompt_reward_tags, dtype=torch.int32)
    #     if tag_tensor.numel() != meta.prompts_num:
    #         raise ValueError("Length of prompt_reward_tags must equal prompts_num")
    #     meta.reward_tag_mask = tag_tensor.repeat_interleave(meta.n_samples_per_prompt,dim=0)

    def get_metrics(self, topic: str):
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        return self.topics[topic].metrics

    def update_metrics(self, topic: str, key="", value=None, cumulate=False):
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        metrics = self.topics[topic].metrics
        if metrics is None:
            return
        metrics.update(key, value, cumulate=cumulate)

    def init_ready(self):
        return True

    def create_timing_item(self, name: str) -> None:
        """
        Create (or ensure) a custom timing item initialized to 0.0 seconds.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Timing item name must be a non-empty string")
        with self._timing_lock:
            if name in self._timings:
                raise ValueError(f"Timing item '{name}' already exists")
            self._timings[name] = 0.0

    def accumulate_timing(self, name: str, seconds: float) -> None:
        """
        Add elapsed time (in seconds) to a timing item.
        Value should come from time.perf_counter() deltas. Stored with 6-decimal precision.
        """
        val = round(float(seconds), 6)
        with self._timing_lock:
            cur = self._timings.get(name, 0.0)
            self._timings[name] = round(cur + val, 6)

    def get_timing(self, name: str) -> float:
        """
        Get accumulated time in seconds (6-decimal precision) for the given name.
        """
        with self._timing_lock:
            if name in self._timings:
                return round(self._timings[name], 6)
        raise ValueError(f"Unknown timing name '{name}'")

    def get_timings(self) -> Dict[str, float]:
        """
        Return a dict of all accumulated timings (seconds, 6-decimal precision),
        including built-ins and any custom items.
        """
        with self._timing_lock:
            return {k: round(v, 6) for k, v in self._timings.items()}

    def reset_timings(self) -> None:
        """
        Reset all accumulated timings to zero.
        """
        with self._timing_lock:
            self._timings = {k: 0.0 for k in self._timings.keys()}

    # def _find_shard_id(self, meta: TopicMeta, global_index: int) -> Optional[int]:
    #     if global_index < 0 or global_index >= meta.max_len:
    #         return None
    #     for i in range(self.nums_tq_data):
    #         lo = meta.shard_sample_offsets[i]
    #         hi = lo + meta.per_shard_max_len_list[i]
    #         if lo <= global_index < hi:
    #             return i
    #     return None

    # def get_max_len(self, topic: str) -> int:
    #     meta = self.topics[topic]
    #     return meta.max_len

    def get_next_group_id(self, topic: str) -> int:
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        return self.topics[topic].next_group_id

    def get_allocation_for_new_groups(self, topic: str, num_new_groups: int):
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if num_new_groups < 0:
            raise ValueError(f"The number of new group {num_new_groups} is negative.")

        meta = self.topics[topic]
        with meta.assign_new_index_lock:
            n_samples = meta.n_samples_per_prompt
            gid_to_shard = meta.gid_to_shard
            current_counts = [len(g) for g in meta.group_ids_per_shard]
            total_count = sum(current_counts) + num_new_groups
            num_shards = meta.nums_tq_data

            # Calculate needed_counts for each shard ('average' - current)
            base_count = total_count // num_shards
            remainder = total_count % num_shards
            target_counts = [base_count + (1 if i < remainder else 0) for i in range(num_shards)]
            needed_counts = [target_counts[i] - current_counts[i] for i in range(num_shards)]

            # Assign group_ids by chunk
            current_new_id = meta.next_group_id
            new_group_ids_per_shard = []

            for i, count in enumerate(needed_counts):
                if count > 0:
                    assigned_ids = range(current_new_id, current_new_id + count)
                    new_group_ids_per_shard.append(list(assigned_ids))
                    for group_id in assigned_ids:
                        start = group_id * n_samples
                        end = start + n_samples
                        # Connect shard_id with global_index
                        for gid in range(start, end):
                            gid_to_shard[gid] = i
                    current_new_id += count
                else:
                    new_group_ids_per_shard.append([])

            # Update group_ids_per_shard and next_group_id
            meta.next_group_id += num_new_groups
            for i in range(num_shards):
                meta.group_ids_per_shard[i].update(new_group_ids_per_shard[i])

        futures = [
            self.data_actors[i].update_owned_groups.remote(topic, new_group_ids_per_shard[i])
            for i in range(num_shards)
        ]
        ray.get(futures)

        return new_group_ids_per_shard

    def get_data_ready_set(self, topic: str, column: str, indexes: List[int]) -> Set[int]:
        # Validate topic and column
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if column not in meta.experience_columns:
            raise ValueError(f"Uregistered column '{column}'.")

        if not indexes:
            return meta.experience_ready[column]
        else:
            return set(indexes) & meta.experience_ready[column]

    def get_data_consumed_set(self, topic: str, consumer: str, indexes: List[int]) -> Set[int]:
        # Validate topic and consumer
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if consumer not in meta.experience_consumers:
            raise ValueError(f"Uregistered consumer '{consumer}'.")

        if not indexes:
            return meta.experience_consumed[consumer]
        else:
            return set(indexes) & meta.experience_consumed[consumer]

    def get_data_ready_counts(self, topic: str, column: str, indexes: List[int]) -> Set[int]:
        # Validate topic and column
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if column not in meta.experience_columns:
            raise ValueError(f"Uregistered column '{column}'.")

        if not indexes:
            return len(meta.experience_ready[column])
        else:
            return len(set(indexes) & meta.experience_ready[column])

    def get_data_consumed_counts(self, topic: str, consumer: str, indexes: List[int]) -> Set[int]:
        # Validate topic and consumer
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        meta = self.topics[topic]
        if consumer not in meta.experience_consumers:
            raise ValueError(f"Uregistered consumer '{consumer}'.")

        if not indexes:
            return len(meta.experience_consumed[consumer])
        else:
            return len(set(indexes) & meta.experience_consumed[consumer])

    def set_reward_tags_to_indexes(self, topic: str, reward_tags: Union[torch.Tensor, List[int]], indexes: List[int]):
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if isinstance(reward_tags, torch.Tensor):
            reward_tags = reward_tags.cpu().tolist()
        if len(reward_tags) == 0 or len(indexes) == 0:
            raise ValueError("Reward tags and indexes must be non-empty for 'set_reward_tags_to_indexes'.")
        if len(reward_tags) != len(indexes):
            raise ValueError(
                f"Length of reward_tags ({len(reward_tags)}) must match length of indexes ({len(indexes)})")

        meta = self.topics[topic]
        # Group indexes by reward_tags
        tag_to_idx_map = {}
        for tag, idx in zip(reward_tags, indexes):
            if tag not in tag_to_idx_map:
                tag_to_idx_map[tag] = []
            tag_to_idx_map[tag].append(idx)

        # Update the index set for each tag
        for tag, idx_list in tag_to_idx_map.items():
            if tag not in meta.reward_tags_to_indexes:
                meta.reward_tags_to_indexes[tag] = set(idx_list)
            else:
                meta.reward_tags_to_indexes[tag].update(idx_list)

    def record_versions(self, topic: str, version: int, indexes: List[int]) -> None:
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if not indexes:
            raise ValueError("No indexes provided for record_versions.")
        self.topics[topic].versions_to_indexes.setdefault(version, set()).update(indexes)

    def get_versions_by_index(self, topic: str, index: int):
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        for version, index_set in self.topics[topic].versions_to_indexes.items():
            if index in index_set:
                return version
        return None

    def reset_versions(self, topic: str, new_version: int, indexes: List[int]) -> None:
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if not indexes:
            raise ValueError("No indexes provided for reset_versions.")
        meta = self.topics[topic]
        for v in meta.versions_to_indexes.keys():
            meta.versions_to_indexes[v] -= set(indexes)
        self.topics[topic].versions_to_indexes.setdefault(new_version, set()).update(indexes)

    def clear_data_by_version(self, topic: str, version: int) -> None:
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if version is None:
            raise ValueError("No iversion provided for clear_data_by_version.")
        meta = self.topics[topic]
        indexes_to_be_pruned = meta.versions_to_indexes.get(version)
        if indexes_to_be_pruned:
            self.prune_topic_by_indexes(topic, indexes_to_be_pruned)

    def clear_data_by_staleness(self, topic: str, allowed_staleness: int, latest_version: int) -> None:
        if topic not in self.topics:
            raise ValueError(f"Unknown topic '{topic}'")
        if not allowed_staleness:
            raise ValueError("No allowed_staleness provided for clear_data_by_staleness.")
        if not latest_version:
            raise ValueError("No latest_version provided for clear_data_by_staleness.")
        for version in range(latest_version - allowed_staleness):
            self.clear_data_by_version(topic, version)
