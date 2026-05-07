# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from transfer_queue.client import TransferQueueClient  # noqa: E402
from transfer_queue.controller import TransferQueueController  # noqa: E402
from transfer_queue.storage.simple_backend import SimpleStorageUnit  # noqa: E402
from transfer_queue.utils.common import get_placement_group  # noqa: E402
from transfer_queue.utils.perf_compare import (  # noqa: E402
    COMMON_CORE_COMPARISON_MODE,
    build_comparison_context,
    build_comparison_result,
    ensure_results_parent,
    resolve_results_output_path,
)
from transfer_queue.utils.trace_utils import VizTracerProfileSession  # noqa: E402
from transfer_queue.utils.zmq_utils import process_zmq_server_info  # noqa: E402

VIZTRACER_AVAILABLE = True
try:
    import viztracer  # noqa: F401
except ImportError:
    VIZTRACER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LEGACY_TEST_MODES = ("ray-obj-store", "ray-remote", "tq-normal", "tq-remote")
COMPARISON_CONFIG_MAP = {
    "debug": {"global_batch_size": 32, "seq_length": 128, "field_num": 2, "desc": "Debug (~32KB)"},
    "tiny": {"global_batch_size": 64, "seq_length": 1024, "field_num": 4, "desc": "Tiny (~1MB)"},
    "small": {"global_batch_size": 512, "seq_length": 12800, "field_num": 4, "desc": "Small (~100MB)"},
    "medium": {"global_batch_size": 1024, "seq_length": 65536, "field_num": 4, "desc": "Medium (~1GB)"},
    "large": {"global_batch_size": 2048, "seq_length": 128000, "field_num": 5, "desc": "Large (~5GB)"},
    "xlarge": {"global_batch_size": 4096, "seq_length": 128000, "field_num": 5, "desc": "X-Large (~10GB)"},
    "huge": {"global_batch_size": 4096, "seq_length": 128000, "field_num": 10, "desc": "Huge (~20GB)"},
}
DTYPE_CONFIGS = [
    {"dtype": torch.float32, "bytes_per_elem": 4},
    {"dtype": torch.int64, "bytes_per_elem": 8},
    {"dtype": torch.float64, "bytes_per_elem": 8},
    {"dtype": torch.int32, "bytes_per_elem": 4},
    {"dtype": torch.float16, "bytes_per_elem": 2},
]

########################################################################
# Please set up Ray cluster before running the legacy script modes.
########################################################################
HEAD_NODE_IP = "NodeA"  # Replace with your head node IP
WORKER_NODE_IP = "NodeB"  # Replace with your worker node IP

config_str = """
  global_batch_size: 1024
  seq_length: 8192
  field_num: 10
  num_global_batch: 1
  num_data_storage_units: 8
"""
dict_conf = OmegaConf.create(config_str)


def create_complex_test_case(
    batch_size: int | None = None, seq_length: int | None = None, field_num: int | None = None
):
    tensor_field_size_bytes = batch_size * seq_length * 4
    tensor_field_size_gb = tensor_field_size_bytes / (1024**3)

    num_tensor_fields = (field_num + 1) // 2
    num_nontensor_fields = field_num // 2

    total_tensor_size_gb = tensor_field_size_gb * num_tensor_fields
    total_nontensor_size_gb = (batch_size * 1024 / (1024**3)) * num_nontensor_fields
    total_size_gb = total_tensor_size_gb + total_nontensor_size_gb

    logger.info(f"Total data size: {total_size_gb:.6f} GB")

    fields = {}
    for field_index in range(field_num):
        field_name = f"field_{field_index}"
        if field_index % 2 == 0:
            fields[field_name] = torch.randn(batch_size, seq_length, dtype=torch.float32)
            continue

        str_length = 1024
        non_tensor_data = [
            "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=str_length))
            for _ in range(batch_size)
        ]
        fields[field_name] = NonTensorData(data=non_tensor_data, batch_size=(batch_size,), device=None)

    return TensorDict(fields, batch_size=(batch_size,), device=None), total_size_gb


def _generate_regular_tensor(batch_size: int, seq_length: int, dtype: torch.dtype) -> torch.Tensor:
    if dtype in (torch.int32, torch.int64):
        return torch.randint(0, 10000, (batch_size, seq_length), dtype=dtype)
    return torch.randn(batch_size, seq_length, dtype=dtype)


def create_common_core_test_case(batch_size: int, seq_length: int, field_num: int) -> tuple[TensorDict, int]:
    total_bytes = 0
    fields = {}

    for field_index in range(field_num):
        dtype_config = DTYPE_CONFIGS[field_index % len(DTYPE_CONFIGS)]
        fields[f"field_{field_index}"] = _generate_regular_tensor(batch_size, seq_length, dtype_config["dtype"])
        total_bytes += batch_size * seq_length * dtype_config["bytes_per_elem"]

    return TensorDict(fields, batch_size=(batch_size,), device=None), total_bytes


def verify_tensor_dict_data(original: TensorDict, fetched: TensorDict) -> tuple[bool, str]:
    if set(original.keys()) != set(fetched.keys()):
        return False, f"keys mismatch: {set(original.keys()) ^ set(fetched.keys())}"

    for field_name in original.keys():
        original_tensor = original[field_name]
        fetched_tensor = fetched[field_name]
        if not isinstance(fetched_tensor, torch.Tensor):
            return False, f"{field_name} type mismatch: {type(fetched_tensor)}"
        if original_tensor.shape != fetched_tensor.shape:
            return False, f"{field_name} shape mismatch: {original_tensor.shape} vs {fetched_tensor.shape}"
        if original_tensor.dtype != fetched_tensor.dtype:
            return False, f"{field_name} dtype mismatch: {original_tensor.dtype} vs {fetched_tensor.dtype}"
        if not torch.equal(original_tensor.cpu(), fetched_tensor.cpu()):
            return False, f"{field_name} tensor values mismatch"

    return True, "✅ PASS"


def build_client_assignments(global_batch_size: int, num_clients: int) -> list[dict[str, int]]:
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")

    active_num_clients = min(global_batch_size, num_clients)
    base_batch_size, remainder = divmod(global_batch_size, active_num_clients)
    assignments = []
    start_idx = 0
    for client_id in range(active_num_clients):
        batch_size = base_batch_size + (1 if client_id < remainder else 0)
        end_idx = start_idx + batch_size
        assignments.append(
            {
                "client_id": client_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "batch_size": batch_size,
            }
        )
        start_idx = end_idx
    return assignments


@ray.remote
class SimpleStorageComparisonWorker:
    def __init__(self, client_id: str, controller_info: Any, storage_unit_infos: Any):
        config = OmegaConf.create({}, flags={"allow_objects": True})
        config.controller_info = controller_info
        config.storage_unit_infos = storage_unit_infos

        self.client = TransferQueueClient(client_id=client_id, controller_info=controller_info)
        self.client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=config)

    def run_round(self, config: dict[str, Any], partition_id: str, verify_round: bool) -> dict[str, Any]:
        data, payload_bytes = create_common_core_test_case(
            batch_size=int(config["global_batch_size"]),
            seq_length=int(config["seq_length"]),
            field_num=int(config["field_num"]),
        )

        start_put = time.time()
        asyncio.run(self.client.async_put(data=data, partition_id=partition_id))
        put_time = time.time() - start_put

        start_get = time.time()
        prompt_meta = asyncio.run(
            self.client.async_get_meta(
                data_fields=list(data.keys()),
                batch_size=int(data.batch_size[0]),
                partition_id=partition_id,
                task_name="generate_sequences",
            )
        )
        fetched = asyncio.run(self.client.async_get_data(prompt_meta))
        get_time = time.time() - start_get

        verified = True
        verify_message = "✅ PASS"
        if verify_round:
            verified, verify_message = verify_tensor_dict_data(data, fetched)

        asyncio.run(self.client.async_clear_partition(partition_id=partition_id))

        return {
            "payload_bytes": payload_bytes,
            "put_time": put_time,
            "get_time": get_time,
            "verified": verified,
            "verify_message": verify_message,
        }


@ray.remote
class RemoteDataStoreObjStore:
    def get_data(self, data_handler):
        start_get = time.time()
        ray.get(data_handler)
        return time.time() - start_get


@ray.remote
class RemoteDataStoreRemote:
    def __init__(self):
        self.stored_data = None

    def put_data(self, data):
        self.stored_data = data

    def get_data(self):
        return self.stored_data

    def clear_data(self):
        self.stored_data = None


class RayBandwidthTester:
    def __init__(self, config, test_mode="obj_store"):
        self.config = config
        self.test_mode = test_mode

        remote_store_cls = RemoteDataStoreObjStore if test_mode == "obj_store" else RemoteDataStoreRemote
        self.remote_store = remote_store_cls.options(num_cpus=10, resources={f"node:{WORKER_NODE_IP}": 0.001}).remote()
        logger.info(f"Remote data store created on worker node {WORKER_NODE_IP}")

    def run_bandwidth_test(self):
        start_create_data = time.time()
        test_data, total_data_size_gb = create_complex_test_case(
            batch_size=self.config.global_batch_size,
            seq_length=self.config.seq_length,
            field_num=self.config.field_num,
        )
        logger.info(f"Data creation time: {time.time() - start_create_data:.8f}s")

        if self.test_mode == "obj_store":
            self._run_obj_store_test(test_data, total_data_size_gb)
            return

        self._run_remote_test(test_data, total_data_size_gb)

    def _run_obj_store_test(self, test_data, total_data_size_gb):
        start_time = time.time()
        data_handler = ray.put(test_data)
        ray.get(self.remote_store.get_data.remote([data_handler]))
        transfer_time = time.time() - start_time
        throughput = (total_data_size_gb * 8) / transfer_time

        logger.info("=" * 60)
        logger.info("RAY OBJECT STORE BANDWIDTH TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"Transfer Time: {transfer_time:.8f}s")
        logger.info(f"Throughput: {throughput:.8f} Gb/s")

    def _run_remote_test(self, test_data, total_data_size_gb):
        logger.info("Starting Ray PUT bandwidth test...")
        start_put = time.time()
        ray.get(self.remote_store.put_data.remote(test_data))
        put_time = time.time() - start_put

        time.sleep(2)

        logger.info("Starting Ray GET bandwidth test...")
        start_get = time.time()
        ray.get(self.remote_store.get_data.remote())
        get_time = time.time() - start_get

        ray.get(self.remote_store.clear_data.remote())

        put_throughput = (total_data_size_gb * 8) / put_time
        get_throughput = (total_data_size_gb * 8) / get_time

        logger.info("=" * 60)
        logger.info("RAY REMOTE ACTOR BANDWIDTH TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"PUT Time: {put_time:.8f}s")
        logger.info(f"GET Time: {get_time:.8f}s")
        logger.info(f"PUT Throughput (Head->Worker): {put_throughput:.8f} Gb/s")
        logger.info(f"GET Throughput (Worker->Head): {get_throughput:.8f} Gb/s")
        logger.info(f"Round-trip Average Throughput: {total_data_size_gb * 16 / (put_time + get_time):.8f} Gb/s")


class TQBandwidthTester:
    def __init__(
        self,
        config,
        remote_mode=False,
        enable_profile: bool = False,
        profile_output_dir: Path | None = None,
        profile_tracer_entries: int = 1_000_000,
        profile_min_duration_us: int = 0,
    ):
        self.config = config
        self.remote_mode = remote_mode
        self.enable_profile = enable_profile
        self.profile_output_dir = profile_output_dir
        self.profile_tracer_entries = profile_tracer_entries
        self.profile_min_duration_us = profile_min_duration_us
        self.profile_session: VizTracerProfileSession | None = None
        self.data_system_storage_units = {}
        self.data_system_controller = None
        self.data_system_controller_info = None
        self.data_system_storage_unit_infos = None
        self.data_system_client = self._initialize_data_system()

    def _initialize_data_system(self):
        total_storage_size = self.config.global_batch_size * self.config.num_global_batch
        if self.remote_mode:
            for storage_unit_rank in range(self.config.num_data_storage_units):
                storage_node = SimpleStorageUnit.options(
                    num_cpus=1,
                    resources={f"node:{WORKER_NODE_IP}": 0.001},
                ).remote(storage_unit_size=math.ceil(total_storage_size / self.config.num_data_storage_units))
                self.data_system_storage_units[storage_unit_rank] = storage_node
        else:
            storage_placement_group = get_placement_group(self.config.num_data_storage_units, num_cpus_per_actor=1)
            self.storage_placement_group = storage_placement_group
            for storage_unit_rank in range(self.config.num_data_storage_units):
                storage_node = SimpleStorageUnit.options(
                    placement_group=storage_placement_group,
                    placement_group_bundle_index=storage_unit_rank,
                ).remote(storage_unit_size=math.ceil(total_storage_size / self.config.num_data_storage_units))
                self.data_system_storage_units[storage_unit_rank] = storage_node

        logger.info(f"TransferQueueStorageSimpleUnit #0 ~ #{storage_unit_rank} has been created.")

        self.data_system_controller = TransferQueueController.remote()
        logger.info("TransferQueueController has been created.")

        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)
        self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        tq_config = OmegaConf.create({}, flags={"allow_objects": True})
        tq_config.controller_info = self.data_system_controller_info
        tq_config.storage_unit_infos = self.data_system_storage_unit_infos
        self.config = OmegaConf.merge(tq_config, self.config)

        client = TransferQueueClient(client_id="Trainer", controller_info=self.data_system_controller_info)
        client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.config)

        if self.enable_profile and self.profile_output_dir is not None:
            self.profile_session = VizTracerProfileSession(
                output_dir=self.profile_output_dir,
                component_name="client",
                enabled=True,
                tracer_entries=self.profile_tracer_entries,
                min_duration_us=self.profile_min_duration_us,
                log_async=True,
            )
            ray.get(
                self.data_system_controller.configure_trace.remote(
                    output_dir=str(self.profile_output_dir),
                    component_name="controller",
                    enabled=True,
                    tracer_entries=self.profile_tracer_entries,
                    min_duration_us=self.profile_min_duration_us,
                )
            )
            ray.get(
                [
                    storage_unit.configure_trace.remote(
                        output_dir=str(self.profile_output_dir),
                        component_name=f"storage_unit_{storage_unit_rank:02d}",
                        enabled=True,
                        tracer_entries=self.profile_tracer_entries,
                        min_duration_us=self.profile_min_duration_us,
                    )
                    for storage_unit_rank, storage_unit in self.data_system_storage_units.items()
                ]
            )
        return client

    def _start_profile_round(self, round_number: int) -> None:
        if not self.enable_profile or self.profile_session is None:
            return

        ray.get(self.data_system_controller.start_trace.remote(round_number))
        ray.get(
            [storage_unit.start_trace.remote(round_number) for storage_unit in self.data_system_storage_units.values()]
        )
        self.profile_session.start(round_number)

    def _stop_profile_round(self, round_number: int) -> None:
        if not self.enable_profile or self.profile_session is None:
            return

        client_output = self.profile_session.stop()
        controller_output = ray.get(self.data_system_controller.stop_trace.remote())
        storage_outputs = ray.get(
            [storage_unit.stop_trace.remote() for storage_unit in self.data_system_storage_units.values()]
        )

        if client_output is not None:
            logger.info("Saved client VizTracer trace for round %s to %s", round_number, client_output)
        if controller_output:
            logger.info("Saved controller VizTracer trace for round %s to %s", round_number, controller_output)
        for storage_output in storage_outputs:
            if storage_output:
                logger.info("Saved storage VizTracer trace for round %s to %s", round_number, storage_output)

    def cleanup(self):
        if self.data_system_controller is not None:
            ray.kill(self.data_system_controller)
            self.data_system_controller = None

        for storage_unit in self.data_system_storage_units.values():
            ray.kill(storage_unit)
        self.data_system_storage_units = {}

        placement_group = getattr(self, "storage_placement_group", None)
        if placement_group is not None:
            ray.util.remove_placement_group(placement_group)
            self.storage_placement_group = None

    def run_bandwidth_test(self):
        logger.info("Creating large batch for bandwidth test...")
        start_create_data = time.time()
        big_input_ids, total_data_size_gb = create_complex_test_case(
            batch_size=self.config.global_batch_size,
            seq_length=self.config.seq_length,
            field_num=self.config.field_num,
        )
        logger.info(f"Data creation time: {time.time() - start_create_data:.8f}s")

        self._start_profile_round(1)
        try:
            logger.info("Starting PUT operation...")
            start_async_put = time.time()
            asyncio.run(self.data_system_client.async_put(data=big_input_ids, partition_id="train_0"))
            put_time = time.time() - start_async_put
            put_throughput_gbps = (total_data_size_gb * 8) / put_time

            time.sleep(2)

            logger.info("Starting GET_META operation...")
            start_async_get_meta = time.time()
            prompt_meta = asyncio.run(
                self.data_system_client.async_get_meta(
                    data_fields=list(big_input_ids.keys()),
                    batch_size=big_input_ids.size(0),
                    partition_id="train_0",
                    task_name="generate_sequences",
                )
            )
            logger.info(f"async_get_meta cost time: {time.time() - start_async_get_meta:.8f}s")

            time.sleep(2)

            logger.info("Starting GET_DATA operation...")
            start_async_get_data = time.time()
            asyncio.run(self.data_system_client.async_get_data(prompt_meta))
            get_time = time.time() - start_async_get_data
            get_throughput_gbps = (total_data_size_gb * 8) / get_time
        finally:
            self._stop_profile_round(1)

        mode_name = "TQ REMOTE" if self.remote_mode else "TQ NORMAL"
        logger.info("=" * 60)
        logger.info(f"{mode_name} BANDWIDTH TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"PUT Time: {put_time:.8f}s")
        logger.info(f"GET Time: {get_time:.8f}s")
        logger.info(f"PUT Throughput: {put_throughput_gbps:.8f} Gb/s")
        logger.info(f"GET Throughput: {get_throughput_gbps:.8f} Gb/s")
        logger.info(f"Network Round-trip Throughput: {(total_data_size_gb * 16) / (put_time + get_time):.8f} Gb/s")

    def run_comparison_rounds(
        self,
        *,
        config_name: str,
        config: dict[str, Any],
        rounds: int,
        num_clients: int,
        comparison_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        assignments = build_client_assignments(
            global_batch_size=int(config["global_batch_size"]),
            num_clients=num_clients,
        )
        active_num_clients = len(assignments)
        payload_bytes_by_round: list[int] = []
        put_seconds_by_round: list[float] = []
        get_seconds_by_round: list[float] = []
        workers = []
        if active_num_clients > 1:
            workers = [
                SimpleStorageComparisonWorker.remote(
                    client_id=f"SimpleStorageBenchClient_{assignment['client_id']}",
                    controller_info=self.data_system_controller_info,
                    storage_unit_infos=self.data_system_storage_unit_infos,
                )
                for assignment in assignments
            ]

        print(
            f"\n🚀 Running Config: [{config_name}] | Mode: [common-core] | "
            f"Clients: {active_num_clients} | Rounds: {rounds}"
        )

        try:
            for round_index in range(rounds):
                verify_round = round_index in (0, rounds - 1)
                if active_num_clients == 1:
                    data, payload_bytes = create_common_core_test_case(
                        batch_size=int(config["global_batch_size"]),
                        seq_length=int(config["seq_length"]),
                        field_num=int(config["field_num"]),
                    )
                    partition_id = f"perf_{config_name}_{round_index}_0"
                    should_profile_round = self.enable_profile and round_index == 0
                    if should_profile_round:
                        self._start_profile_round(round_index + 1)
                    try:
                        start_put = time.time()
                        asyncio.run(self.data_system_client.async_put(data=data, partition_id=partition_id))
                        put_time = time.time() - start_put

                        start_get = time.time()
                        prompt_meta = asyncio.run(
                            self.data_system_client.async_get_meta(
                                data_fields=list(data.keys()),
                                batch_size=int(data.batch_size[0]),
                                partition_id=partition_id,
                                task_name="generate_sequences",
                            )
                        )
                        fetched = asyncio.run(self.data_system_client.async_get_data(prompt_meta))
                        get_time = time.time() - start_get
                    finally:
                        if should_profile_round:
                            self._stop_profile_round(round_index + 1)
                    asyncio.run(self.data_system_client.async_clear_partition(partition_id=partition_id))

                    is_valid = True
                    verify_message = "✅ PASS"
                    if verify_round:
                        is_valid, verify_message = verify_tensor_dict_data(data, fetched)

                    results = [
                        {
                            "payload_bytes": payload_bytes,
                            "put_time": put_time,
                            "get_time": get_time,
                            "verified": is_valid,
                            "verify_message": verify_message,
                        }
                    ]
                else:
                    refs = []
                    for worker, assignment in zip(workers, assignments, strict=True):
                        local_config = dict(config)
                        local_config["global_batch_size"] = assignment["batch_size"]
                        refs.append(
                            worker.run_round.remote(
                                config=local_config,
                                partition_id=f"perf_{config_name}_{round_index}_{assignment['client_id']}",
                                verify_round=verify_round,
                            )
                        )
                    results = ray.get(refs)

                total_payload_bytes = sum(item["payload_bytes"] for item in results)
                put_time = max(item["put_time"] for item in results)
                get_time = max(item["get_time"] for item in results)
                payload_bytes_by_round.append(total_payload_bytes)
                put_seconds_by_round.append(put_time)
                get_seconds_by_round.append(get_time)

                put_gbps = (total_payload_bytes * 8) / put_time / (1024**3)
                get_gbps = (total_payload_bytes * 8) / get_time / (1024**3)

                verification_suffix = ""
                if verify_round:
                    failed_result = next((item for item in results if not item["verified"]), None)
                    verification_suffix = (
                        " | Verify ✅ PASS" if failed_result is None else f" | Verify {failed_result['verify_message']}"
                    )

                print(
                    f"\r  Round {round_index + 1}/{rounds}: PUT {put_gbps:.2f} Gbps | "
                    f"GET {get_gbps:.2f} Gbps ({active_num_clients} clients){verification_suffix}",
                    end="",
                )

            print("\n")
        finally:
            for worker in workers:
                ray.kill(worker)

        notes = list(comparison_context["notes"])
        notes.extend(
            [
                f"storage_units={int(config['num_data_storage_units'])}",
                f"placement={'remote' if self.remote_mode else 'local'}",
                "get_seconds includes async_get_meta + async_get_data",
            ]
        )
        total_bytes = sum(payload_bytes_by_round)
        return [
            build_comparison_result(
                implementation="simple_storage",
                workload_name=str(comparison_context["workload_name"]),
                batch_size=int(config["global_batch_size"]),
                seq_length=int(config["seq_length"]),
                field_count=int(config["field_num"]),
                non_tensor_field_count=0,
                num_shards_or_storage_units=int(config["num_data_storage_units"]),
                num_clients=active_num_clients,
                rounds=rounds,
                total_bytes=total_bytes,
                put_seconds=sum(put_seconds_by_round),
                get_seconds=sum(get_seconds_by_round),
                notes=notes,
                data_mode="tensordict",
                tensor_only=True,
                shared_columns_enabled=False,
                uuid_mode_enabled=False,
                extra_fields={
                    "payload_bytes_per_round": payload_bytes_by_round[0] if payload_bytes_by_round else 0,
                    "put_seconds_by_round": put_seconds_by_round,
                    "get_seconds_by_round": get_seconds_by_round,
                    "put_gbps_by_round": [
                        (payload_bytes * 8) / put_seconds / (1024**3)
                        for payload_bytes, put_seconds in zip(payload_bytes_by_round, put_seconds_by_round, strict=True)
                    ],
                    "get_gbps_by_round": [
                        (payload_bytes * 8) / get_seconds / (1024**3)
                        for payload_bytes, get_seconds in zip(payload_bytes_by_round, get_seconds_by_round, strict=True)
                    ],
                },
            )
        ]


def _ensure_ray_initialized(remote_mode: bool):
    if ray.is_initialized():
        return
    ray.init(address="auto" if remote_mode else None, runtime_env={"working_dir": str(repo_root)})


def _run_legacy_mode(args: argparse.Namespace):
    test_mode = args.test_mode
    if test_mode == "ray-obj-store":
        logger.info("Starting Ray Object Store bandwidth test")
        RayBandwidthTester(config=dict_conf, test_mode="obj_store").run_bandwidth_test()
        logger.info("Ray Object Store bandwidth test completed successfully!")
        return

    if test_mode == "ray-remote":
        logger.info("Starting Ray Remote Actor bandwidth test")
        RayBandwidthTester(config=dict_conf, test_mode="remote").run_bandwidth_test()
        logger.info("Ray Remote Actor bandwidth test completed successfully!")
        return

    if test_mode not in ("tq-normal", "tq-remote"):
        raise ValueError(f"Unknown test mode: {test_mode}")

    remote_mode = test_mode == "tq-remote"
    logger.info(f"Starting {'TQ Remote' if remote_mode else 'TQ Normal'} bandwidth test")
    _ensure_ray_initialized(remote_mode=remote_mode)
    tester = TQBandwidthTester(
        config=dict_conf,
        remote_mode=remote_mode,
        enable_profile=args.profile,
        profile_output_dir=args.profile_output_dir,
        profile_tracer_entries=args.profile_tracer_entries,
        profile_min_duration_us=args.profile_min_duration_us,
    )
    try:
        tester.run_bandwidth_test()
    finally:
        tester.cleanup()
        if ray.is_initialized():
            ray.shutdown()


def _build_comparison_runtime_config(config_name: str, shards: int):
    config = dict(COMPARISON_CONFIG_MAP[config_name])
    config["num_global_batch"] = 1
    config["num_data_storage_units"] = shards
    return OmegaConf.create(config)


def _run_comparison_mode(args: argparse.Namespace):
    if args.test_mode not in ("tq-normal", "tq-remote"):
        raise ValueError("comparison mode currently supports only tq-normal and tq-remote")
    if args.profile and args.num_clients != 1:
        raise ValueError("--profile currently requires --num-clients 1 for stable SimpleStorage tracing")

    remote_mode = args.test_mode == "tq-remote"
    _ensure_ray_initialized(remote_mode=remote_mode)

    run_list = [args.config] if args.config else list(COMPARISON_CONFIG_MAP.keys())
    final_results: list[dict[str, Any]] = []
    tester = None

    try:
        for config_name in run_list:
            comparison_context = build_comparison_context(
                config_name=config_name,
                comparison_mode=args.comparison_mode,
                extra_notes=["comparison_harness=simple_storage"],
            )
            tester = TQBandwidthTester(
                config=_build_comparison_runtime_config(config_name=config_name, shards=args.shards),
                remote_mode=remote_mode,
                enable_profile=args.profile,
                profile_output_dir=args.profile_output_dir,
                profile_tracer_entries=args.profile_tracer_entries,
                profile_min_duration_us=args.profile_min_duration_us,
            )
            final_results.extend(
                tester.run_comparison_rounds(
                    config_name=config_name,
                    config=dict(COMPARISON_CONFIG_MAP[config_name]) | {"num_data_storage_units": args.shards},
                    rounds=args.rounds,
                    num_clients=args.num_clients,
                    comparison_context=comparison_context,
                )
            )
            tester.cleanup()
            tester = None

        output_path = ensure_results_parent(
            resolve_results_output_path(
                implementation="simple_storage",
                workload_name=f"{args.comparison_mode}-{args.config or 'suite'}",
                output_path=args.output,
                output_dir=args.output_dir,
                repo_root=repo_root,
            )
        )
        with open(output_path, "w") as output_file:
            json.dump(final_results, output_file, indent=4)
        print(f"💾 Results saved to {output_path}")
    finally:
        if tester is not None:
            tester.cleanup()
        if ray.is_initialized():
            ray.shutdown()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TransferQueue performance benchmark")
    parser.add_argument("test_mode", nargs="?", choices=LEGACY_TEST_MODES, default="tq-normal")
    parser.add_argument("--comparison-mode", choices=[COMMON_CORE_COMPARISON_MODE], default=None)
    parser.add_argument("--config", choices=list(COMPARISON_CONFIG_MAP.keys()), default=None)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--num-clients", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--profile", action="store_true", help="Enable VizTracer profiling for the first benchmark round"
    )
    parser.add_argument(
        "--profile-output-dir",
        type=Path,
        default=None,
        help="Directory used to save VizTracer trace files",
    )
    parser.add_argument(
        "--profile-tracer-entries",
        type=int,
        default=1_000_000,
        help="VizTracer circular buffer entry count (default: 1000000)",
    )
    parser.add_argument(
        "--profile-min-duration-us",
        type=int,
        default=0,
        help="VizTracer minimum duration in microseconds (default: 0)",
    )
    return parser


def main():
    if len(sys.argv) < 2:
        print("Usage: python performance_test.py <test_mode>")
        print("Available legacy test modes:")
        print("  ray-obj-store    - Ray Object Store bandwidth test")
        print("  ray-remote       - Ray Remote Actor bandwidth test")
        print("  tq-normal        - TQ Normal mode bandwidth test")
        print("  tq-remote        - TQ Remote mode bandwidth test")
        print("Comparison mode example:")
        print("  python performance_test.py tq-normal --comparison-mode common-core --config debug --rounds 1")
        return

    args = _build_parser().parse_args()
    if args.profile and not VIZTRACER_AVAILABLE:
        raise ModuleNotFoundError("viztracer is required for --profile but is not installed in the current environment")
    if args.profile_output_dir is None and args.profile:
        args.profile_output_dir = (
            repo_root / "docs" / "perf" / "traces" / "simple_storage" / f"profile_output_{int(time.time())}"
        )
    if args.comparison_mode:
        _run_comparison_mode(args)
        return

    _run_legacy_mode(args)


if __name__ == "__main__":
    main()
