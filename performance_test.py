# coding:utf-8
import asyncio
import logging
import math
import random
import sys
import time
from pathlib import Path

# from viztracer import VizTracer # Removed as we use py-spy
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from tensordict.utils import LinkedList

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import (
    AsyncTransferQueueClient,
    SimpleStorageUnit,
    TransferQueueController,
    process_zmq_server_info,
)
from transfer_queue.utils.utils import get_placement_group

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ray双机集群配置
HEAD_NODE_IP = "61.28.30.25"
WORKER_NODE_IP = "61.28.30.27"

config_str = """
  global_batch_size: 10240
  seq_length: 81920
  field_num: 10
  num_global_batch: 1
  num_data_storage_units: 8
  num_data_controllers: 1
"""
dict_conf = OmegaConf.create(config_str)


def create_complex_test_case(batch_size=None, seq_length=None, field_num=None):
    # ... (保持原样，省略以节省空间) ...
    tensor_field_size_bytes = batch_size * seq_length * 4
    tensor_field_size_gb = tensor_field_size_bytes / (1024 ** 3)

    num_tensor_fields = (field_num + 1) // 2
    num_nontensor_fields = field_num // 2

    total_tensor_size_gb = tensor_field_size_gb * num_tensor_fields
    total_nontensor_size_gb = (batch_size * 1024 / (1024 ** 3)) * num_nontensor_fields
    total_size_gb = total_tensor_size_gb + total_nontensor_size_gb

    logger.info(f"Total data size: {total_size_gb:.6f} GB")

    fields = {}
    for i in range(field_num):
        field_name = f"field_{i}"
        if i % 2 == 0:
            tensor_data = torch.randn(batch_size, seq_length, dtype=torch.float32)
            fields[field_name] = tensor_data
        else:
            str_length = 1024
            non_tensor_data = [
                ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=str_length))
                for _ in range(batch_size)
            ]
            fields[field_name] = NonTensorData(data=non_tensor_data, batch_size=(batch_size,), device=None)

    batch_size_tuple = (batch_size,)
    prompt_batch = TensorDict(
        fields,
        batch_size=batch_size_tuple,
        device=None,
    )
    return prompt_batch, total_size_gb


def check_data_consistency(original, retrieved, path="root"):
    try:
        if isinstance(original, list) and isinstance(retrieved, LinkedList):
            retrieved = list(retrieved)

        if type(original) != type(retrieved):
            return False, f"[{path}] Type mismatch: {type(original)} vs {type(retrieved)}"

        if isinstance(original, TensorDict):
            if set(original.keys()) != set(retrieved.keys()):
                return False, f"[{path}] Keys mismatch: {original.keys()} vs {retrieved.keys()}"
            for key in original.keys():
                is_valid, msg = check_data_consistency(original[key], retrieved[key], path=f"{path}.{key}")
                if not is_valid:
                    return False, msg
            return True, "Passed"

        elif isinstance(original, torch.Tensor):
            if original.shape != retrieved.shape:
                return False, f"[{path}] Tensor shape mismatch: {original.shape} vs {retrieved.shape}"
            t1 = original.cpu()
            t2 = retrieved.cpu()
            if not torch.equal(t1, t2):
                return False, f"[{path}] Tensor values mismatch"
            return True, "Passed"

        elif 'NonTensorData' in str(type(original)):
            if original.data != retrieved.data:
                return False, f"[{path}] NonTensorData content mismatch"
            return True, "Passed"

        else:
            if original != retrieved:
                return False, f"[{path}] Value mismatch: {original} vs {retrieved}"
            return True, "Passed"

    except Exception as e:
        return False, f"[{path}] Exception during check: {str(e)}"


class TQBandwidthTester:
    def __init__(self, config, remote_mode=False):
        self.config = config
        self.remote_mode = remote_mode
        self.data_system_client = self._initialize_data_system()

    def sync_stage(self, stage_name):
        """
        通用同步函数：
        1. 创建 {stage_name}_ready.flag，通知 Shell 我准备好了。
        2. 阻塞等待 {stage_name}_start.flag，等待 Shell 完成 profiler 操作。
        """
        ready_flag = Path(f"{stage_name}_ready.flag")
        start_flag = Path(f"{stage_name}_start.flag")

        # 清理可能存在的旧 start 信号
        if start_flag.exists():
            start_flag.unlink()

        logger.info(f"[{stage_name}] Waiting for profiler/orchestrator...")
        ready_flag.touch()

        while not start_flag.exists():
            time.sleep(0.1)

        logger.info(f"[{stage_name}] Signal received. Proceeding...")
        if ready_flag.exists():
            ready_flag.unlink()

    def _initialize_data_system(self):
        total_storage_size = (self.config.global_batch_size * self.config.num_global_batch)
        self.data_system_storage_units = {}

        if self.remote_mode:
            for storage_unit_rank in range(self.config.num_data_storage_units):
                storage_node = SimpleStorageUnit.options(
                    num_cpus=1,
                    resources={f"node:{WORKER_NODE_IP}": 0.001},
                    runtime_env={"env_vars": {"OMP_NUM_THREADS": "2"}},
                ).remote(
                    storage_unit_size=math.ceil(total_storage_size / self.config.num_data_storage_units)
                )
                self.data_system_storage_units[storage_unit_rank] = storage_node
        else:
            storage_placement_group = get_placement_group(self.config.num_data_storage_units, num_cpus_per_actor=4)
            for storage_unit_rank in range(self.config.num_data_storage_units):
                storage_node = SimpleStorageUnit.options(
                    placement_group=storage_placement_group,
                    placement_group_bundle_index=storage_unit_rank,
                    runtime_env={"env_vars": {"OMP_NUM_THREADS": "2"}},
                ).remote(
                    storage_unit_size=math.ceil(total_storage_size / self.config.num_data_storage_units)
                )
                self.data_system_storage_units[storage_unit_rank] = storage_node

        logger.info(f"TransferQueueStorageSimpleUnit initialized.")
        self.data_system_controller = TransferQueueController.remote()

        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)
        self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        tq_config = OmegaConf.create({}, flags={"allow_objects": True})
        tq_config.controller_info = self.data_system_controller_info
        tq_config.storage_unit_infos = self.data_system_storage_unit_infos
        self.config = OmegaConf.merge(tq_config, self.config)

        self.data_system_client = AsyncTransferQueueClient(
            client_id='Trainer',
            controller_info=self.data_system_controller_info
        )
        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.config)
        return self.data_system_client

    def run_bandwidth_test(self):
        # 1. 构造数据
        logger.info("Creating large batch for bandwidth test...")
        big_input_ids, total_data_size_gb = create_complex_test_case(
            batch_size=self.config.global_batch_size,
            seq_length=self.config.seq_length,
            field_num=self.config.field_num
        )

        # --- STAGE 1: PUT ---
        self.sync_stage("put_phase")  # 告知 Shell 准备好 PUT，Shell 将启动 PUT Profiler

        logger.info(f"Starting PUT operation...")
        start_async_put = time.time()

        asyncio.run(self.data_system_client.async_put(data=big_input_ids, partition_id=f"train_0"))

        end_async_put = time.time()
        put_time = end_async_put - start_async_put
        put_throughput_gbps = (total_data_size_gb * 8) / put_time
        logger.info(f"async_put cost time: {put_time:.8f}s")
        logger.info(f"PUT Throughput: {put_throughput_gbps:.8f} Gb/s")

        # --- STAGE 2: META (Intermediate) ---
        # 告知 Shell PUT 结束，Shell 将停止 PUT Profiler。
        # 同时等待 Shell 确认后再跑 Meta (防止 Meta 跑太快进入 Profiler 停止的间隙)
        self.sync_stage("put_done_meta_phase")

        logger.info("Starting GET_META operation (Not Profiled)...")
        start_async_get_meta = time.time()
        prompt_meta = asyncio.run(self.data_system_client.async_get_meta(
            data_fields=list(big_input_ids.keys()),
            batch_size=big_input_ids.size(0),
            partition_id=f"train_0",
            task_name='generate_sequences',
        ))
        end_async_get_meta = time.time()
        logger.info(f"async_get_meta cost time: {end_async_get_meta - start_async_get_meta:.8f}s")

        # --- STAGE 3: GET ---
        self.sync_stage("get_phase")  # 告知 Shell 准备好 GET，Shell 将启动 GET Profiler

        logger.info(f"Starting GET_DATA operation...")
        start_async_get_data = time.time()
        data = asyncio.run(self.data_system_client.async_get_data(prompt_meta))
        end_async_get_data = time.time()
        get_time = end_async_get_data - start_async_get_data
        get_throughput_gbps = (total_data_size_gb * 8) / get_time

        logger.info(f"async_get_data cost time: {get_time:.8f}s")
        logger.info(f"GET Throughput: {get_throughput_gbps:.8f} Gb/s")

        # --- 验证 ---
        logger.info("Verifying data integrity (TQ System)...")
        is_consistent, msg = check_data_consistency(big_input_ids, data)
        consistency_status = "PASS" if is_consistent else f"FAIL ({msg})"

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
        logger.info(f"Data Consistency: {consistency_status}")


def main():
    test_mode = "tq-normal"
    # test_mode = "tq-remote"

    if test_mode in ["tq-normal", "tq-remote"]:
        remote_mode = (test_mode == "tq-remote")
        logger.info(f"Starting {test_mode} bandwidth test")
        tester = TQBandwidthTester(config=dict_conf, remote_mode=remote_mode)
        tester.run_bandwidth_test()
    else:
        print(f"Unknown test mode: {test_mode}")


if __name__ == "__main__":
    main()
