# coding:utf-8
import asyncio
import gc
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from tensordict.utils import LinkedList

# --- Path Setup ---
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import (
    AsyncTransferQueueClient,
    SimpleStorageUnit,
    TransferQueueController,
    process_zmq_server_info,
)
from transfer_queue.utils.utils import get_placement_group

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
HEAD_NODE_IP = "61.28.30.25"
WORKER_NODE_IP = "61.28.30.27"

# 统一配置
config_str = """
  global_batch_size: 1024
  seq_length: 8192
  field_num: 10
  num_global_batch: 1
  num_data_storage_units: 8
  num_data_controllers: 1
"""
dict_conf = OmegaConf.create(config_str)


# --- Test Utilities ---

def create_complex_test_case(batch_size=None, seq_length=None, field_num=None):
    """创建测试数据"""
    tensor_field_size_bytes = batch_size * seq_length * 4
    tensor_field_size_gb = tensor_field_size_bytes / (1024 ** 3)
    num_tensor_fields = (field_num + 1) // 2
    num_nontensor_fields = field_num // 2
    total_tensor_size_gb = tensor_field_size_gb * num_tensor_fields
    total_nontensor_size_gb = (batch_size * 1024 / (1024 ** 3)) * num_nontensor_fields
    total_size_gb = total_tensor_size_gb + total_nontensor_size_gb

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

    prompt_batch = TensorDict(fields, batch_size=(batch_size,), device=None)
    return prompt_batch, total_size_gb


def check_data_consistency(original, retrieved, path="root"):
    """数据一致性校验 (简化版)"""
    try:
        if isinstance(original, list) and isinstance(retrieved, LinkedList):
            retrieved = list(retrieved)

        if type(original) != type(retrieved):
            return False, f"Type mismatch: {type(original)} vs {type(retrieved)}"

        if isinstance(original, TensorDict):
            for key in original.keys():
                valid, msg = check_data_consistency(original[key], retrieved[key], path=f"{path}.{key}")
                if not valid: return False, msg
            return True, "Passed"

        elif isinstance(original, torch.Tensor):
            if original.shape != retrieved.shape: return False, "Shape mismatch"
            # 为了速度，只检查少量元素
            if not torch.equal(original.flatten()[:100].cpu(), retrieved.flatten()[:100].cpu()):
                return False, "Tensor values mismatch (sample)"
            return True, "Passed"

        elif 'NonTensorData' in str(type(original)):
            if original.data != retrieved.data: return False, "NonTensorData mismatch"
            return True, "Passed"
        else:
            if original != retrieved: return False, "Value mismatch"
            return True, "Passed"
    except Exception as e:
        return False, str(e)


class BenchmarkStats:
    """统计工具类"""

    def __init__(self, name, data_size_gb):
        self.name = name
        self.data_size_gb = data_size_gb
        self.put_times = []
        self.get_times = []

    def record(self, put_time, get_time):
        self.put_times.append(put_time)
        self.get_times.append(get_time)

    def print_summary(self):
        def _calc(times):
            arr = np.array(times)
            throughput = (self.data_size_gb * 8) / arr  # Gbps
            return {
                "max": np.max(throughput),
                "min": np.min(throughput),
                "avg": np.mean(throughput),
                "p99": np.percentile(throughput, 99)
            }

        put_stats = _calc(self.put_times)
        get_stats = _calc(self.get_times)

        logger.info(f"\n{'=' * 20} [{self.name}] Benchmark Summary (100 runs) {'=' * 20}")
        logger.info(f"Data Size: {self.data_size_gb:.4f} GB")
        logger.info(f"{'Metric':<10} | {'PUT (Gbps)':<15} | {'GET (Gbps)':<15}")
        logger.info("-" * 46)
        logger.info(f"{'Max':<10} | {put_stats['max']:.4f}{'':<9} | {get_stats['max']:.4f}")
        logger.info(f"{'Min':<10} | {put_stats['min']:.4f}{'':<9} | {get_stats['min']:.4f}")
        logger.info(f"{'Avg':<10} | {put_stats['avg']:.4f}{'':<9} | {get_stats['avg']:.4f}")
        logger.info(f"{'P99':<10} | {put_stats['p99']:.4f}{'':<9} | {get_stats['p99']:.4f}")
        logger.info("=" * 66 + "\n")


# --- Benchmark Implementations ---

# 1. Ray Object Store Tester
class RayObjectStoreTester:
    def __init__(self, data, data_size):
        self.data = data
        self.data_size = data_size
        self.ref = None

    async def setup(self):
        pass

    async def run_once(self):
        # PUT
        start = time.time()
        self.ref = ray.put(self.data)
        put_time = time.time() - start

        # GET
        start = time.time()
        _ = ray.get(self.ref)
        get_time = time.time() - start

        return put_time, get_time, _


# 2. Ray Remote Actor Tester
@ray.remote
class DataActor:
    def __init__(self):
        self._data = None

    def put(self, data):
        self._data = data

    def get(self):
        return self._data


class RayActorTester:
    def __init__(self, data, data_size):
        self.data = data
        self.data_size = data_size
        self.actor = None

    async def setup(self):
        self.actor = DataActor.remote()

    async def run_once(self):
        # PUT
        start = time.time()
        ray.get(self.actor.put.remote(self.data))
        put_time = time.time() - start

        # GET
        start = time.time()
        retrieved = ray.get(self.actor.get.remote())
        get_time = time.time() - start

        return put_time, get_time, retrieved


# 3. Transfer Queue Tester
class TQTester:
    def __init__(self, config, data, data_size, remote_mode=False):
        self.config = config
        self.data = data
        self.data_size = data_size
        self.remote_mode = remote_mode
        self.client = None
        self.controller = None
        self.storage_units = {}

    async def setup(self):
        # 初始化 TQ 系统资源
        total_storage_size = (self.config.global_batch_size * self.config.num_global_batch)

        if self.remote_mode:
            for rank in range(self.config.num_data_storage_units):
                self.storage_units[rank] = SimpleStorageUnit.options(
                    num_cpus=1, resources={f"node:{WORKER_NODE_IP}": 0.001}
                ).remote(storage_unit_size=math.ceil(total_storage_size / self.config.num_data_storage_units))
        else:
            pg = get_placement_group(self.config.num_data_storage_units, num_cpus_per_actor=4)
            for rank in range(self.config.num_data_storage_units):
                self.storage_units[rank] = SimpleStorageUnit.options(
                    placement_group=pg, placement_group_bundle_index=rank
                ).remote(storage_unit_size=math.ceil(total_storage_size / self.config.num_data_storage_units))

        self.controller = TransferQueueController.remote()

        # 注册信息
        ctrl_info = process_zmq_server_info(self.controller)
        store_infos = process_zmq_server_info(self.storage_units)

        tq_config = OmegaConf.create({}, flags={"allow_objects": True})
        tq_config.controller_info = ctrl_info
        tq_config.storage_unit_infos = store_infos
        self.full_config = OmegaConf.merge(tq_config, self.config)

        self.client = AsyncTransferQueueClient(client_id='Tester', controller_info=ctrl_info)
        self.client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=self.full_config)
        logger.info(f"TQ System initialized (Remote={self.remote_mode})")

    async def run_once(self):
        partition_id = f"test_{random.randint(0, 100000)}"

        # PUT
        start = time.time()
        await self.client.async_put(data=self.data, partition_id=partition_id)
        put_time = time.time() - start

        # GET META (included in overhead or separate? Usually meta is fast, adding to GET flow)
        meta = await self.client.async_get_meta(
            data_fields=list(self.data.keys()),
            batch_size=self.data.size(0),
            partition_id=partition_id,
            task_name='benchmark'
        )

        # GET
        start = time.time()
        retrieved = await self.client.async_get_data(meta)
        get_time = time.time() - start

        return put_time, get_time, retrieved


# --- Main Runner ---

async def run_benchmark_suite():
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    # 1. 准备数据 (生成一次，复用)
    logger.info("Generating test data...")
    test_data, data_size_gb = create_complex_test_case(
        batch_size=dict_conf.global_batch_size,
        seq_length=dict_conf.seq_length,
        field_num=dict_conf.field_num
    )
    logger.info(f"Test Data Size: {data_size_gb:.4f} GB")

    # 定义测试配置
    configs = [
        ("Ray Object Store", RayObjectStoreTester(test_data, data_size_gb)),
        # ("Ray Remote Actor", RayActorTester(test_data, data_size_gb)), # 可根据需要开启
        ("TQ Normal", TQTester(dict_conf, test_data, data_size_gb, remote_mode=False)),
        # ("TQ Remote", TQTester(dict_conf, test_data, data_size_gb, remote_mode=True)), # 可根据需要开启
    ]

    NUM_RUNS = 100

    for name, tester in configs:
        logger.info(f"\nStarting benchmark for: {name}")
        stats = BenchmarkStats(name, data_size_gb)

        await tester.setup()

        # 预热
        logger.info("Warmup run...")
        _, _, _ = await tester.run_once()

        logger.info(f"Running {NUM_RUNS} iterations...")
        for i in range(NUM_RUNS):
            if i % 10 == 0:
                print(f"Progress: {i}/{NUM_RUNS}", end="\r")

            put_t, get_t, retrieved_data = await tester.run_once()
            stats.record(put_t, get_t)

            # 第一轮检查一致性
            if i == 0:
                ok, msg = check_data_consistency(test_data, retrieved_data)
                logger.info(f"Data Consistency Check: {'PASS' if ok else f'FAIL ({msg})'}")

            # 清理内存，防止 Accumulation
            del retrieved_data
            gc.collect()

        stats.print_summary()


def main():
    try:
        asyncio.run(run_benchmark_suite())
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted.")


if __name__ == "__main__":
    main()
