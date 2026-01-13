import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
import time
import numpy as np
import ray
import torch
from pathlib import Path
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from tensordict.utils import LinkedList

# 添加路径以确保能引用 transfer_queue
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

try:
    from transfer_queue import (
        AsyncTransferQueueClient,
        SimpleStorageUnit,
        TransferQueueController,
        process_zmq_server_info,
    )
    from transfer_queue.utils.utils import get_placement_group
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================================================
# [Configuration Map] 参考 benchmark-td
# =========================================================
CONFIG_MAP = {
    "debug": {
        "global_batch_size": 32,
        "seq_length": 128,
        "field_num": 2,
        "desc": "Debug (~32KB)"
    },
    "tiny": {
        "global_batch_size": 64,
        "seq_length": 1024,
        "field_num": 4,
        "desc": "Tiny (~1MB)"
    },
    "small": {
        "global_batch_size": 512,
        "seq_length": 12800,
        "field_num": 4,
        "desc": "Small (~100MB)"
    },
    "medium": {
        "global_batch_size": 1024,
        "seq_length": 65536,
        "field_num": 4,
        "desc": "Medium (~1GB)"
    },
    "large": {
        "global_batch_size": 2048,
        "seq_length": 128000,
        "field_num": 5,
        "desc": "Large (~5GB)"
    },
    "xlarge": {
        "global_batch_size": 4096,
        "seq_length": 128000,
        "field_num": 5,
        "desc": "X-Large (~10GB)"
    },
    "huge": {
        "global_batch_size": 4096,
        "seq_length": 128000,
        "field_num": 10,
        "desc": "Huge (~20GB)"
    }
}


# =========================================================
# [Helper Functions]
# =========================================================
def calculate_stats(data: list) -> dict:
    if not data:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(data)),
        "max": float(np.max(data)),
        "min": float(np.min(data)),  # Added per request
        "p99": float(np.percentile(data, 99))
    }


def create_complex_test_case(batch_size, seq_length, field_num):
    """
    构造测试数据，计算准确的大小
    """
    tensor_field_size_bytes = batch_size * seq_length * 4
    tensor_field_size_gb = tensor_field_size_bytes / (1024 ** 3)

    num_tensor_fields = (field_num + 1) // 2
    num_nontensor_fields = field_num // 2

    total_tensor_size_gb = tensor_field_size_gb * num_tensor_fields
    # 估算 String NonTensorData 大小 (假设 1024 bytes per sample)
    str_len = 1024
    total_nontensor_size_gb = (batch_size * str_len / (1024 ** 3)) * num_nontensor_fields
    total_size_gb = total_tensor_size_gb + total_nontensor_size_gb

    fields = {}
    for i in range(field_num):
        field_name = f"field_{i}"
        if i % 2 == 0:
            tensor_data = torch.randn(batch_size, seq_length, dtype=torch.float32)
            fields[field_name] = tensor_data
        else:
            non_tensor_data = [
                ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=str_len))
                for _ in range(batch_size)
            ]
            fields[field_name] = NonTensorData(data=non_tensor_data, batch_size=(batch_size,), device=None)

    prompt_batch = TensorDict(
        fields,
        batch_size=(batch_size,),
        device=None,
    )
    return prompt_batch, total_size_gb

def remove_placement_group(placement_group):
    if placement_group is None:
        return

    ray.util.remove_placement_group(placement_group)

def check_data_consistency(original, retrieved, path="root"):
    """
    数据一致性校验
    """
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


# =========================================================
# [Core Tester Class]
# =========================================================
class TQBandwidthTester:
    def __init__(self, target_ip=None, storage_units=8):
        self.target_ip = target_ip
        self.num_storage_units = storage_units
        self.remote_mode = target_ip is not None
        self.data_system_client = None
        self.tq_config = None
        self.data_system_controller = None
        self.data_system_storage_units = {}
        self.storage_placement_group = None

    def initialize_system(self, config_dict):
        """
        根据当前配置初始化 TransferQueue 系统
        """
        # 基础配置转换
        self.tq_config = OmegaConf.create({
            "global_batch_size": config_dict["global_batch_size"],
            "num_global_batch": 1,
            "num_data_storage_units": self.num_storage_units,
            "num_data_controllers": 1
        })

        total_storage_size = self.tq_config.global_batch_size * 2


        logger.info(f"Initializing Storage Units (Remote={self.remote_mode}, Target={self.target_ip})...")

        if self.remote_mode:
            # Remote Mode: Force placement on specific worker IP
            for rank in range(self.num_storage_units):
                self.data_system_storage_units[rank] = SimpleStorageUnit.options(
                    num_cpus=1,
                    resources={f"node:{self.target_ip}": 0.001},
                    runtime_env={"env_vars": {"OMP_NUM_THREADS": "2"}},
                ).remote(
                    storage_unit_size=math.ceil(total_storage_size / self.num_storage_units)
                )
        else:
            # Local Mode: Use placement group
            self.storage_placement_group = get_placement_group(self.num_storage_units, num_cpus_per_actor=2)
            for rank in range(self.num_storage_units):
                self.data_system_storage_units[rank] = SimpleStorageUnit.options(
                    placement_group=self.storage_placement_group,
                    placement_group_bundle_index=rank,
                    runtime_env={"env_vars": {"OMP_NUM_THREADS": "2"}},
                ).remote(
                    storage_unit_size=math.ceil(total_storage_size / self.num_storage_units)
                )

        # Controller Init
        self.data_system_controller = TransferQueueController.remote()

        # Info Collection
        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)
        self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        # Config Merge
        tq_internal_conf = OmegaConf.create({}, flags={"allow_objects": True})
        tq_internal_conf.controller_info = self.data_system_controller_info
        tq_internal_conf.storage_unit_infos = self.data_system_storage_unit_infos
        self.tq_config = OmegaConf.merge(tq_internal_conf, self.tq_config)

        # Client Init
        self.data_system_client = AsyncTransferQueueClient(
            client_id='Trainer',
            controller_info=self.data_system_controller_info
        )
        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager",
                                                           config=self.tq_config)

        return self.data_system_client

    def cleanup(self):
        """显式清理 Ray 资源"""
        logger.info("Cleaning up previous Ray resources...")

        # 1. 销毁 Controller Actor
        if self.data_system_controller:
            ray.kill(self.data_system_controller)
            self.data_system_controller = None

        # 2. 销毁 Storage Unit Actors
        if self.data_system_storage_units:
            for unit in self.data_system_storage_units.values():
                ray.kill(unit)
            self.data_system_storage_units = {}

        # 3. 销毁 Placement Group (释放预留的 CPU/资源束)
        if self.storage_placement_group:
            remove_placement_group(self.storage_placement_group)
            self.storage_placement_group = None

        # 4. 强制垃圾回收
        import gc
        gc.collect()

        # 5. 等待 Ray 调度器更新状态 (防止 Race Condition)
        time.sleep(2)

    def run_benchmark_rounds(self, config_name, config, rounds):
        # 1. Generate Data
        logger.info(f"Generating data for [{config_name}]...")
        big_input_ids, total_gb = create_complex_test_case(
            batch_size=config["global_batch_size"],
            seq_length=config["seq_length"],
            field_num=config["field_num"]
        )
        logger.info(f"Data Size: {total_gb:.4f} GB")

        put_speeds = []
        get_speeds = []

        print(f"\n🚀 Running TQ Config: [{config_name}] | Size: {total_gb:.4f} GB | Rounds: {rounds}")

        for i in range(rounds):
            partition_key = f"bench_{config_name}_{i}"

            # --- PUT ---
            start_put = time.time()
            asyncio.run(self.data_system_client.async_put(data=big_input_ids, partition_id=partition_key))
            put_time = time.time() - start_put

            put_gbps = (total_gb * 8) / put_time
            put_speeds.append(put_gbps)
            time.sleep(10)
            # --- GET META (Necessary for TQ flow but not counted in pure bandwidth usually, but we log time) ---
            # To simulate real flow, we must get meta first
            prompt_meta = asyncio.run(self.data_system_client.async_get_meta(
                data_fields=list(big_input_ids.keys()),
                batch_size=big_input_ids.size(0),
                partition_id=partition_key,
                task_name='generate_sequences',
            ))

            # --- GET DATA ---
            start_get = time.time()
            retrieved_data = asyncio.run(self.data_system_client.async_get_data(prompt_meta))
            get_time = time.time() - start_get

            get_gbps = (total_gb * 8) / get_time
            get_speeds.append(get_gbps)

            print(f"\r  Round {i + 1}/{rounds}: PUT {put_gbps:.2f} Gbps | GET {get_gbps:.2f} Gbps", end="")

            # --- Verification (First and Last Round only) ---
            if i == 0 or i == rounds - 1:
                is_consistent, msg = check_data_consistency(big_input_ids, retrieved_data)
                if not is_consistent:
                    print(f" ❌ FAIL: {msg}")
                else:
                    print(f" ✅ PASS", end="")
            asyncio.run(self.data_system_client.async_clear(partition_id=partition_key))
        print("\n")

        # Result construction
        def make_result(op, speeds):
            return {
                "scenario": "TransferQueue",
                "setting": f"{config_name} (Remote)" if self.remote_mode else f"{config_name} (Local)",
                "data_volume": f"{total_gb * 1024:.2f} MB" if total_gb * 1024 < 10 else f"{total_gb:.4f} GB",
                "operation": op,
                "payload_gb": total_gb,
                "stats_gbps": calculate_stats(speeds)
            }

        return [make_result("PUT", put_speeds), make_result("GET", get_speeds)]


# =========================================================
# [Main]
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="TransferQueue Bandwidth Benchmark")
    parser.add_argument("--ip", type=str, default=None, help="Worker Node IP. Local test if not set.")
    parser.add_argument("--config", type=str, default=None, choices=list(CONFIG_MAP.keys()),
                        help="Specific config to run.")
    parser.add_argument("--output", type=str, default="tq_benchmark_result.json", help="Output JSON file.")
    parser.add_argument("--rounds", type=int, default=20, help="Test rounds per config (default: 10)")
    parser.add_argument("--shards", type=int, default=8, help="Number of storage units (default: 8)")

    args = parser.parse_args()

    # 1. Ray Init
    current_working_dir = os.getcwd()
    if not ray.is_initialized():
        ray.init(
            address="auto" if args.ip else None,
            runtime_env={"working_dir": current_working_dir}
        )

    logger.info(f"Ray Initialized. Remote Target: {args.ip if args.ip else 'Local'}")

    # 2. Setup Tester
    tester = TQBandwidthTester(target_ip=args.ip, storage_units=args.shards)

    # 3. Execution Loop
    run_list = [args.config] if args.config else list(CONFIG_MAP.keys())
    final_results = []

    try:
        for cfg_name in run_list:
            cfg = CONFIG_MAP[cfg_name]

            # Re-initialize system per config to match batch sizes (optional, but safer for memory alloc)
            # Or we can init once if we want to reuse actors.
            # Here we follow TQ pattern: usually TQ is long-lived, but config changes might require reset.
            # To be safe and isolating tests, we re-init client logic, but Actors are persistent if we don't kill them.
            # For this script, let's keep actors alive but re-configure client if needed.
            # Simplified: Call initialize_system each time for cleanliness or do it once if config compatible.
            # Given TQ structure, let's init once per config loop to ensure correct 'storage_unit_size' calculation based on config.

            tester.cleanup()

            tester.initialize_system(cfg)
            res = tester.run_benchmark_rounds(cfg_name, cfg, args.rounds)
            final_results.extend(res)

        # 4. Save Results
        with open(args.output, "w") as f:
            json.dump(final_results, f, indent=4)
        logger.info(f"💾 Results saved to {args.output}")

    except Exception as e:
        logger.error(f"❌ Critical Error: {e}", exc_info=True)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()