import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import ray
import torch

# Local imports - add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from metrics import Metric
from tq_client import get_transferqueue_client, TransferQueueClient
from tq_mgr import TransferQueueManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_MAP = {
    # --- Level 1: Smoke Test (KB/MB) ---
    "debug": {
        "global_batch_size": 32,
        "seq_length": 128,
        "field_num": 2,
        "str_length": 32,
        "desc": "Debug (~32KB) - Quick verification"
    },
    "tiny": {
        "global_batch_size": 64,
        "seq_length": 1024,
        "field_num": 4,
        "str_length": 64,
        "desc": "Tiny (~1MB) - Latency focus"
    },

    # --- Level 2: Throughput Ramp-up (MB) ---
    "small": {
        "global_batch_size": 512,
        "seq_length": 12800,
        "field_num": 4,
        "str_length": 256,
        "desc": "Small (~100MB) - Network warm-up"
    },

    # --- Level 3: Standard Workload (GB) ---
    "medium": {
        "global_batch_size": 1024,
        "seq_length": 65536,  # 64k context
        "field_num": 4,
        "str_length": 512,
        "desc": "Medium (~1GB) - Standard workload"
    },
    "large": {
        "global_batch_size": 2048,
        "seq_length": 128000, # 128k context
        "field_num": 5,
        "str_length": 1024,
        "desc": "Large (~5GB) - High throughput"
    },

    # --- Level 4: Stress Test (10GB+) ---
    "xlarge": {
        "global_batch_size": 4096,
        "seq_length": 128000,
        "field_num": 5,
        "str_length": 1024,
        "desc": "X-Large (~10GB) - Memory bound"
    },
    "huge": {
        "global_batch_size": 4096,
        "seq_length": 128000,
        "field_num": 10,
        "str_length": 1024,
        "desc": "Huge (~20GB) - Full saturation limit"
    }
}

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def calculate_stats(data: List[float]) -> Dict[str, float]:
    if not data:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(data)),
        "max": float(np.max(data)),
        "min": float(np.min(data)),
        "p99": float(np.percentile(data, 99))
    }


def verify_data_integrity(original_data, fetched_data):
    """Verify data integrity: Keys, Shapes, Values"""
    print(" | 🔍 Verifying...", end=" ")
    sys.stdout.flush()

    try:
        if fetched_data is None:
            print("❌ FAIL: Fetched data is None")
            return False

        orig_keys = set(original_data.keys())
        fetched_keys = set(fetched_data.keys())

        if orig_keys != fetched_keys:
            print(f"❌ FAIL: Keys mismatch. Diff: {orig_keys ^ fetched_keys}")
            return False

        for key in orig_keys:
            orig_tensor = original_data[key]
            fet_val = fetched_data[key]

            # Handle List[Tensor] -> Tensor case
            if isinstance(fet_val, list):
                try:
                    fet_tensor = torch.stack(fet_val)
                except:
                    continue
            else:
                fet_tensor = fet_val

            if isinstance(orig_tensor, torch.Tensor) and isinstance(fet_tensor, torch.Tensor):
                # TensorDock may add batch dimension, squeeze if needed
                if orig_tensor.shape != fet_tensor.shape:
                    if fet_tensor.ndim == orig_tensor.ndim + 1:
                        fet_tensor = fet_tensor.squeeze(1)

                    if orig_tensor.shape != fet_tensor.shape:
                        print(f"❌ FAIL: Shape mismatch for {key} (Orig: {orig_tensor.shape}, Fet: {fet_tensor.shape})")
                        return False

                if not torch.equal(orig_tensor.cpu(), fet_tensor.cpu()):
                    print(f"❌ FAIL: Value mismatch for {key}")
                    return False

        print("✅ PASS", end="")
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


# =========================================================
# Data Factory
# =========================================================
def generate_data(config: Dict, mode: str):
    print(f"⏳ Generating test data ({mode})...")

    batch_size = config["global_batch_size"]
    seq_length = config["seq_length"]

    data_payload = {}
    total_gb = 0.0

    if mode == "dict":
        field_num = config["field_num"]
        str_len = config["str_length"]

        num_tensor_fields = (field_num + 1) // 2
        num_nontensor_fields = field_num // 2

        tensor_bytes = batch_size * seq_length * 4 * num_tensor_fields
        nontensor_bytes = batch_size * str_len * 1 * num_nontensor_fields
        total_gb = (tensor_bytes + nontensor_bytes) / (1024 ** 3)

        for i in range(field_num):
            field_name = f"field_{i}"
            if i % 2 == 0:
                data_payload[field_name] = torch.randn(batch_size, seq_length, dtype=torch.float32)
            else:
                data_payload[field_name] = torch.randint(0, 127, (batch_size, str_len), dtype=torch.uint8)

    elif mode == "tensor":
        effective_seq = seq_length * config["field_num"]
        total_gb = (batch_size * effective_seq * 4) / (1024 ** 3)
        data_payload["data"] = torch.randn(batch_size, effective_seq, dtype=torch.float32)

    print(f"✅ Data Ready. Payload: {total_gb:.4f} GB")
    return data_payload, total_gb


# =========================================================
# Core Benchmark Logic
# =========================================================
def run_single_benchmark(td_client: TransferQueueClient, config_name, config, data_mode, test_rounds):
    data_dict, total_gb = generate_data(config, data_mode)

    topic_name = f"perf_{config_name}_{data_mode}_{int(time.time())}"
    field_names = list(data_dict.keys())

    # Register Topic
    td_client.add_topic(
        prompts_num=config["global_batch_size"] * test_rounds * 2,
        n_samples_per_prompt=1,
        experience_columns=field_names,
        experience_consumers=["learner"],
        metrics=Metric(),
        topic=topic_name
    )

    put_speeds = []
    get_speeds = []

    print(
        f"\n🚀 Running Config: [{config_name}] | Mode: [{data_mode}] | Size: {total_gb:.4f} GB | Rounds: {test_rounds}")

    for i in range(test_rounds):
        start_idx = i * config["global_batch_size"]
        end_idx = (i + 1) * config["global_batch_size"]
        current_indices = list(range(start_idx, end_idx))

        # --- PUT ---
        td_client.get_allocation_for_new_groups(topic=topic_name, num_new_groups=config["global_batch_size"])

        start_put = time.time()
        td_client.put_experience(data_dict=data_dict, indexes=current_indices, topic=topic_name)
        put_time = time.time() - start_put

        put_gbps = (total_gb * 8) / put_time
        put_speeds.append(put_gbps)
        print(f"\r  Round {i + 1}/{test_rounds}: PUT {put_gbps:.2f} Gbps", end="")

        # --- GET ---
        time.sleep(2)
        start_get = time.time()

        fetched_data, _ = td_client.get_experience(
            consumer="learner",
            experience_columns=field_names,
            experience_count=config["global_batch_size"],
            indexes=current_indices,
            get_n_samples=False,
            topic=topic_name
        )
        get_time = time.time() - start_get
        if fetched_data:
            get_gbps = (total_gb * 8) / get_time
            get_speeds.append(get_gbps)
            print(f" | GET {get_gbps:.2f} Gbps", end="")

            # Verify on first and last round
            if i == 0 or i == test_rounds - 1:
                verify_data_integrity(data_dict, fetched_data)
        else:
            print(" | GET FAILED", end="")

    print("\n")
    td_client.delete_topic(topic_name)

    # Build result
    def make_result(op, speeds):
        return {
            "scenario": "TensorDock",
            "setting": f"{config_name} ({data_mode})",
            "data_volume": f"{total_gb * 1024:.2f} MB" if total_gb * 1024 < 10 else f"{total_gb:.4f} GB",
            "operation": op,
            "payload_gb": total_gb,
            "stats_gbps": calculate_stats(speeds)
        }

    return [make_result("PUT", put_speeds), make_result("GET", get_speeds)]


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="TensorDock Bandwidth Benchmark")
    parser.add_argument("--ip", type=str, default=None, help="Worker Node IP. Local if not set.")
    parser.add_argument("--config", type=str, default=None, choices=list(CONFIG_MAP.keys()), help="Config name.")
    parser.add_argument("--mode", type=str, default="dict", choices=["dict", "tensor"], help="Data mode.")
    parser.add_argument("--output", type=str, default="res.json", help="Output file.")

    parser.add_argument("--rounds", type=int, default=20, help="Test rounds (default: 20)")
    parser.add_argument("--shards", type=int, default=8, help="nums_td_data (default: 8)")
    parser.add_argument("--cpus", type=int, default=8, help="Manager num_cpus (default: 8)")

    # Dual-node arguments (matching other benchmarks)
    parser.add_argument("--role", type=str, default="single", choices=["single", "head", "worker"], help="Node role")
    parser.add_argument("--head-ip", type=str, help="Head node IP")
    parser.add_argument("--worker-ip", type=str, help="Worker node IP")
    parser.add_argument("--wait-nodes", type=int, default=0, help="Wait for N nodes to be available before starting")

    args = parser.parse_args()

    # 1. Init Ray
    # - For standalone local testing: init without address
    # - For Docker/dual-node: use address="auto" (Ray cluster started externally)
    current_working_dir = os.getcwd()
    if not ray.is_initialized():
        if args.role == "single" and not args.ip and not args.worker_ip:
            # Standalone local mode - start a new local Ray cluster
            ray.init(runtime_env={"working_dir": current_working_dir})
        else:
            # Docker or dual-node mode - connect to existing Ray cluster
            ray.init(address="auto", runtime_env={"working_dir": current_working_dir})

    logger.info(f"Ray initialized. Role: {args.role}")

    # Worker mode: just keep alive
    if args.role == "worker":
        logger.info(f"Worker node started. Connected to Head {args.head_ip}. Waiting for tasks...")
        while True:
            time.sleep(10)

    # Wait for nodes if specified
    if args.wait_nodes > 0:
        logger.info(f"Waiting for {args.wait_nodes} nodes...")
        while len(ray.nodes()) < args.wait_nodes:
            logger.info(f"Current nodes: {len(ray.nodes())}/{args.wait_nodes}")
            time.sleep(2)
        logger.info("All nodes are ready!")

    # 2. Init Manager with placement on worker IP if specified
    if args.worker_ip:
        target_ip = args.worker_ip
        logger.info(f"Placing Manager on worker node: {target_ip}")
        mgr_options = {"num_cpus": args.cpus, "resources": {f"node:{target_ip}": 0.001}}
    elif args.ip:
        target_ip = args.ip
        logger.info(f"Target Worker IP: {target_ip}")
        mgr_options = {"num_cpus": args.cpus, "resources": {f"node:{target_ip}": 0.001}}
    else:
        target_ip = ray.nodes()[0]["NodeManagerAddress"]
        logger.info(f"Local mode. Target IP: {target_ip}")
        mgr_options = {"num_cpus": args.cpus}

    logger.info(f"Manager Config: cpus={args.cpus}, shards={args.shards}")

    mgr = TransferQueueManager.options(**mgr_options).remote(
        nums_tq_data=args.shards,
        base_port=find_free_port()
    )
    ray.get(mgr.init_ready.remote())

    td_client = get_transferqueue_client()

    # 3. Execution
    run_list = [args.config] if args.config else list(CONFIG_MAP.keys())
    final_results = []

    try:
        for cfg_name in run_list:
            cfg = CONFIG_MAP[cfg_name]
            res = run_single_benchmark(td_client, cfg_name, cfg, args.mode, args.rounds)
            final_results.extend(res)

        with open(args.output, "w") as f:
            json.dump(final_results, f, indent=4)
        print(f"💾 Results saved to {args.output}")

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🧹 Cleaning up...")
        try:
            ray.get(mgr.shutdown.remote())
        except:
            pass
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
