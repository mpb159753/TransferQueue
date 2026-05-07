import argparse
import inspect
import json
import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import numpy as np  # noqa: E402
import ray  # noqa: E402
import torch  # noqa: E402
from metrics import Metric  # noqa: E402
from tq_client import TransferQueueClient, get_transferqueue_client  # noqa: E402
from tq_mgr import TransferQueueManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_MAP = {
    "debug": {
        "global_batch_size": 32,
        "seq_length": 128,
        "field_num": 2,
        "str_length": 32,
        "desc": "Debug (~32KB) - Quick verification",
    },
    "tiny": {
        "global_batch_size": 64,
        "seq_length": 1024,
        "field_num": 4,
        "str_length": 64,
        "desc": "Tiny (~1MB) - Latency focus",
    },
    "small": {
        "global_batch_size": 512,
        "seq_length": 12800,
        "field_num": 4,
        "str_length": 256,
        "desc": "Small (~100MB) - Network warm-up",
    },
    "medium": {
        "global_batch_size": 1024,
        "seq_length": 65536,
        "field_num": 4,
        "str_length": 512,
        "desc": "Medium (~1GB) - Standard workload",
    },
    "large": {
        "global_batch_size": 2048,
        "seq_length": 128000,
        "field_num": 5,
        "str_length": 1024,
        "desc": "Large (~5GB) - High throughput",
    },
    "xlarge": {
        "global_batch_size": 4096,
        "seq_length": 128000,
        "field_num": 5,
        "str_length": 1024,
        "desc": "X-Large (~10GB) - Memory bound",
    },
    "huge": {
        "global_batch_size": 4096,
        "seq_length": 128000,
        "field_num": 10,
        "str_length": 1024,
        "desc": "Huge (~20GB) - Full saturation limit",
    },
}

DTYPE_CONFIGS = [
    {"dtype": torch.float32, "bytes_per_elem": 4},
    {"dtype": torch.int64, "bytes_per_elem": 8},
    {"dtype": torch.float64, "bytes_per_elem": 8},
    {"dtype": torch.int32, "bytes_per_elem": 4},
    {"dtype": torch.float16, "bytes_per_elem": 2},
]

MANAGER_TARGET_IP_ARG_NAMES = (
    "storage_node_ips",
    "shard_node_ips",
    "target_ips",
    "node_ips",
    "worker_ips",
)
AUTO_WORKER_HOSTS_ENV = "VC_WORKER_HOSTS"
AUTO_CURRENT_INSTANCE_ENV_NAMES = ("MA-CURRENT_INSTANCE_NAME", "MA_CURRENT_INSTANCE_NAME", "HOSTNAME")
AUTO_HEAD_HOST_SUFFIX = "worker-0"
AUTO_RAY_PORT_ENV_NAMES = ("VC_RAY_PORT", "RAY_PORT")
DEFAULT_RAY_PORT = 6379
RAY_START_RETRY_COUNT = 30
RAY_START_RETRY_INTERVAL_SECONDS = 2
RAY_CONNECT_RETRY_COUNT = 30
RAY_CONNECT_RETRY_INTERVAL_SECONDS = 2


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def sync_stage(flag_to_create: str, flag_to_wait: str) -> None:
    """Synchronize with an external profiler process via flag files."""
    with open(flag_to_create, "w") as flag_file:
        flag_file.write("1")

    while not os.path.exists(flag_to_wait):
        time.sleep(0.05)

    try:
        os.remove(flag_to_wait)
    except OSError:
        pass


def parse_target_ips(raw_target_ips: Any) -> list[str]:
    if not raw_target_ips:
        return []

    if isinstance(raw_target_ips, str):
        candidates = raw_target_ips.split(",")
    else:
        candidates = list(raw_target_ips)

    target_ips = []
    for ip in candidates:
        normalized_ip = str(ip).strip()
        if normalized_ip and normalized_ip not in target_ips:
            target_ips.append(normalized_ip)
    return target_ips


def _get_first_env_entry(env: Mapping[str, str], names: tuple[str, ...]) -> tuple[str | None, str | None]:
    for name in names:
        value = env.get(name)
        if value:
            return name, value.strip()
    return None, None


def _normalize_host_identity(host: str) -> str:
    return host.strip().split(".", 1)[0]


def _resolve_ray_node_address(host: str) -> str:
    candidates = [host.strip(), _normalize_host_identity(host)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return socket.gethostbyname(candidate)
        except socket.gaierror:
            continue
    logger.warning("Unable to resolve host '%s' to IP; using the original value for Ray placement.", host)
    return host.strip()


def _resolve_ray_port(env: Mapping[str, str]) -> int:
    for env_name in AUTO_RAY_PORT_ENV_NAMES:
        raw_value = env.get(env_name)
        if not raw_value:
            continue
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer, got: {raw_value}") from exc
    return DEFAULT_RAY_PORT


def _is_tcp_endpoint_ready(host: str, port: int, timeout_seconds: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _find_head_host(worker_hosts: list[str]) -> str:
    for host in worker_hosts:
        if _normalize_host_identity(host).endswith(AUTO_HEAD_HOST_SUFFIX):
            return host
    raise ValueError(
        f"{AUTO_WORKER_HOSTS_ENV} must include a host ending with '{AUTO_HEAD_HOST_SUFFIX}': {worker_hosts}"
    )


def resolve_env_multi_node_config(
    env: Mapping[str, str] | None = None,
    requested_num_clients: int | None = None,
) -> dict[str, Any] | None:
    env = env or os.environ
    worker_hosts = parse_target_ips(env.get(AUTO_WORKER_HOSTS_ENV))
    current_instance_source, current_instance = _get_first_env_entry(env, AUTO_CURRENT_INSTANCE_ENV_NAMES)

    if not worker_hosts or not current_instance:
        return None

    head_host = _find_head_host(worker_hosts)
    ordered_hosts = [head_host, *[host for host in worker_hosts if host != head_host]]
    normalized_host_map = {_normalize_host_identity(host): host for host in ordered_hosts}
    normalized_current_instance = _normalize_host_identity(current_instance)
    resolved_current_host = normalized_host_map.get(normalized_current_instance)

    if resolved_current_host is None:
        raise ValueError(
            f"Current instance '{current_instance}' is not listed in {AUTO_WORKER_HOSTS_ENV}: {ordered_hosts}"
        )

    is_head = resolved_current_host == head_host
    ray_port = _resolve_ray_port(env)
    resolved_target_ips = [_resolve_ray_node_address(host) for host in ordered_hosts]
    resolved_head_ip = _resolve_ray_node_address(head_host)
    resolved_current_ip = _resolve_ray_node_address(resolved_current_host)
    if requested_num_clients is not None and requested_num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    resolved_num_clients = requested_num_clients if requested_num_clients is not None else len(ordered_hosts)
    return {
        "enabled": True,
        "current_instance": current_instance,
        "current_instance_source": current_instance_source,
        "resolved_current_host": resolved_current_host,
        "resolved_current_ip": resolved_current_ip,
        "head_host": head_host,
        "head_ip": resolved_head_ip,
        "target_ips": resolved_target_ips,
        "target_hosts": ordered_hosts,
        "role": "head" if is_head else "worker",
        "wait_nodes": len(ordered_hosts) if is_head else 0,
        "num_clients": resolved_num_clients,
        "distribute_clients": len(ordered_hosts) > 1,
        "manager_target_ip": resolved_head_ip,
        "ray_port": ray_port,
        "ray_address": f"{resolved_head_ip}:{ray_port}",
    }


def resolve_runtime_config(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    requested_num_clients = getattr(args, "num_clients", None)
    env_multi_node_config = resolve_env_multi_node_config(env, requested_num_clients=requested_num_clients)
    if env_multi_node_config is not None:
        return env_multi_node_config

    target_ips = _resolve_storage_target_ips(args)
    resolved_num_clients = requested_num_clients if requested_num_clients is not None else 1
    return {
        "enabled": False,
        "current_instance": None,
        "current_instance_source": None,
        "resolved_current_host": None,
        "resolved_current_ip": None,
        "head_host": args.head_ip,
        "head_ip": args.head_ip,
        "target_ips": target_ips,
        "target_hosts": target_ips,
        "role": args.role,
        "wait_nodes": args.wait_nodes,
        "num_clients": resolved_num_clients,
        "distribute_clients": args.distribute_clients,
        "manager_target_ip": _resolve_manager_target_ip(args, target_ips),
        "ray_port": None,
        "ray_address": "auto",
    }


def _build_ray_start_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "ray.scripts.scripts", "start", *args]


def _run_ray_start_command(
    command: list[str],
    description: str,
    retry_count: int = 1,
    retry_interval_seconds: int = 0,
) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, retry_count + 1):
        try:
            logger.info("%s via: %s", description, " ".join(command))
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == retry_count:
                break
            logger.info(
                "%s failed on attempt %s/%s. Retrying in %ss.",
                description,
                attempt,
                retry_count,
                retry_interval_seconds,
            )
            time.sleep(retry_interval_seconds)
    if last_error is not None:
        raise RuntimeError(f"{description} failed after {retry_count} attempt(s)") from last_error


def _connect_to_ray_cluster(
    address: str,
    runtime_env: dict[str, Any],
    retry_count: int = RAY_CONNECT_RETRY_COUNT,
    retry_interval_seconds: int = RAY_CONNECT_RETRY_INTERVAL_SECONDS,
) -> None:
    last_error: ConnectionError | None = None
    for attempt in range(1, retry_count + 1):
        try:
            ray.init(address=address, runtime_env=runtime_env)
            return
        except ConnectionError as exc:
            last_error = exc
            if attempt == retry_count:
                break
            logger.info(
                "Connecting to Ray at %s failed on attempt %s/%s. Retrying in %ss.",
                address,
                attempt,
                retry_count,
                retry_interval_seconds,
            )
            time.sleep(retry_interval_seconds)
    if last_error is not None:
        raise last_error


def _ensure_ray_runtime(runtime_config: dict[str, Any], working_dir: str) -> None:
    if ray.is_initialized():
        return

    runtime_env = {"working_dir": working_dir}
    if runtime_config["role"] == "single" and not runtime_config["target_ips"]:
        ray.init(runtime_env=runtime_env)
        return

    ray_address = str(runtime_config["ray_address"])
    if not runtime_config["enabled"]:
        _connect_to_ray_cluster(ray_address, runtime_env)
        return

    if runtime_config["role"] == "head":
        if not _is_tcp_endpoint_ready(str(runtime_config["head_ip"]), int(runtime_config["ray_port"])):
            _run_ray_start_command(
                _build_ray_start_command(
                    "--head",
                    f"--node-ip-address={runtime_config['head_ip']}",
                    f"--port={runtime_config['ray_port']}",
                    "--disable-usage-stats",
                ),
                f"Starting Ray head on {runtime_config['head_ip']}:{runtime_config['ray_port']}",
            )
    else:
        for attempt in range(1, RAY_START_RETRY_COUNT + 1):
            if _is_tcp_endpoint_ready(str(runtime_config["head_ip"]), int(runtime_config["ray_port"])):
                break
            if attempt == RAY_START_RETRY_COUNT:
                raise RuntimeError(
                    f"Ray head {runtime_config['head_ip']}:{runtime_config['ray_port']} did not become reachable"
                )
            logger.info(
                "Waiting for Ray head %s:%s before joining (%s/%s). Retrying in %ss.",
                runtime_config["head_ip"],
                runtime_config["ray_port"],
                attempt,
                RAY_START_RETRY_COUNT,
                RAY_START_RETRY_INTERVAL_SECONDS,
            )
            time.sleep(RAY_START_RETRY_INTERVAL_SECONDS)
        _run_ray_start_command(
            _build_ray_start_command(
                f"--address={ray_address}",
                f"--node-ip-address={runtime_config['resolved_current_ip']}",
                "--disable-usage-stats",
            ),
            f"Joining Ray head at {ray_address} from {runtime_config['resolved_current_ip']}",
            retry_count=RAY_START_RETRY_COUNT,
            retry_interval_seconds=RAY_START_RETRY_INTERVAL_SECONDS,
        )

    _connect_to_ray_cluster(ray_address, runtime_env)


def _build_node_resource_map() -> dict[str, str]:
    node_resource_map: dict[str, str] = {}
    for node in ray.nodes():
        resources = node.get("Resources", {})
        node_resource_key = next((key for key in resources if key.startswith("node:")), None)
        node_address = node.get("NodeManagerAddress")
        if node_resource_key and node_address:
            node_resource_map[str(node_address)] = node_resource_key
    return node_resource_map


def _resolve_node_resource_key(target_ip: str, node_resource_map: Mapping[str, str]) -> str:
    return node_resource_map.get(target_ip, f"node:{target_ip}")


def calculate_stats(data: list[float]) -> dict[str, float]:
    if not data:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(data)),
        "max": float(np.max(data)),
        "min": float(np.min(data)),
        "p99": float(np.percentile(data, 99)),
    }


def _generate_regular_tensor(batch_size: int, seq_length: int, dtype: torch.dtype) -> torch.Tensor:
    return _generate_tensor((batch_size, seq_length), dtype)


def _generate_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if dtype in (torch.int32, torch.int64):
        return torch.randint(0, 10000, shape, dtype=dtype)
    return torch.randn(shape, dtype=dtype)


def _select_jagged_inner_dim(batch_size: int, seq_length: int) -> int:
    total_elements = batch_size * seq_length
    for inner_dim in range(min(seq_length, 16), 1, -1):
        total_rows, remainder = divmod(total_elements, inner_dim)
        if remainder == 0 and total_rows > batch_size:
            return inner_dim
    return 1


def _build_uneven_row_lengths(total_rows: int, batch_size: int, field_index: int) -> list[int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if batch_size == 1:
        return [total_rows]

    row_lengths = [1] * batch_size
    remaining_rows = total_rows - batch_size
    if remaining_rows <= 0:
        return row_lengths

    weights = [field_index + item_index + 1 for item_index in range(batch_size)]
    total_weight = sum(weights)

    for item_index, weight in enumerate(weights):
        extra_rows = (remaining_rows * weight) // total_weight
        row_lengths[item_index] += extra_rows

    assigned_rows = sum(row_lengths)
    leftover_rows = total_rows - assigned_rows
    for offset in range(leftover_rows):
        row_lengths[(field_index + offset) % batch_size] += 1

    if len(set(row_lengths)) == 1 and total_rows > batch_size:
        row_lengths[0] += 1
        row_lengths[-1] -= 1

    return row_lengths


def _generate_jagged_tensor_list(
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype,
    field_index: int,
) -> list[torch.Tensor]:
    inner_dim = _select_jagged_inner_dim(batch_size, seq_length)
    total_elements = batch_size * seq_length
    total_rows = total_elements // inner_dim
    row_lengths = _build_uneven_row_lengths(total_rows, batch_size, field_index)
    trailing_dim = inner_dim if inner_dim > 1 else 1

    return [_generate_tensor((row_count, trailing_dim), dtype) for row_count in row_lengths]


def _payload_bytes_for_config(batch_size: int, seq_length: int, field_num: int, mode: str) -> int:
    bytes_per_sample = seq_length * sum(
        DTYPE_CONFIGS[field_index % len(DTYPE_CONFIGS)]["bytes_per_elem"] for field_index in range(field_num)
    )
    if mode == "dict":
        return batch_size * bytes_per_sample
    if mode == "tensor":
        return batch_size * bytes_per_sample
    raise ValueError(f"Unsupported mode: {mode}")


def _payload_bytes_from_data(data_payload: Any) -> int:
    if isinstance(data_payload, torch.Tensor):
        return data_payload.numel() * data_payload.element_size()
    if isinstance(data_payload, dict):
        return sum(_payload_bytes_from_data(value) for value in data_payload.values())
    if isinstance(data_payload, (list, tuple)):
        return sum(_payload_bytes_from_data(value) for value in data_payload)
    return 0


def _extract_int_list(value: Any) -> list[int] | None:
    if isinstance(value, range):
        return [int(item) for item in value]
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, (int, np.integer)) for item in value):
        return [int(item) for item in value]
    return None


def _extract_allocated_indexes(allocation_result: Any) -> list[int] | None:
    direct_indexes = _extract_int_list(allocation_result)
    if direct_indexes is not None:
        return direct_indexes

    candidate_keys = (
        "indexes",
        "indices",
        "global_indexes",
        "global_indices",
        "gids",
        "group_indexes",
        "group_indices",
    )

    if isinstance(allocation_result, dict):
        for key in candidate_keys:
            if key in allocation_result:
                indexes = _extract_int_list(allocation_result[key])
                if indexes is not None:
                    return indexes
        for value in allocation_result.values():
            indexes = _extract_allocated_indexes(value)
            if indexes is not None:
                return indexes
        return None

    for key in candidate_keys:
        if hasattr(allocation_result, key):
            indexes = _extract_int_list(getattr(allocation_result, key))
            if indexes is not None:
                return indexes

    if isinstance(allocation_result, tuple):
        for item in allocation_result:
            indexes = _extract_allocated_indexes(item)
            if indexes is not None:
                return indexes

    return None


def _allocate_round_indexes(
    td_client: TransferQueueClient,
    topic_name: str,
    allocation_count: int,
    fallback_start_idx: int,
) -> list[int]:
    allocation_result = td_client.get_allocation_for_new_groups(topic=topic_name, num_new_groups=allocation_count)
    allocated_indexes = _extract_allocated_indexes(allocation_result)

    if allocated_indexes is None:
        return list(range(fallback_start_idx, fallback_start_idx + allocation_count))

    if len(allocated_indexes) != allocation_count:
        raise ValueError(
            f"Allocation returned {len(allocated_indexes)} indexes, expected {allocation_count}: {allocated_indexes}"
        )
    return allocated_indexes


def generate_data(config: dict[str, Any], mode: str) -> tuple[dict[str, Any], float]:
    batch_size = int(config["global_batch_size"])
    seq_length = int(config["seq_length"])
    field_num = int(config["field_num"])

    data_payload: dict[str, Any] = {}

    if mode == "dict":
        for field_index in range(field_num):
            dtype = DTYPE_CONFIGS[field_index % len(DTYPE_CONFIGS)]["dtype"]
            data_payload[f"field_{field_index}"] = _generate_regular_tensor(batch_size, seq_length, dtype)
    elif mode == "tensor":
        for field_index in range(field_num):
            dtype = DTYPE_CONFIGS[field_index % len(DTYPE_CONFIGS)]["dtype"]
            data_payload[f"field_{field_index}"] = _generate_jagged_tensor_list(
                batch_size=batch_size,
                seq_length=seq_length,
                dtype=dtype,
                field_index=field_index,
            )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    total_gb = _payload_bytes_from_data(data_payload) / (1024**3)
    return data_payload, total_gb


def _normalize_retrieved_tensor(original_tensor: torch.Tensor, retrieved_value: Any) -> Any:
    if isinstance(retrieved_value, list) and all(isinstance(item, torch.Tensor) for item in retrieved_value):
        try:
            retrieved_value = torch.stack(retrieved_value)
        except RuntimeError:
            return retrieved_value

    if isinstance(retrieved_value, torch.Tensor):
        while retrieved_value.ndim > original_tensor.ndim and 1 in retrieved_value.shape:
            squeezed = False
            for dim_index, dim_size in enumerate(retrieved_value.shape):
                if dim_size == 1:
                    retrieved_value = retrieved_value.squeeze(dim_index)
                    squeezed = True
                    break
            if not squeezed:
                break

    return retrieved_value


def _normalize_retrieved_sequence(retrieved_value: Any) -> Any:
    if isinstance(retrieved_value, tuple):
        return list(retrieved_value)

    if isinstance(retrieved_value, torch.Tensor) and getattr(retrieved_value, "is_nested", False):
        return list(retrieved_value.unbind())

    return retrieved_value


def check_data_consistency(original_data: Any, fetched_data: Any, path: str = "root") -> tuple[bool, str]:
    if fetched_data is None:
        return False, f"[{path}] fetched data is None"

    if isinstance(original_data, dict):
        if not isinstance(fetched_data, dict):
            return False, f"[{path}] type mismatch: {type(original_data)} vs {type(fetched_data)}"
        if set(original_data.keys()) != set(fetched_data.keys()):
            return False, f"[{path}] keys mismatch: {set(original_data.keys()) ^ set(fetched_data.keys())}"
        for key in original_data:
            is_valid, message = check_data_consistency(original_data[key], fetched_data[key], path=f"{path}.{key}")
            if not is_valid:
                return False, message
        return True, "Passed"

    if isinstance(original_data, list):
        fetched_data = _normalize_retrieved_sequence(fetched_data)
        if not isinstance(fetched_data, list):
            return False, f"[{path}] type mismatch: {type(original_data)} vs {type(fetched_data)}"
        if len(original_data) != len(fetched_data):
            return False, f"[{path}] list length mismatch: {len(original_data)} vs {len(fetched_data)}"
        for item_index, original_item in enumerate(original_data):
            is_valid, message = check_data_consistency(
                original_item,
                fetched_data[item_index],
                path=f"{path}[{item_index}]",
            )
            if not is_valid:
                return False, message
        return True, "Passed"

    if isinstance(original_data, torch.Tensor):
        fetched_data = _normalize_retrieved_tensor(original_data, fetched_data)
        if not isinstance(fetched_data, torch.Tensor):
            return False, f"[{path}] type mismatch: {type(original_data)} vs {type(fetched_data)}"
        if original_data.shape != fetched_data.shape:
            return False, f"[{path}] shape mismatch: {original_data.shape} vs {fetched_data.shape}"
        if original_data.dtype != fetched_data.dtype:
            return False, f"[{path}] dtype mismatch: {original_data.dtype} vs {fetched_data.dtype}"
        if not torch.equal(original_data.cpu(), fetched_data.cpu()):
            return False, f"[{path}] tensor values mismatch"
        return True, "Passed"

    if original_data != fetched_data:
        return False, f"[{path}] value mismatch: {original_data} vs {fetched_data}"
    return True, "Passed"


def verify_data_integrity(original_data: Any, fetched_data: Any) -> tuple[bool, str]:
    is_valid, message = check_data_consistency(original_data, fetched_data)
    if is_valid:
        return True, "✅ PASS"
    return False, f"❌ FAIL: {message}"


def _normalize_verification_result(result: Any) -> tuple[bool, str]:
    if isinstance(result, tuple) and len(result) == 2:
        return bool(result[0]), str(result[1])
    return bool(result), "✅ PASS" if result else "❌ FAIL"


def build_client_assignments(
    global_batch_size: int,
    num_clients: int,
    target_ips: list[str] | None = None,
    distribute_clients: bool = False,
) -> list[dict[str, Any]]:
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")

    target_ips = target_ips or []
    active_num_clients = min(global_batch_size, num_clients)
    local_batch_size, remainder = divmod(global_batch_size, active_num_clients)
    assignments = []
    start_idx = 0
    for client_id in range(active_num_clients):
        batch_size = local_batch_size + (1 if client_id < remainder else 0)
        target_ip = target_ips[client_id % len(target_ips)] if distribute_clients and target_ips else None
        end_idx = start_idx + batch_size
        assignments.append(
            {
                "client_id": client_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "batch_size": batch_size,
                "target_ip": target_ip,
            }
        )
        start_idx = end_idx
    return assignments


def attach_indexes_to_assignments(
    assignments: list[dict[str, Any]],
    allocated_indexes: list[int],
) -> list[dict[str, Any]]:
    if sum(int(assignment["batch_size"]) for assignment in assignments) != len(allocated_indexes):
        raise ValueError("Allocated indexes count does not match the total batch size across client assignments")

    assigned_with_indexes = []
    cursor = 0
    for assignment in assignments:
        batch_size = int(assignment["batch_size"])
        assigned_with_indexes.append({**assignment, "indexes": allocated_indexes[cursor : cursor + batch_size]})
        cursor += batch_size
    return assigned_with_indexes


def build_manager_init_kwargs(
    manager_cls: type[Any],
    shard_count: int,
    base_port: int,
    target_ips: list[str] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"nums_tq_data": shard_count, "base_port": base_port}
    if not target_ips:
        return kwargs

    try:
        signature = inspect.signature(manager_cls.__init__)
    except (TypeError, ValueError):
        signature = None

    if signature is None:
        logger.info("Unable to inspect manager signature; shard placement will fall back to actor placement only.")
        return kwargs

    for arg_name in MANAGER_TARGET_IP_ARG_NAMES:
        if arg_name in signature.parameters:
            kwargs[arg_name] = list(target_ips)
            return kwargs

    logger.info("Manager signature has no shard-target parameter; shard placement will use actor placement only.")
    return kwargs


def _get_field_names(config: dict[str, Any], data_mode: str) -> list[str]:
    del data_mode
    field_num = int(config.get("field_num", 1))
    return [f"field_{index}" for index in range(field_num)]


def _format_progress_line(
    round_index: int,
    round_count: int,
    put_gbps: float,
    get_gbps: float,
    num_clients: int,
    verification_suffix: str = "",
) -> str:
    client_suffix = f" ({num_clients} clients)" if num_clients > 1 else ""
    return (
        f"\r  Round {round_index + 1}/{round_count}: PUT {put_gbps:.2f} Gbps | "
        f"GET {get_gbps:.2f} Gbps{client_suffix}{verification_suffix}"
    )


def _make_results(
    config_name: str,
    data_mode: str,
    total_gb: float,
    put_speeds: list[float],
    get_speeds: list[float],
    num_clients: int,
    target_ips: list[str],
) -> list[dict[str, Any]]:
    location = "Remote" if target_ips else "Local"
    client_suffix = ", Multi-Client" if num_clients > 1 else ""
    setting = f"{config_name} ({data_mode}, {location}{client_suffix})"

    def make_result(operation: str, speeds: list[float]) -> dict[str, Any]:
        return {
            "scenario": "TensorDock",
            "setting": setting,
            "data_volume": f"{total_gb * 1024:.2f} MB" if total_gb * 1024 < 10 else f"{total_gb:.4f} GB",
            "operation": operation,
            "payload_gb": total_gb,
            "num_clients": num_clients,
            "stats_gbps": calculate_stats(speeds),
        }

    return [make_result("PUT", put_speeds), make_result("GET", get_speeds)]


@ray.remote
def _run_client_round(
    client_id: int,
    config: dict[str, Any],
    data_mode: str,
    topic_name: str,
    field_names: list[str],
    assignment: dict[str, Any],
    round_offset: int,
    verify_round: bool,
) -> dict[str, Any]:
    td_client = get_transferqueue_client()

    local_config = dict(config)
    local_config["global_batch_size"] = assignment["batch_size"]

    data_dict, total_gb = generate_data(local_config, data_mode)
    current_indices = list(assignment["indexes"])

    start_put = time.time()
    td_client.put_experience(data_dict=data_dict, indexes=current_indices, topic=topic_name)
    put_time = time.time() - start_put

    time.sleep(2)

    start_get = time.time()
    fetched_data, _ = td_client.get_experience(
        consumer="learner",
        experience_columns=field_names,
        experience_count=assignment["batch_size"],
        indexes=current_indices,
        get_n_samples=False,
        topic=topic_name,
    )
    get_time = time.time() - start_get

    is_valid = True
    verify_message = "✅ PASS"
    if verify_round:
        is_valid, verify_message = _normalize_verification_result(verify_data_integrity(data_dict, fetched_data))

    return {
        "client_id": client_id,
        "payload_gb": total_gb,
        "put_time": put_time,
        "get_time": get_time,
        "verified": is_valid,
        "verify_message": verify_message,
    }


def _run_single_client_benchmark(
    td_client: TransferQueueClient,
    topic_name: str,
    config_name: str,
    config: dict[str, Any],
    data_mode: str,
    test_rounds: int,
    num_clients: int,
    target_ips: list[str],
    enable_profile: bool,
) -> list[dict[str, Any]]:
    data_dict, total_gb = generate_data(config, data_mode)
    field_names = list(data_dict.keys())
    put_speeds: list[float] = []
    get_speeds: list[float] = []

    print(
        f"\n🚀 Running Config: [{config_name}] | Mode: [{data_mode}] | Size: {total_gb:.4f} GB | Rounds: {test_rounds}"
    )

    for round_index in range(test_rounds):
        round_offset = round_index * int(config["global_batch_size"])
        current_indices = _allocate_round_indexes(
            td_client=td_client,
            topic_name=topic_name,
            allocation_count=int(config["global_batch_size"]),
            fallback_start_idx=round_offset,
        )

        if round_index == 0 and enable_profile:
            sync_stage("init_ready.flag", "put_start.flag")
        start_put = time.time()
        td_client.put_experience(data_dict=data_dict, indexes=current_indices, topic=topic_name)
        put_time = time.time() - start_put
        if round_index == 0 and enable_profile:
            sync_stage("put_done.flag", "get_prepare.flag")

        time.sleep(2)

        if round_index == 0 and enable_profile:
            sync_stage("get_ready.flag", "get_start.flag")
        start_get = time.time()
        fetched_data, _ = td_client.get_experience(
            consumer="learner",
            experience_columns=field_names,
            experience_count=int(config["global_batch_size"]),
            indexes=current_indices,
            get_n_samples=False,
            topic=topic_name,
        )
        get_time = time.time() - start_get

        put_gbps = (total_gb * 8) / put_time
        get_gbps = (total_gb * 8) / get_time
        put_speeds.append(put_gbps)
        get_speeds.append(get_gbps)

        verification_suffix = ""
        if round_index in (0, test_rounds - 1):
            is_valid, verify_message = _normalize_verification_result(verify_data_integrity(data_dict, fetched_data))
            verification_suffix = f" | Verify {verify_message}"
            if not is_valid:
                verification_suffix = f" | Verify {verify_message}"

        print(
            _format_progress_line(
                round_index=round_index,
                round_count=test_rounds,
                put_gbps=put_gbps,
                get_gbps=get_gbps,
                num_clients=num_clients,
                verification_suffix=verification_suffix,
            ),
            end="",
        )

    print("\n")
    return _make_results(config_name, data_mode, total_gb, put_speeds, get_speeds, num_clients, target_ips)


def _run_multi_client_benchmark(
    td_client: TransferQueueClient,
    topic_name: str,
    config_name: str,
    config: dict[str, Any],
    data_mode: str,
    test_rounds: int,
    num_clients: int,
    target_ips: list[str],
    distribute_clients: bool,
    node_resource_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    assignments = build_client_assignments(
        global_batch_size=int(config["global_batch_size"]),
        num_clients=num_clients,
        target_ips=target_ips,
        distribute_clients=distribute_clients,
    )
    active_num_clients = len(assignments)
    field_names = _get_field_names(config, data_mode)
    total_gb = _payload_bytes_for_config(
        batch_size=int(config["global_batch_size"]),
        seq_length=int(config["seq_length"]),
        field_num=int(config["field_num"]),
        mode=data_mode,
    ) / (1024**3)
    put_speeds: list[float] = []
    get_speeds: list[float] = []

    print(
        f"\n🚀 Running Config: [{config_name}] | Mode: [{data_mode}] | Size: {total_gb:.4f} GB | "
        f"Clients: {active_num_clients} | Rounds: {test_rounds}"
    )

    for round_index in range(test_rounds):
        round_offset = round_index * int(config["global_batch_size"])
        verify_round = round_index in (0, test_rounds - 1)
        refs = []
        round_indexes = _allocate_round_indexes(
            td_client=td_client,
            topic_name=topic_name,
            allocation_count=int(config["global_batch_size"]),
            fallback_start_idx=round_offset,
        )
        round_assignments = attach_indexes_to_assignments(assignments, round_indexes)

        for assignment in round_assignments:
            remote_call = _run_client_round
            if assignment["target_ip"]:
                node_resource_key = _resolve_node_resource_key(assignment["target_ip"], node_resource_map or {})
                remote_call = _run_client_round.options(resources={node_resource_key: 0.001})

            refs.append(
                remote_call.remote(
                    assignment["client_id"],
                    config,
                    data_mode,
                    topic_name,
                    field_names,
                    assignment,
                    round_offset,
                    verify_round,
                )
            )

        results = ray.get(refs)

        total_payload_gb = sum(item["payload_gb"] for item in results)
        put_time = max(item["put_time"] for item in results)
        get_time = max(item["get_time"] for item in results)
        put_gbps = (total_payload_gb * 8) / put_time
        get_gbps = (total_payload_gb * 8) / get_time
        put_speeds.append(put_gbps)
        get_speeds.append(get_gbps)

        verification_suffix = ""
        if verify_round:
            failed_result = next((item for item in results if not item["verified"]), None)
            if failed_result is None:
                verification_suffix = " | Verify ✅ PASS"
            else:
                verification_suffix = f" | Verify {failed_result['verify_message']}"

        print(
            _format_progress_line(
                round_index=round_index,
                round_count=test_rounds,
                put_gbps=put_gbps,
                get_gbps=get_gbps,
                num_clients=active_num_clients,
                verification_suffix=verification_suffix,
            ),
            end="",
        )

    print("\n")
    return _make_results(config_name, data_mode, total_gb, put_speeds, get_speeds, active_num_clients, target_ips)


def run_single_benchmark(
    td_client: TransferQueueClient,
    config_name: str,
    config: dict[str, Any],
    data_mode: str,
    test_rounds: int,
    num_clients: int = 1,
    target_ips: list[str] | None = None,
    distribute_clients: bool = False,
    enable_profile: bool = False,
    node_resource_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    target_ips = target_ips or []
    topic_name = f"perf_{config_name}_{data_mode}_{int(time.time())}"
    field_names = _get_field_names(config, data_mode)

    td_client.add_topic(
        prompts_num=int(config["global_batch_size"]) * test_rounds * 2,
        n_samples_per_prompt=1,
        experience_columns=field_names,
        experience_consumers=["learner"],
        metrics=Metric(),
        topic=topic_name,
    )

    try:
        if num_clients == 1:
            return _run_single_client_benchmark(
                td_client=td_client,
                topic_name=topic_name,
                config_name=config_name,
                config=config,
                data_mode=data_mode,
                test_rounds=test_rounds,
                num_clients=num_clients,
                target_ips=target_ips,
                enable_profile=enable_profile,
            )
        if enable_profile:
            logger.warning("Profile sync is only supported for single-client mode, matching put_benchmark.py.")
        return _run_multi_client_benchmark(
            td_client=td_client,
            topic_name=topic_name,
            config_name=config_name,
            config=config,
            data_mode=data_mode,
            test_rounds=test_rounds,
            num_clients=num_clients,
            target_ips=target_ips,
            distribute_clients=distribute_clients,
            node_resource_map=node_resource_map,
        )
    finally:
        td_client.delete_topic(topic_name)


def _resolve_storage_target_ips(args: argparse.Namespace) -> list[str]:
    if args.storage_node_ips:
        return parse_target_ips(args.storage_node_ips)
    if args.role == "head" and args.head_ip and args.worker_ip:
        return parse_target_ips([args.head_ip, args.worker_ip])
    if args.worker_ip:
        return parse_target_ips(args.worker_ip)
    return parse_target_ips(args.ip)


def _resolve_manager_target_ip(args: argparse.Namespace, target_ips: list[str]) -> str | None:
    if args.worker_ip:
        return args.worker_ip
    if args.ip:
        return args.ip
    if target_ips:
        return target_ips[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorDock Bandwidth Benchmark")
    parser.add_argument("--ip", type=str, default=None, help="Deprecated alias for a single target node IP")
    parser.add_argument(
        "--storage-node-ips",
        type=str,
        default=None,
        help="Comma-separated node IPs used for shard placement and optional client pinning",
    )
    parser.add_argument("--config", type=str, default=None, choices=list(CONFIG_MAP.keys()), help="Config name")
    parser.add_argument("--mode", type=str, default="dict", choices=["dict", "tensor"], help="Data mode")
    parser.add_argument("--output", type=str, default="res.json", help="Output file")
    parser.add_argument("--rounds", type=int, default=20, help="Test rounds (default: 20)")
    parser.add_argument("--shards", type=int, default=8, help="Number of manager shards (default: 8)")
    parser.add_argument("--cpus", type=int, default=8, help="Manager num_cpus (default: 8)")
    parser.add_argument("--profile", action="store_true", help="Enable profile sync (requires external profiler)")
    parser.add_argument(
        "--num-clients",
        type=int,
        default=None,
        help="Number of concurrent benchmark clients (defaults to node count in auto multi-node mode)",
    )
    parser.add_argument(
        "--distribute-clients",
        action="store_true",
        help="Pin benchmark clients to --storage-node-ips via round-robin",
    )
    parser.add_argument("--role", type=str, default="single", choices=["single", "head", "worker"], help="Node role")
    parser.add_argument("--head-ip", type=str, help="Head node IP")
    parser.add_argument("--worker-ip", type=str, help="Worker node IP")
    parser.add_argument("--wait-nodes", type=int, default=0, help="Wait for N nodes to be available before starting")

    args = parser.parse_args()
    runtime_config = resolve_runtime_config(args)
    target_ips = list(runtime_config["target_ips"])

    current_working_dir = os.getcwd()
    if runtime_config["enabled"]:
        logger.info(
            "Auto multi-node mode enabled from %s/%s. current=%s, resolved_current=%s, current_ip=%s, head=%s, head_ip=%s, nodes=%s, clients=%s, ray_address=%s",
            AUTO_WORKER_HOSTS_ENV,
            runtime_config["current_instance_source"],
            runtime_config["current_instance"],
            runtime_config["resolved_current_host"],
            runtime_config["resolved_current_ip"],
            runtime_config["head_host"],
            runtime_config["head_ip"],
            runtime_config["target_hosts"],
            runtime_config["num_clients"],
            runtime_config["ray_address"],
        )

    _ensure_ray_runtime(runtime_config, current_working_dir)
    logger.info(f"Ray initialized. Role: {runtime_config['role']}")

    node_resource_map = _build_node_resource_map()
    logger.info("Detected Ray node resources: %s", node_resource_map)

    if runtime_config["role"] == "worker":
        logger.info(f"Worker node started. Connected to Head {runtime_config['head_host']}. Waiting for tasks...")
        while True:
            time.sleep(10)

    if runtime_config["wait_nodes"] > 0:
        logger.info(f"Waiting for {runtime_config['wait_nodes']} nodes...")
        while len(ray.nodes()) < runtime_config["wait_nodes"]:
            logger.info(f"Current nodes: {len(ray.nodes())}/{runtime_config['wait_nodes']}")
            time.sleep(2)
        logger.info("All nodes are ready!")

    manager_target_ip = runtime_config["manager_target_ip"]
    manager_options: dict[str, Any] = {"num_cpus": args.cpus}
    if manager_target_ip:
        manager_resource_key = _resolve_node_resource_key(manager_target_ip, node_resource_map)
        logger.info("Placing manager on node: %s (%s)", manager_target_ip, manager_resource_key)
        manager_options["resources"] = {manager_resource_key: 0.001}
    else:
        logger.info("Local mode. Manager will run on the local Ray node.")

    manager_init_kwargs = build_manager_init_kwargs(
        manager_cls=TransferQueueManager,
        shard_count=args.shards,
        base_port=find_free_port(),
        target_ips=target_ips,
    )

    logger.info(
        f"Manager Config: cpus={args.cpus}, shards={args.shards}, "
        f"clients={runtime_config['num_clients']}, storage_targets={target_ips or ['local']}"
    )

    mgr = TransferQueueManager.options(**manager_options).remote(**manager_init_kwargs)
    ray.get(mgr.init_ready.remote())

    td_client = get_transferqueue_client()
    run_list = [args.config] if args.config else list(CONFIG_MAP.keys())
    final_results: list[dict[str, Any]] = []

    try:
        for config_name in run_list:
            config = CONFIG_MAP[config_name]
            results = run_single_benchmark(
                td_client=td_client,
                config_name=config_name,
                config=config,
                data_mode=args.mode,
                test_rounds=args.rounds,
                num_clients=runtime_config["num_clients"],
                target_ips=target_ips,
                distribute_clients=runtime_config["distribute_clients"],
                enable_profile=args.profile,
                node_resource_map=node_resource_map,
            )
            final_results.extend(results)

        with open(args.output, "w") as output_file:
            json.dump(final_results, output_file, indent=4)
        print(f"💾 Results saved to {args.output}")
    except Exception as exc:
        print(f"❌ Critical Error: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        print("🧹 Cleaning up...")
        try:
            ray.get(mgr.shutdown.remote())
        except Exception:
            pass
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
