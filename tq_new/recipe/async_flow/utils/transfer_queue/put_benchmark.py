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
from contextlib import nullcontext
from pathlib import Path
from typing import Any

tq_new_root = Path(__file__).resolve().parents[4]
repo_root = Path(__file__).resolve().parents[5]
sys.path.append(str(tq_new_root))
sys.path.append(str(repo_root))

import numpy as np
import ray
import torch

import recipe.async_flow.utils.transfer_queue.tq_client as tq_client_module
import recipe.async_flow.utils.transfer_queue.tq_config as tq_config_module
import recipe.async_flow.utils.transfer_queue.tq_mgr as tq_mgr_module
from transfer_queue.utils.perf_compare import (
    COMMON_CORE_COMPARISON_MODE,
    build_comparison_context,
    build_comparison_result,
    ensure_results_parent,
    resolve_results_output_path,
)

VIZTRACER_AVAILABLE = True
try:
    from viztracer_tools import VizTracerProfileSession
except ModuleNotFoundError as exc:
    if exc.name != "viztracer":
        raise
    VIZTRACER_AVAILABLE = False
    VizTracerProfileSession = Any  # type: ignore[assignment,misc]
from recipe.async_flow.utils.transfer_queue.tq_client import TransferQueueClient, get_transferqueue_client
from recipe.async_flow.utils.transfer_queue.tq_config import (
    TransferQueueAblationConfig,
    add_ablation_arguments,
    add_shared_column_arguments,
    format_group_shared_columns,
    normalize_ablation_config,
    parse_ablation_config,
    parse_group_shared_columns,
)
from recipe.async_flow.utils.transfer_queue.tq_mgr import TransferQueueManager

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
    "match1": {
        "global_batch_size": 8192,
        "seq_length": 256000,
        "field_num": 12,
        "str_length": 2048,
        "desc": "match to pangu",
    },
    "match2": {
        "global_batch_size": 8192,
        "seq_length": 65536,
        "field_num": 4,
        "str_length": 512,
        "desc": "match to pangu with same batch size",
    },
    "match3": {
        "global_batch_size": 8192,
        "seq_length": 100,
        "field_num": 4,
        "str_length": 512,
        "desc": "match to pangu with same batch size",
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


def _resolve_effective_data_mode(
    requested_mode: str,
    ablation: TransferQueueAblationConfig | Mapping[str, Any] | None,
) -> str:
    resolved_ablation = normalize_ablation_config(ablation)
    if resolved_ablation.force_tensor_only_raw_path and requested_mode == "uuid":
        return "dict"
    return requested_mode


def _ablation_notes(
    ablation: TransferQueueAblationConfig | Mapping[str, Any] | None,
) -> list[str]:
    resolved_ablation = normalize_ablation_config(ablation)
    if not resolved_ablation.active_flags():
        return ["ablation=baseline"]
    return [f"ablation={resolved_ablation.label()}"]


def _apply_group_shared_columns(shared_columns: set[str]) -> list[str]:
    formatted_columns = sorted(shared_columns)
    os.environ[tq_config_module.TQ_GROUP_SHARED_COLUMNS_ENV_VAR] = format_group_shared_columns(formatted_columns)
    tq_config_module.GROUP_SHARED_COLUMNS = set(formatted_columns)
    tq_client_module.GROUP_SHARED_COLUMNS = set(formatted_columns)
    tq_mgr_module.GROUP_SHARED_COLUMNS = set(formatted_columns)
    return formatted_columns


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


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


def parse_profile_round_spec(raw_spec: str | None, total_rounds: int) -> set[int]:
    if total_rounds < 1:
        raise ValueError("total_rounds must be >= 1")

    if raw_spec is None or not raw_spec.strip():
        return {1}

    normalized_spec = raw_spec.strip().lower()
    if normalized_spec == "all":
        return set(range(1, total_rounds + 1))

    selected_rounds: set[int] = set()
    for raw_part in raw_spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                raise ValueError(f"Invalid profile round range: {part}")
            selected_rounds.update(range(start, end + 1))
        else:
            selected_rounds.add(int(part))

    invalid_rounds = sorted(
        round_number for round_number in selected_rounds if round_number < 1 or round_number > total_rounds
    )
    if invalid_rounds:
        raise ValueError(f"Profile rounds {invalid_rounds} are out of range for total_rounds={total_rounds}")
    if not selected_rounds:
        raise ValueError("No profile rounds were selected")
    return selected_rounds


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

    python_path_entries = [str(repo_root), str(tq_new_root)]
    existing_python_path = os.environ.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    runtime_env = {
        "working_dir": working_dir,
        "env_vars": {"PYTHONPATH": os.pathsep.join(python_path_entries)},
    }
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
    if mode == "uuid":
        # UUID mode: add UUID field overhead (approximately 36 bytes per UUID string)
        uuid_overhead = batch_size * 36
        return batch_size * bytes_per_sample + uuid_overhead
    raise ValueError(f"Unsupported mode: {mode}")


def _payload_bytes_from_data(data_payload: Any) -> int:
    if isinstance(data_payload, torch.Tensor):
        return data_payload.numel() * data_payload.element_size()
    if isinstance(data_payload, dict):
        return sum(_payload_bytes_from_data(value) for value in data_payload.values())
    if isinstance(data_payload, (list, tuple)):
        # Check if it's a list of strings (like UUIDs)
        if data_payload and isinstance(data_payload[0], str):
            return sum(len(s) for s in data_payload)
        return sum(_payload_bytes_from_data(value) for value in data_payload)
    return 0


def generate_data(
    config: dict[str, Any],
    mode: str,
    n_samples_per_prompt: int = 1,
) -> tuple[dict[str, Any], float]:
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
    elif mode == "uuid":
        # UUID mode: simulate multiple samples per prompt using UUIDs
        # Generate prompt and response fields
        for field_index in range(field_num):
            dtype = DTYPE_CONFIGS[field_index % len(DTYPE_CONFIGS)]["dtype"]
            data_payload[f"field_{field_index}"] = _generate_regular_tensor(batch_size, seq_length, dtype)

        data_payload["prompt_uuid"] = _generate_prompt_uuid_values(
            batch_size=batch_size,
            n_samples_per_prompt=n_samples_per_prompt,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    total_gb = _payload_bytes_from_data(data_payload) / (1024**3)
    return data_payload, total_gb


def _generate_prompt_uuid_values(
    batch_size: int,
    n_samples_per_prompt: int = 1,
) -> list[str]:
    import uuid

    if n_samples_per_prompt > 1:
        if batch_size % n_samples_per_prompt != 0:
            raise ValueError("UUID group mode requires global_batch_size to be divisible by n_samples_per_prompt")
        prompt_count = batch_size // n_samples_per_prompt
        prompt_uuids = [str(uuid.uuid4()) for _ in range(prompt_count)]
        return [prompt_uuids[sample_index // n_samples_per_prompt] for sample_index in range(batch_size)]

    return [str(uuid.uuid4()) for _ in range(batch_size)]


def _refresh_uuid_payload_in_place(
    data_payload: dict[str, Any],
    *,
    n_samples_per_prompt: int = 1,
) -> dict[str, Any]:
    if "prompt_uuid" not in data_payload:
        return data_payload

    batch_size = len(data_payload["prompt_uuid"])
    data_payload["prompt_uuid"] = _generate_prompt_uuid_values(
        batch_size=batch_size,
        n_samples_per_prompt=n_samples_per_prompt,
    )
    return data_payload


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
    field_num = int(config.get("field_num", 1))
    field_names = [f"field_{index}" for index in range(field_num)]

    # Add prompt_uuid field for UUID mode
    if data_mode == "uuid":
        field_names.append("prompt_uuid")

    return field_names


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
    n_samples_per_prompt: int = 1,
    shared_columns: list[str] | None = None,
    ablation: TransferQueueAblationConfig | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    location = "Remote" if target_ips else "Local"
    client_suffix = ", Multi-Client" if num_clients > 1 else ""
    setting = f"{config_name} ({data_mode}, {location}{client_suffix})"
    resolved_ablation = normalize_ablation_config(ablation)

    def make_result(operation: str, speeds: list[float]) -> dict[str, Any]:
        return {
            "scenario": "TensorDock",
            "setting": setting,
            "data_volume": f"{total_gb * 1024:.2f} MB" if total_gb * 1024 < 10 else f"{total_gb:.4f} GB",
            "operation": operation,
            "payload_gb": total_gb,
            "num_clients": num_clients,
            "n_samples_per_prompt": int(n_samples_per_prompt),
            "shared_columns": list(shared_columns or []),
            "stats_gbps": calculate_stats(speeds),
            "ablation": resolved_ablation.to_metadata(),
        }

    return [make_result("PUT", put_speeds), make_result("GET", get_speeds)]


def _build_benchmark_summary(
    *,
    config_name: str,
    config: dict[str, Any],
    data_mode: str,
    num_clients: int,
    target_ips: list[str],
    payload_bytes_by_round: list[int],
    put_seconds_by_round: list[float],
    get_seconds_by_round: list[float],
    n_samples_per_prompt: int = 1,
    shared_columns: list[str] | None = None,
    ablation: TransferQueueAblationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    put_gbps_by_round = [
        float((payload_bytes * 8) / put_seconds / (1024**3)) if put_seconds > 0 else 0.0
        for payload_bytes, put_seconds in zip(payload_bytes_by_round, put_seconds_by_round, strict=True)
    ]
    get_gbps_by_round = [
        float((payload_bytes * 8) / get_seconds / (1024**3)) if get_seconds > 0 else 0.0
        for payload_bytes, get_seconds in zip(payload_bytes_by_round, get_seconds_by_round, strict=True)
    ]
    return {
        "config_name": config_name,
        "config": dict(config),
        "data_mode": data_mode,
        "num_clients": num_clients,
        "target_ips": list(target_ips),
        "n_samples_per_prompt": int(n_samples_per_prompt),
        "shared_columns": list(shared_columns or []),
        "ablation": normalize_ablation_config(ablation).to_metadata(),
        "payload_bytes_by_round": payload_bytes_by_round,
        "put_seconds_by_round": put_seconds_by_round,
        "get_seconds_by_round": get_seconds_by_round,
        "put_gbps_by_round": put_gbps_by_round,
        "get_gbps_by_round": get_gbps_by_round,
    }


def _legacy_results_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    payload_bytes_by_round = list(summary["payload_bytes_by_round"])
    payload_gb = (payload_bytes_by_round[0] / (1024**3)) if payload_bytes_by_round else 0.0
    return _make_results(
        config_name=str(summary["config_name"]),
        data_mode=str(summary["data_mode"]),
        total_gb=payload_gb,
        put_speeds=list(summary["put_gbps_by_round"]),
        get_speeds=list(summary["get_gbps_by_round"]),
        num_clients=int(summary["num_clients"]),
        target_ips=list(summary["target_ips"]),
        n_samples_per_prompt=int(summary.get("n_samples_per_prompt", 1)),
        shared_columns=list(summary.get("shared_columns", [])),
        ablation=summary.get("ablation"),
    )


def _comparison_results_from_summary(
    summary: dict[str, Any],
    comparison_context: dict[str, Any],
    shard_count: int,
    ablation: TransferQueueAblationConfig | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    resolved_ablation = normalize_ablation_config(ablation or summary.get("ablation"))
    config = dict(summary["config"])
    payload_bytes_by_round = list(summary["payload_bytes_by_round"])
    put_seconds_by_round = list(summary["put_seconds_by_round"])
    get_seconds_by_round = list(summary["get_seconds_by_round"])
    shared_columns = sorted(str(column) for column in summary.get("shared_columns", []) if str(column).strip())
    n_samples_per_prompt = int(summary.get("n_samples_per_prompt", 1))
    notes = list(comparison_context["notes"])
    notes.append(f"shards={shard_count}")
    notes.append(f"placement={'remote' if summary['target_ips'] else 'local'}")
    notes.append(f"n_samples_per_prompt={n_samples_per_prompt}")
    if shared_columns:
        notes.append(f"shared_columns={','.join(shared_columns)}")
    notes.extend(_ablation_notes(resolved_ablation))

    if comparison_context["resolved_data_mode"] != summary["data_mode"]:
        notes.append(f"effective_data_mode={summary['data_mode']}")

    total_bytes = sum(payload_bytes_by_round)
    non_tensor_field_count = 1 if str(summary["data_mode"]) == "uuid" else 0
    result = build_comparison_result(
        implementation="tq_new",
        workload_name=str(comparison_context["workload_name"]),
        batch_size=int(config["global_batch_size"]),
        seq_length=int(config["seq_length"]),
        field_count=int(config["field_num"]),
        non_tensor_field_count=non_tensor_field_count,
        num_shards_or_storage_units=shard_count,
        num_clients=int(summary["num_clients"]),
        rounds=len(payload_bytes_by_round),
        total_bytes=total_bytes,
        put_seconds=sum(put_seconds_by_round),
        get_seconds=sum(get_seconds_by_round),
        notes=notes,
        data_mode=str(comparison_context["resolved_data_mode"] or summary["data_mode"]),
        tensor_only=bool(comparison_context["tensor_only"]),
        shared_columns_enabled=bool(comparison_context["shared_columns_enabled"] or shared_columns),
        uuid_mode_enabled=bool(comparison_context["uuid_mode_enabled"]),
        extra_fields={
            "ablation": resolved_ablation.to_metadata(),
            "n_samples_per_prompt": n_samples_per_prompt,
            "shared_columns": shared_columns,
            "payload_bytes_per_round": payload_bytes_by_round[0] if payload_bytes_by_round else 0,
            "put_seconds_by_round": put_seconds_by_round,
            "get_seconds_by_round": get_seconds_by_round,
            "put_gbps_by_round": list(summary["put_gbps_by_round"]),
            "get_gbps_by_round": list(summary["get_gbps_by_round"]),
        },
    )
    return [result]


@ray.remote
def _run_client_round(
    client_id: int,
    config: dict[str, Any],
    data_mode: str,
    topic_name: str,
    field_names: list[str],
    assignment: dict[str, Any],
    verify_round: bool,
    n_samples_per_prompt: int = 1,
    ablation: dict[str, bool] | None = None,
) -> dict[str, Any]:
    resolved_ablation = normalize_ablation_config(ablation)
    td_client = get_transferqueue_client(ablation=resolved_ablation)

    local_config = dict(config)
    local_config["global_batch_size"] = assignment["batch_size"]

    data_dict, total_gb = generate_data(local_config, data_mode, n_samples_per_prompt=n_samples_per_prompt)
    payload_bytes = int(round(total_gb * (1024**3)))

    preallocated_indexes = None
    if resolved_ablation.preallocate_indexes:
        preallocated_indexes = td_client.allocate_put_indexes(
            data_dict=data_dict,
            topic=topic_name,
        )

    start_put = time.time()
    current_indices = td_client.put_experience(
        data_dict=data_dict,
        indexes=preallocated_indexes,
        topic=topic_name,
    )
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
        "payload_bytes": payload_bytes,
        "put_time": put_time,
        "get_time": get_time,
        "verified": is_valid,
        "verify_message": verify_message,
    }


def _run_single_client_benchmark(
    td_client: TransferQueueClient,
    manager_handle: Any | None,
    topic_name: str,
    config_name: str,
    config: dict[str, Any],
    data_mode: str,
    test_rounds: int,
    num_clients: int,
    target_ips: list[str],
    enable_profile: bool,
    profile_session: VizTracerProfileSession | None = None,
    profile_rounds: set[int] | None = None,
    n_samples_per_prompt: int = 1,
    shared_columns: list[str] | None = None,
    ablation: TransferQueueAblationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_ablation = normalize_ablation_config(ablation)
    payload_bytes_per_round = 0
    total_gb = 0.0

    # For large datasets (>10GB), regenerate data each round to avoid OOM
    # For small datasets, generate once and reuse
    MEMORY_THRESHOLD_GB = 30.0
    regenerate_each_round = total_gb > MEMORY_THRESHOLD_GB

    if regenerate_each_round:
        logger.info(
            f"Large dataset detected ({total_gb:.2f} GB > {MEMORY_THRESHOLD_GB} GB). "
            f"Will regenerate data each round to avoid OOM."
        )
        # Generate first batch to get field names
        data_dict, _ = generate_data(config, data_mode, n_samples_per_prompt=n_samples_per_prompt)
        field_names = list(data_dict.keys())
        payload_bytes_per_round = _payload_bytes_from_data(data_dict)
        total_gb = payload_bytes_per_round / (1024**3)
        # Free memory immediately
        del data_dict
    else:
        data_dict, total_gb = generate_data(config, data_mode, n_samples_per_prompt=n_samples_per_prompt)
        field_names = list(data_dict.keys())
        payload_bytes_per_round = _payload_bytes_from_data(data_dict)
        total_gb = payload_bytes_per_round / (1024**3)

    payload_bytes_by_round: list[int] = []
    put_seconds_by_round: list[float] = []
    get_seconds_by_round: list[float] = []

    print(
        f"\n🚀 Running Config: [{config_name}] | Mode: [{data_mode}] | "
        f"Size: {total_gb:.4f} GB | Rounds: {test_rounds} | Ablation: {resolved_ablation.label()}"
    )

    for round_index in range(test_rounds):
        round_number = round_index + 1
        should_profile_round = enable_profile and round_number in (profile_rounds or {1})

        # Generate data for this round if needed
        if regenerate_each_round:
            data_dict, _ = generate_data(config, data_mode, n_samples_per_prompt=n_samples_per_prompt)
            payload_bytes_per_round = _payload_bytes_from_data(data_dict)
        elif data_mode == "uuid":
            # Reuse large tensor payloads across rounds, but refresh prompt UUIDs so
            # each round maps to fresh groups in manager-side UUID allocation.
            _refresh_uuid_payload_in_place(
                data_dict,
                n_samples_per_prompt=n_samples_per_prompt,
            )

        manager_profile_outputs = None
        if should_profile_round and profile_session is not None:
            if manager_handle is not None:
                manager_profile_outputs = ray.get(manager_handle.start_profile.remote(round_number))
            profile_session.start(round_number)

        preallocated_indexes = None
        if resolved_ablation.preallocate_indexes:
            preallocated_indexes = td_client.allocate_put_indexes(
                data_dict=data_dict,
                topic=topic_name,
            )

        start_put = time.time()
        try:
            round_marker_context = (
                profile_session.marker("client_round", round=round_number, config=config_name, mode=data_mode)
                if should_profile_round and profile_session is not None
                else nullcontext()
            )
            with round_marker_context:
                marker_context = (
                    profile_session.marker("put_phase", round=round_number)
                    if should_profile_round and profile_session is not None
                    else nullcontext()
                )
                with marker_context:
                    current_indices = td_client.put_experience(
                        data_dict=data_dict,
                        indexes=preallocated_indexes,
                        topic=topic_name,
                    )
            put_time = time.time() - start_put

            time.sleep(2)

            start_get = time.time()
            marker_context = (
                profile_session.marker("get_phase", round=round_number)
                if should_profile_round and profile_session is not None
                else nullcontext()
            )
            with marker_context:
                fetched_data, _ = td_client.get_experience(
                    consumer="learner",
                    experience_columns=field_names,
                    experience_count=int(config["global_batch_size"]),
                    indexes=current_indices,
                    get_n_samples=False,
                    topic=topic_name,
                )
            get_time = time.time() - start_get
        finally:
            if should_profile_round and profile_session is not None:
                output_path = profile_session.stop()
                if output_path is not None:
                    logger.info("Saved client VizTracer trace for round %s to %s", round_number, output_path)
                if manager_handle is not None:
                    manager_profile_outputs = ray.get(manager_handle.stop_profile.remote())
                    manager_output = manager_profile_outputs.get("manager") if manager_profile_outputs else None
                    if manager_output:
                        logger.info("Saved manager VizTracer trace for round %s to %s", round_number, manager_output)
                    for shard_output in manager_profile_outputs.get("shards", []) if manager_profile_outputs else []:
                        if shard_output:
                            logger.info("Saved shard VizTracer trace for round %s to %s", round_number, shard_output)

        put_gbps = (total_gb * 8) / put_time
        get_gbps = (total_gb * 8) / get_time
        payload_bytes_by_round.append(payload_bytes_per_round)
        put_seconds_by_round.append(put_time)
        get_seconds_by_round.append(get_time)

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

        # Free memory for this round if regenerating
        if regenerate_each_round:
            del data_dict
            del fetched_data

    print("\n")

    return _build_benchmark_summary(
        config_name=config_name,
        config=config,
        data_mode=data_mode,
        num_clients=num_clients,
        target_ips=target_ips,
        n_samples_per_prompt=n_samples_per_prompt,
        shared_columns=shared_columns,
        ablation=resolved_ablation,
        payload_bytes_by_round=payload_bytes_by_round,
        put_seconds_by_round=put_seconds_by_round,
        get_seconds_by_round=get_seconds_by_round,
    )


def _run_multi_client_benchmark(
    topic_name: str,
    config_name: str,
    config: dict[str, Any],
    data_mode: str,
    test_rounds: int,
    num_clients: int,
    target_ips: list[str],
    distribute_clients: bool,
    node_resource_map: Mapping[str, str] | None = None,
    n_samples_per_prompt: int = 1,
    shared_columns: list[str] | None = None,
    ablation: TransferQueueAblationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_ablation = normalize_ablation_config(ablation)
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
    payload_bytes_by_round: list[int] = []
    put_seconds_by_round: list[float] = []
    get_seconds_by_round: list[float] = []

    print(
        f"\n🚀 Running Config: [{config_name}] | Mode: [{data_mode}] | Size: {total_gb:.4f} GB | "
        f"Clients: {active_num_clients} | Rounds: {test_rounds} | Ablation: {resolved_ablation.label()}"
    )

    for round_index in range(test_rounds):
        verify_round = round_index in (0, test_rounds - 1)
        refs = []

        for assignment in assignments:
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
                    verify_round,
                    n_samples_per_prompt,
                    resolved_ablation.as_dict(),
                )
            )

        results = ray.get(refs)

        total_payload_bytes = sum(item["payload_bytes"] for item in results)
        put_time = max(item["put_time"] for item in results)
        get_time = max(item["get_time"] for item in results)
        total_payload_gb = total_payload_bytes / (1024**3)
        put_gbps = (total_payload_gb * 8) / put_time
        get_gbps = (total_payload_gb * 8) / get_time
        payload_bytes_by_round.append(total_payload_bytes)
        put_seconds_by_round.append(put_time)
        get_seconds_by_round.append(get_time)

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

    return _build_benchmark_summary(
        config_name=config_name,
        config=config,
        data_mode=data_mode,
        num_clients=active_num_clients,
        target_ips=target_ips,
        n_samples_per_prompt=n_samples_per_prompt,
        shared_columns=shared_columns,
        ablation=resolved_ablation,
        payload_bytes_by_round=payload_bytes_by_round,
        put_seconds_by_round=put_seconds_by_round,
        get_seconds_by_round=get_seconds_by_round,
    )


def run_single_benchmark(
    td_client: TransferQueueClient,
    manager_handle: Any | None,
    config_name: str,
    config: dict[str, Any],
    data_mode: str,
    test_rounds: int,
    num_clients: int = 1,
    target_ips: list[str] | None = None,
    distribute_clients: bool = False,
    enable_profile: bool = False,
    node_resource_map: Mapping[str, str] | None = None,
    profile_session: VizTracerProfileSession | None = None,
    profile_rounds: set[int] | None = None,
    comparison_context: dict[str, Any] | None = None,
    shard_count: int | None = None,
    n_samples_per_prompt: int = 1,
    shared_columns: list[str] | None = None,
    ablation: TransferQueueAblationConfig | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    resolved_ablation = normalize_ablation_config(ablation)
    target_ips = target_ips or []
    topic_name = f"perf_{config_name}_{data_mode}_{int(time.time())}"
    field_names = _get_field_names(config, data_mode)

    td_client.add_topic(
        prompts_num=int(config["global_batch_size"]) * test_rounds * 2,
        n_samples_per_prompt=n_samples_per_prompt,
        experience_columns=field_names,
        experience_consumers=["learner"],
        # metrics=Metric(),
        topic=topic_name,
    )

    try:
        if num_clients == 1:
            summary = _run_single_client_benchmark(
                td_client=td_client,
                manager_handle=manager_handle,
                topic_name=topic_name,
                config_name=config_name,
                config=config,
                data_mode=data_mode,
                test_rounds=test_rounds,
                num_clients=num_clients,
                target_ips=target_ips,
                enable_profile=enable_profile,
                profile_session=profile_session,
                profile_rounds=profile_rounds,
                n_samples_per_prompt=n_samples_per_prompt,
                shared_columns=shared_columns,
                ablation=resolved_ablation,
            )
        else:
            if enable_profile:
                logger.warning("Profile sync is only supported for single-client mode, matching put_benchmark.py.")
            summary = _run_multi_client_benchmark(
                topic_name=topic_name,
                config_name=config_name,
                config=config,
                data_mode=data_mode,
                test_rounds=test_rounds,
                num_clients=num_clients,
                target_ips=target_ips,
                distribute_clients=distribute_clients,
                node_resource_map=node_resource_map,
                n_samples_per_prompt=n_samples_per_prompt,
                shared_columns=shared_columns,
                ablation=resolved_ablation,
            )

        if comparison_context is None:
            return _legacy_results_from_summary(summary)

        if shard_count is None:
            raise ValueError("shard_count is required when comparison_context is provided")
        return _comparison_results_from_summary(
            summary,
            comparison_context=comparison_context,
            shard_count=shard_count,
            ablation=resolved_ablation,
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
    parser.add_argument("--mode", type=str, default="dict", choices=["dict", "tensor", "uuid"], help="Data mode")
    parser.add_argument("--output", type=str, default="res.json", help="Output file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for comparison JSON output (comparison mode only)",
    )
    parser.add_argument(
        "--comparison-mode",
        type=str,
        default=None,
        choices=[COMMON_CORE_COMPARISON_MODE],
        help="Emit unified comparison JSON and normalize workload semantics",
    )
    parser.add_argument("--rounds", type=int, default=20, help="Test rounds (default: 20)")
    parser.add_argument("--shards", type=int, default=8, help="Number of manager shards (default: 8)")
    parser.add_argument("--cpus", type=int, default=8, help="Manager num_cpus (default: 8)")
    parser.add_argument(
        "--n-samples-per-prompt",
        type=int,
        default=1,
        help="Group width for tq_new topic metadata. UUID mode reuses prompt_uuid within each group when > 1.",
    )
    parser.add_argument("--profile", action="store_true", help="Enable VizTracer profiling for selected rounds")
    parser.add_argument(
        "--profile-output-dir",
        type=str,
        default=None,
        help="Directory used to save VizTracer trace files",
    )
    parser.add_argument(
        "--profile-rounds",
        type=str,
        default=None,
        help="1-based rounds to profile, e.g. 1,3-5 or all (default: 1)",
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
    add_shared_column_arguments(parser)
    add_ablation_arguments(parser)

    args = parser.parse_args()
    ablation = parse_ablation_config(args)
    shared_columns = _apply_group_shared_columns(parse_group_shared_columns(args=args))
    if args.n_samples_per_prompt < 1:
        raise ValueError("--n-samples-per-prompt must be >= 1")
    if args.profile and not VIZTRACER_AVAILABLE:
        raise ModuleNotFoundError("viztracer is required for --profile but is not installed in the current environment")
    if args.comparison_mode and args.profile:
        logger.info("Comparison mode enabled; profiling remains available but output schema switches to unified JSON.")
    if args.comparison_mode and (args.n_samples_per_prompt != 1 or shared_columns):
        raise ValueError("common-core comparison mode requires --n-samples-per-prompt 1 and no --shared-columns")
    logger.info("Benchmark ablations: %s", ablation.label())
    logger.info("Shared columns: %s", shared_columns or ["<none>"])

    runtime_config = resolve_runtime_config(args)
    target_ips = list(runtime_config["target_ips"])
    if args.profile and runtime_config["num_clients"] != 1:
        raise ValueError("--profile currently requires --num-clients 1 so client/shard/manager profiling stays stable")
    profile_rounds = parse_profile_round_spec(args.profile_rounds, args.rounds) if args.profile else set()

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
        f"clients={runtime_config['num_clients']}, storage_targets={target_ips or ['local']}, "
        f"n_samples_per_prompt={args.n_samples_per_prompt}, shared_columns={shared_columns or ['<none>']}, "
        f"ablation={ablation.label()}"
    )

    manager_init_kwargs["ablation"] = ablation.as_dict()
    mgr = TransferQueueManager.options(**manager_options).remote(**manager_init_kwargs)
    ray.get(mgr.init_ready.remote())

    td_client = get_transferqueue_client(ablation=ablation)
    profile_session = None
    if args.profile:
        profile_output_dir = (
            (Path(args.profile_output_dir).resolve())
            if args.profile_output_dir
            else Path(current_working_dir).resolve() / f"profile_output_{int(time.time())}"
        )
        profile_session = VizTracerProfileSession(
            output_dir=profile_output_dir,
            component_name="client",
            enabled=True,
            tracer_entries=args.profile_tracer_entries,
            min_duration_us=args.profile_min_duration_us,
            log_async=True,
        )
        logger.info(
            "VizTracer profiling enabled at %s for rounds=%s",
            profile_output_dir,
            sorted(profile_rounds),
        )
        ray.get(
            mgr.configure_profile.remote(
                output_dir=str(profile_output_dir),
                enabled=True,
                tracer_entries=args.profile_tracer_entries,
                min_duration_us=args.profile_min_duration_us,
            )
        )
    run_list = [args.config] if args.config else list(CONFIG_MAP.keys())
    final_results: list[dict[str, Any]] = []
    output_path = args.output

    exit_code = 0
    try:
        for config_name in run_list:
            config = CONFIG_MAP[config_name]
            comparison_context = None
            run_data_mode = args.mode
            if args.comparison_mode:
                comparison_context = build_comparison_context(
                    config_name=config_name,
                    comparison_mode=args.comparison_mode,
                    requested_data_mode=args.mode,
                    extra_notes=[f"storage_targets={target_ips or ['local']}"],
                )
                run_data_mode = str(comparison_context["resolved_data_mode"] or args.mode)
                if run_data_mode != args.mode:
                    logger.info(
                        "Comparison mode %s overrides data mode %s -> %s",
                        args.comparison_mode,
                        args.mode,
                        run_data_mode,
                    )
            effective_data_mode = _resolve_effective_data_mode(run_data_mode, ablation)
            if effective_data_mode != run_data_mode:
                logger.info(
                    "Ablation forces data mode %s -> %s to stay on tensor-only raw path",
                    run_data_mode,
                    effective_data_mode,
                )
            results = run_single_benchmark(
                td_client=td_client,
                manager_handle=mgr,
                config_name=config_name,
                config=config,
                data_mode=effective_data_mode,
                test_rounds=args.rounds,
                num_clients=runtime_config["num_clients"],
                target_ips=target_ips,
                distribute_clients=runtime_config["distribute_clients"],
                enable_profile=args.profile,
                node_resource_map=node_resource_map,
                profile_session=profile_session,
                profile_rounds=profile_rounds,
                comparison_context=comparison_context,
                shard_count=args.shards,
                n_samples_per_prompt=args.n_samples_per_prompt,
                shared_columns=shared_columns,
                ablation=ablation,
            )
            final_results.extend(results)

        if args.comparison_mode:
            explicit_output = None if args.output == "res.json" else args.output
            output_path = str(
                ensure_results_parent(
                    resolve_results_output_path(
                        implementation="tq_new",
                        workload_name=f"{args.comparison_mode}-{args.config or 'suite'}",
                        output_path=explicit_output,
                        output_dir=args.output_dir,
                        repo_root=repo_root,
                    )
                )
            )

        with open(output_path, "w") as output_file:
            json.dump(final_results, output_file, indent=4)
        print(f"💾 Results saved to {output_path}")
    except Exception as exc:
        print(f"❌ Critical Error: {exc}")
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        print("🧹 Cleaning up...")
        try:
            ray.get(mgr.shutdown.remote())
        except Exception:
            pass
        if ray.is_initialized():
            ray.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
