from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

COMMON_CORE_COMPARISON_MODE = "common-core"


def _dedupe_notes(notes: Iterable[str] | None) -> list[str]:
    deduped: list[str] = []
    for note in notes or []:
        normalized_note = str(note).strip()
        if normalized_note and normalized_note not in deduped:
            deduped.append(normalized_note)
    return deduped


def _safe_workload_name(workload_name: str) -> str:
    sanitized = workload_name.strip()
    for source, target in (("/", "-"), ("\\", "-"), (" ", "-"), (":", "-")):
        sanitized = sanitized.replace(source, target)
    return sanitized or "benchmark"


def _throughput_gbps(total_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float((total_bytes * 8) / seconds / (1024**3))


def build_comparison_context(
    config_name: str,
    comparison_mode: str | None,
    requested_data_mode: str | None = None,
    extra_notes: Iterable[str] | None = None,
) -> dict[str, Any]:
    notes = list(extra_notes or [])
    resolved_data_mode = requested_data_mode
    tensor_only = requested_data_mode in {"dict", "tensor", None}
    shared_columns_enabled = False
    uuid_mode_enabled = requested_data_mode == "uuid"
    workload_name = config_name

    if comparison_mode == COMMON_CORE_COMPARISON_MODE:
        workload_name = f"{COMMON_CORE_COMPARISON_MODE}-{config_name}"
        if requested_data_mode and requested_data_mode != "dict":
            notes.append(
                f"requested_data_mode={requested_data_mode} overridden to dict for {COMMON_CORE_COMPARISON_MODE}"
            )
        resolved_data_mode = "dict"
        tensor_only = True
        shared_columns_enabled = False
        uuid_mode_enabled = False
        notes.extend(
            [
                f"comparison_mode={COMMON_CORE_COMPARISON_MODE}",
                "n_samples_per_prompt=1",
                "tensor_only",
                "shared_columns disabled",
                "UUID/group/version extras disabled",
            ]
        )

    return {
        "comparison_mode": comparison_mode,
        "workload_name": workload_name,
        "resolved_data_mode": resolved_data_mode,
        "tensor_only": tensor_only,
        "shared_columns_enabled": shared_columns_enabled,
        "uuid_mode_enabled": uuid_mode_enabled,
        "notes": _dedupe_notes(notes),
    }


def default_results_output_path(
    implementation: str,
    workload_name: str,
    repo_root: Path | str | None = None,
    timestamp: str | None = None,
) -> Path:
    resolved_repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    resolved_timestamp = timestamp or datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"{_safe_workload_name(workload_name)}_{resolved_timestamp}.json"
    return resolved_repo_root / "docs" / "perf" / "results" / implementation / filename


def resolve_results_output_path(
    implementation: str,
    workload_name: str,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    repo_root: Path | str | None = None,
) -> Path:
    if output_path:
        return Path(output_path)
    if output_dir:
        return Path(output_dir) / f"{_safe_workload_name(workload_name)}.json"
    return default_results_output_path(
        implementation=implementation,
        workload_name=workload_name,
        repo_root=repo_root,
    )


def ensure_results_parent(output_path: str | Path) -> Path:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def build_comparison_result(
    *,
    implementation: str,
    workload_name: str,
    batch_size: int,
    seq_length: int,
    field_count: int,
    non_tensor_field_count: int,
    num_shards_or_storage_units: int,
    num_clients: int,
    rounds: int,
    total_bytes: int,
    put_seconds: float,
    get_seconds: float,
    notes: Iterable[str] | None = None,
    data_mode: str | None = None,
    tensor_only: bool | None = None,
    shared_columns_enabled: bool | None = None,
    uuid_mode_enabled: bool | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    round_trip_seconds = float(put_seconds + get_seconds)
    result: dict[str, Any] = {
        "implementation": implementation,
        "workload_name": workload_name,
        "batch_size": int(batch_size),
        "seq_length": int(seq_length),
        "field_count": int(field_count),
        "non_tensor_field_count": int(non_tensor_field_count),
        "num_shards_or_storage_units": int(num_shards_or_storage_units),
        "num_clients": int(num_clients),
        "rounds": int(rounds),
        "total_bytes": int(total_bytes),
        "put_seconds": float(put_seconds),
        "get_seconds": float(get_seconds),
        "round_trip_seconds": round_trip_seconds,
        "put_gbps": _throughput_gbps(int(total_bytes), float(put_seconds)),
        "get_gbps": _throughput_gbps(int(total_bytes), float(get_seconds)),
        "round_trip_gbps": _throughput_gbps(int(total_bytes) * 2, round_trip_seconds),
        "notes": _dedupe_notes(notes),
    }

    optional_fields = {
        "data_mode": data_mode,
        "tensor_only": tensor_only,
        "shared_columns_enabled": shared_columns_enabled,
        "uuid_mode_enabled": uuid_mode_enabled,
    }
    for key, value in optional_fields.items():
        if value is not None:
            result[key] = value

    for key, value in (extra_fields or {}).items():
        result[key] = value

    return result
