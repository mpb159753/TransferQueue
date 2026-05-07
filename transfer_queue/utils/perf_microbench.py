from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Sequence

MICROBENCH_RESULTS_DIR = Path("docs") / "perf" / "results" / "microbench"


def _dedupe_notes(notes: Iterable[str] | None) -> list[str]:
    deduped: list[str] = []
    for note in notes or []:
        normalized = str(note).strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if quantile <= 0:
        return float(min(values))
    if quantile >= 1:
        return float(max(values))

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    index = (len(ordered) - 1) * quantile
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = index - lower_index
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def _throughput(quantity: int | float, seconds: float) -> float:
    if quantity <= 0 or seconds <= 0:
        return 0.0
    return float(quantity) / float(seconds)


def default_microbench_output_path(
    repo_root: Path | str | None = None,
    timestamp: str | None = None,
) -> Path:
    resolved_repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    resolved_timestamp = timestamp or datetime.now().strftime("%Y%m%dT%H%M%S")
    return resolved_repo_root / MICROBENCH_RESULTS_DIR / f"microbench_{resolved_timestamp}.json"


def ensure_microbench_parent(output_path: Path | str) -> Path:
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_microbench_dimensions(
    *,
    base_dimensions: Mapping[str, Any],
    put_benchmark_config: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = dict(base_dimensions)

    if put_benchmark_config is not None:
        field_count = put_benchmark_config.get("field_num")
        global_batch_size = put_benchmark_config.get("global_batch_size")
        seq_length = put_benchmark_config.get("seq_length")

        if field_count is not None:
            resolved["field_count"] = int(field_count)

        if global_batch_size is not None:
            batch_size = int(global_batch_size)
            for key in (
                "serialize_batch_size",
                "storage_batch_size",
                "selection_fetch_count",
                "aggregation_batch_size",
            ):
                resolved[key] = batch_size
            current_selection_sample_count = int(resolved.get("selection_sample_count", 0))
            resolved["selection_sample_count"] = max(current_selection_sample_count, batch_size * 32)

        if seq_length is not None:
            resolved_seq_length = int(seq_length)
            for key in (
                "serialize_seq_length",
                "storage_seq_length",
                "aggregation_seq_length",
            ):
                resolved[key] = resolved_seq_length

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        resolved[key] = value

    return resolved


def build_microbench_summary(
    *,
    samples_seconds: Sequence[float],
    bytes_processed: int = 0,
    items_processed: int = 0,
) -> dict[str, Any]:
    if not samples_seconds:
        raise ValueError("samples_seconds must not be empty")

    normalized_samples = [float(sample) for sample in samples_seconds]
    median_seconds = float(median(normalized_samples))

    return {
        "iterations": len(normalized_samples),
        "min_seconds": float(min(normalized_samples)),
        "max_seconds": float(max(normalized_samples)),
        "mean_seconds": float(mean(normalized_samples)),
        "median_seconds": median_seconds,
        "p95_seconds": _linear_percentile(normalized_samples, 0.95),
        "stddev_seconds": float(pstdev(normalized_samples)) if len(normalized_samples) > 1 else 0.0,
        "bytes_processed": int(bytes_processed),
        "items_processed": int(items_processed),
        "median_bytes_per_second": _throughput(bytes_processed, median_seconds),
        "median_items_per_second": _throughput(items_processed, median_seconds),
        "samples_seconds": normalized_samples,
    }


def build_microbench_result(
    *,
    comparison_key: str,
    group: str,
    implementation: str,
    case_name: str,
    samples_seconds: Sequence[float],
    bytes_processed: int = 0,
    items_processed: int = 0,
    notes: Iterable[str] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "comparison_key": comparison_key,
        "group": group,
        "implementation": implementation,
        "case_name": case_name,
        "notes": _dedupe_notes(notes),
    }
    result.update(
        build_microbench_summary(
            samples_seconds=samples_seconds,
            bytes_processed=bytes_processed,
            items_processed=items_processed,
        )
    )
    result.update(extra_fields or {})
    return result
