from pathlib import Path

import pytest

from transfer_queue.utils.perf_microbench import (
    build_microbench_summary,
    default_microbench_output_path,
    resolve_microbench_dimensions,
)


def test_default_microbench_output_path_uses_microbench_directory() -> None:
    output_path = default_microbench_output_path(
        repo_root=Path("/repo"),
        timestamp="20260418T180000",
    )

    assert output_path == Path("/repo/docs/perf/results/microbench/microbench_20260418T180000.json")


def test_build_microbench_summary_reports_percentiles_and_throughput() -> None:
    summary = build_microbench_summary(
        samples_seconds=[0.010, 0.020, 0.030, 0.040],
        bytes_processed=2048,
        items_processed=128,
    )

    assert summary["iterations"] == 4
    assert summary["median_seconds"] == pytest.approx(0.025)
    assert summary["p95_seconds"] == pytest.approx(0.0385)
    assert summary["bytes_processed"] == 2048
    assert summary["items_processed"] == 128
    assert summary["median_bytes_per_second"] == pytest.approx(2048 / 0.025)
    assert summary["median_items_per_second"] == pytest.approx(128 / 0.025)


def test_resolve_microbench_dimensions_aligns_core_sizes_to_put_benchmark() -> None:
    resolved = resolve_microbench_dimensions(
        base_dimensions={
            "field_count": 4,
            "serialize_batch_size": 256,
            "serialize_seq_length": 1024,
            "storage_batch_size": 512,
            "storage_seq_length": 1024,
            "selection_sample_count": 50_000,
            "selection_fetch_count": 2_048,
            "aggregation_batch_size": 1_024,
            "aggregation_seq_length": 512,
            "shard_count": 4,
        },
        put_benchmark_config={
            "global_batch_size": 2_048,
            "seq_length": 128_000,
            "field_num": 5,
        },
    )

    assert resolved["field_count"] == 5
    assert resolved["serialize_batch_size"] == 2_048
    assert resolved["storage_batch_size"] == 2_048
    assert resolved["aggregation_batch_size"] == 2_048
    assert resolved["selection_fetch_count"] == 2_048
    assert resolved["serialize_seq_length"] == 128_000
    assert resolved["storage_seq_length"] == 128_000
    assert resolved["aggregation_seq_length"] == 128_000
    assert resolved["selection_sample_count"] == 65_536


def test_resolve_microbench_dimensions_prefers_explicit_overrides() -> None:
    resolved = resolve_microbench_dimensions(
        base_dimensions={
            "field_count": 4,
            "serialize_batch_size": 256,
            "serialize_seq_length": 1024,
            "storage_batch_size": 512,
            "storage_seq_length": 1024,
            "selection_sample_count": 50_000,
            "selection_fetch_count": 2_048,
            "aggregation_batch_size": 1_024,
            "aggregation_seq_length": 512,
            "shard_count": 4,
        },
        put_benchmark_config={
            "global_batch_size": 2_048,
            "seq_length": 128_000,
            "field_num": 5,
        },
        overrides={
            "aggregation_batch_size": 512,
            "selection_sample_count": 200_000,
        },
    )

    assert resolved["field_count"] == 5
    assert resolved["aggregation_batch_size"] == 512
    assert resolved["selection_sample_count"] == 200_000
