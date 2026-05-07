from pathlib import Path

import pytest

from transfer_queue.utils.perf_compare import (
    COMMON_CORE_COMPARISON_MODE,
    build_comparison_context,
    build_comparison_result,
    default_results_output_path,
)


def test_build_comparison_result_uses_total_directional_bytes_for_throughput() -> None:
    total_bytes = 2 * 1024**3

    result = build_comparison_result(
        implementation="tq_new",
        workload_name="common-core-debug",
        batch_size=32,
        seq_length=128,
        field_count=4,
        non_tensor_field_count=0,
        num_shards_or_storage_units=2,
        num_clients=1,
        rounds=2,
        total_bytes=total_bytes,
        put_seconds=2.0,
        get_seconds=4.0,
        notes=["common-core"],
        data_mode="dict",
        tensor_only=True,
        shared_columns_enabled=False,
        uuid_mode_enabled=False,
    )

    assert result["implementation"] == "tq_new"
    assert result["total_bytes"] == total_bytes
    assert result["put_gbps"] == pytest.approx(8.0)
    assert result["get_gbps"] == pytest.approx(4.0)
    assert result["round_trip_seconds"] == pytest.approx(6.0)
    assert result["round_trip_gbps"] == pytest.approx(32.0 / 6.0)
    assert result["notes"] == ["common-core"]


def test_default_results_output_path_uses_implementation_subdirectory() -> None:
    output_path = default_results_output_path(
        implementation="simple_storage",
        workload_name="common-core-debug",
        repo_root=Path("/repo"),
        timestamp="20260418T120000",
    )

    assert output_path == Path("/repo/docs/perf/results/simple_storage/common-core-debug_20260418T120000.json")


def test_build_comparison_context_for_common_core_disables_extra_semantics() -> None:
    context = build_comparison_context(
        config_name="debug",
        comparison_mode=COMMON_CORE_COMPARISON_MODE,
        requested_data_mode="uuid",
        extra_notes=["local run"],
    )

    assert context["workload_name"] == "common-core-debug"
    assert context["resolved_data_mode"] == "dict"
    assert context["tensor_only"] is True
    assert context["shared_columns_enabled"] is False
    assert context["uuid_mode_enabled"] is False
    assert "n_samples_per_prompt=1" in context["notes"]
    assert "local run" in context["notes"]
