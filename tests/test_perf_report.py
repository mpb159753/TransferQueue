import json
from pathlib import Path

import pytest

from transfer_queue.utils.perf_report import (
    extract_stage_name,
    pair_common_core_results,
    summarize_viztracer_round,
)


def test_extract_stage_name_strips_viztracer_suffix() -> None:
    raw_name = (
        "send_request [endpoint=tcp://127.0.0.1:1234, indexes=32] "
        "(/Users/mpb/WorkSpace/TransferQueue/transfer_queue/utils/trace_utils.py:30)"
    )

    assert extract_stage_name(raw_name) == "send_request"


def test_summarize_viztracer_round_aggregates_stage_durations(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_01"
    round_dir.mkdir()

    (round_dir / "client.json").write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"ph": "X", "name": "allocate_indexes [count=32] (/tmp/x.py:1)", "dur": 2_000.0},
                    {"ph": "X", "name": "send_request [indexes=32] (/tmp/x.py:2)", "dur": 3_000.0},
                    {"ph": "X", "name": "plain_python_call", "dur": 9_999.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    (round_dir / "shard_00.json").write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"ph": "X", "name": "send_request [batches=1] (/tmp/y.py:3)", "dur": 1_000.0},
                    {"ph": "X", "name": "storage_put [indexes=32] (/tmp/y.py:4)", "dur": 5_000.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_viztracer_round(round_dir)

    assert summary["trace_files"] == 2
    assert summary["stage_breakdown_ms"]["allocate_indexes"] == pytest.approx(2.0)
    assert summary["stage_breakdown_ms"]["send_request"] == pytest.approx(4.0)
    assert summary["stage_breakdown_ms"]["storage_put"] == pytest.approx(5.0)
    assert summary["stage_breakdown_pct"]["storage_put"] == pytest.approx(5.0 / 11.0 * 100.0)
    assert summary["event_counts_by_stage"]["send_request"] == 2


def test_pair_common_core_results_matches_by_shape_not_data_mode() -> None:
    pairs = pair_common_core_results(
        [
            {
                "implementation": "tq_new",
                "workload_name": "common-core-tiny",
                "batch_size": 64,
                "seq_length": 1024,
                "field_count": 4,
                "num_shards_or_storage_units": 4,
                "num_clients": 1,
                "data_mode": "dict",
            },
            {
                "implementation": "simple_storage",
                "workload_name": "common-core-tiny",
                "batch_size": 64,
                "seq_length": 1024,
                "field_count": 4,
                "num_shards_or_storage_units": 4,
                "num_clients": 1,
                "data_mode": "tensordict",
            },
        ]
    )

    assert len(pairs) == 1
    assert pairs[0]["tq_new"]["implementation"] == "tq_new"
    assert pairs[0]["simple_storage"]["implementation"] == "simple_storage"
