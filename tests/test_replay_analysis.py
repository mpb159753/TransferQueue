# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import subprocess
import sys
from pathlib import Path

from transfer_queue.utils.replay_recorder import ReplayConfig, ReplayRecorder

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_replay.py"


def run_analyzer(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        text=True,
        capture_output=True,
    )


def write_events(path, events):
    with path.open("w", encoding="utf-8") as stream:
        for event in events:
            if isinstance(event, str):
                stream.write(event + "\n")
            else:
                stream.write(json.dumps(event) + "\n")


def test_raw_only_jsonl_produces_summary_with_na_compression(tmp_path):
    events_file = tmp_path / "events.jsonl"
    write_events(
        events_file,
        [
            {"event_type": "put_raw", "pid": 0, "indexes": [0, 1], "raw_tensor_bytes": 1000, "elapsed_ms": 2.0},
            {"event_type": "get_raw", "pid": 0, "indexes": [0], "raw_estimated_bytes": 400, "elapsed_ms": 4.0},
            {"event_type": "custom_meta_set", "pid": 0, "elapsed_ms": 1.0},
            {"event_type": "future_kv_replay_event", "pid": 0},
        ],
    )

    result = run_analyzer("--events-file", str(events_file))

    assert result.returncode == 0, result.stderr
    assert "Replay Analysis Summary" in result.stdout
    assert "Processed events: 4" in result.stdout
    assert "Unknown events ignored: 1" in result.stdout
    assert "partition=0 put_count=1 get_count=1 raw_bytes=1400" in result.stdout
    assert "wire_frame_ratio=N/A compressed_ratio=N/A ratio_basis=N/A" in result.stdout
    assert "custom_meta_set count=1 avg_ms=1.00 p99_ms=1.00" in result.stdout
    assert "get_raw count=1 avg_ms=4.00 p99_ms=4.00" in result.stdout
    assert "Storage Unit Distribution\nN/A" in result.stdout


def test_real_replay_recorder_event_field_is_recognized(tmp_path):
    recorder = ReplayRecorder(
        ReplayConfig(
            record_dir=tmp_path,
            dump_data=False,
            buf_size=10,
            dump_max_bytes=None,
            record_wire=False,
        ),
        role="test",
        component_id="unit",
    )
    recorder.record_event(
        "put_raw",
        {"pid": "real", "indexes": [0, 1], "raw_tensor_bytes": 1000, "elapsed_ms": 2.0},
    )
    recorder.record_event(
        "get_raw",
        {"pid": "real", "indexes": [0], "raw_estimated_bytes": 400, "elapsed_ms": 4.0},
    )
    recorder.close()

    result = run_analyzer("--events-file", str(tmp_path / "events.jsonl"))

    assert result.returncode == 0, result.stderr
    assert "Processed events: 2" in result.stdout
    assert "Unknown events ignored: 0" in result.stdout
    assert "partition=real put_count=1 get_count=1 raw_bytes=1400" in result.stdout
    assert "put_raw count=1 avg_ms=2.00 p99_ms=2.00" in result.stdout


def test_real_recorder_ts_seconds_are_converted_for_throughput(tmp_path):
    events_file = tmp_path / "events.jsonl"
    write_events(
        events_file,
        [
            {"event": "put_raw", "pid": "p0", "ts": 1000.0, "raw_tensor_bytes": 1024 * 1024},
            {"event": "get_raw", "pid": "p0", "ts": 1001.0},
        ],
    )

    result = run_analyzer("--events-file", str(events_file))

    assert result.returncode == 0, result.stderr
    assert "raw_mib_per_s=1.00 event_rate_per_s=2.00" in result.stdout


def test_mixed_raw_and_wire_events_report_matched_compression_ratios(tmp_path):
    events_file = tmp_path / "events.jsonl"
    write_events(
        events_file,
        [
            {
                "event_type": "put_raw",
                "pid": "p0",
                "indexes": [0, 1, 2, 3],
                "timestamp_ms": 1000,
                "raw_tensor_bytes": 1000,
            },
            {
                "event_type": "put_wire",
                "pid": "p0",
                "indexes": [0, 1],
                "timestamp_ms": 1001,
                "wire_frame_bytes": 300,
                "compressed_tensor_bytes": 250,
                "compression_algorithm": "zstd",
                "target_storage_unit": "su-a",
            },
            {
                "event_type": "put_wire",
                "pid": "p0",
                "indexes": [2, 3],
                "timestamp_ms": 1002,
                "wire_frame_bytes": 200,
                "compressed_tensor_bytes": 150,
                "compression_algorithm": "zstd",
                "target_storage_unit": "su-b",
            },
        ],
    )

    result = run_analyzer("--events-file", str(events_file))

    assert result.returncode == 0, result.stderr
    assert "partition=p0 put_count=1 get_count=0 raw_bytes=1000" in result.stdout
    assert "wire_frame_ratio=2.00 compressed_ratio=2.50 ratio_basis=matched" in result.stdout
    assert "storage_unit=su-a put_wire_count=1 wire_frame_bytes=300 compressed_tensor_bytes=250" in result.stdout
    assert "storage_unit=su-b put_wire_count=1 wire_frame_bytes=200 compressed_tensor_bytes=150" in result.stdout


def test_partial_wire_coverage_does_not_report_matched_ratio(tmp_path):
    events_file = tmp_path / "events.jsonl"
    write_events(
        events_file,
        [
            {
                "event_type": "put_raw",
                "pid": "p0",
                "indexes": [0, 1, 2, 3],
                "timestamp_ms": 1000,
                "raw_tensor_bytes": 1000,
            },
            {
                "event_type": "put_wire",
                "pid": "p0",
                "indexes": [0, 1],
                "timestamp_ms": 1001,
                "wire_frame_bytes": 300,
                "compressed_tensor_bytes": 250,
            },
        ],
    )

    result = run_analyzer("--events-file", str(events_file))

    assert result.returncode == 0, result.stderr
    assert "partition=p0 put_count=1 get_count=0 raw_bytes=1000" in result.stdout
    assert "wire_frame_ratio=N/A compressed_ratio=N/A ratio_basis=partial" in result.stdout


def test_get_wire_without_compressed_bytes_does_not_dilute_compressed_ratio(tmp_path):
    events_file = tmp_path / "events.jsonl"
    write_events(
        events_file,
        [
            {"event_type": "put_raw", "pid": "p0", "indexes": [0], "timestamp_ms": 1000, "raw_tensor_bytes": 1000},
            {
                "event_type": "put_wire",
                "pid": "p0",
                "indexes": [0],
                "timestamp_ms": 1001,
                "wire_frame_bytes": 500,
                "compressed_tensor_bytes": 250,
            },
            {"event_type": "get_raw", "pid": "p0", "indexes": [1], "timestamp_ms": 1002, "raw_estimated_bytes": 1000},
            {
                "event_type": "get_wire",
                "pid": "p0",
                "indexes": [1],
                "timestamp_ms": 1003,
                "wire_frame_bytes": 500,
            },
        ],
    )

    result = run_analyzer("--events-file", str(events_file))

    assert result.returncode == 0, result.stderr
    assert "partition=p0 put_count=1 get_count=1 raw_bytes=2000" in result.stdout
    assert "wire_frame_ratio=2.00 compressed_ratio=4.00 ratio_basis=matched" in result.stdout


def test_ambiguous_wire_events_fall_back_to_partition_aggregate_ratio(tmp_path):
    events_file = tmp_path / "events.jsonl"
    write_events(
        events_file,
        [
            {"event_type": "put_raw", "pid": "p0", "indexes": [0, 1], "timestamp_ms": 1000, "raw_tensor_bytes": 1000},
            {"event_type": "put_raw", "pid": "p0", "indexes": [0, 1], "timestamp_ms": 1001, "raw_tensor_bytes": 800},
            {
                "event_type": "put_wire",
                "pid": "p0",
                "indexes": [0, 1],
                "timestamp_ms": 1002,
                "wire_frame_bytes": 600,
                "compressed_tensor_bytes": 300,
                "target_storage_unit": "su-a",
            },
        ],
    )

    result = run_analyzer("--events-file", str(events_file))

    assert result.returncode == 0, result.stderr
    assert "partition=p0 put_count=2 get_count=0 raw_bytes=1800" in result.stdout
    assert "wire_frame_ratio=3.00 compressed_ratio=6.00 ratio_basis=aggregate" in result.stdout


def test_malformed_jsonl_lines_are_skipped_with_warning(tmp_path):
    events_file = tmp_path / "events.jsonl"
    write_events(
        events_file,
        [
            {"event_type": "put_raw", "pid": 1, "indexes": [0], "raw_tensor_bytes": 256},
            "{not valid json",
            {"event_type": "get_raw", "pid": 1},
        ],
    )

    result = run_analyzer("--events-file", str(events_file))

    assert result.returncode == 0, result.stderr
    assert "Malformed lines skipped: 1" in result.stdout
    assert "partition=1 put_count=1 get_count=1 raw_bytes=256" in result.stdout
    assert "WARNING: skipped malformed JSONL line 2" in result.stderr


def test_empty_replay_directory_returns_clear_error(tmp_path):
    result = run_analyzer("--replay-dir", str(tmp_path))

    assert result.returncode == 2
    assert "No replay events found" in result.stderr
    assert str(tmp_path / "events.jsonl") in result.stderr
