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

"""Unit tests for ``transfer_queue.utils.replay_recorder``."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import NonTensorStack

from transfer_queue.utils import replay_recorder as replay_module
from transfer_queue.utils.replay_recorder import ReplayRecorder


def _enable_replay(monkeypatch, tmp_path: Path, **env: str) -> None:
    monkeypatch.setenv("TQ_REPLAY_DIR", str(tmp_path))
    for key in (
        "TQ_REPLAY_BUF_SIZE",
        "TQ_REPLAY_DUMP_DATA",
        "TQ_REPLAY_DUMP_MAX_BYTES",
        "TQ_REPLAY_RECORD_WIRE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_from_env_disabled_returns_none_and_creates_no_files(monkeypatch, tmp_path):
    monkeypatch.delenv("TQ_REPLAY_DIR", raising=False)

    recorder = ReplayRecorder.from_env(role="storage_manager", component_id="TQ_CLIENT_1")

    assert recorder is None
    assert list(tmp_path.iterdir()) == []


def test_jsonl_events_are_one_valid_object_per_line(monkeypatch, tmp_path):
    _enable_replay(monkeypatch, tmp_path)
    recorder = ReplayRecorder.from_env(role="storage_manager", component_id="TQ_CLIENT_1")
    assert recorder is not None

    recorder.record_event(
        "put_raw",
        {
            "partition_id": Path("train"),
            "dtype": torch.float32,
            "np_scalar": np.int64(3),
        },
    )

    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "put_raw"
    assert event["role"] == "storage_manager"
    assert event["component_id"] == "TQ_CLIENT_1"
    assert isinstance(event["ts"], float)
    assert event["partition_id"] == "train"
    assert event["dtype"] == "torch.float32"
    assert event["np_scalar"] == 3


def test_buffer_size_flushes_at_expected_boundary(monkeypatch, tmp_path):
    _enable_replay(monkeypatch, tmp_path, TQ_REPLAY_BUF_SIZE="2")
    recorder = ReplayRecorder.from_env(role="controller")
    assert recorder is not None

    recorder.record_event("first", {"seq": 1})
    assert not (tmp_path / "events.jsonl").exists()

    recorder.record_event("second", {"seq": 2})
    events = _read_events(tmp_path / "events.jsonl")
    assert [event["seq"] for event in events] == [1, 2]


def test_threaded_writes_do_not_corrupt_jsonl(monkeypatch, tmp_path):
    _enable_replay(monkeypatch, tmp_path, TQ_REPLAY_BUF_SIZE="1")
    recorder = ReplayRecorder.from_env(role="storage_manager", component_id="threaded")
    assert recorder is not None

    count_per_thread = 40
    thread_count = 8

    def write_events(thread_idx: int) -> None:
        for seq in range(count_per_thread):
            recorder.record_event("thread_event", {"thread": thread_idx, "seq": seq})

    threads = [threading.Thread(target=write_events, args=(idx,)) for idx in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    events = _read_events(tmp_path / "events.jsonl")
    assert len(events) == thread_count * count_per_thread
    assert {(event["thread"], event["seq"]) for event in events} == {
        (thread_idx, seq) for thread_idx in range(thread_count) for seq in range(count_per_thread)
    }


def test_extract_fields_info_covers_supported_field_types(monkeypatch, tmp_path):
    _enable_replay(monkeypatch, tmp_path)
    recorder = ReplayRecorder.from_env(role="storage_manager")
    assert recorder is not None

    nested = torch.nested.as_nested_tensor(
        [torch.ones(3, dtype=torch.int16), torch.ones(1, dtype=torch.int16)],
        layout=torch.jagged,
    )
    sparse = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [1, 0]]),
        values=torch.tensor([1.0, 2.0], dtype=torch.float32),
        size=(3, 3),
    )
    unknown = object()

    fields = recorder.extract_fields_info(
        {
            "tensor": torch.zeros((2, 3), dtype=torch.float32),
            "nested": nested,
            "sparse": sparse,
            "non_tensor": NonTensorStack.from_list(["a", "bc"]),
            "list": ["abc", b"de"],
            "string": "hello",
            "bytes": b"abc",
            "array": np.zeros((2, 3), dtype=np.int16),
            "unknown": unknown,
        }
    )

    assert fields["tensor"]["kind"] == "tensor"
    assert fields["tensor"]["dtype"] == "torch.float32"
    assert fields["tensor"]["shape"] == [2, 3]
    assert fields["tensor"]["raw_tensor_bytes"] == 24
    assert fields["tensor"]["raw_estimated_bytes"] == 24

    assert fields["nested"]["kind"] == "nested_tensor"
    assert fields["nested"]["raw_tensor_bytes"] == 8
    assert fields["nested"]["raw_estimated_bytes"] == 8

    assert fields["sparse"]["kind"] == "sparse_tensor"
    assert fields["sparse"]["shape"] == [3, 3]
    assert fields["sparse"]["raw_tensor_bytes"] == 40
    assert fields["sparse"]["raw_estimated_bytes"] == 40

    assert fields["non_tensor"]["kind"] == "non_tensor_stack"
    assert fields["non_tensor"]["length"] == 2
    assert fields["non_tensor"]["raw_tensor_bytes"] is None
    assert fields["non_tensor"]["raw_estimated_bytes"] == 3

    assert fields["list"]["kind"] == "list"
    assert fields["list"]["raw_estimated_bytes"] == 5
    assert fields["string"]["kind"] == "str"
    assert fields["string"]["raw_estimated_bytes"] == 5
    assert fields["bytes"]["kind"] == "bytes"
    assert fields["bytes"]["raw_estimated_bytes"] == 3
    assert fields["array"]["kind"] == "numpy_array"
    assert fields["array"]["shape"] == [2, 3]
    assert fields["array"]["raw_estimated_bytes"] == 12
    assert fields["unknown"]["kind"] == "object"
    assert fields["unknown"]["raw_estimated_bytes"] is None


def test_extract_fields_info_preserves_exact_bytes_for_list_of_tensors(monkeypatch, tmp_path):
    _enable_replay(monkeypatch, tmp_path)
    recorder = ReplayRecorder.from_env(role="storage_manager")
    assert recorder is not None

    fields = recorder.extract_fields_info(
        {
            "rows": [
                torch.ones(3, dtype=torch.int16),
                torch.ones(1, dtype=torch.int16),
            ]
        }
    )

    assert fields["rows"]["kind"] == "list"
    assert fields["rows"]["raw_tensor_bytes"] == 8
    assert fields["rows"]["raw_estimated_bytes"] == 8


def test_dump_path_is_sanitized_and_dump_limit_is_applied(monkeypatch, tmp_path):
    _enable_replay(
        monkeypatch,
        tmp_path,
        TQ_REPLAY_DUMP_DATA="1",
        TQ_REPLAY_DUMP_MAX_BYTES="10",
    )
    recorder = ReplayRecorder.from_env(role="storage_manager")
    assert recorder is not None

    first = recorder.make_dump_path("../bad/partition")
    second = recorder.make_dump_path("../bad/partition")

    assert first == tmp_path / "data" / "bad_partition" / "put_000001.pt"
    assert second == tmp_path / "data" / "bad_partition" / "put_000002.pt"
    assert first is not None
    first.relative_to(tmp_path)
    assert ".." not in first.parts
    assert recorder.should_dump(raw_estimated_bytes=10) is True
    assert recorder.should_dump(raw_estimated_bytes=11) is False
    assert recorder.should_dump(raw_estimated_bytes=None) is False


def test_dump_paths_do_not_conflict_between_components(monkeypatch, tmp_path):
    _enable_replay(monkeypatch, tmp_path, TQ_REPLAY_DUMP_DATA="1")
    first_recorder = ReplayRecorder.from_env(role="storage_manager", component_id="client/one")
    second_recorder = ReplayRecorder.from_env(role="storage_manager", component_id="client/two")
    assert first_recorder is not None
    assert second_recorder is not None

    first = first_recorder.make_dump_path("../bad/partition")
    second = second_recorder.make_dump_path("../bad/partition")

    assert first is not None
    assert second is not None
    assert first != second
    assert first.parent.name == "client_one"
    assert second.parent.name == "client_two"
    assert first.parent.parent == tmp_path / "data" / "bad_partition"
    assert second.parent.parent == tmp_path / "data" / "bad_partition"


def test_make_dump_path_returns_none_when_data_dumping_is_disabled(monkeypatch, tmp_path):
    _enable_replay(monkeypatch, tmp_path)
    recorder = ReplayRecorder.from_env(role="storage_manager")
    assert recorder is not None

    assert recorder.make_dump_path("train") is None
    assert recorder.should_dump(raw_estimated_bytes=1) is False


@pytest.mark.parametrize("env_key", ["TQ_REPLAY_BUF_SIZE", "TQ_REPLAY_DUMP_MAX_BYTES"])
def test_invalid_numeric_env_values_disable_replay(monkeypatch, tmp_path, env_key):
    _enable_replay(monkeypatch, tmp_path, **{env_key: "not-an-int"})

    assert ReplayRecorder.from_env(role="storage_manager") is None


def test_recorder_write_failures_do_not_raise(monkeypatch, tmp_path):
    _enable_replay(monkeypatch, tmp_path)
    recorder = ReplayRecorder.from_env(role="storage_manager")
    assert recorder is not None

    def fail_open(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(replay_module.os, "open", fail_open)

    recorder.record_event("put_raw", {"seq": 1})


def test_buf_size_one_produces_readable_jsonl_without_close(monkeypatch, tmp_path):
    _enable_replay(monkeypatch, tmp_path, TQ_REPLAY_BUF_SIZE="1")
    recorder = ReplayRecorder.from_env(role="storage_manager")
    assert recorder is not None

    recorder.record_event("put_raw", {"seq": 1})

    events = _read_events(tmp_path / "events.jsonl")
    assert len(events) == 1
    assert events[0]["event"] == "put_raw"
    assert events[0]["seq"] == 1
