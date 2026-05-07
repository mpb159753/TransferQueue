import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _load_put_benchmark2_module():
    module_path = Path(__file__).resolve().parents[1] / "put_benchmark2.py"

    fake_metrics = types.ModuleType("metrics")
    fake_metrics.Metric = type("Metric", (), {})

    fake_tq_client = types.ModuleType("tq_client")
    fake_tq_client.TransferQueueClient = type("TransferQueueClient", (), {})
    fake_tq_client.get_transferqueue_client = lambda: None

    fake_tq_mgr = types.ModuleType("tq_mgr")
    fake_tq_mgr.TransferQueueManager = type("TransferQueueManager", (), {})

    for name, module in {
        "metrics": fake_metrics,
        "tq_client": fake_tq_client,
        "tq_mgr": fake_tq_mgr,
    }.items():
        sys.modules[name] = module

    spec = importlib.util.spec_from_file_location("tq_test_put_benchmark2", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_configures_sys_path_before_local_imports():
    source = (Path(__file__).resolve().parents[1] / "put_benchmark2.py").read_text()

    assert source.index("sys.path") < source.index("from metrics import Metric")
    assert source.index("sys.path") < source.index("from tq_client import")
    assert source.index("sys.path") < source.index("from tq_mgr import")


def test_generate_data_dict_returns_batched_2d_field_tensors_with_expected_payload_bytes():
    put_benchmark2 = _load_put_benchmark2_module()
    config = {"global_batch_size": 3, "seq_length": 12, "field_num": 3, "str_length": 1}

    data_payload, total_gb = put_benchmark2.generate_data(config, "dict")

    actual_bytes = put_benchmark2._payload_bytes_from_data(data_payload)
    expected_bytes = 3 * 12 * (4 + 8 + 8)

    assert actual_bytes == expected_bytes
    assert total_gb == pytest.approx(expected_bytes / (1024**3))
    assert [data_payload[f"field_{i}"].shape for i in range(3)] == [(3, 12), (3, 12), (3, 12)]
    assert [data_payload[f"field_{i}"].dtype for i in range(3)] == [
        torch.float32,
        torch.int64,
        torch.float64,
    ]


def test_generate_data_tensor_returns_jagged_tensor_lists_with_expected_payload_bytes():
    put_benchmark2 = _load_put_benchmark2_module()
    config = {"global_batch_size": 3, "seq_length": 12, "field_num": 3, "str_length": 1}

    data_payload, total_gb = put_benchmark2.generate_data(config, "tensor")

    actual_bytes = put_benchmark2._payload_bytes_from_data(data_payload)
    expected_bytes = 3 * 12 * (4 + 8 + 8)

    assert actual_bytes == expected_bytes
    assert total_gb == pytest.approx(expected_bytes / (1024**3))
    assert all(isinstance(data_payload[f"field_{i}"], list) for i in range(3))
    assert all(len(data_payload[f"field_{i}"]) == 3 for i in range(3))
    assert all(
        all(isinstance(item, torch.Tensor) and item.ndim == 2 for item in data_payload[f"field_{i}"]) for i in range(3)
    )
    assert len({tensor.shape[0] for tensor in data_payload["field_0"]}) > 1
    assert [data_payload[f"field_{i}"][0].dtype for i in range(3)] == [
        torch.float32,
        torch.int64,
        torch.float64,
    ]


def test_verify_data_integrity_accepts_nested_tensor_for_jagged_field():
    put_benchmark2 = _load_put_benchmark2_module()
    original_data = {
        "field_0": [
            torch.arange(2, dtype=torch.float32).reshape(1, 2),
            torch.arange(6, dtype=torch.float32).reshape(3, 2),
        ]
    }
    fetched_data = {
        "field_0": torch.nested.as_nested_tensor(original_data["field_0"], layout=torch.jagged),
    }

    is_valid, message = put_benchmark2.verify_data_integrity(original_data, fetched_data)

    assert is_valid is True
    assert message == "✅ PASS"


def test_build_client_assignments_splits_evenly_and_round_robins_nodes():
    put_benchmark2 = _load_put_benchmark2_module()

    assignments = put_benchmark2.build_client_assignments(
        global_batch_size=12,
        num_clients=3,
        target_ips=["10.0.0.1", "10.0.0.2"],
        distribute_clients=True,
    )

    assert assignments == [
        {"client_id": 0, "start_idx": 0, "end_idx": 4, "batch_size": 4, "target_ip": "10.0.0.1"},
        {"client_id": 1, "start_idx": 4, "end_idx": 8, "batch_size": 4, "target_ip": "10.0.0.2"},
        {"client_id": 2, "start_idx": 8, "end_idx": 12, "batch_size": 4, "target_ip": "10.0.0.1"},
    ]


def test_build_client_assignments_distributes_remainder_across_first_clients():
    put_benchmark2 = _load_put_benchmark2_module()

    assignments = put_benchmark2.build_client_assignments(global_batch_size=10, num_clients=3)

    assert [assignment["batch_size"] for assignment in assignments] == [4, 3, 3]


def test_build_client_assignments_evenly_spreads_clients_when_clients_exceed_nodes():
    put_benchmark2 = _load_put_benchmark2_module()

    assignments = put_benchmark2.build_client_assignments(
        global_batch_size=8,
        num_clients=8,
        target_ips=["10.0.0.1", "10.0.0.2"],
        distribute_clients=True,
    )

    assert [assignment["target_ip"] for assignment in assignments] == [
        "10.0.0.1",
        "10.0.0.2",
        "10.0.0.1",
        "10.0.0.2",
        "10.0.0.1",
        "10.0.0.2",
        "10.0.0.1",
        "10.0.0.2",
    ]


def test_resolve_runtime_config_auto_mode_keeps_explicit_num_clients():
    put_benchmark2 = _load_put_benchmark2_module()
    args = types.SimpleNamespace(
        storage_node_ips=None,
        role="single",
        head_ip=None,
        worker_ip=None,
        ip=None,
        wait_nodes=0,
        num_clients=512,
        distribute_clients=False,
    )
    env = {
        put_benchmark2.AUTO_WORKER_HOSTS_ENV: "worker-0,worker-1",
        put_benchmark2.AUTO_CURRENT_INSTANCE_ENV_NAMES[0]: "worker-0",
    }

    runtime_config = put_benchmark2.resolve_runtime_config(args, env)

    assert runtime_config["enabled"] is True
    assert runtime_config["target_hosts"] == ["worker-0", "worker-1"]
    assert runtime_config["num_clients"] == 512
    assert runtime_config["distribute_clients"] is True


def test_build_manager_init_kwargs_uses_supported_storage_node_parameter():
    put_benchmark2 = _load_put_benchmark2_module()

    class FakeManager:
        def __init__(self, nums_tq_data, base_port, storage_node_ips=None):
            pass

    kwargs = put_benchmark2.build_manager_init_kwargs(
        manager_cls=FakeManager,
        shard_count=8,
        base_port=1234,
        target_ips=["10.0.0.1", "10.0.0.2"],
    )

    assert kwargs == {
        "nums_tq_data": 8,
        "base_port": 1234,
        "storage_node_ips": ["10.0.0.1", "10.0.0.2"],
    }


def test_run_single_benchmark_uses_single_line_progress_output(monkeypatch, capsys):
    put_benchmark2 = _load_put_benchmark2_module()
    fake_data = {"field_0": torch.ones(2, 2)}

    class FakeClient:
        def add_topic(self, **kwargs):
            return None

        def get_allocation_for_new_groups(self, **kwargs):
            return None

        def put_experience(self, **kwargs):
            return None

        def get_experience(self, **kwargs):
            return fake_data, None

        def delete_topic(self, topic):
            return None

    class FakeClock:
        def __init__(self):
            self.current = 0

        def time(self):
            self.current += 1
            return float(self.current)

    fake_clock = FakeClock()

    monkeypatch.setattr(put_benchmark2, "generate_data", lambda config, mode: (fake_data, 0.25))
    monkeypatch.setattr(put_benchmark2, "verify_data_integrity", lambda *args, **kwargs: True)
    monkeypatch.setattr(put_benchmark2.time, "time", fake_clock.time)
    monkeypatch.setattr(put_benchmark2.time, "sleep", lambda *_args, **_kwargs: None)

    put_benchmark2.run_single_benchmark(
        FakeClient(),
        "debug",
        {"global_batch_size": 2},
        "dict",
        test_rounds=2,
    )

    output = capsys.readouterr().out

    assert "allocation time:" not in output
    assert "\r  Round 1/2: PUT" in output
    assert "\r  Round 2/2: PUT" in output
    assert output.count("\n") <= 4


def test_single_client_uses_allocated_indexes_from_client(monkeypatch):
    put_benchmark2 = _load_put_benchmark2_module()
    fake_data = {"field_0": torch.ones(2, 2)}
    captured = {"put": None, "get": None}

    class FakeClock:
        def __init__(self):
            self.current = 0

        def time(self):
            self.current += 1
            return float(self.current)

    class FakeClient:
        def add_topic(self, **kwargs):
            return None

        def get_allocation_for_new_groups(self, **kwargs):
            return {"indexes": [7, 8]}

        def put_experience(self, **kwargs):
            captured["put"] = kwargs["indexes"]
            return None

        def get_experience(self, **kwargs):
            captured["get"] = kwargs["indexes"]
            return fake_data, None

        def delete_topic(self, topic):
            return None

    monkeypatch.setattr(put_benchmark2, "generate_data", lambda config, mode: (fake_data, 0.25))
    monkeypatch.setattr(put_benchmark2, "verify_data_integrity", lambda *args, **kwargs: True)
    monkeypatch.setattr(put_benchmark2.time, "time", FakeClock().time)
    monkeypatch.setattr(put_benchmark2.time, "sleep", lambda *_args, **_kwargs: None)

    put_benchmark2.run_single_benchmark(
        FakeClient(),
        "debug",
        {"global_batch_size": 2},
        "dict",
        test_rounds=1,
    )

    assert captured["put"] == [7, 8]
    assert captured["get"] == [7, 8]


def test_multi_client_splits_one_round_of_allocated_indexes_across_workers(monkeypatch):
    put_benchmark2 = _load_put_benchmark2_module()
    fake_data = {"field_0": torch.ones(2, 2)}
    captured_assignments = []

    class FakeRemote:
        def remote(self, *args, **kwargs):
            captured_assignments.append(args[5]["indexes"])
            return f"ref_{len(captured_assignments)}"

        def options(self, **kwargs):
            return self

    class FakeClient:
        def get_allocation_for_new_groups(self, **kwargs):
            return {"indexes": [100, 101, 102, 103]}

    monkeypatch.setattr(put_benchmark2, "_run_client_round", FakeRemote())
    monkeypatch.setattr(put_benchmark2, "generate_data", lambda config, mode: (fake_data, 0.25))
    monkeypatch.setattr(
        put_benchmark2.ray,
        "get",
        lambda refs: [
            {"payload_gb": 0.25, "put_time": 1.0, "get_time": 1.0, "verified": True, "verify_message": "✅ PASS"}
            for _ in refs
        ],
    )

    put_benchmark2._run_multi_client_benchmark(
        td_client=FakeClient(),
        topic_name="topic",
        config_name="debug",
        config={"global_batch_size": 4, "seq_length": 2, "field_num": 1},
        data_mode="dict",
        test_rounds=1,
        num_clients=2,
        target_ips=[],
        distribute_clients=False,
    )

    assert captured_assignments == [[100, 101], [102, 103]]


def test_single_client_profile_sync_marks_only_first_round(monkeypatch):
    put_benchmark2 = _load_put_benchmark2_module()
    fake_data = {"field_0": torch.ones(2, 2)}
    sync_calls = []

    class FakeClock:
        def __init__(self):
            self.current = 0

        def time(self):
            self.current += 1
            return float(self.current)

    class FakeClient:
        def add_topic(self, **kwargs):
            return None

        def get_allocation_for_new_groups(self, **kwargs):
            return {"indexes": [0, 1]}

        def put_experience(self, **kwargs):
            return None

        def get_experience(self, **kwargs):
            return fake_data, None

        def delete_topic(self, topic):
            return None

    monkeypatch.setattr(put_benchmark2, "generate_data", lambda config, mode: (fake_data, 0.25))
    monkeypatch.setattr(put_benchmark2, "verify_data_integrity", lambda *args, **kwargs: True)
    monkeypatch.setattr(put_benchmark2.time, "time", FakeClock().time)
    monkeypatch.setattr(put_benchmark2.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(put_benchmark2, "sync_stage", lambda create, wait: sync_calls.append((create, wait)))

    put_benchmark2.run_single_benchmark(
        FakeClient(),
        "debug",
        {"global_batch_size": 2},
        "dict",
        test_rounds=2,
        enable_profile=True,
    )

    assert sync_calls == [
        ("init_ready.flag", "put_start.flag"),
        ("put_done.flag", "get_prepare.flag"),
        ("get_ready.flag", "get_start.flag"),
    ]
