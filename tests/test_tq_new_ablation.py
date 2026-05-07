import argparse
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import torch


def _install_fake_tq_new_dependencies() -> None:
    def _fake_remote(obj=None, **_kwargs):
        if obj is None:
            return lambda actual_obj: actual_obj
        return obj

    sys.modules["ray"] = types.SimpleNamespace(
        get=lambda value: value,
        remote=_fake_remote,
        is_initialized=lambda: False,
        nodes=lambda: [],
        init=lambda *args, **kwargs: None,
        shutdown=lambda: None,
    )

    fake_transfer_queue = types.ModuleType("transfer_queue")
    fake_transfer_queue.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault("transfer_queue", fake_transfer_queue)

    fake_transfer_queue_utils = types.ModuleType("transfer_queue.utils")
    fake_transfer_queue_utils.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault("transfer_queue.utils", fake_transfer_queue_utils)

    fake_perf_compare = types.ModuleType("transfer_queue.utils.perf_compare")
    fake_perf_compare.COMMON_CORE_COMPARISON_MODE = "common-core"
    fake_perf_compare.build_comparison_context = lambda **kwargs: kwargs

    def _fake_build_comparison_result(**kwargs):
        result = dict(kwargs)
        extra_fields = result.pop("extra_fields", {})
        result.update(extra_fields)
        return result

    fake_perf_compare.build_comparison_result = _fake_build_comparison_result
    fake_perf_compare.ensure_results_parent = lambda path: path
    fake_perf_compare.resolve_results_output_path = lambda **kwargs: Path("/tmp/res.json")
    sys.modules.setdefault("transfer_queue.utils.perf_compare", fake_perf_compare)

    fake_recipe = types.ModuleType("recipe")
    fake_recipe.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault("recipe", fake_recipe)

    fake_async_flow = types.ModuleType("recipe.async_flow")
    fake_async_flow.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault("recipe.async_flow", fake_async_flow)

    fake_utils_pkg = types.ModuleType("recipe.async_flow.utils")
    fake_utils_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault("recipe.async_flow.utils", fake_utils_pkg)

    fake_tq_pkg = types.ModuleType("recipe.async_flow.utils.transfer_queue")
    fake_tq_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault("recipe.async_flow.utils.transfer_queue", fake_tq_pkg)

    fake_tq_sampler = types.ModuleType("recipe.async_flow.utils.transfer_queue.tq_sampler")
    fake_tq_sampler.BaseSampler = type("BaseSampler", (), {})
    sys.modules.setdefault("recipe.async_flow.utils.transfer_queue.tq_sampler", fake_tq_sampler)

    fake_tq_utils = types.ModuleType("recipe.async_flow.utils.transfer_queue.tq_utils")
    fake_tq_utils.serialize_batch = lambda *args, **kwargs: (memoryview(b"raw"), [1], "float32", [(1,)])
    fake_tq_utils.serialize_batch_pickle = lambda batch_objects: (memoryview(b"pickle"), [len(batch_objects)], "pickle")
    fake_tq_utils.deserialize_column_from_frame = lambda *args, **kwargs: None
    fake_tq_utils.deserialize_column_pickle_from_frame = lambda *args, **kwargs: None
    fake_tq_utils.torch_to_numpy = lambda tensor: tensor.detach().cpu().numpy()

    class FakeZMQClient:
        def __init__(self, *args, **kwargs):
            self.calls = []

        async def call(self, method, **kwargs):
            self.calls.append((method, kwargs))
            return []

    fake_tq_utils.ZMQClient = FakeZMQClient
    sys.modules.setdefault("recipe.async_flow.utils.transfer_queue.tq_utils", fake_tq_utils)

    fake_trace = types.ModuleType("recipe.async_flow.utils.transfer_queue.viztracer_tools")

    class _TraceMarker:
        @staticmethod
        def scope(*args, **kwargs):
            class _Scope:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Scope()

    fake_trace.TraceMarker = _TraceMarker
    fake_trace.VizTracerProfileSession = type("VizTracerProfileSession", (), {})
    sys.modules.setdefault("recipe.async_flow.utils.transfer_queue.viztracer_tools", fake_trace)
    sys.modules.setdefault("viztracer_tools", fake_trace)

    fake_tq_mgr = types.ModuleType("recipe.async_flow.utils.transfer_queue.tq_mgr")
    fake_tq_mgr.TransferQueueManager = type("TransferQueueManager", (), {})
    sys.modules.setdefault("recipe.async_flow.utils.transfer_queue.tq_mgr", fake_tq_mgr)


def _load_module(module_name: str, relative_path: str):
    original_ray = sys.modules.get("ray")
    _install_fake_tq_new_dependencies()
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if original_ray is None:
            sys.modules.pop("ray", None)
        else:
            sys.modules["ray"] = original_ray
    return module


@pytest.fixture
def tq_config_module():
    return _load_module(
        "tq_new_test_config",
        "tq_new/recipe/async_flow/utils/transfer_queue/tq_config.py",
    )


@pytest.fixture
def tq_client_module(tq_config_module):
    sys.modules["recipe.async_flow.utils.transfer_queue.tq_config"] = tq_config_module
    return _load_module(
        "tq_new_test_client",
        "tq_new/recipe/async_flow/utils/transfer_queue/tq_client.py",
    )


@pytest.fixture
def tq_put_benchmark_module(tq_config_module, tq_client_module):
    sys.modules["recipe.async_flow.utils.transfer_queue.tq_config"] = tq_config_module
    sys.modules["recipe.async_flow.utils.transfer_queue.tq_client"] = tq_client_module
    return _load_module(
        "tq_new_test_put_benchmark",
        "tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py",
    )


def test_parse_ablation_flags_from_args_and_env(tq_config_module, monkeypatch):
    monkeypatch.setenv(
        "TQ_NEW_ABLATIONS",
        "disable_shared_column_mark_writes,preallocate_indexes",
    )
    args = argparse.Namespace(
        disable_shared_column_written_checks=True,
        disable_shared_column_mark_writes=False,
        disable_version_recording=False,
        bypass_uuid_allocation=False,
        preallocate_indexes=False,
        force_tensor_only_raw_path=False,
    )

    ablation = tq_config_module.parse_ablation_config(args)

    assert ablation.disable_shared_column_written_checks is True
    assert ablation.disable_shared_column_mark_writes is True
    assert ablation.preallocate_indexes is True
    assert ablation.disable_version_recording is False
    assert ablation.active_flags() == [
        "disable_shared_column_written_checks",
        "disable_shared_column_mark_writes",
        "preallocate_indexes",
    ]


@pytest.mark.asyncio
async def test_filter_and_split_data_skips_written_check_when_disabled(tq_client_module, tq_config_module):
    tq_config_module.GROUP_SHARED_COLUMNS = {"prompt"}
    tq_client_module.GROUP_SHARED_COLUMNS = {"prompt"}

    class FakeAsyncRemoteMethod:
        def __init__(self, func):
            self.func = func

        async def remote(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    class FakeSyncRemoteMethod:
        def __init__(self, func):
            self.func = func

        def remote(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    class FakeManager:
        def __init__(self):
            self.batch_check_columns_written = FakeAsyncRemoteMethod(self._batch_check_columns_written)
            self.get_manager_endpoint = FakeSyncRemoteMethod(lambda: "tcp://127.0.0.1:9000")

        def _batch_check_columns_written(self, topic, queries):
            raise AssertionError("written checks should be bypassed when ablation is enabled")

    client = tq_client_module.TransferQueueClient(
        FakeManager(),
        logger=logging.getLogger("test"),
        ablation=tq_config_module.TransferQueueAblationConfig(
            disable_shared_column_written_checks=True,
        ),
    )

    shared_dict, non_shared_dict, groups_to_mark = await client._filter_and_split_data(
        data_dict={
            "prompt": ["p0", "p1"],
            "payload": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        },
        indexes=[0, 1],
        topic="topic",
        data_len=2,
        n_samples_per_prompt=1,
        needs_expansion=False,
    )

    assert shared_dict == {"prompt": ["p0", "p1"]}
    assert "payload" in non_shared_dict
    assert groups_to_mark == {0, 1}


def test_comparison_results_include_ablation_metadata(tq_put_benchmark_module, tq_config_module):
    summary = {
        "config_name": "debug",
        "config": {
            "global_batch_size": 8,
            "seq_length": 16,
            "field_num": 2,
        },
        "data_mode": "dict",
        "num_clients": 1,
        "target_ips": [],
        "payload_bytes_by_round": [1024, 1024],
        "put_seconds_by_round": [0.1, 0.2],
        "get_seconds_by_round": [0.3, 0.4],
        "put_gbps_by_round": [1.0, 2.0],
        "get_gbps_by_round": [3.0, 4.0],
    }
    comparison_context = {
        "workload_name": "common-core-debug",
        "resolved_data_mode": "dict",
        "tensor_only": True,
        "shared_columns_enabled": False,
        "uuid_mode_enabled": False,
        "notes": ["local run"],
    }
    ablation = tq_config_module.TransferQueueAblationConfig(
        disable_version_recording=True,
        preallocate_indexes=True,
    )

    results = tq_put_benchmark_module._comparison_results_from_summary(
        summary,
        comparison_context=comparison_context,
        shard_count=4,
        ablation=ablation,
    )

    assert len(results) == 1
    assert results[0]["ablation"] == ablation.to_metadata()
    assert "ablation=disable_version_recording,preallocate_indexes" in results[0]["notes"]
