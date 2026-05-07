import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
from test_tq_new_ablation import _install_fake_tq_new_dependencies


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
        "tq_new_shared_test_config",
        "tq_new/recipe/async_flow/utils/transfer_queue/tq_config.py",
    )


@pytest.fixture
def tq_client_module(tq_config_module):
    import sys

    sys.modules["recipe.async_flow.utils.transfer_queue.tq_config"] = tq_config_module
    return _load_module(
        "tq_new_shared_test_client",
        "tq_new/recipe/async_flow/utils/transfer_queue/tq_client.py",
    )


@pytest.fixture
def tq_put_benchmark_module(tq_config_module, tq_client_module):
    import sys

    sys.modules["recipe.async_flow.utils.transfer_queue.tq_config"] = tq_config_module
    sys.modules["recipe.async_flow.utils.transfer_queue.tq_client"] = tq_client_module
    return _load_module(
        "tq_new_shared_test_put_benchmark",
        "tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py",
    )


def test_parse_group_shared_columns_from_args_and_env(tq_config_module):
    args = argparse.Namespace(shared_columns=["prompt_uuid,prompt", "prompt_uuid"])

    resolved = tq_config_module.parse_group_shared_columns(
        args=args,
        env={tq_config_module.TQ_GROUP_SHARED_COLUMNS_ENV_VAR: "prompt_length"},
    )

    assert resolved == {"prompt", "prompt_length", "prompt_uuid"}


def test_comparison_results_include_group_workload_metadata(tq_put_benchmark_module):
    comparison_context = {
        "workload_name": "group-tiny",
        "resolved_data_mode": "uuid",
        "tensor_only": False,
        "shared_columns_enabled": True,
        "uuid_mode_enabled": True,
        "notes": ["group workload"],
    }
    summary = {
        "config_name": "tiny",
        "config": {
            "global_batch_size": 64,
            "seq_length": 1024,
            "field_num": 4,
        },
        "data_mode": "uuid",
        "num_clients": 1,
        "target_ips": [],
        "payload_bytes_by_round": [2_048],
        "put_seconds_by_round": [0.5],
        "get_seconds_by_round": [0.25],
        "put_gbps_by_round": [0.03],
        "get_gbps_by_round": [0.06],
        "n_samples_per_prompt": 4,
        "shared_columns": ["prompt_uuid"],
    }

    results = tq_put_benchmark_module._comparison_results_from_summary(
        summary,
        comparison_context=comparison_context,
        shard_count=4,
    )

    assert len(results) == 1
    assert results[0]["non_tensor_field_count"] == 1
    assert results[0]["n_samples_per_prompt"] == 4
    assert results[0]["shared_columns"] == ["prompt_uuid"]
    assert "n_samples_per_prompt=4" in results[0]["notes"]
    assert "shared_columns=prompt_uuid" in results[0]["notes"]


def test_refresh_uuid_payload_in_place_replaces_prompt_uuids_only(tq_put_benchmark_module):
    config = {
        "global_batch_size": 8,
        "seq_length": 4,
        "field_num": 2,
    }
    data_payload, _ = tq_put_benchmark_module.generate_data(
        config,
        "uuid",
        n_samples_per_prompt=1,
    )

    tensor_id = id(data_payload["field_0"])
    original_prompt_uuid = list(data_payload["prompt_uuid"])

    tq_put_benchmark_module._refresh_uuid_payload_in_place(
        data_payload,
        n_samples_per_prompt=1,
    )

    assert id(data_payload["field_0"]) == tensor_id
    assert data_payload["prompt_uuid"] != original_prompt_uuid
    assert len(set(data_payload["prompt_uuid"])) == config["global_batch_size"]


def test_refresh_uuid_payload_in_place_preserves_group_structure(tq_put_benchmark_module):
    config = {
        "global_batch_size": 8,
        "seq_length": 4,
        "field_num": 2,
    }
    data_payload, _ = tq_put_benchmark_module.generate_data(
        config,
        "uuid",
        n_samples_per_prompt=4,
    )

    tq_put_benchmark_module._refresh_uuid_payload_in_place(
        data_payload,
        n_samples_per_prompt=4,
    )

    prompt_uuids = data_payload["prompt_uuid"]
    assert len(prompt_uuids) == config["global_batch_size"]
    assert prompt_uuids[0] == prompt_uuids[1] == prompt_uuids[2] == prompt_uuids[3]
    assert prompt_uuids[4] == prompt_uuids[5] == prompt_uuids[6] == prompt_uuids[7]
    assert prompt_uuids[0] != prompt_uuids[4]


def test_select_batch_values_handles_non_contiguous_local_indices(tq_client_module):
    selected = tq_client_module._select_batch_values(
        ["g0", "g1", "g2", "g3", "g4", "g5"],
        [0, 2, 5],
    )

    assert selected == ["g0", "g2", "g5"]


def test_select_batch_values_handles_non_contiguous_tensor_rows(tq_client_module):
    tensor = tq_client_module.torch.arange(12).reshape(6, 2)

    selected = tq_client_module._select_batch_values(tensor, [1, 3, 5])

    assert selected.shape == (3, 2)
    assert selected.tolist() == [[2, 3], [6, 7], [10, 11]]


@pytest.mark.asyncio
async def test_filter_and_split_data_maps_shared_columns_by_batch_local_index(
    tq_client_module,
    tq_config_module,
):
    tq_config_module.GROUP_SHARED_COLUMNS = {"prompt_uuid"}
    tq_client_module.GROUP_SHARED_COLUMNS = {"prompt_uuid"}

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
            return {query: False for query in queries}

    client = tq_client_module.TransferQueueClient(
        FakeManager(),
        logger=__import__("logging").getLogger("test"),
    )

    shared_dict, non_shared_dict, groups_to_mark = await client._filter_and_split_data(
        data_dict={
            "prompt_uuid": ["g0", "g0", "g0", "g0", "g1", "g1", "g1", "g1"],
            "payload": tq_client_module.torch.arange(16).reshape(8, 2),
        },
        indexes=[8, 9, 10, 11, 12, 13, 14, 15],
        topic="topic",
        data_len=8,
        n_samples_per_prompt=4,
        needs_expansion=False,
    )

    assert shared_dict == {"prompt_uuid": ["g0", "g1"]}
    assert "payload" in non_shared_dict
    assert groups_to_mark == {2, 3}
