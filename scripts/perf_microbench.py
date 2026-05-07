from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TQ_NEW_ROOT = REPO_ROOT / "tq_new"
if str(TQ_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(TQ_NEW_ROOT))

import torch
import zmq
from tensordict import TensorDict

from recipe.async_flow.utils.transfer_queue.tq_mgr import TopicMeta, TransferQueueManager
from recipe.async_flow.utils.transfer_queue.tq_structures import ExperienceTable
from recipe.async_flow.utils.transfer_queue.tq_utils import serialize_batch, torch_to_numpy
from transfer_queue.controller import DataPartitionStatus
from transfer_queue.storage.managers.simple_backend_manager import AsyncSimpleStorageManager, RoutingGroup
from transfer_queue.storage.simple_backend import StorageUnitData
from transfer_queue.utils.perf_microbench import (
    build_microbench_result,
    default_microbench_output_path,
    ensure_microbench_parent,
    resolve_microbench_dimensions,
)
from transfer_queue.utils.serial_utils import encode

LOGGER = logging.getLogger("perf_microbench")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DTYPES = (torch.float32, torch.int64, torch.float64, torch.int32)
TOPIC_NAME = "microbench-topic"
CONSUMER_NAME = "trainer"
TQ_MANAGER_IMPL = TransferQueueManager.__ray_metadata__.modified_class


@dataclass(frozen=True)
class MicrobenchPreset:
    name: str
    field_count: int
    serialize_batch_size: int
    serialize_seq_length: int
    storage_batch_size: int
    storage_seq_length: int
    selection_sample_count: int
    selection_fetch_count: int
    aggregation_batch_size: int
    aggregation_seq_length: int
    shard_count: int


PRESETS: dict[str, MicrobenchPreset] = {
    "debug": MicrobenchPreset(
        name="debug",
        field_count=4,
        serialize_batch_size=64,
        serialize_seq_length=512,
        storage_batch_size=128,
        storage_seq_length=512,
        selection_sample_count=12_000,
        selection_fetch_count=1_024,
        aggregation_batch_size=256,
        aggregation_seq_length=256,
        shard_count=4,
    ),
    "default": MicrobenchPreset(
        name="default",
        field_count=4,
        serialize_batch_size=256,
        serialize_seq_length=1_024,
        storage_batch_size=512,
        storage_seq_length=1_024,
        selection_sample_count=50_000,
        selection_fetch_count=2_048,
        aggregation_batch_size=1_024,
        aggregation_seq_length=512,
        shard_count=4,
    ),
}

PUT_BENCHMARK_CONFIGS: dict[str, dict[str, int]] = {
    "debug": {"global_batch_size": 32, "seq_length": 128, "field_num": 2},
    "tiny": {"global_batch_size": 64, "seq_length": 1024, "field_num": 4},
    "small": {"global_batch_size": 512, "seq_length": 12800, "field_num": 4},
    "medium": {"global_batch_size": 1024, "seq_length": 65536, "field_num": 4},
    "large": {"global_batch_size": 2048, "seq_length": 128000, "field_num": 5},
    "xlarge": {"global_batch_size": 4096, "seq_length": 128000, "field_num": 5},
    "huge": {"global_batch_size": 4096, "seq_length": 128000, "field_num": 10},
}


def _buffer_nbytes(buffer: Any) -> int:
    if hasattr(buffer, "nbytes"):
        return int(buffer.nbytes)
    if isinstance(buffer, memoryview):
        return int(buffer.nbytes)
    return len(buffer)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _logical_tensor_dict_bytes(columnar_data: dict[str, torch.Tensor]) -> int:
    return sum(_tensor_nbytes(value) for value in columnar_data.values())


def _build_columnar_tensor_batch(
    batch_size: int,
    seq_length: int,
    field_count: int,
) -> dict[str, torch.Tensor]:
    total_elements = batch_size * seq_length
    base = torch.arange(total_elements, dtype=torch.int64).reshape(batch_size, seq_length)
    columns: dict[str, torch.Tensor] = {}
    for index in range(field_count):
        dtype = DTYPES[index % len(DTYPES)]
        tensor = base + index * 17
        columns[f"field_{index}"] = tensor.to(dtype=dtype)
    return columns


def _build_row_oriented_batch(columnar_data: dict[str, torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    return {field_name: [row.clone() for row in tensor.unbind(0)] for field_name, tensor in columnar_data.items()}


def _serialize_tq_request(
    columnar_data: dict[str, torch.Tensor],
    global_indexes: list[int],
) -> tuple[dict[str, Any], list[Any], int]:
    header = {"topic": TOPIC_NAME, "indexes": global_indexes, "columns": {}, "order": []}
    payload_buffers: list[Any] = []

    for field_name, raw_data in columnar_data.items():
        batch_data = torch_to_numpy(raw_data)
        final_buffer, lengths, dtype_name, shapes = serialize_batch(batch_data, None, None, None)
        header["order"].append(field_name)
        header["columns"][field_name] = {
            "dtype": dtype_name,
            "lengths": lengths,
            "shapes": shapes,
            "ref_multiplier": 1,
            "encoding": "raw",
        }
        payload_buffers.append(final_buffer)

    header_bytes = pickle.dumps(header, protocol=pickle.HIGHEST_PROTOCOL)
    encoded_bytes = len(header_bytes) + sum(_buffer_nbytes(buffer) for buffer in payload_buffers)
    return header, payload_buffers, encoded_bytes


def _wrap_payload_frames(payload_buffers: Iterable[Any]) -> list[zmq.Frame]:
    return [zmq.Frame(memoryview(buffer)) for buffer in payload_buffers]


def _build_selection_sets(
    sample_count: int,
    field_names: list[str],
) -> tuple[dict[str, set[int]], set[int]]:
    ready_by_field: dict[str, set[int]] = {}
    for field_index, field_name in enumerate(field_names):
        stride = 17 + field_index * 2
        ready_by_field[field_name] = {
            sample_index for sample_index in range(sample_count) if (sample_index + field_index) % stride != 0
        }

    usable_seed = sorted(set.intersection(*(ready_by_field[field_name] for field_name in field_names)))
    consumed_indexes = {sample_index for position, sample_index in enumerate(usable_seed) if position % 7 == 0}
    return ready_by_field, consumed_indexes


def _build_partition_schema(field_name: str, seq_length: int) -> dict[str, dict[str, Any]]:
    return {
        field_name: {
            "dtype": "float32",
            "shape": (seq_length,),
            "is_nested": False,
            "is_non_tensor": False,
        }
    }


def _build_tq_selection_manager(
    sample_count: int,
    field_names: list[str],
) -> Any:
    ready_by_field, consumed_indexes = _build_selection_sets(sample_count, field_names)
    topic_meta = TopicMeta(
        nums_tq_data=4,
        prompts_num=sample_count,
        n_samples_per_prompt=1,
        experience_columns=field_names,
        experience_consumers=[CONSUMER_NAME],
    )
    for field_name, ready_indexes in ready_by_field.items():
        topic_meta.experience_ready[field_name].update(ready_indexes)
    topic_meta.experience_consumed[CONSUMER_NAME].update(consumed_indexes)

    manager = TQ_MANAGER_IMPL.__new__(TQ_MANAGER_IMPL)
    manager.logger = LOGGER
    manager.topics = {TOPIC_NAME: topic_meta}
    return manager


def _build_simple_partition(
    sample_count: int,
    field_names: list[str],
    seq_length: int,
) -> DataPartitionStatus:
    ready_by_field, consumed_indexes = _build_selection_sets(sample_count, field_names)
    partition = DataPartitionStatus(partition_id=TOPIC_NAME)
    for field_name, ready_indexes in ready_by_field.items():
        partition.update_production_status(
            global_indices=sorted(ready_indexes),
            field_names=[field_name],
            field_schema=_build_partition_schema(field_name, seq_length),
        )
    partition.mark_consumed(CONSUMER_NAME, sorted(consumed_indexes))
    return partition


def _build_aggregation_fixture(
    batch_size: int,
    seq_length: int,
    field_count: int,
    shard_count: int,
) -> dict[str, Any]:
    field_names = [f"field_{index}" for index in range(field_count)]
    columnar_data = _build_columnar_tensor_batch(batch_size, seq_length, field_count)
    row_oriented_data = _build_row_oriented_batch(columnar_data)
    global_indexes = list(range(batch_size))
    target_order = global_indexes[::2] + global_indexes[1::2]

    routing: dict[str, RoutingGroup] = {}
    shard_results: list[dict[str, Any]] = []
    simple_results: list[tuple[list[str], dict[str, list[torch.Tensor]]]] = []
    for shard_id in range(shard_count):
        endpoint = f"shard-{shard_id}"
        shard_positions = [
            position for position, global_index in enumerate(target_order) if global_index % shard_count == shard_id
        ]
        shard_indexes = [target_order[position] for position in shard_positions]
        routing[endpoint] = RoutingGroup(global_indexes=shard_indexes, batch_positions=shard_positions)
        shard_data = {
            field_name: [row_oriented_data[field_name][global_index] for global_index in shard_indexes]
            for field_name in field_names
        }
        shard_results.append({"indexes": shard_indexes, "data": shard_data})
        simple_results.append((field_names, shard_data))

    return {
        "field_names": field_names,
        "global_indexes": target_order,
        "routing": routing,
        "tq_shard_results": shard_results,
        "simple_results": simple_results,
        "logical_bytes": _logical_tensor_dict_bytes(columnar_data),
    }


def _aggregate_tq_client_results(
    shard_results: list[dict[str, Any]],
    field_names: list[str],
    indexes: list[int],
) -> tuple[dict[str, list[Any]], list[int]]:
    id_to_data_map: dict[int, dict[str, Any]] = {}
    for result in shard_results:
        returned_ids = result["indexes"]
        shard_data = result["data"]
        for item_index, global_index in enumerate(returned_ids):
            sample_pack = {
                field_name: shard_data[field_name][item_index] for field_name in field_names if field_name in shard_data
            }
            id_to_data_map[global_index] = sample_pack

    final_indexes: list[int] = []
    final_columns = {field_name: [] for field_name in field_names}
    for target_index in indexes:
        if target_index not in id_to_data_map:
            continue
        final_indexes.append(target_index)
        sample = id_to_data_map[target_index]
        for field_name in field_names:
            final_columns[field_name].append(sample[field_name])
    return final_columns, final_indexes


def _aggregate_simple_storage_results(
    routing: dict[str, RoutingGroup],
    results: list[tuple[list[str], dict[str, list[Any]]]],
    field_names: list[str],
    global_indexes: list[int],
) -> TensorDict:
    ordered_data: dict[str, list[Any]] = {field_name: [None] * len(global_indexes) for field_name in field_names}
    for (_endpoint, group), (result_fields, storage_data) in zip(routing.items(), results, strict=True):
        for field_name in result_fields:
            for item_index, batch_position in enumerate(group.batch_positions):
                ordered_data[field_name][batch_position] = storage_data[field_name][item_index]

    tensor_data = {
        field_name: AsyncSimpleStorageManager._pack_field_values(values) for field_name, values in ordered_data.items()
    }
    return TensorDict(tensor_data, batch_size=len(global_indexes))


def _run_benchmark(
    timed_fn: Callable[[Any], Any],
    *,
    repeats: int,
    warmup: int,
    setup_fn: Callable[[], Any] | None = None,
) -> list[float]:
    samples: list[float] = []
    total_runs = warmup + repeats
    for run_index in range(total_runs):
        state = setup_fn() if setup_fn is not None else None
        gc.collect()
        start = time.perf_counter()
        timed_fn(state)
        elapsed = time.perf_counter() - start
        if run_index >= warmup:
            samples.append(elapsed)
    return samples


def _build_resolved_preset(args: argparse.Namespace) -> MicrobenchPreset:
    base_dimensions = asdict(PRESETS[args.preset])
    base_dimensions.pop("name", None)
    put_benchmark_config = PUT_BENCHMARK_CONFIGS.get(args.put_config)
    overrides = {
        "field_count": args.field_count,
        "serialize_batch_size": args.serialize_batch_size,
        "serialize_seq_length": args.serialize_seq_length,
        "storage_batch_size": args.storage_batch_size,
        "storage_seq_length": args.storage_seq_length,
        "selection_sample_count": args.selection_sample_count,
        "selection_fetch_count": args.selection_fetch_count,
        "aggregation_batch_size": args.aggregation_batch_size,
        "aggregation_seq_length": args.aggregation_seq_length,
        "shard_count": args.shard_count,
    }
    resolved_dimensions = resolve_microbench_dimensions(
        base_dimensions=base_dimensions,
        put_benchmark_config=put_benchmark_config,
        overrides=overrides,
    )

    name_parts = [args.preset]
    if args.put_config:
        name_parts.append(f"put-{args.put_config}")
    if any(value is not None for value in overrides.values()):
        name_parts.append("custom")

    return MicrobenchPreset(name="+".join(name_parts), **resolved_dimensions)


def _build_benchmarks(
    preset: MicrobenchPreset,
    repeats: int,
    warmup: int,
) -> list[dict[str, Any]]:
    field_names = [f"field_{index}" for index in range(preset.field_count)]
    benchmarks: list[dict[str, Any]] = []

    serialize_columns = _build_columnar_tensor_batch(
        preset.serialize_batch_size,
        preset.serialize_seq_length,
        preset.field_count,
    )
    serialize_indexes = list(range(preset.serialize_batch_size))
    serialize_bytes = _logical_tensor_dict_bytes(serialize_columns)

    tq_serialize_samples = _run_benchmark(
        lambda _state: _serialize_tq_request(serialize_columns, serialize_indexes),
        repeats=repeats,
        warmup=warmup,
    )
    tq_header, tq_payload_buffers, tq_encoded_bytes = _serialize_tq_request(serialize_columns, serialize_indexes)
    benchmarks.append(
        build_microbench_result(
            comparison_key="serialization",
            group="serialization",
            implementation="tq_new",
            case_name="serialize_batch",
            samples_seconds=tq_serialize_samples,
            bytes_processed=serialize_bytes,
            items_processed=preset.serialize_batch_size,
            notes=[
                "Measures tq_new put-path column serialization with torch_to_numpy + serialize_batch.",
                "Network send/recv and Ray are excluded.",
            ],
            extra_fields={
                "field_count": preset.field_count,
                "batch_size": preset.serialize_batch_size,
                "seq_length": preset.serialize_seq_length,
                "encoded_bytes": tq_encoded_bytes,
                "frame_count": len(tq_payload_buffers) + 1,
                "source": "tq_new/recipe/async_flow/utils/transfer_queue/tq_utils.py::serialize_batch",
            },
        )
    )

    simple_serialize_samples = _run_benchmark(
        lambda _state: encode({"global_indexes": serialize_indexes, "data": serialize_columns}),
        repeats=repeats,
        warmup=warmup,
    )
    simple_frames = encode({"global_indexes": serialize_indexes, "data": serialize_columns})
    simple_encoded_bytes = sum(_buffer_nbytes(frame) for frame in simple_frames)
    benchmarks.append(
        build_microbench_result(
            comparison_key="serialization",
            group="serialization",
            implementation="simple_storage",
            case_name="encode",
            samples_seconds=simple_serialize_samples,
            bytes_processed=serialize_bytes,
            items_processed=preset.serialize_batch_size,
            notes=[
                "Measures serial_utils.encode on the SimpleStorage put payload body.",
                "Network send/recv and ZMQMessage wrapper are excluded.",
            ],
            extra_fields={
                "field_count": preset.field_count,
                "batch_size": preset.serialize_batch_size,
                "seq_length": preset.serialize_seq_length,
                "encoded_bytes": simple_encoded_bytes,
                "frame_count": len(simple_frames),
                "source": "transfer_queue/utils/serial_utils.py::encode",
            },
        )
    )

    storage_columns = _build_columnar_tensor_batch(
        preset.storage_batch_size,
        preset.storage_seq_length,
        preset.field_count,
    )
    storage_indexes = list(range(preset.storage_batch_size))
    storage_bytes = _logical_tensor_dict_bytes(storage_columns)
    storage_rows = _build_row_oriented_batch(storage_columns)
    tq_put_header, tq_put_buffers, _ = _serialize_tq_request(storage_columns, storage_indexes)

    def build_tq_put_state() -> tuple[ExperienceTable, dict[str, Any], list[zmq.Frame]]:
        return (
            ExperienceTable(n_samples_per_prompt=1, experience_columns=field_names),
            tq_put_header,
            _wrap_payload_frames(tq_put_buffers),
        )

    tq_put_samples = _run_benchmark(
        lambda state: state[0].put_batch(
            global_ids=storage_indexes,
            col_order=field_names,
            col_inputs_meta=state[1],
            payload_frames=state[2],
        ),
        repeats=repeats,
        warmup=warmup,
        setup_fn=build_tq_put_state,
    )
    benchmarks.append(
        build_microbench_result(
            comparison_key="storage_put",
            group="storage",
            implementation="tq_new",
            case_name="ExperienceTable.put_batch",
            samples_seconds=tq_put_samples,
            bytes_processed=storage_bytes,
            items_processed=preset.storage_batch_size,
            notes=["Serialization is precomputed outside the timed section; measures table insertion only."],
            extra_fields={
                "field_count": preset.field_count,
                "batch_size": preset.storage_batch_size,
                "seq_length": preset.storage_seq_length,
                "source": "tq_new/recipe/async_flow/utils/transfer_queue/tq_structures.py::ExperienceTable.put_batch",
            },
        )
    )

    simple_put_samples = _run_benchmark(
        lambda storage: storage.put_data(storage_rows, storage_indexes),
        repeats=repeats,
        warmup=warmup,
        setup_fn=lambda: StorageUnitData(storage_size=preset.storage_batch_size * 2),
    )
    benchmarks.append(
        build_microbench_result(
            comparison_key="storage_put",
            group="storage",
            implementation="simple_storage",
            case_name="StorageUnitData.put_data",
            samples_seconds=simple_put_samples,
            bytes_processed=storage_bytes,
            items_processed=preset.storage_batch_size,
            notes=["Measures in-memory dict-backed storage writes with row-oriented sample values."],
            extra_fields={
                "field_count": preset.field_count,
                "batch_size": preset.storage_batch_size,
                "seq_length": preset.storage_seq_length,
                "source": "transfer_queue/storage/simple_backend.py::StorageUnitData.put_data",
            },
        )
    )

    def build_tq_get_state() -> tuple[ExperienceTable, list[int], list[str]]:
        table = ExperienceTable(n_samples_per_prompt=1, experience_columns=field_names)
        table.put_batch(
            global_ids=storage_indexes,
            col_order=field_names,
            col_inputs_meta=tq_put_header,
            payload_frames=_wrap_payload_frames(tq_put_buffers),
        )
        return table, storage_indexes, field_names

    tq_get_samples = _run_benchmark(
        lambda state: state[0].get_batch(state[1], state[2]),
        repeats=repeats,
        warmup=warmup,
        setup_fn=build_tq_get_state,
    )
    benchmarks.append(
        build_microbench_result(
            comparison_key="storage_get",
            group="storage",
            implementation="tq_new",
            case_name="ExperienceTable.get_batch",
            samples_seconds=tq_get_samples,
            bytes_processed=storage_bytes,
            items_processed=preset.storage_batch_size,
            notes=["Measures slab reconstruction and metadata assembly inside ExperienceTable.get_batch."],
            extra_fields={
                "field_count": preset.field_count,
                "batch_size": preset.storage_batch_size,
                "seq_length": preset.storage_seq_length,
                "source": "tq_new/recipe/async_flow/utils/transfer_queue/tq_structures.py::ExperienceTable.get_batch",
            },
        )
    )

    def build_simple_get_state() -> StorageUnitData:
        storage = StorageUnitData(storage_size=preset.storage_batch_size * 2)
        storage.put_data(storage_rows, storage_indexes)
        return storage

    simple_get_samples = _run_benchmark(
        lambda storage: storage.get_data(field_names, storage_indexes),
        repeats=repeats,
        warmup=warmup,
        setup_fn=build_simple_get_state,
    )
    benchmarks.append(
        build_microbench_result(
            comparison_key="storage_get",
            group="storage",
            implementation="simple_storage",
            case_name="StorageUnitData.get_data",
            samples_seconds=simple_get_samples,
            bytes_processed=storage_bytes,
            items_processed=preset.storage_batch_size,
            notes=["Measures in-memory dict-backed storage reads with row-oriented sample values."],
            extra_fields={
                "field_count": preset.field_count,
                "batch_size": preset.storage_batch_size,
                "seq_length": preset.storage_seq_length,
                "source": "transfer_queue/storage/simple_backend.py::StorageUnitData.get_data",
            },
        )
    )

    tq_selection_samples = _run_benchmark(
        lambda manager: TQ_MANAGER_IMPL._sample_ready_index(
            manager,
            topic=TOPIC_NAME,
            consumer=CONSUMER_NAME,
            experience_count=preset.selection_fetch_count,
            experience_columns=field_names,
        ),
        repeats=repeats,
        warmup=warmup,
        setup_fn=lambda: _build_tq_selection_manager(preset.selection_sample_count, field_names),
    )
    benchmarks.append(
        build_microbench_result(
            comparison_key="selection",
            group="selection",
            implementation="tq_new",
            case_name="ready_usable_selection",
            samples_seconds=tq_selection_samples,
            items_processed=preset.selection_sample_count,
            notes=["Benchmarks the current tq_mgr set/dict/sort ready/usable path with n_samples_per_prompt=1."],
            extra_fields={
                "field_count": preset.field_count,
                "sample_count": preset.selection_sample_count,
                "requested_batch_size": preset.selection_fetch_count,
                "source": "tq_new/recipe/async_flow/utils/transfer_queue/tq_mgr.py::_sample_ready_index",
            },
        )
    )

    simple_selection_samples = _run_benchmark(
        lambda partition: partition.scan_data_status(field_names, CONSUMER_NAME),
        repeats=repeats,
        warmup=warmup,
        setup_fn=lambda: _build_simple_partition(
            preset.selection_sample_count,
            field_names,
            preset.storage_seq_length,
        ),
    )
    benchmarks.append(
        build_microbench_result(
            comparison_key="selection",
            group="selection",
            implementation="simple_storage",
            case_name="scan_data_status",
            samples_seconds=simple_selection_samples,
            items_processed=preset.selection_sample_count,
            notes=[
                "Benchmarks DataPartitionStatus.scan_data_status, which backs TransferQueueController.scan_data_status."
            ],
            extra_fields={
                "field_count": preset.field_count,
                "sample_count": preset.selection_sample_count,
                "requested_batch_size": preset.selection_fetch_count,
                "source": "transfer_queue/controller.py::DataPartitionStatus.scan_data_status",
            },
        )
    )

    aggregation_fixture = _build_aggregation_fixture(
        preset.aggregation_batch_size,
        preset.aggregation_seq_length,
        preset.field_count,
        preset.shard_count,
    )
    tq_aggregation_samples = _run_benchmark(
        lambda _state: _aggregate_tq_client_results(
            aggregation_fixture["tq_shard_results"],
            aggregation_fixture["field_names"],
            aggregation_fixture["global_indexes"],
        ),
        repeats=repeats,
        warmup=warmup,
    )
    benchmarks.append(
        build_microbench_result(
            comparison_key="aggregation",
            group="aggregation",
            implementation="tq_new",
            case_name="shard_result_aggregation",
            samples_seconds=tq_aggregation_samples,
            bytes_processed=aggregation_fixture["logical_bytes"],
            items_processed=preset.aggregation_batch_size,
            notes=["Mirrors tq_client.get_experience_async aggregation after shard fetches complete."],
            extra_fields={
                "field_count": preset.field_count,
                "batch_size": preset.aggregation_batch_size,
                "seq_length": preset.aggregation_seq_length,
                "shard_count": preset.shard_count,
                "source": "tq_new/recipe/async_flow/utils/transfer_queue/tq_client.py::get_experience_async aggregate_results",
            },
        )
    )

    simple_aggregation_samples = _run_benchmark(
        lambda _state: _aggregate_simple_storage_results(
            aggregation_fixture["routing"],
            aggregation_fixture["simple_results"],
            aggregation_fixture["field_names"],
            aggregation_fixture["global_indexes"],
        ),
        repeats=repeats,
        warmup=warmup,
    )
    benchmarks.append(
        build_microbench_result(
            comparison_key="aggregation",
            group="aggregation",
            implementation="simple_storage",
            case_name="get_post_processing",
            samples_seconds=simple_aggregation_samples,
            bytes_processed=aggregation_fixture["logical_bytes"],
            items_processed=preset.aggregation_batch_size,
            notes=["Mirrors AsyncSimpleStorageManager.get_data aggregation and final TensorDict packing."],
            extra_fields={
                "field_count": preset.field_count,
                "batch_size": preset.aggregation_batch_size,
                "seq_length": preset.aggregation_seq_length,
                "shard_count": preset.shard_count,
                "source": "transfer_queue/storage/managers/simple_backend_manager.py::get_data aggregate_results",
            },
        )
    )

    return benchmarks


def _build_comparisons(benchmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(entry["comparison_key"], entry["implementation"]): entry for entry in benchmarks}
    comparisons: list[dict[str, Any]] = []
    for comparison_key in ("serialization", "storage_put", "storage_get", "selection", "aggregation"):
        tq_entry = indexed.get((comparison_key, "tq_new"))
        simple_entry = indexed.get((comparison_key, "simple_storage"))
        if tq_entry is None or simple_entry is None:
            continue

        tq_median = tq_entry["median_seconds"]
        simple_median = simple_entry["median_seconds"]
        tq_rate = tq_entry["median_bytes_per_second"] or tq_entry["median_items_per_second"]
        simple_rate = simple_entry["median_bytes_per_second"] or simple_entry["median_items_per_second"]

        if tq_median < simple_median:
            faster = "tq_new"
        elif tq_median > simple_median:
            faster = "simple_storage"
        else:
            faster = "tie"

        comparisons.append(
            {
                "comparison_key": comparison_key,
                "tq_new_case": tq_entry["case_name"],
                "simple_storage_case": simple_entry["case_name"],
                "tq_new_median_seconds": tq_median,
                "simple_storage_median_seconds": simple_median,
                "median_time_ratio_tq_over_simple": (tq_median / simple_median) if simple_median > 0 else None,
                "median_throughput_ratio_tq_over_simple": (tq_rate / simple_rate) if simple_rate > 0 else None,
                "faster_implementation": faster,
            }
        )
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated tq_new vs SimpleStorage microbenchmarks.")
    parser.add_argument(
        "--preset",
        default="debug",
        choices=sorted(PRESETS),
        help="Workload preset for the microbench suite.",
    )
    parser.add_argument(
        "--put-config",
        default=None,
        choices=sorted(PUT_BENCHMARK_CONFIGS),
        help="Align core batch/seq/field sizes to the tq_new put_benchmark CONFIG_MAP before applying overrides.",
    )
    parser.add_argument("--repeats", type=int, default=7, help="Measured iterations per benchmark.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per benchmark.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--field-count", type=int, default=None, help="Override field count for all benchmark groups.")
    parser.add_argument("--serialize-batch-size", type=int, default=None, help="Override serialization batch size.")
    parser.add_argument(
        "--serialize-seq-length", type=int, default=None, help="Override serialization sequence length."
    )
    parser.add_argument("--storage-batch-size", type=int, default=None, help="Override storage put/get batch size.")
    parser.add_argument(
        "--storage-seq-length", type=int, default=None, help="Override storage put/get sequence length."
    )
    parser.add_argument(
        "--selection-sample-count",
        type=int,
        default=None,
        help="Override total candidate sample count for selection logic benchmark.",
    )
    parser.add_argument(
        "--selection-fetch-count",
        type=int,
        default=None,
        help="Override requested batch size for the selection logic benchmark.",
    )
    parser.add_argument("--aggregation-batch-size", type=int, default=None, help="Override aggregation batch size.")
    parser.add_argument(
        "--aggregation-seq-length", type=int, default=None, help="Override aggregation sequence length."
    )
    parser.add_argument("--shard-count", type=int, default=None, help="Override shard/storage-unit count.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    for key, value in vars(args).items():
        if (
            key
            in {
                "field_count",
                "serialize_batch_size",
                "serialize_seq_length",
                "storage_batch_size",
                "storage_seq_length",
                "selection_sample_count",
                "selection_fetch_count",
                "aggregation_batch_size",
                "aggregation_seq_length",
                "shard_count",
            }
            and value is not None
            and value < 1
        ):
            raise ValueError(f"--{key.replace('_', '-')} must be >= 1")

    torch.set_num_threads(1)
    preset = _build_resolved_preset(args)
    output_path = args.output or default_microbench_output_path(repo_root=REPO_ROOT)
    output_path = ensure_microbench_parent(output_path)

    LOGGER.info(
        "Running perf microbench preset=%s repeats=%s warmup=%s output=%s",
        preset.name,
        args.repeats,
        args.warmup,
        output_path,
    )

    benchmarks = _build_benchmarks(preset, repeats=args.repeats, warmup=args.warmup)
    comparisons = _build_comparisons(benchmarks)

    payload = {
        "script": str(Path(__file__).resolve()),
        "repo_root": str(REPO_ROOT),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "preset": asdict(preset),
        "requested_preset": args.preset,
        "requested_put_config": args.put_config,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "benchmarks": benchmarks,
        "comparisons": comparisons,
        "notes": [
            "All benchmarks are local-only and exclude Ray actor startup, ZMQ network transfer, and distributed scheduling.",
            "tq_client aggregation and SimpleStorage get post-processing are mirrored from current source loops to isolate the hot aggregation logic.",
            "Selection benchmark targets DataPartitionStatus.scan_data_status for SimpleStorage because TransferQueueController delegates to it.",
        ],
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote microbench results to %s", output_path)

    for comparison in comparisons:
        LOGGER.info(
            "[%s] faster=%s tq/simple=%.3f",
            comparison["comparison_key"],
            comparison["faster_implementation"],
            comparison["median_time_ratio_tq_over_simple"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
