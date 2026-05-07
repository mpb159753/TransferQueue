from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import zmq

TQ_NEW_ROOT = Path(__file__).resolve().parents[4]
if str(TQ_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(TQ_NEW_ROOT))

from recipe.async_flow.utils.transfer_queue.tq_structures import ExperienceTable
from recipe.async_flow.utils.transfer_queue.tq_utils import (
    deserialize_column_from_frame,
    deserialize_column_pickle_from_frame,
    serialize_batch,
    serialize_batch_pickle,
)


def _build_raw_column(
    arrays: list[np.ndarray],
    *,
    ref_multiplier: int = 1,
    group_ids: list[int] | None = None,
) -> tuple[dict[str, object], zmq.Frame]:
    normalized = [np.asarray(arr) for arr in arrays]
    final_buffer, lengths, dtype_str, shapes = serialize_batch(normalized)
    column_meta: dict[str, object] = {
        "dtype": dtype_str,
        "lengths": lengths,
        "shapes": shapes,
        "ref_multiplier": ref_multiplier,
        "encoding": "raw",
    }
    if group_ids is not None:
        column_meta["group_ids"] = group_ids
    return column_meta, zmq.Frame(final_buffer.tobytes())


def _build_pickle_column(
    objects: list[object],
    *,
    ref_multiplier: int = 1,
    group_ids: list[int] | None = None,
) -> tuple[dict[str, object], zmq.Frame]:
    final_buffer, lengths, dtype_str = serialize_batch_pickle(objects)
    column_meta: dict[str, object] = {
        "dtype": dtype_str,
        "lengths": lengths,
        "shapes": None,
        "ref_multiplier": ref_multiplier,
        "encoding": "pickle",
    }
    if group_ids is not None:
        column_meta["group_ids"] = group_ids
    return column_meta, zmq.Frame(final_buffer)


def _decode_raw(payload: bytes | bytearray, meta: dict[str, object]) -> list[list[object]]:
    tensors = deserialize_column_from_frame(
        payload,
        meta["dtype"],
        meta["lengths"],
        copy=True,
        shapes=meta["shapes"],
    )
    return [tensor.tolist() for tensor in tensors]


def _decode_pickle(payload: bytes | bytearray, lengths: list[int]) -> list[object]:
    return deserialize_column_pickle_from_frame(zmq.Frame(bytes(payload)), lengths)


def test_put_and_get_non_shared_raw_column_round_trip():
    table = ExperienceTable(n_samples_per_prompt=2, experience_columns=["tokens"])
    column_meta, frame = _build_raw_column(
        [
            np.asarray([1, 2], dtype=np.int32),
            np.asarray([3, 4, 5], dtype=np.int32),
        ]
    )

    table.put_batch(
        global_ids=[0, 1],
        col_order=["tokens"],
        col_inputs_meta={"columns": {"tokens": column_meta}},
        payload_frames=[frame],
    )

    result_meta, result_payloads = table.get_batch([1, 0], ["tokens"])
    tokens_meta = result_meta["columns"]["tokens"]

    assert result_meta["indexes"] == [1, 0]
    assert result_meta["order"] == ["tokens"]
    assert tokens_meta["dtype"] == "int32"
    assert tokens_meta["encoding"] == "raw"
    assert tokens_meta["lengths"] == [3, 2]
    assert tokens_meta["shapes"] == [(3,), (2,)]
    assert _decode_raw(result_payloads[0], tokens_meta) == [[3, 4, 5], [1, 2]]


def test_shared_column_stores_once_per_group_and_reads_for_each_index():
    table = ExperienceTable(n_samples_per_prompt=2, experience_columns=["prompt"])
    column_meta, frame = _build_raw_column(
        [
            np.asarray([10, 11], dtype=np.int64),
            np.asarray([20, 21, 22], dtype=np.int64),
        ],
        ref_multiplier=2,
        group_ids=[0, 1],
    )

    table.put_batch(
        global_ids=[0, 1, 2, 3],
        col_order=["prompt"],
        col_inputs_meta={"columns": {"prompt": column_meta}},
        payload_frames=[frame],
    )

    assert len(table.column_entries["prompt"]) == 2

    result_meta, result_payloads = table.get_batch([0, 1, 2, 3], ["prompt"])
    prompt_meta = result_meta["columns"]["prompt"]

    assert prompt_meta["lengths"] == [2, 2, 3, 3]
    assert prompt_meta["shapes"] == [(2,), (2,), (3,), (3,)]
    assert _decode_raw(result_payloads[0], prompt_meta) == [
        [10, 11],
        [10, 11],
        [20, 21, 22],
        [20, 21, 22],
    ]


def test_mixed_raw_and_pickle_columns_round_trip():
    table = ExperienceTable(n_samples_per_prompt=2, experience_columns=["tokens", "metadata"])
    raw_meta, raw_frame = _build_raw_column(
        [
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[5.0, 6.0]], dtype=np.float32),
        ]
    )
    pickle_meta, pickle_frame = _build_pickle_column(
        [
            {"id": "sample-0", "score": 0.5},
            ["tag-a", "tag-b"],
        ]
    )

    table.put_batch(
        global_ids=[4, 5],
        col_order=["tokens", "metadata"],
        col_inputs_meta={"columns": {"tokens": raw_meta, "metadata": pickle_meta}},
        payload_frames=[raw_frame, pickle_frame],
    )

    result_meta, result_payloads = table.get_batch([4, 5], ["tokens", "metadata"])

    assert result_meta["order"] == ["tokens", "metadata"]
    assert _decode_raw(result_payloads[0], result_meta["columns"]["tokens"]) == [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0]],
    ]
    assert _decode_pickle(
        result_payloads[1],
        result_meta["columns"]["metadata"]["lengths"],
    ) == [{"id": "sample-0", "score": 0.5}, ["tag-a", "tag-b"]]
    assert result_meta["columns"]["metadata"]["encoding"] == "pickle"
    assert result_meta["columns"]["metadata"]["shapes"] == [None, None]


def test_overwrite_replaces_existing_index_and_group_payloads():
    table = ExperienceTable(n_samples_per_prompt=2, experience_columns=["reward", "prompt"])

    reward_meta, reward_frame = _build_raw_column(
        [
            np.asarray([1.0], dtype=np.float32),
            np.asarray([2.0], dtype=np.float32),
        ]
    )
    table.put_batch(
        global_ids=[0, 1],
        col_order=["reward"],
        col_inputs_meta={"columns": {"reward": reward_meta}},
        payload_frames=[reward_frame],
    )

    shared_meta, shared_frame = _build_raw_column(
        [np.asarray([101, 102], dtype=np.int32)],
        ref_multiplier=2,
        group_ids=[0],
    )
    table.put_batch(
        global_ids=[0, 1],
        col_order=["prompt"],
        col_inputs_meta={"columns": {"prompt": shared_meta}},
        payload_frames=[shared_frame],
    )

    replacement_reward_meta, replacement_reward_frame = _build_raw_column(
        [
            np.asarray([10.0], dtype=np.float32),
            np.asarray([20.0], dtype=np.float32),
        ]
    )
    replacement_shared_meta, replacement_shared_frame = _build_raw_column(
        [np.asarray([201, 202, 203], dtype=np.int32)],
        ref_multiplier=2,
        group_ids=[0],
    )
    table.put_batch(
        global_ids=[0, 1],
        col_order=["reward", "prompt"],
        col_inputs_meta={
            "columns": {
                "reward": replacement_reward_meta,
                "prompt": replacement_shared_meta,
            }
        },
        payload_frames=[replacement_reward_frame, replacement_shared_frame],
    )

    result_meta, result_payloads = table.get_batch([0, 1], ["reward", "prompt"])

    assert _decode_raw(result_payloads[0], result_meta["columns"]["reward"]) == [[10.0], [20.0]]
    assert _decode_raw(result_payloads[1], result_meta["columns"]["prompt"]) == [
        [201, 202, 203],
        [201, 202, 203],
    ]
    assert len(table.column_entries["prompt"]) == 1


def test_put_batch_validates_lengths_group_ids_and_column_metadata():
    table = ExperienceTable(n_samples_per_prompt=2, experience_columns=["tokens", "meta"])

    valid_meta, valid_frame = _build_raw_column([np.asarray([1, 2], dtype=np.int32), np.asarray([3], dtype=np.int32)])
    table.put_batch(
        global_ids=[0, 1],
        col_order=["tokens"],
        col_inputs_meta={"columns": {"tokens": valid_meta}},
        payload_frames=[valid_frame],
    )

    bad_lengths_meta = dict(valid_meta)
    bad_lengths_meta["lengths"] = [2]
    with pytest.raises(ValueError, match="Length mismatch for col 'tokens'"):
        table.put_batch(
            global_ids=[0, 1],
            col_order=["tokens"],
            col_inputs_meta={"columns": {"tokens": bad_lengths_meta}},
            payload_frames=[valid_frame],
        )

    shared_meta, shared_frame = _build_raw_column(
        [np.asarray([10, 11], dtype=np.int32)],
        ref_multiplier=2,
        group_ids=[0, 1],
    )
    with pytest.raises(ValueError, match="Length mismatch for shared col 'meta'"):
        table.put_batch(
            global_ids=[0, 1, 2, 3],
            col_order=["meta"],
            col_inputs_meta={"columns": {"meta": shared_meta}},
            payload_frames=[shared_frame],
        )

    wrong_dtype_meta, wrong_dtype_frame = _build_raw_column(
        [np.asarray([1.5, 2.5], dtype=np.float32), np.asarray([3.5], dtype=np.float32)]
    )
    with pytest.raises(ValueError, match="Dtype mismatch for column 'tokens'"):
        table.put_batch(
            global_ids=[0, 1],
            col_order=["tokens"],
            col_inputs_meta={"columns": {"tokens": wrong_dtype_meta}},
            payload_frames=[wrong_dtype_frame],
        )

    pickle_meta, pickle_frame = _build_pickle_column([{"value": 1}, {"value": 2}])
    with pytest.raises(ValueError, match="Encoding mismatch for column 'tokens'"):
        table.put_batch(
            global_ids=[0, 1],
            col_order=["tokens"],
            col_inputs_meta={"columns": {"tokens": pickle_meta}},
            payload_frames=[pickle_frame],
        )


def test_prune_removes_shared_group_and_invalidates_all_group_indexes():
    table = ExperienceTable(n_samples_per_prompt=2, experience_columns=["prompt", "reward"])
    table.owned_groups.update({0, 1})

    shared_meta, shared_frame = _build_raw_column(
        [
            np.asarray([1, 2], dtype=np.int32),
            np.asarray([3, 4, 5], dtype=np.int32),
        ],
        ref_multiplier=2,
        group_ids=[0, 1],
    )
    reward_meta, reward_frame = _build_raw_column(
        [
            np.asarray([0.1], dtype=np.float32),
            np.asarray([0.2], dtype=np.float32),
            np.asarray([0.3], dtype=np.float32),
            np.asarray([0.4], dtype=np.float32),
        ]
    )
    table.put_batch(
        global_ids=[0, 1, 2, 3],
        col_order=["prompt", "reward"],
        col_inputs_meta={"columns": {"prompt": shared_meta, "reward": reward_meta}},
        payload_frames=[shared_frame, reward_frame],
    )

    table.prune([0, 1])

    assert 0 not in table.column_entries["prompt"]
    assert 0 not in table.owned_groups
    assert 1 in table.owned_groups

    with pytest.raises(KeyError, match="Global ID 0 missing"):
        table.get_batch([0], ["prompt"])
    with pytest.raises(KeyError, match="Global ID 1 missing"):
        table.get_batch([1], ["reward"])

    result_meta, result_payloads = table.get_batch([2, 3], ["prompt", "reward"])
    assert _decode_raw(result_payloads[0], result_meta["columns"]["prompt"]) == [
        [3, 4, 5],
        [3, 4, 5],
    ]
    reward_values = _decode_raw(result_payloads[1], result_meta["columns"]["reward"])
    assert [value[0] for value in reward_values] == pytest.approx([0.3, 0.4])


def test_clear_resets_table_and_allows_fresh_writes():
    table = ExperienceTable(n_samples_per_prompt=2, experience_columns=["tokens"])
    initial_meta, initial_frame = _build_raw_column([np.asarray([1, 2], dtype=np.int32)])
    table.put_batch(
        global_ids=[0],
        col_order=["tokens"],
        col_inputs_meta={"columns": {"tokens": initial_meta}},
        payload_frames=[initial_frame],
    )

    table.clear()

    assert table.column_entries == {}
    assert table.col_metas == {}
    assert table.owned_groups == set()

    replacement_meta, replacement_frame = _build_raw_column([np.asarray([1.5, 2.5], dtype=np.float32)])
    table.put_batch(
        global_ids=[0],
        col_order=["tokens"],
        col_inputs_meta={"columns": {"tokens": replacement_meta}},
        payload_frames=[replacement_frame],
    )

    result_meta, result_payloads = table.get_batch([0], ["tokens"])
    tokens_meta = result_meta["columns"]["tokens"]
    assert tokens_meta["dtype"] == "float32"
    assert _decode_raw(result_payloads[0], tokens_meta) == [[1.5, 2.5]]
