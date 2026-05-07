# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import zmq

from recipe.async_flow.utils.transfer_queue.tq_utils import setup_logger


@dataclass(slots=True, frozen=True)
class StoredBlob:
    payload: memoryview
    logical_length: int
    shape: tuple[int, ...] | None


@dataclass(slots=True, frozen=True)
class ColumnStorageMeta:
    item_size: int
    dtype: str
    encoding: str
    is_shared: bool


@dataclass(slots=True, frozen=True)
class PreparedColumnWrite:
    col_name: str
    item_size: int
    dtype: str
    encoding: str
    is_shared: bool
    entries: dict[int, StoredBlob]


class ExperienceTable:
    """
    Column-oriented experience table backed by Python-owned blobs keyed by global index / group id.
    """

    def __init__(self, n_samples_per_prompt: int, experience_columns: list[str]):
        if n_samples_per_prompt <= 0:
            raise ValueError("n_samples_per_prompt must be a positive integer")
        if not experience_columns:
            raise ValueError("experience_columns must be provided")

        self.n_samples_per_prompt = n_samples_per_prompt
        self.experience_columns = experience_columns
        self.logger = setup_logger("ExperienceTable")
        self.logger.debug(f"ExperienceTable initialized successfully, n_samples_per_prompt={self.n_samples_per_prompt}")

        self.column_entries: dict[str, dict[int, StoredBlob]] = {}
        self.col_metas: dict[str, ColumnStorageMeta] = {}
        self.owned_groups: set[int] = set()
        self._lock = threading.Lock()

    @staticmethod
    def _get_item_size(dtype_str: str) -> int:
        if dtype_str in {"bfloat16", "bf16"}:
            return 2
        if dtype_str in {"float8", "fp8", "float8_e4m3fn", "fp8_e4m3", "float8_e5m2", "fp8_e5m2"}:
            return 1
        return np.dtype(dtype_str).itemsize

    def _storage_key(self, global_idx: int, *, is_shared: bool) -> int:
        if not is_shared:
            return global_idx
        return global_idx // self.n_samples_per_prompt

    def _normalize_shape(self, shape: Any) -> tuple[int, ...] | None:
        if shape is None:
            return None
        return tuple(int(dim) for dim in shape)

    def _resolve_shared_group_ids(self, col_name: str, col_info: dict[str, Any], global_ids: list[int]) -> list[int]:
        group_ids = col_info.get("group_ids")
        if group_ids is None:
            group_ids = sorted({idx // self.n_samples_per_prompt for idx in global_ids})
        if len(col_info["lengths"]) != len(group_ids):
            raise ValueError(
                f"Length mismatch for shared col '{col_name}': "
                f"len(elem_lengths)={len(col_info['lengths'])}, group_ids={len(group_ids)}"
            )
        return [int(group_id) for group_id in group_ids]

    def _expected_bytes(self, *, encoding: str, item_size: int, lengths: list[int]) -> int:
        if encoding == "pickle":
            return sum(lengths)
        return sum(lengths) * item_size

    def _validate_column_meta(self, write: PreparedColumnWrite) -> None:
        existing_meta = self.col_metas.get(write.col_name)
        if existing_meta is None:
            self.col_metas[write.col_name] = ColumnStorageMeta(
                item_size=write.item_size,
                dtype=write.dtype,
                encoding=write.encoding,
                is_shared=write.is_shared,
            )
            return

        if existing_meta.encoding != write.encoding:
            raise ValueError(
                f"Encoding mismatch for column '{write.col_name}': "
                f"existing={existing_meta.encoding}, new={write.encoding}"
            )
        if existing_meta.dtype != write.dtype:
            raise ValueError(
                f"Dtype mismatch for column '{write.col_name}': existing={existing_meta.dtype}, new={write.dtype}"
            )
        if existing_meta.is_shared != write.is_shared:
            raise ValueError(
                f"Sharedness mismatch for column '{write.col_name}': "
                f"existing={existing_meta.is_shared}, new={write.is_shared}"
            )

    def _prepare_column_write(
        self,
        *,
        col_name: str,
        frame: zmq.Frame,
        col_info: dict[str, Any],
        global_ids: list[int],
    ) -> PreparedColumnWrite:
        ref_multiplier = int(col_info.get("ref_multiplier", 1))
        is_shared = ref_multiplier > 1
        lengths = [int(length) for length in col_info["lengths"]]
        dtype_str = str(col_info["dtype"])
        encoding = str(col_info.get("encoding", "raw"))
        shapes = col_info.get("shapes")

        if not is_shared and len(lengths) != len(global_ids):
            raise ValueError(
                f"Length mismatch for col '{col_name}': len(elem_lengths)={len(lengths)} // batch_size={len(global_ids)}"
            )

        storage_keys = (
            self._resolve_shared_group_ids(col_name, col_info, global_ids)
            if is_shared
            else [int(global_idx) for global_idx in global_ids]
        )

        item_size = 1 if encoding == "pickle" else self._get_item_size(dtype_str)
        expected_bytes = self._expected_bytes(encoding=encoding, item_size=item_size, lengths=lengths)
        frame_bytes = frame.bytes
        if expected_bytes != len(frame_bytes):
            raise ValueError(
                f"Size Mismatch '{col_name}': Meta {expected_bytes} bytes != Frame {len(frame_bytes)} bytes"
            )

        owned_blob = bytes(frame_bytes)
        blob_view = memoryview(owned_blob)
        cursor = 0
        entries: dict[int, StoredBlob] = {}

        for idx, storage_key in enumerate(storage_keys):
            logical_length = lengths[idx]
            byte_length = logical_length if encoding == "pickle" else logical_length * item_size
            shape = None
            if shapes is not None and idx < len(shapes):
                shape = self._normalize_shape(shapes[idx])
            entries[storage_key] = StoredBlob(
                payload=blob_view[cursor : cursor + byte_length],
                logical_length=logical_length,
                shape=shape,
            )
            cursor += byte_length

        return PreparedColumnWrite(
            col_name=col_name,
            item_size=item_size,
            dtype=dtype_str,
            encoding=encoding,
            is_shared=is_shared,
            entries=entries,
        )

    def put_batch(
        self,
        global_ids: list[int],
        col_order: list[str],
        col_inputs_meta: dict[str, Any],
        payload_frames: list[zmq.Frame],
    ) -> None:
        if not global_ids:
            return
        if len(payload_frames) < len(col_order):
            missing_col = col_order[len(payload_frames)]
            raise ValueError(f"Missing frame for column {missing_col}")

        cols_info = col_inputs_meta["columns"]
        prepared_writes = [
            self._prepare_column_write(
                col_name=col_name,
                frame=payload_frames[idx],
                col_info=cols_info[col_name],
                global_ids=global_ids,
            )
            for idx, col_name in enumerate(col_order)
        ]

        with self._lock:
            for write in prepared_writes:
                self._validate_column_meta(write)
                col_entries = self.column_entries.setdefault(write.col_name, {})
                col_entries.update(write.entries)

    def get_batch(
        self,
        target_global_idxs: list[int],
        target_cols: list[str],
    ) -> tuple[dict[str, Any], list[bytes]]:
        result_meta = {
            "indexes": target_global_idxs,
            "columns": {col: {"lengths": []} for col in target_cols},
            "order": target_cols,
        }

        column_snapshots: dict[str, tuple[ColumnStorageMeta, list[StoredBlob]]] = {}
        with self._lock:
            for col_name in target_cols:
                col_meta = self.col_metas.get(col_name)
                if col_meta is None:
                    raise KeyError(f"Column {col_name} not found.")

                col_entries = self.column_entries.get(col_name)
                if col_entries is None:
                    raise KeyError(f"Column index {col_name} is empty or missing.")

                blobs: list[StoredBlob] = []
                for global_idx in target_global_idxs:
                    storage_key = self._storage_key(global_idx, is_shared=col_meta.is_shared)
                    blob = col_entries.get(storage_key)
                    if blob is None:
                        raise KeyError(f"Global ID {global_idx} missing in column {col_name}.")
                    blobs.append(blob)
                column_snapshots[col_name] = (col_meta, blobs)

        result_frames: list[bytes] = []
        for col_name in target_cols:
            col_meta, blobs = column_snapshots[col_name]
            result_meta["columns"][col_name]["dtype"] = col_meta.dtype
            result_meta["columns"][col_name]["encoding"] = col_meta.encoding
            result_meta["columns"][col_name]["lengths"] = [blob.logical_length for blob in blobs]
            result_meta["columns"][col_name]["shapes"] = [blob.shape for blob in blobs]

            total_bytes = sum(len(blob.payload) for blob in blobs)
            dest_buffer = bytearray(total_bytes)
            cursor = 0
            for blob in blobs:
                byte_length = len(blob.payload)
                dest_buffer[cursor : cursor + byte_length] = blob.payload
                cursor += byte_length
            result_frames.append(bytes(dest_buffer))

        return result_meta, result_frames

    def prune(self, global_idxs_to_remove: list[int]) -> None:
        if not global_idxs_to_remove:
            return

        group_ids_to_remove = {idx // self.n_samples_per_prompt for idx in global_idxs_to_remove}
        with self._lock:
            for col_name, col_entries in self.column_entries.items():
                col_meta = self.col_metas.get(col_name)
                if col_meta is None:
                    continue
                keys_to_remove = group_ids_to_remove if col_meta.is_shared else set(global_idxs_to_remove)
                for storage_key in keys_to_remove:
                    col_entries.pop(storage_key, None)

            self.owned_groups.difference_update(group_ids_to_remove)

    def clear(self) -> None:
        with self._lock:
            self.column_entries.clear()
            self.col_metas.clear()
            self.owned_groups.clear()
