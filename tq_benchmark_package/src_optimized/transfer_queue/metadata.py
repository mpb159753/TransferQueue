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

import copy
import dataclasses
import itertools
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


@dataclass
class BatchMeta:
    """Records the metadata of a batch of data samples with optimized field-level schema.

    This is the O(BxF) optimized version that stores field metadata at the field level
    instead of per-sample, reducing storage from O(B*F) to O(F).

    Attributes:
        global_indexes: List of global sample indices in this batch.
        partition_ids: List of partition IDs corresponding to each sample.
        field_schema: Field-level metadata {field_name: {dtype, shape, is_nested, is_non_tensor, per_sample_shapes}}.
        production_status: Vectorized production status, shape (B,) where B is batch size.
        extra_info: Additional batch-level information.
        _custom_meta: Per-sample custom metadata for storage backends.
    """

    global_indexes: list[int]
    partition_ids: list[str]
    # O(F) field-level metadata: {field_name: {dtype, shape, is_nested, is_non_tensor, per_sample_shapes}}
    field_schema: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    # O(B) vectorized production status
    production_status: Optional[np.ndarray] = None
    extra_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    # internal data for different storage backends: _custom_meta[global_index][field]
    _custom_meta: dict[int, dict[str, Any]] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """Initialize all computed properties during initialization"""
        self.global_indexes = copy.deepcopy(self.global_indexes)
        self.partition_ids = copy.deepcopy(self.partition_ids)
        self.field_schema = copy.deepcopy(self.field_schema)
        self.extra_info = copy.deepcopy(self.extra_info)

        # Validation
        if len(self.global_indexes) != len(self.partition_ids):
            raise ValueError(
                f"Length mismatch: global_indexes has {len(self.global_indexes)}, "
                f"partition_ids has {len(self.partition_ids)}"
            )

        batch_size = len(self.global_indexes)

        # Validate production_status if provided
        if self.production_status is not None:
            if isinstance(self.production_status, np.ndarray):
                self.production_status = self.production_status.copy()
            elif isinstance(self.production_status, torch.Tensor):
                self.production_status = self.production_status.numpy().copy()
            elif isinstance(self.production_status, list):
                self.production_status = np.array(self.production_status, dtype=np.int8)

            if len(self.production_status) != batch_size:
                raise ValueError(f"production_status length {len(self.production_status)} != batch_size {batch_size}")
        else:
            # Default: all NOT_PRODUCED
            self.production_status = np.zeros(batch_size, dtype=np.int8) if batch_size > 0 else None

        # Validate per_sample_shapes in field_schema
        for field_name, meta in self.field_schema.items():
            if meta.get("per_sample_shapes") is not None:
                if len(meta["per_sample_shapes"]) != batch_size:
                    raise ValueError(
                        f"Field '{field_name}' per_sample_shapes length {len(meta['per_sample_shapes'])} "
                        f"!= batch_size {batch_size}"
                    )

        self._size = batch_size
        self._field_names = sorted(self.field_schema.keys())

        # Check if is_ready (all production_status == READY_FOR_CONSUME i.e. 1)
        is_ready = False
        if batch_size > 0 and self.production_status is not None:
            is_ready = bool(np.all(self.production_status == 1))

        self._is_ready = is_ready

    @property
    def size(self) -> int:
        """Return the number of samples in this batch"""
        return getattr(self, "_size", 0)

    @property
    def field_names(self) -> list[str]:
        """Get all unique field names in this batch"""
        return getattr(self, "_field_names", [])

    @property
    def is_ready(self) -> bool:
        """Check if all samples in this batch are ready for consumption"""
        return getattr(self, "_is_ready", False)

    # Custom meta methods for different storage backends
    def get_all_custom_meta(self) -> dict[int, dict[str, Any]]:
        """Get the entire custom meta dictionary"""
        return copy.deepcopy(self._custom_meta)

    def update_custom_meta(self, new_custom_meta: Optional[dict[int, dict[str, Any]]]):
        """Update custom meta with a new dictionary"""
        if new_custom_meta:
            for idx, meta in new_custom_meta.items():
                if idx not in self._custom_meta:
                    self._custom_meta[idx] = {}
                self._custom_meta[idx].update(meta)

    # Extra info interface methods
    def get_extra_info(self, key: str, default: Any = None) -> Any:
        """Get extra info by key"""
        return self.extra_info.get(key, default)

    def set_extra_info(self, key: str, value: Any) -> None:
        """Set extra info by key"""
        self.extra_info[key] = value

    def update_extra_info(self, info_dict: dict[str, Any]) -> None:
        """Update extra info with multiple key-value pairs"""
        self.extra_info.update(info_dict)

    def remove_extra_info(self, key: str) -> Any:
        """Remove extra info by key and return its value"""
        return self.extra_info.pop(key, None)

    def clear_extra_info(self) -> None:
        """Clear all extra info"""
        self.extra_info.clear()

    def has_extra_info(self, key: str) -> bool:
        """Check if extra info contains a specific key"""
        return key in self.extra_info

    def get_all_extra_info(self) -> dict[str, Any]:
        """Get all extra info as a dictionary"""
        return copy.deepcopy(self.extra_info)

    def add_fields(self, tensor_dict: TensorDict, set_all_ready: bool = True) -> "BatchMeta":
        """
        Add new fields from a TensorDict to all samples in this batch.
        This modifies the batch in-place to include the new fields.

        Args:
            tensor_dict (TensorDict): The input TensorDict containing new fields.
            set_all_ready (bool): If True, set all production_status to READY_FOR_CONSUME. Default is True.
        """
        batch_size = tensor_dict.batch_size[0]
        if batch_size != self.size:
            raise ValueError(f"add_fields batch size mismatch: self.size={self.size} vs tensor_dict={batch_size}")

        for name, value in tensor_dict.items():
            # Determine if this is a nested tensor
            is_nested = isinstance(value, torch.Tensor) and value.is_nested

            first_item = None
            if is_nested:
                unbound = value.unbind()
                first_item = unbound[0] if unbound else None
            else:
                first_item = value[0] if len(value) > 0 else None

            # Determine if this is non-tensor data
            is_non_tensor = not isinstance(first_item, torch.Tensor) if first_item is not None else False

            field_meta = {
                "dtype": getattr(first_item, "dtype", type(first_item) if first_item is not None else None),
                "shape": getattr(first_item, "shape", None) if not is_nested else None,
                "is_nested": is_nested,
                "is_non_tensor": is_non_tensor,
            }

            # For nested tensors, record per-sample shapes
            if is_nested:
                field_meta["per_sample_shapes"] = [tuple(t.shape) for t in value.unbind()]

            self.field_schema[name] = field_meta

        # Update production status if set_all_ready
        if set_all_ready and self.production_status is not None:
            self.production_status[:] = 1

        # Update cached properties
        self._field_names = sorted(self.field_schema.keys())

        # Re-evaluate is_ready
        is_ready = False
        if self.size > 0 and self.production_status is not None:
            is_ready = bool(np.all(self.production_status == 1))
        self._is_ready = is_ready

        return self

    def select_samples(self, sample_indices: list[int]) -> "BatchMeta":
        """
        Select specific samples from this batch.
        This will construct a new BatchMeta instance containing only the specified samples.

        Args:
            sample_indices (list[int]): List of sample indices to retain.

        Returns:
            BatchMeta: A new BatchMeta instance containing only the specified samples.
        """
        if any(i < 0 or i >= self.size for i in sample_indices):
            raise ValueError(f"Sample indices must be in range [0, {self.size})")

        new_global_indexes = [self.global_indexes[i] for i in sample_indices]
        new_partition_ids = [self.partition_ids[i] for i in sample_indices]

        # Select production_status
        new_production_status = None
        if self.production_status is not None:
            new_production_status = self.production_status[sample_indices]

        # Select per_sample_shapes in field_schema
        new_field_schema = {}
        for fname, meta in self.field_schema.items():
            new_meta = copy.deepcopy(meta)
            if meta.get("per_sample_shapes") is not None:
                new_meta["per_sample_shapes"] = [meta["per_sample_shapes"][i] for i in sample_indices]
            new_field_schema[fname] = new_meta

        # Select custom meta
        new_custom_meta = {}
        for idx in new_global_indexes:
            if idx in self._custom_meta:
                new_custom_meta[idx] = copy.deepcopy(self._custom_meta[idx])

        return BatchMeta(
            global_indexes=new_global_indexes,
            partition_ids=new_partition_ids,
            field_schema=new_field_schema,
            production_status=new_production_status,
            extra_info=self.extra_info,
            _custom_meta=new_custom_meta,
        )

    def select_fields(self, field_names: list[str]) -> "BatchMeta":
        """
        Select specific fields from all samples in this batch.
        This will construct a new BatchMeta instance containing only the specified fields.

        Args:
            field_names (list[str]): List of field names to retain.

        Returns:
            BatchMeta: A new BatchMeta instance containing only the specified fields.
        """
        new_field_schema = {}
        for fname in field_names:
            if fname in self.field_schema:
                new_field_schema[fname] = copy.deepcopy(self.field_schema[fname])

        return BatchMeta(
            global_indexes=self.global_indexes,
            partition_ids=self.partition_ids,
            field_schema=new_field_schema,
            production_status=self.production_status.copy() if self.production_status is not None else None,
            extra_info=self.extra_info,
            _custom_meta=self._custom_meta,
        )

    def __len__(self) -> int:
        """Return the number of samples in this batch."""
        return self.size

    def __getitem__(self, item) -> "BatchMeta":
        if isinstance(item, int | np.integer):
            if item < 0:
                item += self.size
            if item < 0 or item >= self.size:
                raise IndexError("BatchMeta index out of range")
            return self.select_samples([item])
        elif isinstance(item, slice):
            start, stop, step = item.indices(self.size)
            indices = list(range(start, stop, step))
            return self.select_samples(indices)
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported.")

    def chunk(self, chunks: int) -> list["BatchMeta"]:
        """
        Split this batch into smaller chunks.

        Args:
            chunks: number of chunks

        Return:
            List of smaller BatchMeta chunks
        """
        chunk_list = []
        n = self.size

        if n < chunks:
            logger.warning(
                f"Chunk size {chunks} > number of samples in BatchMeta {n}, this will return some "
                f"empty BatchMeta chunks."
            )

        # Calculate the base size and remainder of each chunk
        base_size = n // chunks
        remainder = n % chunks

        start = 0
        for i in range(chunks):
            current_chunk_size = base_size + 1 if i < remainder else base_size
            end = start + current_chunk_size
            indices = list(range(start, end))
            chunk = self.select_samples(indices)
            chunk_list.append(chunk)
            start = end
        return chunk_list

    def union(self, other: "BatchMeta") -> "BatchMeta":
        """
        Return the union of this BatchMeta and another BatchMeta.
        Samples with global_indexes already present in this batch are ignored from the other batch.

        Args:
            other: The other BatchMeta to merge with.

        Returns:
            BatchMeta: A new merged BatchMeta.
        """
        if not other or other.size == 0:
            return self
        if self.size == 0:
            return other

        self_indexes = set(self.global_indexes)
        # Find indices in 'other' that are NOT in 'self'
        unique_indices_in_other = [i for i, idx in enumerate(other.global_indexes) if idx not in self_indexes]

        if not unique_indices_in_other:
            return self

        # improved performance: if all are unique, just concat
        if len(unique_indices_in_other) == other.size:
            return BatchMeta.concat([self, other])

        # otherwise, select unique samples and concat
        other_unique = other.select_samples(unique_indices_in_other)
        return BatchMeta.concat([self, other_unique])

    @classmethod
    def concat(cls, data: list["BatchMeta"], validate: bool = True) -> "BatchMeta":
        """
        Concatenate multiple BatchMeta chunks into one large batch.

        Args:
            data: List of BatchMeta chunks to concatenate
            validate: Whether to validate concatenation conditions

        Returns:
            Concatenated BatchMeta

        Raises:
            ValueError: If validation fails (e.g., field names do not match)
        """
        if not data:
            logger.warning("Try to concat empty BatchMeta chunks. Returning empty BatchMeta.")
            return BatchMeta.empty()

        # skip empty chunks
        data = [chunk for chunk in data if chunk and chunk.size > 0]

        if len(data) == 0:
            logger.warning("No valid BatchMeta chunks to concatenate. Returning empty BatchMeta.")
            return BatchMeta.empty()

        if validate:
            base_fields = data[0].field_names

            for chunk in data:
                if chunk.field_names != base_fields:
                    raise ValueError("Error: Field names do not match for concatenation.")

        # Combine lists
        all_global_indexes = list(itertools.chain.from_iterable(chunk.global_indexes for chunk in data))
        all_partition_ids = list(itertools.chain.from_iterable(chunk.partition_ids for chunk in data))

        # Combine production_status
        all_production_status = None
        status_arrays = [chunk.production_status for chunk in data if chunk.production_status is not None]
        if status_arrays:
            all_production_status = np.concatenate(status_arrays)

        # Combine field_schema (merge per_sample_shapes)
        all_field_schema: dict[str, dict[str, Any]] = {}
        first_chunk = data[0]
        for fname, meta in first_chunk.field_schema.items():
            all_field_schema[fname] = {
                "dtype": meta.get("dtype"),
                "shape": meta.get("shape"),
                "is_nested": meta.get("is_nested", False),
                "is_non_tensor": meta.get("is_non_tensor", False),
            }
            # Concatenate per_sample_shapes if any chunk has them
            if any(chunk.field_schema.get(fname, {}).get("per_sample_shapes") for chunk in data):
                all_shapes = []
                for chunk in data:
                    chunk_meta = chunk.field_schema.get(fname, {})
                    chunk_shapes = chunk_meta.get("per_sample_shapes")
                    if chunk_shapes:
                        all_shapes.extend(chunk_shapes)
                    else:
                        # Fill with None for chunks without per_sample_shapes
                        all_shapes.extend([None] * chunk.size)
                all_field_schema[fname]["per_sample_shapes"] = all_shapes

        # Combine custom meta
        all_custom_meta = {}
        for chunk in data:
            all_custom_meta.update(chunk.get_all_custom_meta())

        # Merge all extra_info dictionaries from the chunks
        merged_extra_info = dict()

        values_by_key = defaultdict(list)
        for chunk in data:
            for key, value in chunk.extra_info.items():
                values_by_key[key].append(value)
        for key, values in values_by_key.items():
            if all(isinstance(v, torch.Tensor) for v in values):
                try:
                    if all(v.dim() == 0 for v in values):
                        merged_extra_info[key] = torch.cat([v.unsqueeze(0) for v in values], dim=0)
                    else:
                        merged_extra_info[key] = torch.cat(values, dim=0)
                except RuntimeError as e:
                    logger.warning(
                        f"BatchMeta.concat try to use torch.cat(dim=0) to merge extra_info key '{key}'"
                        f" fails, with RuntimeError {e}. Falling back to use list."
                    )
                    merged_extra_info[key] = values
            elif all(isinstance(v, NonTensorStack | NonTensorData) for v in values):
                merged_extra_info[key] = torch.stack(values)
            elif all(isinstance(v, list) for v in values):
                merged_extra_info[key] = list(itertools.chain.from_iterable(values))
            else:
                merged_extra_info[key] = values[-1]

        return BatchMeta(
            global_indexes=all_global_indexes,
            partition_ids=all_partition_ids,
            field_schema=all_field_schema,
            production_status=all_production_status,
            extra_info=merged_extra_info,
            _custom_meta=all_custom_meta,
        )

    def reorder(self, indices: list[int]):
        """
        Reorder the samples in the BatchMeta according to the given indices.
        The operation is performed in-place.
        """
        if len(indices) != self.size:
            raise ValueError(f"Indices length {len(indices)} mismatch batch size {self.size}")

        if len(set(indices)) != self.size:
            raise ValueError("Indices contain duplicates")

        if any(i < 0 or i >= self.size for i in indices):
            raise ValueError(f"Reorder indices must be in range [0, {self.size})")

        self.global_indexes = [self.global_indexes[i] for i in indices]
        self.partition_ids = [self.partition_ids[i] for i in indices]

        if self.production_status is not None:
            self.production_status = self.production_status[indices]

        # Reorder per_sample_shapes in field_schema
        for fname, meta in self.field_schema.items():
            if meta.get("per_sample_shapes") is not None:
                meta["per_sample_shapes"] = [meta["per_sample_shapes"][i] for i in indices]

    @classmethod
    def empty(cls, extra_info: Optional[dict[str, Any]] = None) -> "BatchMeta":
        """
        Create an empty BatchMeta with no samples.

        Args:
            extra_info: Optional additional information to store with the batch

        Returns:
            Empty BatchMeta instance

        Example:
            >>> empty_batch = BatchMeta.empty()
        """
        if extra_info is None:
            extra_info = {}
        return cls(
            global_indexes=[],
            partition_ids=[],
            field_schema={},
            production_status=None,
            extra_info=extra_info,
        )

    def __str__(self):
        return (
            f"BatchMeta(size={self.size}, field_names={self.field_names}, is_ready={self.is_ready}, "
            f"global_indexes={self.global_indexes}, extra_info={self.extra_info})"
        )

    def to_dict(self) -> dict:
        """Convert BatchMeta to dict for serialization.

        Note: Actual serialization (including dtype encoding) is handled by serial_utils.
        """
        serialized_schema = {}
        for fname, meta in self.field_schema.items():
            serialized_schema[fname] = {
                "dtype": meta.get("dtype"),  # Will be handled by serial_utils
                "shape": list(meta["shape"]) if meta.get("shape") else None,
                "is_nested": meta.get("is_nested", False),
                "is_non_tensor": meta.get("is_non_tensor", False),
            }
            if meta.get("per_sample_shapes"):
                serialized_schema[fname]["per_sample_shapes"] = [list(s) for s in meta["per_sample_shapes"]]

        return {
            "global_indexes": self.global_indexes,
            "partition_ids": self.partition_ids,
            "field_schema": serialized_schema,
            "production_status": self.production_status.tolist() if self.production_status is not None else None,
            "extra_info": self.extra_info,
            "custom_meta": self._custom_meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchMeta":
        """Create BatchMeta from dictionary.

        Note: Actual deserialization (including dtype decoding) is handled by serial_utils.
        """
        # Reconstruct field_schema
        field_schema = {}
        for fname, meta in data.get("field_schema", {}).items():
            field_schema[fname] = {
                "dtype": meta.get("dtype"),  # Already decoded by serial_utils
                "shape": tuple(meta["shape"]) if meta.get("shape") else None,
                "is_nested": meta.get("is_nested", False),
                "is_non_tensor": meta.get("is_non_tensor", False),
            }
            if meta.get("per_sample_shapes"):
                field_schema[fname]["per_sample_shapes"] = [tuple(s) for s in meta["per_sample_shapes"]]

        # Reconstruct production_status
        production_status = None
        if data.get("production_status") is not None:
            production_status = np.array(data["production_status"], dtype=np.int8)

        return cls(
            global_indexes=data["global_indexes"],
            partition_ids=data["partition_ids"],
            field_schema=field_schema,
            production_status=production_status,
            extra_info=data.get("extra_info", {}),
            _custom_meta=data.get("custom_meta", {}),
        )
