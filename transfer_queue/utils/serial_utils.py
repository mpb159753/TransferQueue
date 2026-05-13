# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
# Copyright 2025 The vLLM project
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

# This implementation is inspired by https://github.com/vllm-project/vllm/blob/main/vllm/v1/serial_utils.py


import pickle
import warnings
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, TypeAlias

import cloudpickle
import numpy as np
import torch
import zmq
from msgspec import msgpack
from tensordict import TensorDictBase

from transfer_queue.utils.compression import CompressedTensor, TensorCompressor
from transfer_queue.utils.logging_utils import get_logger

CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_TENSOR = 3  # For tensor with buffer reference
CUSTOM_TYPE_NESTED_TENSOR = 4  # For nested tensor (strided or jagged)
CUSTOM_TYPE_NUMPY = 5  # For numpy ndarray with buffer reference
CUSTOM_TYPE_COMPRESSED_TENSOR = 6  # For per-row compressed tensors (SimpleStorage compression)

# 0xC1 is permanently reserved (invalid) in msgpack spec — safe to use as pickle fallback sentinel.
_PICKLE_FALLBACK_SENTINEL = b"\xc1\xfe\xed"

bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame

logger = get_logger(__name__)

# Ignore warnings about non-writable buffers from torch.frombuffer. Upper codes will ensure
# the tensors are writable to users.
warnings.filterwarnings(action="ignore", message=r"The given buffer is not writable*", category=UserWarning)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _compressed_tensor_raw_nbytes(ct: CompressedTensor) -> int | None:
    try:
        torch_dtype = getattr(torch, ct.dtype)
        item_size = torch.empty((), dtype=torch_dtype).element_size()
        return int(reduce(mul, ct.shape, 1) * item_size)
    except (AttributeError, TypeError, RuntimeError):
        return None


# ContextVar for thread/coroutine-safe buffer storage during serialization/deserialization
# This enables the global _encoder/_decoder instances to be safely used across threads
_encoder_aux_buffers: ContextVar[list[bytestr] | None] = ContextVar("encoder_aux_buffers", default=None)
_decoder_aux_buffers: ContextVar[Sequence[bytestr] | None] = ContextVar("decoder_aux_buffers", default=None)
_encoder_stats: ContextVar["SerializationStats | None"] = ContextVar("encoder_stats", default=None)


@dataclass
class SerializationStats:
    """Optional serialization telemetry for replay wire metrics."""

    raw_tensor_bytes: int = 0
    compressed_tensor_bytes: int | None = 0
    serialization_fallback: bool = False

    def add_raw_tensor(self, nbytes: int) -> None:
        self.raw_tensor_bytes += int(nbytes)

    def add_compressed_tensor(self, nbytes: int) -> None:
        if self.compressed_tensor_bytes is not None:
            self.compressed_tensor_bytes += int(nbytes)

    def mark_fallback(self) -> None:
        self.serialization_fallback = True
        self.compressed_tensor_bytes = None


class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    This implementation uses ContextVar for thread-safe buffer storage,
    allowing the global encoder instance to be safely used across multiple
    threads and async coroutines.

    """

    def __init__(self, compressor: TensorCompressor | None = None):
        """Build an encoder. Pass a ``TensorCompressor`` to enable per-row compression of large tensor fields."""
        self.compressor = compressor
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)

    @property
    def aux_buffers(self) -> list[bytestr]:
        """Get the current context's aux_buffers."""
        buffers = _encoder_aux_buffers.get()
        assert buffers is not None, "aux_buffers accessed outside of encode() context"
        return buffers

    def encode(self, obj: Any, stats: SerializationStats | None = None) -> Sequence[bytestr]:
        """Encode a given object to a byte array."""

        bufs: list[bytestr] = [b""]
        token = _encoder_aux_buffers.set(bufs)
        stats_token = _encoder_stats.set(stats)
        try:
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
        finally:
            _encoder_stats.reset(stats_token)
            _encoder_aux_buffers.reset(token)

    @property
    def stats(self) -> SerializationStats | None:
        """Get optional stats for the current encode context."""
        return _encoder_stats.get()

    def enc_hook(self, obj: Any) -> Any:
        """Custom encoding hook for types msgspec doesn't natively support.

        For zero-copy tensor serialization, we need to handle:
        - torch.Tensor: Extract buffer, store metadata
        - TensorDict: Convert to dict structure for recursive processing
        - numpy.ndarray: Convert to tensor for unified handling

        """
        if isinstance(obj, CompressedTensor):
            return self._encode_compressed_tensor(obj)

        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        # Handle TensorDict explicitly for recursive zero-copy
        if isinstance(obj, TensorDictBase):
            return self._encode_tensordict(obj)

        # Numpy arrays: serialize natively unless the dtype contains Python objects.
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind != "O" and not obj.dtype.hasobject:
                try:
                    return self._encode_numpy(obj)
                except (TypeError, RuntimeError, ValueError):
                    # Fallback to pickle for platforms that don't support the view
                    pass
            # Only true object arrays (or structured dtypes with object fields) reach here
            self._mark_serialization_fallback()
            return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

        if callable(obj):
            # cloudpickle for arbitrary callables (functions, lambdas, functools.partial,
            # callable class instances, bound methods, etc.)
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        # Fallback to pickle for unknown types
        self._mark_serialization_fallback()
        return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _mark_serialization_fallback(self) -> None:
        stats = self.stats
        if stats is not None:
            stats.mark_fallback()

    def _record_raw_tensor(self, tensor: torch.Tensor) -> None:
        stats = self.stats
        if stats is not None:
            stats.add_raw_tensor(_tensor_nbytes(tensor))

    def _record_compressed_tensor(self, nbytes: int) -> None:
        stats = self.stats
        if stats is not None:
            stats.add_compressed_tensor(nbytes)

    def _encode_tensordict(self, obj: Any) -> dict:
        """Convert TensorDict to a dict structure for recursive msgpack processing.

        This allows msgpack to recursively call enc_hook for each tensor inside,
        enabling zero-copy serialization of nested tensors.
        """
        # Convert to dict, preserving structure
        # TensorDict.to_dict() returns nested dicts with tensors as leaves
        data_dict = dict(obj.items())

        # Return a marked dict that decoder will recognize
        return {
            "__tq_tensordict__": True,
            "batch_size": list(obj.batch_size),  # torch.Size -> list for msgpack
            "data": data_dict,
        }

    def _encode_tensor(self, obj: torch.Tensor) -> msgpack.Ext | list[msgpack.Ext]:
        """Encode tensor with zero-copy buffer extraction (handles GPU, non-contiguous, nested).

        Returns a single ``Ext`` for the regular zero-copy path or a list of
        ``Ext(6)`` rows when the configured compressor splits the tensor by row.
        """
        assert len(self.aux_buffers) > 0

        # Handle nested tensors (strided or jagged) via unbind
        if obj.is_nested:
            return self._encode_nested_tensor(obj)

        return self._encode_regular_tensor(obj)

    def _encode_nested_tensor(self, obj: torch.Tensor) -> msgpack.Ext:
        """Encode nested tensor by unbinding into sub-tensors for zero-copy."""
        # Unbind nested tensor into list of regular tensors
        sub_tensors = obj.unbind()

        # Encode each sub-tensor with zero-copy
        encoded_sub_tensors = []
        for t in sub_tensors:
            # Get tensor metadata (dtype, shape, buffer_idx)
            meta = self._encode_regular_tensor_meta(t)
            encoded_sub_tensors.append(meta)

        # Pack: layout type + list of tensor metas
        layout = "jagged" if obj.layout == torch.jagged else "strided"
        nested_meta = {
            "layout": layout,
            "tensors": encoded_sub_tensors,
        }
        return msgpack.Ext(CUSTOM_TYPE_NESTED_TENSOR, pickle.dumps(nested_meta, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_regular_tensor_meta(self, obj: torch.Tensor) -> tuple:
        """Encode a regular tensor and return its metadata tuple."""
        # Handle non-contiguous tensors

        if not obj.is_contiguous():
            obj = obj.contiguous()

        # Handle GPU tensors
        if obj.device.type != "cpu":
            obj = obj.cpu()

        self._record_raw_tensor(obj)

        # Zero-copy buffer extraction via uint8 view
        arr = obj.flatten().view(torch.uint8).numpy()
        buf = memoryview(arr)
        idx = len(self.aux_buffers)
        self.aux_buffers.append(buf)

        dtype = str(obj.dtype).removeprefix("torch.")
        return (dtype, tuple(obj.shape), idx)

    def _encode_regular_tensor(self, obj: torch.Tensor) -> msgpack.Ext | list[msgpack.Ext]:
        """Encode a regular (non-nested) tensor with zero-copy."""
        if obj.is_sparse:
            # Sparse tensors fallback to pickle
            self._mark_serialization_fallback()
            return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

        if self.compressor is not None and self.compressor.should_compress_field(obj):
            return self._encode_compressed_rows(obj)

        # Handle non-contiguous tensors

        if not obj.is_contiguous():
            obj = obj.contiguous()

        # Handle GPU tensors
        if obj.device.type != "cpu":
            obj = obj.cpu()

        self._record_raw_tensor(obj)

        # Note: view(uint8) is a byte-level view, NOT a value conversion.
        arr = obj.flatten().view(torch.uint8).numpy()
        buf = memoryview(arr)
        idx = len(self.aux_buffers)
        self.aux_buffers.append(buf)

        # Pack tensor metadata as Ext type
        dtype = str(obj.dtype).removeprefix("torch.")
        meta = (dtype, tuple(obj.shape), idx)
        return msgpack.Ext(CUSTOM_TYPE_TENSOR, pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_compressed_rows(self, tensor: torch.Tensor) -> list[msgpack.Ext]:
        """Compress a batched tensor row-wise into N ``Ext(6)`` slots.

        Done once on the manager during PUT so storage units never see the raw
        bytes. The dim-0 row layout matches how the storage unit indexes
        samples — each ``Ext(6)`` is one sample.
        """
        assert self.compressor is not None
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        dtype = str(tensor.dtype).removeprefix("torch.")
        row_shape = tuple(tensor.shape[1:])
        results: list[msgpack.Ext] = []
        for row in tensor:
            row_view = memoryview(row.flatten().view(torch.uint8).numpy())
            compressed = self.compressor.compress_bytes(row_view)
            self._record_raw_tensor(row)
            self._record_compressed_tensor(len(compressed))
            idx = len(self.aux_buffers)
            self.aux_buffers.append(memoryview(compressed))
            meta = (dtype, row_shape, idx, self.compressor.algorithm)
            results.append(
                msgpack.Ext(CUSTOM_TYPE_COMPRESSED_TENSOR, pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
            )
        return results

    def _encode_compressed_tensor(self, ct: CompressedTensor) -> msgpack.Ext:
        """Forward a ``CompressedTensor`` as ``Ext(6)`` — SU-side GET path, no zstd."""
        raw_nbytes = _compressed_tensor_raw_nbytes(ct)
        if raw_nbytes is not None:
            stats = self.stats
            if stats is not None:
                stats.add_raw_tensor(raw_nbytes)
        self._record_compressed_tensor(len(ct.data))
        idx = len(self.aux_buffers)
        self.aux_buffers.append(memoryview(ct.data))
        meta = (ct.dtype, ct.shape, idx, ct.algorithm)
        return msgpack.Ext(CUSTOM_TYPE_COMPRESSED_TENSOR, pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_numpy(self, obj: np.ndarray) -> msgpack.Ext:
        """Encode numpy array with zero-copy buffer extraction."""
        # Ensure C-contiguous layout; no-op when already contiguous
        if not obj.flags["C_CONTIGUOUS"]:
            obj = np.ascontiguousarray(obj)

        # Byte-level view as uint8 then ravel → 1-D C-contiguous raw-bytes array
        buf = memoryview(obj.view(np.uint8).ravel())
        idx = len(self.aux_buffers)
        self.aux_buffers.append(buf)

        meta = (str(obj.dtype), tuple(obj.shape), idx)
        return msgpack.Ext(CUSTOM_TYPE_NUMPY, pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))


class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    This implementation uses ContextVar for thread-safe buffer storage,
    allowing the global decoder instance to be safely used across multiple
    threads and async coroutines.
    """

    def __init__(self, compressor: TensorCompressor | None = None):
        """Pass a ``TensorCompressor`` to decompress ``Ext(6)`` rows; without one they stay as ``CompressedTensor``."""
        self.compressor = compressor
        self.decoder = msgpack.Decoder(ext_hook=self.ext_hook)

    @property
    def aux_buffers(self) -> Sequence[bytestr]:
        """Get the current context's aux_buffers."""
        buffers = _decoder_aux_buffers.get()
        assert buffers is not None, "aux_buffers accessed outside of decode() context"
        return buffers

    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        """Decode a list of bytes."""
        if isinstance(bufs, bytestr):
            result = self.decoder.decode(bufs)
        else:
            token = _decoder_aux_buffers.set(bufs)
            try:
                result = self.decoder.decode(bufs[0])  # type: ignore[index,arg-type]
            finally:
                _decoder_aux_buffers.reset(token)

        # Post-process to reconstruct TensorDict from marked dicts
        return self._reconstruct_special_types(result)

    def _reconstruct_special_types(self, obj: Any) -> Any:
        """Recursively reconstruct special types (TensorDict) from their dict representation."""
        if isinstance(obj, dict):
            # Check if this is a TensorDict marker
            if obj.get("__tq_tensordict__"):
                return self._reconstruct_tensordict(obj)
            # Recursively process dict values
            return {k: self._reconstruct_special_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._reconstruct_special_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._reconstruct_special_types(item) for item in obj)
        return obj

    def _reconstruct_tensordict(self, obj: dict) -> Any:
        """Reconstruct TensorDict from marked dict structure."""
        try:
            from tensordict import TensorDict

            batch_size = obj["batch_size"]
            data = obj["data"]
            # Recursively process nested data
            processed_data = self._reconstruct_special_types(data)
            return TensorDict(processed_data, batch_size=batch_size)
        except ImportError:
            # If tensordict not available, return as dict
            return obj

    def _decode_tensor(self, meta: tuple) -> torch.Tensor:
        """Decode tensor from (dtype, shape, buffer_idx) tuple."""
        dtype, shape, idx = meta
        buffer = self.aux_buffers[idx]
        torch_dtype = getattr(torch, dtype)

        if not buffer:  # Handle empty tensors
            return torch.empty(shape, dtype=torch_dtype)

        # Create uint8 tensor from buffer, then view as original dtype and reshape
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    def _decode_nested_tensor(self, nested_meta: dict) -> torch.Tensor:
        """Decode nested tensor from serialized sub-tensors."""
        layout = nested_meta["layout"]
        tensor_metas = nested_meta["tensors"]

        # Decode each sub-tensor
        sub_tensors = [self._decode_tensor(meta) for meta in tensor_metas]

        # Reconstruct nested tensor with appropriate layout
        if layout == "jagged":
            return torch.nested.as_nested_tensor(sub_tensors, layout=torch.jagged)
        else:  # strided
            return torch.nested.as_nested_tensor(sub_tensors, layout=torch.strided)

    def _decode_compressed_row(self, meta: tuple) -> CompressedTensor | torch.Tensor:
        """Materialize a single ``Ext(6)`` row.

        Manager side (with compressor): decompress and rebuild a row tensor.
        Storage-unit side (no compressor): wrap the bytes in a
        ``CompressedTensor`` so the SU can index it by ``global_index``.
        """
        dtype_str, shape, idx, algorithm = meta
        buffer = self.aux_buffers[idx]

        if self.compressor is None:
            return CompressedTensor(
                data=bytes(buffer),
                dtype=dtype_str,
                shape=shape,
                algorithm=algorithm,
            )

        # Manager side: decompress to tensor
        raw = self.compressor.decompress_bytes(buffer)
        torch_dtype = getattr(torch, dtype_str)
        arr = torch.frombuffer(raw, dtype=torch.uint8)
        return arr.view(torch_dtype).view(shape)

    def _decode_numpy(self, meta: tuple) -> np.ndarray:
        """Decode numpy array from (dtype_str, shape, buffer_idx) tuple."""
        dtype_str, shape, idx = meta
        buffer = self.aux_buffers[idx]
        np_dtype = np.dtype(dtype_str)

        if not buffer:  # empty array
            return np.empty(shape, dtype=np_dtype)

        # Reconstruct from raw bytes: uint8 view → reinterpret as original dtype
        arr = np.frombuffer(buffer, dtype=np.uint8)
        return arr.view(np_dtype).reshape(shape)

    def ext_hook(self, code: int, data: memoryview) -> Any:
        """Custom decoding hook for types msgspec doesn't natively support.

        For zero-copy tensor serialization, we need to handle:
        - torch.Tensor: Extract buffer, store metadata
        - TensorDict: Convert to dict structure for recursive processing
        - numpy.ndarray: Convert to tensor for unified handling
        """
        if code == CUSTOM_TYPE_PICKLE:
            return pickle.loads(data)
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)
        if code == CUSTOM_TYPE_TENSOR:
            meta = pickle.loads(data)
            return self._decode_tensor(meta)
        if code == CUSTOM_TYPE_NESTED_TENSOR:
            nested_meta = pickle.loads(data)
            return self._decode_nested_tensor(nested_meta)
        if code == CUSTOM_TYPE_NUMPY:
            meta = pickle.loads(data)
            return self._decode_numpy(meta)
        if code == CUSTOM_TYPE_COMPRESSED_TENSOR:
            meta = pickle.loads(data)
            return self._decode_compressed_row(meta)

        raise NotImplementedError(f"Extension type code {code} is not supported")


# Default uncompressed encoder/decoder; never rebound at runtime.
_encoder = MsgpackEncoder()
_decoder = MsgpackDecoder()


def encode(
    obj: Any,
    encoder: MsgpackEncoder | None = None,
    stats: SerializationStats | None = None,
) -> list[bytestr]:
    """Encode an object via msgpack zero-copy; falls back to pickle on failure.

    The pickle path is a normal degradation path (e.g. body contains torch.dtype
    objects). Use this as the single entry point for all ZMQ message serialization.
    Pass ``encoder`` to use a caller-owned (e.g. compression-aware) encoder.
    """
    enc = encoder if encoder is not None else _encoder
    try:
        return list(enc.encode(obj, stats=stats))
    except (TypeError, ValueError) as e:
        logger.debug(
            "encode: msgpack failed (%s), falling back to pickle.",
            type(e).__name__,
        )
        if stats is not None:
            stats.mark_fallback()
        return [_PICKLE_FALLBACK_SENTINEL, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)]


def decode(frames: list, decoder: MsgpackDecoder | None = None) -> Any:
    """Decode frames produced by encode.

    Transparently handles both the msgpack zero-copy path and the pickle
    fallback path based on the leading sentinel frame. Pass ``decoder`` to use
    a caller-owned (e.g. compression-aware) decoder.
    """
    if len(frames) >= 2 and frames[0] == _PICKLE_FALLBACK_SENTINEL:
        return pickle.loads(frames[1])
    dec = decoder if decoder is not None else _decoder
    return dec.decode(frames)
