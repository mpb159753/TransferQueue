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

"""Pluggable per-row tensor compression for the SimpleStorage backend.

Compression is performed once on the manager during PUT, and reversed on the
manager during GET. The storage unit only stores opaque ``CompressedTensor``
rows and never invokes the compression backend itself.
"""

from __future__ import annotations

from typing import Callable, TypeAlias

import torch

BytesLike: TypeAlias = bytes | bytearray | memoryview

_SUPPORTED_ALGORITHMS = frozenset({"none", "zstd"})


class CompressedTensor:
    """A single batched-tensor row in compressed form.

    Storage units treat instances as opaque values keyed by ``global_index``.
    Only the manager-side decoder reconstructs a ``torch.Tensor`` from one.

    Implemented as a plain class (not a ``dataclass``) so msgspec's msgpack
    encoder does not auto-serialize it as a map — instead it falls through to
    our ``enc_hook`` which packs an ``Ext(6)`` frame referencing the bytes
    zero-copy.
    """

    __slots__ = ("data", "dtype", "shape", "algorithm")

    def __init__(self, data: bytes, dtype: str, shape: tuple, algorithm: str) -> None:
        self.data = data
        self.dtype = dtype
        self.shape = shape
        self.algorithm = algorithm

    def __repr__(self) -> str:
        """Omits ``data`` to avoid logging large blobs."""
        return (
            f"CompressedTensor(dtype={self.dtype!r}, shape={self.shape!r}, "
            f"algorithm={self.algorithm!r}, len={len(self.data)})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompressedTensor):
            return NotImplemented
        return (
            self.data == other.data
            and self.dtype == other.dtype
            and self.shape == other.shape
            and self.algorithm == other.algorithm
        )

    def __hash__(self) -> int:
        """Hash matching ``__eq__`` so instances can live in sets/dicts."""
        return hash((self.data, self.dtype, self.shape, self.algorithm))


_CompressFn = Callable[[BytesLike, int], bytes]
_DecompressFn = Callable[[BytesLike], bytes]


class TensorCompressor:
    """Algorithm-agnostic compressor used by the manager-side encoder/decoder."""

    def __init__(
        self,
        algorithm: str = "none",
        level: int = 3,
        min_bytes: int = 1024,
    ) -> None:
        """Configure a compressor.

        Args:
            algorithm: ``"none"`` disables compression; ``"zstd"`` uses zstandard.
            level: Compression level (zstd: 1-22).
            min_bytes: Skip rows whose per-row payload is smaller than this.
        """
        if algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported compression algorithm: {algorithm!r}. Supported: {sorted(_SUPPORTED_ALGORITHMS)}"
            )
        if level < 1:
            raise ValueError(f"Compression level must be >= 1, got {level}")
        if min_bytes < 0:
            raise ValueError(f"min_bytes must be >= 0, got {min_bytes}")
        self._algorithm = algorithm
        self._level = level
        self._min_bytes = min_bytes
        self._backend: tuple[_CompressFn, _DecompressFn] | None = None

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @property
    def level(self) -> int:
        return self._level

    @property
    def enabled(self) -> bool:
        """Whether compression is active."""
        return self._algorithm != "none"

    @property
    def min_bytes(self) -> int:
        return self._min_bytes

    def _get_backend(self) -> tuple[_CompressFn, _DecompressFn]:
        if self._backend is not None:
            return self._backend
        if self._algorithm == "zstd":
            try:
                import zstandard as zstd
            except ImportError as e:
                raise ImportError(
                    "zstandard is required for compression='zstd'. "
                    "Install with: pip install 'transfer_queue[compression]'"
                ) from e

            def _compress(raw: BytesLike, level: int) -> bytes:
                return zstd.compress(raw, level)

            def _decompress(compressed: BytesLike) -> bytes:
                return zstd.decompress(compressed)

            self._backend = (_compress, _decompress)
            return self._backend
        # algorithm == "none"
        self._backend = (_identity_compress, _identity_decompress)
        return self._backend

    def should_compress_field(self, tensor: torch.Tensor) -> bool:
        """Whether a batched tensor field should go through the compressed encode path.

        Returns False for any tensor that cannot be cleanly split per row
        (nested / sparse / 0-dim / empty batch) or whose per-row payload is
        below ``min_bytes``.
        """
        if not self.enabled:
            return False
        if tensor.is_nested or tensor.is_sparse:
            return False
        if tensor.ndim < 1 or tensor.shape[0] == 0:
            return False
        per_row_bytes = tensor[0].nbytes
        return per_row_bytes >= self._min_bytes

    def compress_bytes(self, raw: BytesLike) -> bytes:
        compress_fn, _ = self._get_backend()
        return compress_fn(raw, self._level)

    def decompress_bytes(self, compressed: BytesLike) -> bytes:
        _, decompress_fn = self._get_backend()
        return decompress_fn(compressed)


def _identity_compress(raw: BytesLike, level: int) -> bytes:
    return raw


def _identity_decompress(compressed: BytesLike) -> bytes:
    return compressed
