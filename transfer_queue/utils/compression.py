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

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

_SUPPORTED_ALGORITHMS = frozenset({"none", "zstd"})


@dataclass(frozen=True, eq=True, slots=True)
class CompressedTensor:
    data: bytes
    dtype: str
    shape: tuple
    algorithm: str
    level: int


class TensorCompressor:
    def __init__(
        self,
        algorithm: str = "none",
        level: int = 3,
        min_bytes: int | None = None,
    ) -> None:
        if algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported compression algorithm: {algorithm!r}. Supported: {sorted(_SUPPORTED_ALGORITHMS)}"
            )
        self._algorithm = algorithm
        self._level = level
        self._min_bytes = min_bytes if min_bytes is not None else 1024
        self._backend: tuple[Callable[[bytes, int], bytes], Callable[[bytes], bytes]] | None = None

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @property
    def level(self) -> int:
        return self._level

    @property
    def enabled(self) -> bool:
        return self._algorithm != "none"

    @property
    def min_bytes(self) -> int:
        return self._min_bytes

    def _get_backend(self) -> tuple[Callable[[bytes, int], bytes], Callable[[bytes], bytes]]:
        if self._backend is not None:
            return self._backend
        if self._algorithm == "zstd":
            import zstandard as zstd

            def _compress(raw: bytes, level: int) -> bytes:
                return zstd.compress(raw, level)

            def _decompress(compressed: bytes) -> bytes:
                return zstd.decompress(compressed)

            self._backend = (_compress, _decompress)
            return self._backend
        if self._algorithm == "none":
            self._backend = (_identity_compress, _identity_decompress)
            return self._backend
        raise ValueError(f"Unsupported compression algorithm: {self._algorithm!r}")

    def should_compress_field(self, tensor: torch.Tensor) -> bool:
        if not self.enabled:
            return False
        if tensor.is_nested:
            return False
        if tensor.is_sparse:
            return False
        if tensor.ndim < 1:
            return False
        if tensor.shape[0] == 0:
            return False
        per_row_bytes = tensor[0].nbytes
        if per_row_bytes < self._min_bytes:
            return False
        return True

    def compress_bytes(self, raw: bytes) -> bytes:
        _compress, _ = self._get_backend()
        return _compress(raw, self._level)

    def decompress_bytes(self, compressed: bytes) -> bytes:
        _, _decompress = self._get_backend()
        return _decompress(compressed)


def _identity_compress(raw: bytes, level: int) -> bytes:
    return raw


def _identity_decompress(compressed: bytes) -> bytes:
    return compressed
