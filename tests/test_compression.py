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

from dataclasses import FrozenInstanceError

import pytest
import torch

from transfer_queue.utils.compression import CompressedTensor, TensorCompressor


class TestCompressorDisabled:
    def test_compressor_disabled_enabled_false(self):
        c = TensorCompressor(algorithm="none")
        assert c.enabled is False

    def test_compressor_disabled_should_compress_always_false(self):
        c = TensorCompressor(algorithm="none")
        t = torch.randn(10, 10000)
        assert c.should_compress_field(t) is False

    def test_compressor_disabled_min_bytes_default(self):
        c = TensorCompressor(algorithm="none")
        assert c.min_bytes == 1024


class TestShouldCompressField:
    @pytest.mark.parametrize("row_bytes,expected", [(2048, True), (512, False), (1023, False)])
    def test_min_bytes_threshold(self, row_bytes, expected):
        c = TensorCompressor(algorithm="zstd", min_bytes=1024)
        elements = row_bytes // 4
        t = torch.randn(5, elements, dtype=torch.float32)
        assert c.should_compress_field(t) is expected

    def test_min_bytes_none_defaults_to_1024(self):
        c = TensorCompressor(algorithm="zstd", min_bytes=None)
        assert c.min_bytes == 1024

    def test_skip_nested(self):
        c = TensorCompressor(algorithm="zstd", min_bytes=1)
        t = torch.nested.as_nested_tensor([torch.randn(4, 1000), torch.randn(2, 1000)], layout=torch.jagged)
        assert c.should_compress_field(t) is False

    def test_skip_sparse(self):
        c = TensorCompressor(algorithm="zstd", min_bytes=1)
        t = torch.randn(5, 1000).to_sparse()
        assert c.should_compress_field(t) is False

    def test_skip_0d(self):
        c = TensorCompressor(algorithm="zstd", min_bytes=1)
        t = torch.tensor(3.14)
        assert c.should_compress_field(t) is False

    def test_skip_empty_batch(self):
        c = TensorCompressor(algorithm="zstd", min_bytes=1)
        t = torch.randn(0, 1000)
        assert c.should_compress_field(t) is False


class TestZstdRoundtrip:
    @staticmethod
    def _raw_bytes(tensor):
        return tensor.flatten().view(torch.uint8).numpy().tobytes()

    @staticmethod
    def _from_raw(raw, dtype, shape):
        torch_dtype = getattr(torch, dtype.replace("torch.", ""))
        arr = torch.frombuffer(raw, dtype=torch.uint8)
        return arr.view(torch_dtype).view(shape)

    def _roundtrip(self, tensor, algorithm, level):
        c = TensorCompressor(algorithm=algorithm, level=level)
        raw = self._raw_bytes(tensor)
        compressed = c.compress_bytes(raw)
        decompressed = c.decompress_bytes(compressed)
        result = self._from_raw(decompressed, str(tensor.dtype), tensor.shape)
        assert torch.equal(tensor, result)
        return compressed

    def test_float32(self):
        t = torch.randn(4, 128, dtype=torch.float32)
        self._roundtrip(t, "zstd", 3)

    def test_float16(self):
        t = torch.randn(4, 128, dtype=torch.float16)
        self._roundtrip(t, "zstd", 3)

    def test_int64(self):
        t = torch.randint(-1000, 1000, (4, 128), dtype=torch.int64)
        self._roundtrip(t, "zstd", 3)

    def test_int8(self):
        t = torch.randint(-128, 127, (4, 128), dtype=torch.int8)
        self._roundtrip(t, "zstd", 3)

    def test_bool(self):
        t = torch.randint(0, 2, (4, 128), dtype=torch.bool)
        self._roundtrip(t, "zstd", 3)

    def test_bfloat16(self):
        t = torch.randn(4, 128, dtype=torch.bfloat16)
        self._roundtrip(t, "zstd", 3)

    def test_float64(self):
        t = torch.randn(4, 64, dtype=torch.float64)
        self._roundtrip(t, "zstd", 3)

    def test_high_dim_row(self):
        t = torch.randn(2, 128, 768, dtype=torch.float32)
        self._roundtrip(t, "zstd", 3)

    def test_zstd_reduces_size(self):
        t = torch.zeros(4, 1024, dtype=torch.float32)
        raw = self._raw_bytes(t)
        c = TensorCompressor(algorithm="zstd", level=3)
        compressed = c.compress_bytes(raw)
        assert len(compressed) < len(raw)

    @pytest.mark.parametrize("level", list(range(1, 23)))
    def test_level_range(self, level):
        t = torch.randn(4, 128, dtype=torch.float32)
        self._roundtrip(t, "zstd", level)


class TestCompressedTensorFields:
    def test_fields_correct(self):
        ct = CompressedTensor(
            data=b"test",
            dtype="float32",
            shape=(128, 768),
            algorithm="zstd",
            level=3,
        )
        assert ct.data == b"test"
        assert ct.dtype == "float32"
        assert ct.shape == (128, 768)
        assert ct.algorithm == "zstd"
        assert ct.level == 3

    def test_frozen_immutable(self):
        ct = CompressedTensor(
            data=b"test",
            dtype="float32",
            shape=(10,),
            algorithm="zstd",
            level=3,
        )
        with pytest.raises(FrozenInstanceError):
            ct.dtype = "int64"  # type: ignore[misc]

    def test_equality(self):
        a = CompressedTensor(data=b"x", dtype="float32", shape=(2,), algorithm="zstd", level=3)
        b = CompressedTensor(data=b"x", dtype="float32", shape=(2,), algorithm="zstd", level=3)
        c = CompressedTensor(data=b"y", dtype="float32", shape=(2,), algorithm="zstd", level=3)
        assert a == b
        assert a != c

    def test_hashable(self):
        a = CompressedTensor(data=b"x", dtype="float32", shape=(2,), algorithm="zstd", level=3)
        b = CompressedTensor(data=b"x", dtype="float32", shape=(2,), algorithm="zstd", level=3)
        assert hash(a) == hash(b)


class TestTensorCompressorInit:
    def test_unsupported_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            TensorCompressor(algorithm="lz4")

    def test_custom_min_bytes(self):
        c = TensorCompressor(algorithm="zstd", min_bytes=5000)
        assert c.min_bytes == 5000

    def test_default_algorithm_none(self):
        c = TensorCompressor()
        assert c.algorithm == "none"

    def test_default_level(self):
        c = TensorCompressor(algorithm="zstd")
        assert c.level == 3

    def test_algorithm_property(self):
        c = TensorCompressor(algorithm="zstd")
        assert c.algorithm == "zstd"
