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

"""Unit tests for ``transfer_queue.utils.compression``."""

import pytest
import torch

zstandard = pytest.importorskip("zstandard")

from transfer_queue.utils.compression import CompressedTensor, TensorCompressor  # noqa: E402


class TestTensorCompressorBasics:
    def test_disabled_default(self):
        compressor = TensorCompressor()
        assert compressor.algorithm == "none"
        assert compressor.enabled is False
        # should_compress_field always False when disabled
        t = torch.zeros(8, 1024)
        assert compressor.should_compress_field(t) is False

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unsupported compression algorithm"):
            TensorCompressor(algorithm="snappy")

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="level must be"):
            TensorCompressor(algorithm="zstd", level=0)

    def test_invalid_min_bytes_raises(self):
        with pytest.raises(ValueError, match="min_bytes"):
            TensorCompressor(algorithm="zstd", min_bytes=-1)

    def test_zstd_enabled_properties(self):
        compressor = TensorCompressor(algorithm="zstd", level=5, min_bytes=512)
        assert compressor.algorithm == "zstd"
        assert compressor.level == 5
        assert compressor.min_bytes == 512
        assert compressor.enabled is True


class TestShouldCompressField:
    def setup_method(self):
        self.compressor = TensorCompressor(algorithm="zstd", level=3, min_bytes=1024)

    def test_above_threshold(self):
        # float32 [4, 256] -> per-row 1024 bytes (== min_bytes)
        t = torch.zeros(4, 256, dtype=torch.float32)
        assert self.compressor.should_compress_field(t) is True

    def test_below_threshold(self):
        t = torch.zeros(4, 100, dtype=torch.float32)
        assert self.compressor.should_compress_field(t) is False

    def test_zero_dim_tensor(self):
        t = torch.tensor(1.0)
        assert self.compressor.should_compress_field(t) is False

    def test_empty_batch(self):
        t = torch.zeros(0, 1024, dtype=torch.float32)
        assert self.compressor.should_compress_field(t) is False

    def test_sparse_tensor(self):
        indices = torch.tensor([[0, 1], [1, 0]])
        values = torch.tensor([1.0, 2.0])
        t = torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()
        assert self.compressor.should_compress_field(t) is False

    def test_nested_tensor(self):
        t = torch.nested.as_nested_tensor(
            [torch.zeros(512, dtype=torch.float32), torch.zeros(512, dtype=torch.float32)],
            layout=torch.jagged,
        )
        assert self.compressor.should_compress_field(t) is False


class TestCompressDecompressRoundtrip:
    @pytest.mark.parametrize("level", [1, 3, 9])
    def test_roundtrip_random_bytes(self, level):
        compressor = TensorCompressor(algorithm="zstd", level=level)
        raw = torch.randn(1024, dtype=torch.float32).numpy().tobytes()
        compressed = compressor.compress_bytes(raw)
        assert compressor.decompress_bytes(compressed) == raw

    def test_accepts_memoryview_input(self):
        compressor = TensorCompressor(algorithm="zstd")
        raw = torch.arange(2048, dtype=torch.int64).contiguous().view(torch.uint8).numpy()
        # Pass a memoryview directly — must not require .tobytes()
        compressed = compressor.compress_bytes(memoryview(raw))
        roundtripped = compressor.decompress_bytes(memoryview(compressed))
        assert roundtripped == bytes(raw)

    def test_identity_when_disabled(self):
        compressor = TensorCompressor(algorithm="none")
        raw = b"hello world"
        assert compressor.compress_bytes(raw) == raw
        assert compressor.decompress_bytes(raw) == raw


class TestCompressedTensorBasics:
    def test_fields(self):
        ct = CompressedTensor(data=b"\x00\x01", dtype="float32", shape=(2,), algorithm="zstd")
        assert ct.data == b"\x00\x01"
        assert ct.dtype == "float32"
        assert ct.shape == (2,)
        assert ct.algorithm == "zstd"

    def test_no_extra_attributes(self):
        # __slots__ prevents accidental extra attributes — important so msgspec
        # never grows a __dict__ that could be auto-serialized as a map.
        ct = CompressedTensor(data=b"x", dtype="int8", shape=(1,), algorithm="zstd")
        with pytest.raises(AttributeError):
            ct.extra = 42  # type: ignore[attr-defined]

    def test_eq_and_hash(self):
        a = CompressedTensor(data=b"x", dtype="int8", shape=(1,), algorithm="zstd")
        b = CompressedTensor(data=b"x", dtype="int8", shape=(1,), algorithm="zstd")
        c = CompressedTensor(data=b"y", dtype="int8", shape=(1,), algorithm="zstd")
        assert a == b
        assert hash(a) == hash(b)
        assert a != c


def test_zstd_missing_raises_clear_error(monkeypatch):
    """If zstandard is not installed, manager-time backend resolution raises a clear ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "zstandard":
            raise ImportError("simulated missing zstandard")
        return real_import(name, *args, **kwargs)

    compressor = TensorCompressor(algorithm="zstd")
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="zstandard is required"):
        compressor.compress_bytes(b"hello")


class TestLz4Roundtrip:
    lz4 = pytest.importorskip("lz4")

    @pytest.mark.parametrize("level", [1, 6, 9])
    def test_roundtrip_random_bytes(self, level):
        compressor = TensorCompressor(algorithm="lz4", level=level)
        raw = torch.randn(1024, dtype=torch.float32).numpy().tobytes()
        compressed = compressor.compress_bytes(raw)
        assert compressor.decompress_bytes(compressed) == raw

    def test_accepts_memoryview_input(self):
        compressor = TensorCompressor(algorithm="lz4")
        raw = torch.arange(2048, dtype=torch.int64).contiguous().view(torch.uint8).numpy()
        compressed = compressor.compress_bytes(memoryview(raw))
        roundtripped = compressor.decompress_bytes(memoryview(compressed))
        assert roundtripped == bytes(raw)

    def test_lz4_enabled_properties(self):
        compressor = TensorCompressor(algorithm="lz4", level=5, min_bytes=512)
        assert compressor.algorithm == "lz4"
        assert compressor.level == 5
        assert compressor.min_bytes == 512
        assert compressor.enabled is True


def test_lz4_missing_raises_clear_error(monkeypatch):
    """If lz4 is not installed, manager-time backend resolution raises a clear ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lz4.frame":
            raise ImportError("simulated missing lz4")
        return real_import(name, *args, **kwargs)

    compressor = TensorCompressor(algorithm="lz4")
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="lz4 is required"):
        compressor.compress_bytes(b"hello")
