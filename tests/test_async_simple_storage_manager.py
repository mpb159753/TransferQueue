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

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import pytest_asyncio
import torch
import zmq
from tensordict import NonTensorStack, TensorDict

from transfer_queue.metadata import BatchMeta
from transfer_queue.storage import AsyncSimpleStorageManager
from transfer_queue.utils.compression import TensorCompressor
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo


def _enable_replay(monkeypatch, tmp_path: Path, **env: str) -> None:
    monkeypatch.setenv("TQ_REPLAY_DIR", str(tmp_path))
    monkeypatch.setenv("TQ_REPLAY_BUF_SIZE", "1")
    for key in ("TQ_REPLAY_DUMP_DATA", "TQ_REPLAY_DUMP_MAX_BYTES", "TQ_REPLAY_RECORD_WIRE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _read_replay_events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _storage_infos(count: int = 1) -> dict[str, ZMQServerInfo]:
    return {
        f"storage_{idx}": ZMQServerInfo(
            role=Role.STORAGE,
            id=f"storage_{idx}",
            ip="127.0.0.1",
            ports={"put_get_socket": 12345 + idx},
        )
        for idx in range(count)
    }


def _controller_info() -> ZMQServerInfo:
    return ZMQServerInfo(
        role=Role.CONTROLLER,
        id="controller_0",
        ip="127.0.0.1",
        ports={"handshake_socket": 22345, "data_status_update_socket": 22346},
    )


def _make_replay_manager(
    monkeypatch,
    tmp_path: Path,
    *,
    storage_count: int = 1,
    replay_env: dict[str, str] | None = None,
    config_extra: dict | None = None,
    mock_io: bool = True,
) -> AsyncSimpleStorageManager:
    _enable_replay(monkeypatch, tmp_path, **(replay_env or {}))
    for key in ("TQ_COMPRESSION_ALGORITHM", "TQ_COMPRESSION_LEVEL", "TQ_COMPRESSION_MIN_BYTES"):
        monkeypatch.delenv(key, raising=False)
    storage_unit_infos = _storage_infos(storage_count)
    config = {"zmq_info": storage_unit_infos}
    if config_extra:
        config.update(config_extra)

    with patch("transfer_queue.storage.managers.base.StorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager(_controller_info(), config)

    if mock_io:
        manager._put_to_single_storage_unit = AsyncMock()
        manager.notify_data_update = AsyncMock()
    return manager


class _FakeAsyncSocket:
    def __init__(self, response_frames: list):
        self.response_frames = response_frames
        self.sent_frames: list | None = None
        self.send_copy: bool | None = None
        self.recv_copy: bool | None = None

    async def send_multipart(self, frames, copy=False):
        self.sent_frames = list(frames)
        self.send_copy = copy

    async def recv_multipart(self, copy=False):
        self.recv_copy = copy
        return self.response_frames


def _frames_nbytes(frames: list) -> int:
    total = 0
    for frame in frames:
        if isinstance(frame, zmq.Frame):
            total += len(frame.bytes)
        elif isinstance(frame, memoryview):
            total += frame.nbytes
        else:
            total += len(frame)
    return total


@pytest_asyncio.fixture
async def mock_async_storage_manager():
    """Create a mock AsyncSimpleStorageManager for testing."""

    # Mock storage unit infos
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=Role.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 12345},
        ),
        "storage_1": ZMQServerInfo(
            role=Role.STORAGE,
            id="storage_1",
            ip="127.0.0.1",
            ports={"put_get_socket": 12346},
        ),
    }

    # Mock controller info
    controller_info = ZMQServerInfo(
        role=Role.CONTROLLER,
        id="controller_0",
        ip="127.0.0.1",
        ports={"handshake_socket": 12347, "data_status_update_socket": 12348},
    )

    config = {
        "zmq_info": storage_unit_infos,
    }

    # Mock the handshake process entirely to avoid ZMQ complexity
    with patch("transfer_queue.storage.managers.base.StorageManager._connect_to_controller") as mock_connect:
        # Mock the manager without actually connecting
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_storage_manager"
        manager.config = config
        manager.controller_info = controller_info
        manager.storage_unit_infos = storage_unit_infos
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None

        # Mock essential methods
        manager._connect_to_controller = mock_connect

        yield manager


@pytest.mark.asyncio
async def test_async_storage_manager_initialization(mock_async_storage_manager):
    """Test AsyncSimpleStorageManager initialization."""
    manager = mock_async_storage_manager

    # Test basic properties
    assert len(manager.storage_unit_infos) == 2
    assert "storage_0" in manager.storage_unit_infos
    assert "storage_1" in manager.storage_unit_infos


@pytest.mark.asyncio
async def test_async_storage_manager_mock_operations(mock_async_storage_manager):
    """Test AsyncSimpleStorageManager operations with mocked ZMQ."""
    manager = mock_async_storage_manager

    # Create test metadata using columnar API
    batch_meta = BatchMeta(
        global_indexes=[0, 1],
        partition_ids=["0", "0"],
        field_schema={
            "test_field": {
                "dtype": torch.float32,
                "shape": (2,),
                "is_nested": False,
                "is_non_tensor": False,
            }
        },
        production_status=np.ones(2, dtype=np.int8),
    )

    # Create test data
    test_data = TensorDict(
        {
            "test_field": torch.stack([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]),
        },
        batch_size=2,
    )

    manager._put_to_single_storage_unit = AsyncMock()
    manager._get_from_single_storage_unit = AsyncMock(
        return_value=(
            ["test_field"],
            {"test_field": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]},
        )
    )
    manager._clear_single_storage_unit = AsyncMock()
    manager.notify_data_update = AsyncMock()

    # Test put_data (should not raise exceptions)
    await manager.put_data(test_data, batch_meta)
    manager.notify_data_update.assert_awaited_once()

    # Test get_data
    retrieved_data = await manager.get_data(batch_meta)
    assert "test_field" in retrieved_data

    # Test clear_data
    await manager.clear_data(batch_meta)


@pytest.mark.asyncio
async def test_async_storage_manager_error_handling():
    """Test AsyncSimpleStorageManager error handling."""

    # Mock storage unit infos
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=Role.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 12345},
        ),
    }

    # Mock controller info
    controller_info = ZMQServerInfo(
        role=Role.CONTROLLER,
        id="controller_0",
        ip="127.0.0.1",
        ports={"handshake_socket": 12346, "data_status_update_socket": 12347},
    )

    config = {
        "zmq_info": storage_unit_infos,
    }

    # Mock ZMQ operations
    with (
        patch("transfer_queue.storage.managers.base.create_zmq_socket") as mock_create_socket,
        patch("zmq.Poller") as mock_poller,
    ):
        # Create mock socket with proper sync methods
        mock_socket = Mock()
        mock_socket.connect = Mock()  # sync method
        mock_socket.send = Mock()  # sync method
        mock_create_socket.return_value = mock_socket

        # Mock poller with sync methods
        mock_poller_instance = Mock()
        mock_poller_instance.register = Mock()  # sync method
        # Return mock socket in poll to simulate handshake response
        mock_poller_instance.poll = Mock(return_value=[(mock_socket, zmq.POLLIN)])  # sync method
        mock_poller.return_value = mock_poller_instance

        # Mock handshake response
        handshake_response = ZMQMessage.create(
            request_type=ZMQRequestType.HANDSHAKE_ACK,  # type: ignore[arg-type]
            sender_id="controller_0",
            body={"message": "Handshake successful"},
        )
        mock_socket.recv_multipart = Mock(return_value=handshake_response.serialize())

        # Create manager
        manager = AsyncSimpleStorageManager(controller_info, config)

        # Mock operations that raise exceptions
        manager._put_to_single_storage_unit = AsyncMock(side_effect=RuntimeError("Mock PUT error"))
        manager._get_from_single_storage_unit = AsyncMock(side_effect=RuntimeError("Mock GET error"))
        manager._clear_single_storage_unit = AsyncMock(side_effect=RuntimeError("Mock CLEAR error"))
        manager.notify_data_update = AsyncMock()

        # Create test metadata using columnar API
        batch_meta = BatchMeta(
            global_indexes=[0],
            partition_ids=["0"],
            field_schema={
                "test_field": {
                    "dtype": torch.float32,
                    "shape": (2,),
                    "is_nested": False,
                    "is_non_tensor": False,
                }
            },
            production_status=np.ones(1, dtype=np.int8),
        )

        # Create test data
        test_data = TensorDict(
            {
                "test_field": torch.tensor([[1.0, 2.0]]),
            },
            batch_size=1,
        )

        # Test that exceptions are properly raised
        with pytest.raises(RuntimeError, match="Mock PUT error"):
            await manager.put_data(test_data, batch_meta)

        with pytest.raises(RuntimeError, match="Mock GET error"):
            await manager.get_data(batch_meta)

        # Note: clear_data uses return_exceptions=True, so it doesn't raise exceptions directly
        # Instead, we can verify that the clear operation was attempted
        await manager.clear_data(batch_meta)  # Should not raise due to return_exceptions=True


@pytest.mark.asyncio
async def test_get_data_routes_from_hash():
    """get_data should route using global_idx % num_su (hash routing)."""
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=Role.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 19010},
        ),
        "storage_1": ZMQServerInfo(
            role=Role.STORAGE,
            id="storage_1",
            ip="127.0.0.1",
            ports={"put_get_socket": 19011},
        ),
    }
    with patch("transfer_queue.storage.managers.base.StorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_get"
        manager.storage_unit_infos = storage_unit_infos
        manager.controller_info = None
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None

    # global_index 0,2 → storage_0 (even % 2 = 0); 1,3 → storage_1 (odd % 2 = 1)
    batch_meta = BatchMeta(
        global_indexes=[0, 1, 2, 3],
        partition_ids=["p0"] * 4,
        field_schema={"f": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(4, dtype=np.int8),
    )

    # Mock _get_from_single_storage_unit to record which su_id and global_index were requested
    called_with: dict[str, list] = {}

    async def fake_get(global_indexes, fields, target_storage_unit=None, **kwargs):
        su = target_storage_unit
        called_with[su] = list(global_indexes)
        tensors = [torch.zeros(2) for _ in global_indexes]
        return fields, {"f": tensors}

    manager._get_from_single_storage_unit = fake_get

    await manager.get_data(batch_meta)

    assert "storage_0" in called_with, "storage_0 was not called by get"
    assert "storage_1" in called_with, "storage_1 was not called by get"
    assert set(called_with["storage_0"]) == {0, 2}
    assert set(called_with["storage_1"]) == {1, 3}


@pytest.mark.asyncio
async def test_clear_data_routes_from_hash():
    """clear_data should route using global_idx % num_su (hash routing)."""
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=Role.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 19020},
        ),
        "storage_1": ZMQServerInfo(
            role=Role.STORAGE,
            id="storage_1",
            ip="127.0.0.1",
            ports={"put_get_socket": 19021},
        ),
    }
    with patch("transfer_queue.storage.managers.base.StorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_clear"
        manager.storage_unit_infos = storage_unit_infos
        manager.controller_info = None
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None

    # global_index 0,2 → storage_0 (even); 1,3 → storage_1 (odd)
    batch_meta = BatchMeta(
        global_indexes=[0, 1, 2, 3],
        partition_ids=["p0"] * 4,
        field_schema={"f": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(4, dtype=np.int8),
    )

    called_with: dict[str, list] = {}

    async def fake_clear(global_indexes, target_storage_unit=None, **kwargs):
        called_with[target_storage_unit] = list(global_indexes)

    manager._clear_single_storage_unit = fake_clear

    await manager.clear_data(batch_meta)

    assert set(called_with.get("storage_0", [])) == {0, 2}
    assert set(called_with.get("storage_1", [])) == {1, 3}


@pytest.mark.asyncio
async def test_hash_routing_stable_across_batch_sizes():
    """Hash routing must produce the same SU assignment regardless of batch size.

    Put 10 samples in one batch vs two batches of 5 — each global_idx must route
    to the same SU in both cases.
    """
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=Role.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 19030},
        ),
        "storage_1": ZMQServerInfo(
            role=Role.STORAGE,
            id="storage_1",
            ip="127.0.0.1",
            ports={"put_get_socket": 19031},
        ),
    }
    with patch("transfer_queue.storage.managers.base.StorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_hash_batch"
        manager.storage_unit_infos = storage_unit_infos
        manager.controller_info = None
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None

    all_indexes = list(range(10))
    full_routing = manager._group_by_hash(all_indexes)

    # Build per-index mapping from the full-batch result
    idx_to_su_full: dict[int, str] = {}
    for su_id, group in full_routing.items():
        for gi in group.global_indexes:
            idx_to_su_full[gi] = su_id

    # Route as two batches of 5
    batch_a_routing = manager._group_by_hash(all_indexes[:5])
    batch_b_routing = manager._group_by_hash(all_indexes[5:])

    idx_to_su_split: dict[int, str] = {}
    for su_id, group in batch_a_routing.items():
        for gi in group.global_indexes:
            idx_to_su_split[gi] = su_id
    for su_id, group in batch_b_routing.items():
        for gi in group.global_indexes:
            idx_to_su_split[gi] = su_id

    assert idx_to_su_full == idx_to_su_split, (
        f"Routing differs between full batch and split batches:\n  full:  {idx_to_su_full}\n  split: {idx_to_su_split}"
    )

    # Verify RoutingGroup carries correct batch_positions alongside global_indexes
    for su_id, group in full_routing.items():
        assert len(group.global_indexes) == len(group.batch_positions)
        for gi, pos in zip(group.global_indexes, group.batch_positions, strict=False):
            assert all_indexes[pos] == gi


@pytest.mark.asyncio
async def test_hash_routing_stable_reversed_order():
    """Hash routing must produce the same SU assignment regardless of key order.

    Forward order [0..9] and reversed order [9..0] must yield identical routing.
    """
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=Role.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 19040},
        ),
        "storage_1": ZMQServerInfo(
            role=Role.STORAGE,
            id="storage_1",
            ip="127.0.0.1",
            ports={"put_get_socket": 19041},
        ),
    }
    with patch("transfer_queue.storage.managers.base.StorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_hash_order"
        manager.storage_unit_infos = storage_unit_infos
        manager.controller_info = None
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None

    forward = list(range(10))
    reversed_indexes = list(reversed(forward))

    routing_fwd = manager._group_by_hash(forward)
    routing_rev = manager._group_by_hash(reversed_indexes)

    # Build per-index mapping
    def _to_idx_map(routing):
        m = {}
        for su_id, group in routing.items():
            for gi in group.global_indexes:
                m[gi] = su_id
        return m

    assert _to_idx_map(routing_fwd) == _to_idx_map(routing_rev), "Hash routing should be order-independent"


class TestSelectByPositions:
    """Test _select_by_positions static method for all field types."""

    def test_regular_tensor(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = AsyncSimpleStorageManager._select_by_positions(t, [0, 2])
        assert torch.equal(result, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))

    def test_nested_tensor(self):
        t = torch.nested.as_nested_tensor(
            [torch.tensor([1.0]), torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])],
            layout=torch.jagged,
        )
        result = AsyncSimpleStorageManager._select_by_positions(t, [0, 2])
        assert isinstance(result, list)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([1.0]))
        assert torch.equal(result[1], torch.tensor([4.0, 5.0, 6.0]))

    def test_non_tensor_stack(self):
        nts = NonTensorStack("a", "b", "c")
        result = AsyncSimpleStorageManager._select_by_positions(nts, [1, 2])
        assert isinstance(result, NonTensorStack)
        assert result.tolist() == ["b", "c"]

    def test_list(self):
        data = [{"x": 1}, {"x": 2}, {"x": 3}]
        result = AsyncSimpleStorageManager._select_by_positions(data, [0, 2])
        assert result == [{"x": 1}, {"x": 3}]

    def test_numpy_array(self):
        arr = np.array([10, 20, 30])
        result = AsyncSimpleStorageManager._select_by_positions(arr, [0, 2])
        np.testing.assert_array_equal(result, np.array([10, 30]))

    def test_regular_tensor_single_element(self):
        """Case 1: Single element selection returns a single-row view."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = AsyncSimpleStorageManager._select_by_positions(t, [1])
        assert result.shape == (1, 2)
        assert torch.equal(result, torch.tensor([[3.0, 4.0]]))

    def test_regular_tensor_strided_slice(self):
        """Case 2: Constant stride (step > 1) uses Python slicing for zero-copy view."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        result = AsyncSimpleStorageManager._select_by_positions(t, [0, 2, 4])
        # positions form constant stride of 2
        expected = torch.tensor([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]])
        assert torch.equal(result, expected)

    def test_regular_tensor_irregular_indices_fallback(self):
        """Case 3: Irregular indices fall back to index_select to avoid ZMQ frame fragmentation."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        # positions [0, 2, 3] have irregular gaps (2, then 1) - not constant stride
        result = AsyncSimpleStorageManager._select_by_positions(t, [0, 2, 3])
        expected = torch.tensor([[1.0, 2.0], [5.0, 6.0], [7.0, 8.0]])
        assert torch.equal(result, expected)

    def test_regular_tensor_irregular_reverse_order(self):
        """Irregular indices in reverse order also falls back to index_select."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        result = AsyncSimpleStorageManager._select_by_positions(t, [3, 1, 0])
        expected = torch.tensor([[7.0, 8.0], [3.0, 4.0], [1.0, 2.0]])
        assert torch.equal(result, expected)

    def test_nested_tensor_single_element(self):
        """Single element from nested tensor uses the lambda path."""
        t = torch.nested.as_nested_tensor(
            [torch.tensor([1.0]), torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])],
            layout=torch.jagged,
        )
        result = AsyncSimpleStorageManager._select_by_positions(t, [1])
        assert isinstance(result, list)
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor([2.0, 3.0]))

    def test_empty_positions_raises_error(self):
        """Empty positions list should raise ValueError."""
        t = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="No positions specified"):
            AsyncSimpleStorageManager._select_by_positions(t, [])

    def test_regular_tensor_negative_stride_rejected(self):
        """Negative stride (reversed order) should fall back to index_select."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        # positions [2, 1, 0] have step = -1 (negative)
        result = AsyncSimpleStorageManager._select_by_positions(t, [2, 1, 0])
        expected = torch.tensor([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])
        assert torch.equal(result, expected)


class TestPackFieldValues:
    """Test _pack_field_values static method packing logic."""

    def test_uniform_tensors_to_nested(self):
        """Same-shape tensors → nested tensor (default)."""
        values = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = AsyncSimpleStorageManager._pack_field_values(values)  # type: ignore[attr-defined]
        assert isinstance(result, torch.Tensor)
        assert result.is_nested

    def test_variable_length_tensors_to_nested(self):
        """Different-shape tensors → nested tensor."""
        values = [torch.tensor([1.0]), torch.tensor([2.0, 3.0])]
        result = AsyncSimpleStorageManager._pack_field_values(values)  # type: ignore[attr-defined]
        assert isinstance(result, torch.Tensor)
        assert result.is_nested

    def test_non_tensors_to_nontensorstack(self):
        """Non-tensor values → NonTensorStack."""
        values = ["hello", "world"]
        result = AsyncSimpleStorageManager._pack_field_values(values)  # type: ignore[attr-defined]
        assert isinstance(result, NonTensorStack)
        assert result.tolist() == ["hello", "world"]

    def test_mixed_tensors_and_none_to_nontensorstack(self):
        """Mixed tensor + None values should stay as NonTensorStack (no nested tensor)."""
        t0 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0])
        values = [t0, None, t2]

        result = AsyncSimpleStorageManager._pack_field_values(values)  # type: ignore[attr-defined]

        assert isinstance(result, NonTensorStack)
        unpacked = result.tolist()
        assert len(unpacked) == 3
        assert torch.equal(unpacked[0], t0)
        assert unpacked[1] is None
        assert torch.equal(unpacked[2], t2)

    def test_all_none_to_nontensorstack(self):
        """All-None values should be preserved in NonTensorStack."""
        values = [None, None]

        result = AsyncSimpleStorageManager._pack_field_values(values)  # type: ignore[attr-defined]

        assert isinstance(result, NonTensorStack)
        assert result.tolist() == [None, None]


@pytest.mark.asyncio
async def test_put_data_raises_when_data_parser_combined_with_compression():
    """put_data rejects data_parser when tensor compression is enabled."""
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=Role.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 12345},
        ),
    }

    controller_info = ZMQServerInfo(
        role=Role.CONTROLLER,
        id="controller_0",
        ip="127.0.0.1",
        ports={"handshake_socket": 12346, "data_status_update_socket": 12347},
    )

    config = {"zmq_info": storage_unit_infos}

    with patch("transfer_queue.storage.managers.base.StorageManager._connect_to_controller") as mock_connect:
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_storage_manager"
        manager.config = config
        manager.controller_info = controller_info
        manager.storage_unit_infos = storage_unit_infos
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None
        manager._connect_to_controller = mock_connect
        manager.compressor = TensorCompressor(algorithm="zstd", level=3, min_bytes=1)

        batch_meta = BatchMeta(
            global_indexes=[0],
            partition_ids=["0"],
            field_schema={
                "x": {
                    "dtype": torch.float32,
                    "shape": (1,),
                    "is_nested": False,
                    "is_non_tensor": False,
                }
            },
            production_status=np.ones(1, dtype=np.int8),
        )

        data = TensorDict({"x": torch.randn(1, 1)}, batch_size=1)

        with pytest.raises(ValueError, match="data_parser is not supported"):
            await manager.put_data(data, batch_meta, data_parser=lambda d: d)


@pytest.mark.asyncio
async def test_put_data_records_raw_replay_event(monkeypatch, tmp_path):
    manager = _make_replay_manager(monkeypatch, tmp_path)
    metadata = BatchMeta(
        global_indexes=[0, 1],
        partition_ids=["train_0", "train_0"],
        field_schema={
            "input_ids": {"dtype": torch.int64, "shape": (4,), "is_nested": False, "is_non_tensor": False},
            "attention_mask": {"dtype": torch.bool, "shape": (4,), "is_nested": False, "is_non_tensor": False},
        },
        production_status=np.ones(2, dtype=np.int8),
    )
    data = TensorDict(
        {
            "input_ids": torch.arange(8, dtype=torch.int64).reshape(2, 4),
            "attention_mask": torch.ones((2, 4), dtype=torch.bool),
        },
        batch_size=2,
    )

    await manager.put_data(data, metadata)

    events = _read_replay_events(tmp_path)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "put_raw"
    assert event["role"] == "storage_manager"
    assert event["component_id"] == manager.storage_manager_id
    assert event["pid"] == "train_0"
    assert event["partition_id"] == "train_0"
    assert event["indexes"] == [0, 1]
    assert event["fields"]["input_ids"]["dtype"] == "torch.int64"
    assert event["fields"]["input_ids"]["shape"] == [2, 4]
    assert event["fields"]["input_ids"]["raw_tensor_bytes"] == 64
    assert event["fields"]["attention_mask"]["raw_tensor_bytes"] == 8
    assert event["raw_tensor_bytes"] == 72
    assert event["raw_estimated_bytes"] == 72
    assert event["data_parser_stage"] == "none"
    assert event["elapsed_ms"] >= 0
    assert not (tmp_path / "data").exists()


@pytest.mark.asyncio
async def test_put_data_records_exact_raw_bytes_for_nested_tensor(monkeypatch, tmp_path):
    manager = _make_replay_manager(monkeypatch, tmp_path)
    nested = torch.nested.as_nested_tensor(
        [torch.ones(3, dtype=torch.int16), torch.ones(1, dtype=torch.int16)],
        layout=torch.jagged,
    )
    metadata = BatchMeta(
        global_indexes=[0, 1],
        partition_ids=["train_0", "train_0"],
        field_schema={"x": {"dtype": torch.int16, "shape": None, "is_nested": True, "is_non_tensor": False}},
        production_status=np.ones(2, dtype=np.int8),
    )
    data = TensorDict({"x": nested}, batch_size=2)

    await manager.put_data(data, metadata)

    event = _read_replay_events(tmp_path)[0]
    assert event["event"] == "put_raw"
    assert event["fields"]["x"]["kind"] == "list"
    assert event["fields"]["x"]["raw_tensor_bytes"] == 8
    assert event["fields"]["x"]["raw_estimated_bytes"] == 8
    assert event["raw_tensor_bytes"] == 8
    assert event["raw_estimated_bytes"] == 8


@pytest.mark.asyncio
async def test_get_data_records_raw_replay_from_reconstructed_data(monkeypatch, tmp_path):
    manager = _make_replay_manager(monkeypatch, tmp_path)
    manager._get_from_single_storage_unit = AsyncMock(
        return_value=(
            ["input_ids"],
            {"input_ids": [torch.tensor([1, 2, 3], dtype=torch.int16), torch.tensor([4, 5, 6], dtype=torch.int16)]},
        )
    )
    metadata = BatchMeta(
        global_indexes=[10, 11],
        partition_ids=["train_0", "train_0"],
        field_schema={"input_ids": {"dtype": torch.int16, "shape": (3,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(2, dtype=np.int8),
    )

    result = await manager.get_data(metadata)

    assert torch.equal(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int16))
    events = _read_replay_events(tmp_path)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "get_raw"
    assert event["pid"] == "train_0"
    assert event["indexes"] == [10, 11]
    assert event["fields"]["input_ids"]["shape"] == [2, 3]
    assert event["raw_tensor_bytes"] == 12
    assert event["raw_estimated_bytes"] == 12
    assert event["elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_put_data_replay_dump_writes_loadable_pt(monkeypatch, tmp_path):
    manager = _make_replay_manager(monkeypatch, tmp_path, replay_env={"TQ_REPLAY_DUMP_DATA": "1"})
    metadata = BatchMeta(
        global_indexes=[3],
        partition_ids=["train_0"],
        field_schema={"x": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(1, dtype=np.int8),
    )
    data = TensorDict({"x": torch.tensor([[1.5, 2.5]], dtype=torch.float32)}, batch_size=1)

    await manager.put_data(data, metadata)

    event = _read_replay_events(tmp_path)[0]
    assert event["dump"] == f"data/train_0/{manager.storage_manager_id}/put_000001.pt"
    dump = torch.load(tmp_path / event["dump"], weights_only=False)
    assert dump["partition_id"] == "train_0"
    assert dump["global_indexes"] == [3]
    assert dump["batch_seq"] == 1
    assert dump["raw_tensor_bytes"] == 8
    assert torch.equal(dump["fields"]["x"], data["x"])
    assert dump["field_schema"]["x"]["dtype"] == torch.float32


@pytest.mark.asyncio
async def test_put_data_replay_dump_max_bytes_skips_large_dump(monkeypatch, tmp_path):
    manager = _make_replay_manager(
        monkeypatch,
        tmp_path,
        replay_env={"TQ_REPLAY_DUMP_DATA": "1", "TQ_REPLAY_DUMP_MAX_BYTES": "4"},
    )
    metadata = BatchMeta(
        global_indexes=[3],
        partition_ids=["train_0"],
        field_schema={"x": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(1, dtype=np.int8),
    )
    data = TensorDict({"x": torch.tensor([[1.5, 2.5]], dtype=torch.float32)}, batch_size=1)

    await manager.put_data(data, metadata)

    event = _read_replay_events(tmp_path)[0]
    assert "dump" not in event
    if (tmp_path / "data").exists():
        assert not list((tmp_path / "data").glob("**/*.pt"))


@pytest.mark.asyncio
async def test_put_data_replay_marks_data_parser_stage(monkeypatch, tmp_path):
    manager = _make_replay_manager(monkeypatch, tmp_path)
    metadata = BatchMeta(
        global_indexes=[0],
        partition_ids=["train_0"],
        field_schema={"x": {"dtype": torch.float32, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(1, dtype=np.int8),
    )
    data = TensorDict({"x": torch.tensor([[1.0]])}, batch_size=1)

    await manager.put_data(data, metadata, data_parser=lambda values: values)

    event = _read_replay_events(tmp_path)[0]
    assert event["data_parser_stage"] == "before_storage_unit_parser"


@pytest.mark.asyncio
async def test_put_data_rejects_multi_partition_metadata(monkeypatch, tmp_path):
    manager = _make_replay_manager(monkeypatch, tmp_path)
    metadata = BatchMeta(
        global_indexes=[0, 1],
        partition_ids=["train_0", "train_1"],
        field_schema={"x": {"dtype": torch.int32, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(2, dtype=np.int8),
    )
    data = TensorDict({"x": torch.tensor([[7], [8]], dtype=torch.int32)}, batch_size=2)

    with pytest.raises(ValueError, match="single partition"):
        await manager.put_data(data, metadata)

    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.asyncio
async def test_get_data_rejects_multi_partition_metadata(monkeypatch, tmp_path):
    manager = _make_replay_manager(monkeypatch, tmp_path)
    manager._get_from_single_storage_unit = AsyncMock(
        return_value=(["x"], {"x": [torch.tensor([7], dtype=torch.int32), torch.tensor([8], dtype=torch.int32)]})
    )
    metadata = BatchMeta(
        global_indexes=[0, 1],
        partition_ids=["train_0", "train_1"],
        field_schema={"x": {"dtype": torch.int32, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(2, dtype=np.int8),
    )

    with pytest.raises(ValueError, match="single partition"):
        await manager.get_data(metadata)

    manager._get_from_single_storage_unit.assert_not_awaited()
    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.asyncio
async def test_put_data_replay_records_raw_bytes_when_compression_enabled(monkeypatch, tmp_path):
    manager = _make_replay_manager(
        monkeypatch,
        tmp_path,
        config_extra={"compression": {"algorithm": "zstd", "level": 3, "min_bytes": 1}},
    )
    metadata = BatchMeta(
        global_indexes=[0],
        partition_ids=["train_0"],
        field_schema={"x": {"dtype": torch.float32, "shape": (4,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(1, dtype=np.int8),
    )
    data = TensorDict({"x": torch.zeros((1, 4), dtype=torch.float32)}, batch_size=1)

    await manager.put_data(data, metadata)

    event = _read_replay_events(tmp_path)[0]
    assert event["raw_tensor_bytes"] == 16
    assert event["fields"]["x"]["raw_tensor_bytes"] == 16


@pytest.mark.asyncio
async def test_put_to_single_storage_unit_does_not_record_wire_by_default(monkeypatch, tmp_path):
    manager = _make_replay_manager(monkeypatch, tmp_path, mock_io=False)
    response = ZMQMessage.create(
        request_type=ZMQRequestType.PUT_DATA_RESPONSE,
        sender_id="storage_0",
        receiver_id=manager.storage_manager_id,
        body={"message": "ok"},
    )
    socket = _FakeAsyncSocket(response.serialize(encoder=manager._encoder))

    await AsyncSimpleStorageManager._put_to_single_storage_unit.__wrapped__(
        manager,
        [0],
        {"x": torch.ones((1, 2), dtype=torch.float32)},
        target_storage_unit="storage_0",
        partition_id="train_0",
        socket=socket,
    )

    assert socket.sent_frames is not None
    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.asyncio
async def test_put_to_single_storage_unit_records_wire_event_when_enabled(monkeypatch, tmp_path):
    pytest.importorskip("zstandard")
    manager = _make_replay_manager(
        monkeypatch,
        tmp_path,
        replay_env={"TQ_REPLAY_RECORD_WIRE": "1"},
        config_extra={"compression": {"algorithm": "zstd", "level": 3, "min_bytes": 1}},
        mock_io=False,
    )
    response = ZMQMessage.create(
        request_type=ZMQRequestType.PUT_DATA_RESPONSE,
        sender_id="storage_0",
        receiver_id=manager.storage_manager_id,
        body={"message": "ok"},
    )
    socket = _FakeAsyncSocket(response.serialize(encoder=manager._encoder))
    tensor = torch.zeros((2, 1024), dtype=torch.float32)

    await AsyncSimpleStorageManager._put_to_single_storage_unit.__wrapped__(
        manager,
        [0, 2],
        {"x": tensor},
        target_storage_unit="storage_0",
        partition_id="train_0",
        socket=socket,
    )

    assert socket.sent_frames is not None
    [event] = _read_replay_events(tmp_path)
    assert event["event"] == "put_wire"
    assert event["pid"] == "train_0"
    assert event["partition_id"] == "train_0"
    assert event["target_storage_unit"] == "storage_0"
    assert event["indexes"] == [0, 2]
    assert event["compression_algorithm"] == "zstd"
    assert event["raw_tensor_bytes"] == tensor.numel() * tensor.element_size()
    assert 0 < event["compressed_tensor_bytes"] < event["raw_tensor_bytes"]
    assert event["wire_frame_bytes"] == _frames_nbytes(socket.sent_frames)
    assert event["elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_put_wire_event_labels_data_parser_payload(monkeypatch, tmp_path):
    manager = _make_replay_manager(
        monkeypatch,
        tmp_path,
        replay_env={"TQ_REPLAY_RECORD_WIRE": "1"},
        mock_io=False,
    )
    response = ZMQMessage.create(
        request_type=ZMQRequestType.PUT_DATA_RESPONSE,
        sender_id="storage_0",
        receiver_id=manager.storage_manager_id,
        body={"message": "ok"},
    )
    socket = _FakeAsyncSocket(response.serialize(encoder=manager._encoder))

    await AsyncSimpleStorageManager._put_to_single_storage_unit.__wrapped__(
        manager,
        [0],
        {"x": torch.ones((1, 1), dtype=torch.float32)},
        target_storage_unit="storage_0",
        partition_id="train_0",
        data_parser=lambda values: values,
        socket=socket,
    )

    [event] = _read_replay_events(tmp_path)
    assert event["event"] == "put_wire"
    assert event["data_parser_stage"] == "before_storage_unit_parser"


@pytest.mark.asyncio
async def test_get_from_single_storage_unit_records_response_wire_event(monkeypatch, tmp_path):
    manager = _make_replay_manager(
        monkeypatch,
        tmp_path,
        replay_env={"TQ_REPLAY_RECORD_WIRE": "1"},
        mock_io=False,
    )
    response = ZMQMessage.create(
        request_type=ZMQRequestType.GET_DATA_RESPONSE,
        sender_id="storage_0",
        receiver_id=manager.storage_manager_id,
        body={"data": {"x": [torch.tensor([1, 2], dtype=torch.int16)]}},
    )
    response_frames = response.serialize(encoder=manager._encoder)
    socket = _FakeAsyncSocket(response_frames)

    fields, data = await AsyncSimpleStorageManager._get_from_single_storage_unit.__wrapped__(
        manager,
        [5],
        ["x"],
        target_storage_unit="storage_0",
        partition_id="train_0",
        socket=socket,
    )

    assert fields == ["x"]
    assert torch.equal(data["x"][0], torch.tensor([1, 2], dtype=torch.int16))
    [event] = _read_replay_events(tmp_path)
    assert event["event"] == "get_wire"
    assert event["pid"] == "train_0"
    assert event["partition_id"] == "train_0"
    assert event["target_storage_unit"] == "storage_0"
    assert event["indexes"] == [5]
    assert event["wire_frame_bytes"] == _frames_nbytes(response_frames)
    assert event["elapsed_ms"] >= 0
