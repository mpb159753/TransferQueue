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
import io
import logging
import os
import pickle
import socket
import time
from dataclasses import dataclass
from typing import Any, Optional, TypeAlias, Union
from uuid import uuid4

import psutil
import torch
import zmq

try:
    from torch.distributed.rpc.internal import _internal_rpc_pickler

    HAS_RPC_PICKLER = True
except ImportError:
    HAS_RPC_PICKLER = False

from transfer_queue.utils.serial_utils import MsgpackDecoder, MsgpackEncoder
from transfer_queue.utils.utils import (
    ExplicitEnum,
    TransferQueueRole,
    get_env_bool,
)
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

TQ_ZERO_COPY_SERIALIZATION = get_env_bool("TQ_ZERO_COPY_SERIALIZATION", default=False) and HAS_RPC_PICKLER
_encoder = MsgpackEncoder()
_decoder = MsgpackDecoder(torch.Tensor)

bytestr: TypeAlias = bytes | bytearray | memoryview


class ZMQRequestType(ExplicitEnum):
    # HANDSHAKE
    HANDSHAKE = "HANDSHAKE"  # TransferQueueStorageUnit -> TransferQueueController
    HANDSHAKE_ACK = "HANDSHAKE_ACK"  # TransferQueueController  -> TransferQueueStorageUnit

    # DATA_OPERATION
    GET_DATA = "GET"
    PUT_DATA = "PUT"
    GET_DATA_RESPONSE = "GET_DATA_RESPONSE"
    PUT_DATA_RESPONSE = "PUT_DATA_RESPONSE"
    CLEAR_DATA = "CLEAR_DATA"
    CLEAR_DATA_RESPONSE = "CLEAR_DATA_RESPONSE"

    PUT_GET_OPERATION_ERROR = "PUT_GET_OPERATION_ERROR"
    PUT_GET_ERROR = "PUT_GET_ERROR"
    PUT_ERROR = "PUT_ERROR"
    GET_ERROR = "GET_ERROR"
    CLEAR_DATA_ERROR = "CLEAR_DATA_ERROR"

    # META_OPERATION
    GET_META = "GET_META"
    GET_META_RESPONSE = "GET_META_RESPONSE"
    GET_CLEAR_META = "GET_CLEAR_META"
    GET_CLEAR_META_RESPONSE = "GET_CLEAR_META_RESPONSE"
    CLEAR_META = "CLEAR_META"
    CLEAR_META_RESPONSE = "CLEAR_META_RESPONSE"

    # CHECK_CONSUMPTION
    CHECK_CONSUMPTION = "CHECK_CONSUMPTION"
    CONSUMPTION_RESPONSE = "CONSUMPTION_RESPONSE"

    # NOTIFY_DATA_UPDATE
    NOTIFY_DATA_UPDATE = "NOTIFY_DATA_UPDATE"
    NOTIFY_DATA_UPDATE_ACK = "NOTIFY_DATA_UPDATE_ACK"
    NOTIFY_DATA_UPDATE_ERROR = "NOTIFY_DATA_UPDATE_ERROR"


class ZMQServerInfo:
    def __init__(self, role: TransferQueueRole, id: str, ip: str, ports: dict[str, str]):
        self.role = role
        self.id = id
        self.ip = ip
        self.ports = ports

    def to_addr(self, port_name: str) -> str:
        return f"tcp://{self.ip}:{self.ports[port_name]}"

    def to_dict(self):
        return {
            "role": self.role,
            "id": self.id,
            "ip": self.ip,
            "ports": self.ports,
        }

    def __str__(self) -> str:
        return f"ZMQSocketInfo(role={self.role}, id={self.id}, ip={self.ip}, ports={self.ports})"


class _InternalRPCUnpickler(pickle.Unpickler):
    def __init__(self, file, tensors: list[torch.Tensor]):
        super().__init__(file)
        self.tensors = tensors

    def persistent_load(self, pid: int) -> torch.Tensor:
        return self.tensors[pid]

class ZMQPickler(pickle.Pickler):
    def __init__(self, file):
        super().__init__(file)
        self.tensors = []

    def persistent_id(self, obj):
        if isinstance(obj, torch.Tensor):
            self.tensors.append(obj)
            return len(self.tensors) - 1
        return None

@dataclass
class ZMQMessage:
    request_type: ZMQRequestType
    sender_id: str
    receiver_id: str | None
    body: dict[str, Any]
    request_id: str
    timestamp: float

    @classmethod
    def create(
            cls,
            request_type: ZMQRequestType,
            sender_id: str,
            body: dict[str, Any],
            receiver_id: Optional[str] = None,
    ) -> "ZMQMessage":
        return cls(
            request_type=request_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
            body=body,
            request_id=str(uuid4().hex[:8]),
            timestamp=time.time(),
        )

    def serialize(self) -> list[bytestr]:
        """
        Serializes the ZMQMessage object.

        Returns:
            list[bytestr]: If TQ_ZERO_COPY_SERIALIZATION is enabled, returns a list where the first element
            is the pickled bytes of the message, followed by the flattened serialized tensor parts as
            [pickled_bytes, <bytes>, |<bytes>, <memoryview>, |<bytes>, <memoryview>|...].
            From the third element, two elements is a group that will be used to restore a tensor.

            If TQ_ZERO_COPY_SERIALIZATION is disabled, returns a single-element list containing only the pickled bytes
            through pickle.
        """
        logger.info(f"Serializing ZMQMessage with TQ_ZERO_COPY_SERIALIZATION={TQ_ZERO_COPY_SERIALIZATION}")
        if TQ_ZERO_COPY_SERIALIZATION:
            # pickled_bytes 是骨架，tensors 是被剥离出的张量列表
            pickled_bytes, tensors = _internal_rpc_pickler.serialize(self)

            tensor_headers = []
            payload_frames = []

            current_payload_idx = 0

            for t in tensors:
                # 移除这里的全局 t = t.detach().cpu()，改为按需移动

                # --- 分支 A: Nested Tensor ---
                if t.is_nested:
                    try:
                        # 尝试直接获取 (Jagged Layout)
                        values = t.values()
                        offsets = t.offsets()
                    except AttributeError:
                        # 兼容 Strided Layout
                        sub_tensors = t.unbind()

                        if len(sub_tensors) > 0:
                            values = torch.cat(sub_tensors, dim=0)
                        else:
                            values = torch.tensor([], dtype=t.dtype, device=t.device)

                        lengths = [st.size(0) for st in sub_tensors]
                        offsets = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(dim=0)

                    values_np = values.detach().cpu().numpy()

                    if isinstance(offsets, torch.Tensor):
                        offsets_np = offsets.detach().cpu().numpy()
                    else:
                        offsets_np = offsets.numpy()

                    # 构建 Header
                    header = {
                        "type": "nested",
                        "dtype": str(t.dtype).split(".")[-1],
                        "values_shape": values_np.shape,
                        "offsets_shape": offsets_np.shape,
                        "offsets_dtype": str(offsets_np.dtype),
                        "frame_idx_values": current_payload_idx,
                        "frame_idx_offsets": current_payload_idx + 1
                    }
                    tensor_headers.append(header)
                    payload_frames.append(values_np)
                    payload_frames.append(offsets_np)
                    current_payload_idx += 2

                # Dense Tensor
                else:
                    if t.is_cuda:
                        t = t.detach().cpu()

                    if not t.is_contiguous():
                        t = t.contiguous()

                    t_np = t.numpy()

                    header = {
                        "type": "dense",
                        "dtype": str(t.dtype).split(".")[-1],
                        "shape": tuple(t.shape),
                        "frame_idx": current_payload_idx
                    }

                    tensor_headers.append(header)
                    payload_frames.append(t_np)

                    current_payload_idx += 1

            # 序列化 Header
            header_bytes = pickle.dumps(tensor_headers)
            # 返回最终帧列表 (ZMQ 将以此顺序发送，payload 部分为 0-copy)
            return [pickled_bytes, header_bytes, *payload_frames]
        else:
            return [pickle.dumps(self)]

    @classmethod
    def deserialize(cls, data: Union[list[bytes], bytes]) -> "ZMQMessage":
        """Deserialize a ZMQMessage object from serialized data."""
        logger.info(f"Deserializing ZMQMessage with TQ_ZERO_COPY_SERIALIZATION={TQ_ZERO_COPY_SERIALIZATION}")
        if isinstance(data, bytes):
            return pickle.loads(data)
        if isinstance(data, list) and len(data) == 1:
            return pickle.loads(data[0])

        # data 结构: [pickled_skeleton, header_bytes, frame_0, frame_1, ...]
        pickled_skeleton = data[0]
        header_bytes = data[1]
        payload_frames = data[2:]

        # 解析 Tensor 元数据头
        tensor_headers = pickle.loads(header_bytes)
        reconstructed_tensors = []

        # 重组 Tensors
        for header in tensor_headers:
            t_type = header["type"]

            # 辅助函数：将字节流转为 Tensor (Zero-Copy View)
            def buffer_to_tensor(frame_idx, dtype_str, shape):
                np_dtype = getattr(np, dtype_str)
                arr = np.frombuffer(payload_frames[frame_idx], dtype=np_dtype).reshape(shape)
                return torch.from_numpy(arr)

            if t_type == "dense":
                # 重组普通 Tensor
                tensor = buffer_to_tensor(
                    header["frame_idx"],
                    header["dtype"],
                    header["shape"]
                )
                reconstructed_tensors.append(tensor)

            elif t_type == "nested":
                # 重组 Nested Tensor (Values + Offsets)
                # 还原 values 大张量
                values = buffer_to_tensor(
                    header["frame_idx_values"],
                    header["dtype"],
                    header["values_shape"]
                )
                # 还原 offsets 张量
                offsets = buffer_to_tensor(
                    header["frame_idx_offsets"],
                    header["offsets_dtype"],
                    header["offsets_shape"]
                )

                # 使用 PyTorch 高效 API 重组
                nested_t = torch.nested.nested_tensor_from_jagged(values, offsets)
                reconstructed_tensors.append(nested_t)
            else:
                raise ValueError(f"Unknown tensor type in header: {t_type}")

        # 最终反序列化：将 Tensor 挂载回对象骨架
        f = io.BytesIO(pickled_skeleton)
        unpickler = _InternalRPCUnpickler(f, reconstructed_tensors)
        obj = unpickler.load()

        return obj


def get_free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def create_zmq_socket(
        ctx: zmq.Context,
        socket_type: Any,
        identity: Optional[bytestr] = None,
) -> zmq.Socket:
    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024 ** 3
    available_mem = mem.available / 1024 ** 3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024 ** 3)  # 0.5GB in bytes
    else:
        buf_size = -1  # Use system default buffer size

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)
    return socket
