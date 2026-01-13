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

import logging
import os
import pickle
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TypeAlias
from uuid import uuid4

import psutil
import torch
import zmq

from transfer_queue.utils.serial_utils import MsgpackDecoder, MsgpackEncoder
from transfer_queue.utils.utils import (
    ExplicitEnum,
    TransferQueueRole,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

_encoder = MsgpackEncoder()
_decoder = MsgpackDecoder(torch.Tensor)

bytestr: TypeAlias = bytes | bytearray | memoryview
META_KEY = "__tq_meta__"
LARGE_OBJECT_THRESHOLD = 10 * 1024


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


def _pack_data(data: Any, buffers: list[memoryview]) -> Any:
    """
    递归遍历数据结构。
    1. 将所有 Tensor 提取为 buffer 并存入 buffers 列表。
    2. 在原数据结构位置替换为元数据描述符（占位符）。
    """
    if isinstance(data, torch.Tensor):
        if not data.is_contiguous():
            data = data.contiguous()

        if data.device.type != 'cpu':
            data = data.cpu()

        # 记录元数据
        meta = {
            META_KEY: "tensor",
            # 获取 buffer 长度, 作为当前片段的 idx
            "idx": len(buffers),
            "dtype": data.dtype,
            "shape": data.shape,
        }

        # 获取零拷贝视图
        buf = memoryview(data.numpy())
        buffers.append(buf)
        return meta
    elif isinstance(data, str):
        if len(data) > LARGE_OBJECT_THRESHOLD:
            encoded_data = data.encode("utf-8")
            meta = {
                META_KEY: "str",
                "idx": len(buffers),
                "encoding": "utf-8"
            }
            buffers.append(memoryview(encoded_data))
            return meta
        return data
    elif isinstance(data, dict):
        return {k: _pack_data(v, buffers) for k, v in data.items()}

    elif isinstance(data, list):
        return [_pack_data(v, buffers) for v in data]

    elif isinstance(data, tuple):
        return tuple(_pack_data(v, buffers) for v in data)

    # 其他类型直接返回 (将在 Header 中被 pickle)
    return data


def _unpack_data(data: Any, buffers: list[bytestr]) -> Any:
    """
    递归还原数据结构。
    """
    if isinstance(data, dict):
        # 检查是否是特殊元数据包
        if META_KEY in data:
            obj_type = data[META_KEY]
            idx = data["idx"]
            raw_buffer = buffers[idx]

            # Case A: Tensor
            if obj_type == "tensor":
                dtype = data["dtype"]
                shape = data["shape"]
                # Zero-Copy View
                tensor = torch.frombuffer(raw_buffer, dtype=dtype)
                return tensor.reshape(shape)

            # Case B: Bytes
            elif obj_type == "bytes":
                # 将 memoryview 转回 bytes
                # 注意：如果用户能接受 memoryview，直接返回 raw_buffer 性能最好
                # 这里为了兼容性转为 bytes (可能发生拷贝，取决于具体实现)
                return bytes(raw_buffer)

            # Case C: String
            elif obj_type == "str":
                encoding = data["encoding"]
                # Decode bytes -> str
                return bytes(raw_buffer).decode(encoding)

        # 常规字典递归
        return {k: _unpack_data(v, buffers) for k, v in data.items()}

    elif isinstance(data, list):
        return [_unpack_data(v, buffers) for v in data]

    elif isinstance(data, tuple):
        return tuple(_unpack_data(v, buffers) for v in data)

    return data


@dataclass
class ZMQMessage:
    request_type: Any  # ZMQRequestType
    sender_id: str
    receiver_id: str | None
    body: dict[str, Any]  # 这里的 body 是未处理的原始数据
    request_id: str
    timestamp: float

    # 仅在序列化/反序列化过程中使用的临时容器，不需要在构造时传入
    _buffers: list[memoryview | bytestr] = field(default_factory=list, repr=False)

    @classmethod
    def create(
            cls,
            request_type: Any,
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
        将消息序列化为 ZMQ Multipart 帧列表。
        Frame 0: Pickled Header (包含结构树和非 Tensor 数据)
        Frame 1...N: Raw Tensor Buffers
        """
        self._buffers = []

        # 1. 提取所有 Tensor 到 self._buffers，并将 body 转换为轻量级结构
        packed_body = _pack_data(self.body, self._buffers)

        # 2. 构建 Header
        header = {
            "request_type": self.request_type,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "body_structure": packed_body,  # 仅包含结构和元数据
        }

        # 3. 序列化 Header
        header_bytes = pickle.dumps(header)

        # 4. 组装最终发送列表：[Header, Buffer_0, Buffer_1, ...]
        return [header_bytes, *self._buffers]

    @classmethod
    def deserialize(cls, frames: list[bytestr]) -> "ZMQMessage":
        """
        从 ZMQ Multipart 帧列表反序列化消息。
        """
        if not frames:
            raise ValueError("Empty frames received")

        # 1. 解析 Header
        header_bytes = frames[0]
        header = pickle.loads(header_bytes)

        # 2. 获取数据帧 (Frames 1...N)
        raw_buffers = frames[1:]

        # 3. 递归重构 Body (Zero-Copy)
        body_structure = header["body_structure"]
        restored_body = _unpack_data(body_structure, raw_buffers)

        # 4. 构建对象
        msg = cls(
            request_type=header["request_type"],
            sender_id=header["sender_id"],
            receiver_id=header["receiver_id"],
            body=restored_body,
            request_id=header["request_id"],
            timestamp=header["timestamp"]
        )
        return msg


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
