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

"""Standalone JSONL replay recorder utilities.

The recorder is intentionally not wired into Controller or StorageManager here.
It is enabled only by ``TQ_REPLAY_DIR`` and treats write/inspection failures as
non-fatal so normal TransferQueue behavior can continue.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from transfer_queue.utils.logging_utils import get_logger

fcntl: Any
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised only on platforms without fcntl.
    fcntl = None
else:
    fcntl = _fcntl

np: Any
try:
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is available in the supported test environment.
    np = None
else:
    np = _np

torch: Any
try:
    import torch as _torch
except ImportError:  # pragma: no cover - torch is a TransferQueue dependency.
    torch = None
else:
    torch = _torch

NonTensorStack: Any
try:
    from tensordict import NonTensorStack as _NonTensorStack
except ImportError:  # pragma: no cover - tensordict is a TransferQueue dependency.
    NonTensorStack = None
else:
    NonTensorStack = _NonTensorStack

logger = get_logger(__name__)

_EVENTS_FILE = "events.jsonl"
_DUMP_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class ReplayConfig:
    record_dir: Path
    dump_data: bool
    buf_size: int
    dump_max_bytes: int | None
    record_wire: bool


class ReplayRecorder:
    """Small best-effort recorder for replay metadata events."""

    _append_lock = threading.Lock()

    def __init__(self, config: ReplayConfig, *, role: str, component_id: str | None = None) -> None:
        self.config = config
        self.role = role
        self.component_id = component_id
        self._buffer: list[str] = []
        self._lock = threading.RLock()
        self._closed = False
        self._dump_counters: dict[str, int] = {}

    @classmethod
    def from_env(cls, *, role: str, component_id: str | None = None) -> ReplayRecorder | None:
        record_dir = os.getenv("TQ_REPLAY_DIR")
        if not record_dir:
            return None

        try:
            buf_size = max(1, _parse_int_env("TQ_REPLAY_BUF_SIZE", default=1))
            dump_max_bytes = _parse_optional_int_env("TQ_REPLAY_DUMP_MAX_BYTES")
            config = ReplayConfig(
                record_dir=Path(record_dir).expanduser(),
                dump_data=_parse_bool_env("TQ_REPLAY_DUMP_DATA", default=False),
                buf_size=buf_size,
                dump_max_bytes=dump_max_bytes,
                record_wire=_parse_bool_env("TQ_REPLAY_RECORD_WIRE", default=False),
            )
        except ValueError as exc:
            logger.warning("Disabling replay recorder due to invalid environment: %s", exc)
            return None

        return cls(config, role=role, component_id=component_id)

    def record_event(self, event: str, payload: dict[str, Any]) -> None:
        """Record one JSONL event, logging and suppressing recorder failures."""
        try:
            common_fields = {
                "event": event,
                "role": self.role,
                "component_id": self.component_id,
                "ts": time.time(),
            }
            event_payload = _json_safe(payload)
            if not isinstance(event_payload, dict):
                event_payload = {"payload": event_payload}
            event_payload.update(common_fields)
            line = json.dumps(event_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"

            with self._lock:
                if self._closed:
                    return
                self._buffer.append(line)
                if len(self._buffer) >= self.config.buf_size:
                    self._flush_locked()
        except Exception as exc:  # pragma: no cover - defensive by design.
            logger.warning("Failed to record replay event %r: %s", event, exc)

    def extract_fields_info(self, data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Extract JSON-safe field metadata and cheap byte estimates."""
        fields: dict[str, dict[str, Any]] = {}
        for field_name, value in data.items():
            try:
                fields[str(field_name)] = _field_info(value)
            except Exception as exc:  # pragma: no cover - defensive by design.
                logger.warning("Failed to inspect replay field %r: %s", field_name, exc)
                fields[str(field_name)] = _unknown_info(value)
        return fields

    def make_dump_path(self, partition_id: str) -> Path | None:
        """Return the next sanitized dump path for a partition, if dumping is enabled."""
        if not self.config.dump_data:
            return None

        try:
            safe_partition = _sanitize_partition_id(partition_id)
            safe_component = _sanitize_component_id(self.component_id)
            with self._lock:
                counter = self._dump_counters.get(safe_partition, 0) + 1
                self._dump_counters[safe_partition] = counter
            partition_dir = self.config.record_dir / "data" / safe_partition
            if safe_component is not None:
                partition_dir = partition_dir / safe_component
            partition_dir.mkdir(parents=True, exist_ok=True)
            return partition_dir / f"put_{counter:06d}.pt"
        except Exception as exc:  # pragma: no cover - defensive by design.
            logger.warning("Failed to create replay dump path for partition %r: %s", partition_id, exc)
            return None

    def should_dump(self, raw_estimated_bytes: int | None = None) -> bool:
        if not self.config.dump_data:
            return False
        if self.config.dump_max_bytes is None:
            return True
        if raw_estimated_bytes is None:
            return False
        return raw_estimated_bytes <= self.config.dump_max_bytes

    def flush(self) -> None:
        """Flush buffered events, suppressing recorder write failures."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush buffered events and stop accepting new events."""
        with self._lock:
            if self._closed:
                return
            self._flush_locked()
            self._closed = True

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        lines = self._buffer
        self._buffer = []
        try:
            self._append_lines(lines)
        except Exception as exc:
            logger.warning("Failed to flush replay events: %s", exc)

    def _append_lines(self, lines: list[str]) -> None:
        self.config.record_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.record_dir / _EVENTS_FILE
        payload = "".join(lines).encode("utf-8")

        with self._append_lock:
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            lock_acquired = False
            try:
                if fcntl is not None:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX)
                        lock_acquired = True
                    except OSError as exc:
                        logger.warning("Could not lock replay event file %s: %s", path, exc)
                _write_all(fd, payload)
            finally:
                if lock_acquired and fcntl is not None:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except OSError as exc:
                        logger.warning("Could not unlock replay event file %s: %s", path, exc)
                os.close(fd)


def _parse_int_env(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _parse_optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _parse_bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be 0 or 1, got {raw!r}")


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        chunk_written = os.write(fd, view[written:])
        if chunk_written == 0:
            raise OSError("short write while appending replay events")
        written += chunk_written


def _field_info(value: Any) -> dict[str, Any]:
    if _is_tensor(value):
        if getattr(value, "is_nested", False):
            return _nested_tensor_info(value)
        if _is_sparse_tensor(value):
            return _sparse_tensor_info(value)
        return _regular_tensor_info(value)

    if NonTensorStack is not None and isinstance(value, NonTensorStack):
        estimated = _estimate_value_bytes(value)
        return {
            "kind": "non_tensor_stack",
            "length": len(value),
            "raw_tensor_bytes": _tensor_payload_bytes(value),
            "raw_estimated_bytes": estimated,
        }

    if np is not None and isinstance(value, np.ndarray):
        return {
            "kind": "numpy_array",
            "dtype": str(value.dtype),
            "shape": [int(dim) for dim in value.shape],
            "raw_tensor_bytes": None,
            "raw_estimated_bytes": int(value.nbytes),
        }

    if isinstance(value, str):
        return _estimated_info("str", len(value.encode("utf-8")))
    if isinstance(value, bytes | bytearray | memoryview):
        return _estimated_info("bytes", _bytes_len(value))
    if isinstance(value, list):
        return _sequence_info("list", value)
    if isinstance(value, tuple):
        return _sequence_info("tuple", value)
    if np is not None and isinstance(value, np.generic):
        return {
            "kind": "numpy_scalar",
            "dtype": str(value.dtype),
            "raw_tensor_bytes": None,
            "raw_estimated_bytes": int(value.dtype.itemsize),
        }

    return _unknown_info(value)


def _regular_tensor_info(tensor: Any) -> dict[str, Any]:
    nbytes = _regular_tensor_bytes(tensor)
    return {
        "kind": "tensor",
        "dtype": str(tensor.dtype),
        "shape": _safe_shape(tensor),
        "device": str(tensor.device),
        "raw_tensor_bytes": nbytes,
        "raw_estimated_bytes": nbytes,
    }


def _nested_tensor_info(tensor: Any) -> dict[str, Any]:
    rows = list(tensor.unbind())
    row_bytes = [_regular_tensor_bytes(row) for row in rows]
    total_bytes = sum(row_bytes)
    return {
        "kind": "nested_tensor",
        "dtype": str(tensor.dtype),
        "shape": _safe_shape(tensor),
        "layout": str(tensor.layout),
        "row_shapes": [_safe_shape(row) for row in rows],
        "raw_tensor_bytes": total_bytes,
        "raw_estimated_bytes": total_bytes,
    }


def _sparse_tensor_info(tensor: Any) -> dict[str, Any]:
    component_bytes = _sparse_component_bytes(tensor)
    return {
        "kind": "sparse_tensor",
        "dtype": str(tensor.dtype),
        "shape": _safe_shape(tensor),
        "layout": str(tensor.layout),
        "raw_tensor_bytes": component_bytes,
        "raw_estimated_bytes": component_bytes,
    }


def _sparse_component_bytes(tensor: Any) -> int | None:
    for method_group in (
        ("_values", "_indices"),
        ("values", "indices"),
        ("values", "crow_indices", "col_indices"),
        ("values", "ccol_indices", "row_indices"),
    ):
        components: list[Any] = []
        for method_name in method_group:
            method = getattr(tensor, method_name, None)
            if method is None:
                components = []
                break
            try:
                component = method()
            except Exception:
                components = []
                break
            if not _is_tensor(component) or getattr(component, "is_nested", False) or _is_sparse_tensor(component):
                components = []
                break
            components.append(component)
        if components:
            return sum(_regular_tensor_bytes(component) for component in components)
    return None


def _regular_tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _sequence_info(kind: str, value: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
    return {
        "kind": kind,
        "length": len(value),
        "raw_tensor_bytes": _tensor_payload_bytes(value),
        "raw_estimated_bytes": _estimate_value_bytes(value),
    }


def _estimated_info(kind: str, estimated_bytes: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "raw_tensor_bytes": None,
        "raw_estimated_bytes": estimated_bytes,
    }


def _unknown_info(value: Any) -> dict[str, Any]:
    return {
        "kind": type(value).__name__,
        "raw_tensor_bytes": None,
        "raw_estimated_bytes": None,
    }


def _estimate_value_bytes(value: Any, seen: set[int] | None = None) -> int | None:
    if seen is None:
        seen = set()

    value_id = id(value)
    if value_id in seen:
        return None

    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, bytes | bytearray | memoryview):
        return _bytes_len(value)
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return 8
    if isinstance(value, float):
        return 8
    if np is not None and isinstance(value, np.generic):
        return int(value.dtype.itemsize)
    if np is not None and isinstance(value, np.ndarray):
        return int(value.nbytes)
    if _is_tensor(value):
        if getattr(value, "is_nested", False):
            return _nested_tensor_info(value)["raw_estimated_bytes"]
        if _is_sparse_tensor(value):
            return _sparse_component_bytes(value)
        return _regular_tensor_bytes(value)
    if NonTensorStack is not None and isinstance(value, NonTensorStack):
        try:
            value = value.tolist()
        except Exception:
            return None

    if isinstance(value, Mapping):
        seen.add(value_id)
        total = 0
        for key, item in value.items():
            key_size = _estimate_value_bytes(key, seen)
            item_size = _estimate_value_bytes(item, seen)
            if key_size is None or item_size is None:
                return None
            total += key_size + item_size
        seen.remove(value_id)
        return total

    if isinstance(value, list | tuple | set | frozenset):
        seen.add(value_id)
        total = 0
        for item in value:
            item_size = _estimate_value_bytes(item, seen)
            if item_size is None:
                return None
            total += item_size
        seen.remove(value_id)
        return total

    return None


def _tensor_payload_bytes(value: Any, seen: set[int] | None = None) -> int | None:
    if seen is None:
        seen = set()

    if _is_tensor(value):
        if getattr(value, "is_nested", False):
            return _nested_tensor_info(value)["raw_tensor_bytes"]
        if _is_sparse_tensor(value):
            return _sparse_component_bytes(value)
        return _regular_tensor_bytes(value)

    value_id = id(value)
    if value_id in seen:
        return None

    if NonTensorStack is not None and isinstance(value, NonTensorStack):
        try:
            value = value.tolist()
        except Exception:
            return None

    if isinstance(value, Mapping):
        seen.add(value_id)
        total = 0
        found = False
        for key, item in value.items():
            for child in (key, item):
                child_bytes = _tensor_payload_bytes(child, seen)
                if child_bytes is not None:
                    total += child_bytes
                    found = True
        seen.remove(value_id)
        return total if found else None

    if isinstance(value, list | tuple | set | frozenset):
        seen.add(value_id)
        total = 0
        found = False
        for item in value:
            item_bytes = _tensor_payload_bytes(item, seen)
            if item_bytes is not None:
                total += item_bytes
                found = True
        seen.remove(value_id)
        return total if found else None

    return None


def _json_safe(value: Any, seen: set[int] | None = None) -> Any:
    if seen is None:
        seen = set()

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return {"kind": "bytes", "length": _bytes_len(value)}
    if torch is not None and isinstance(value, torch.dtype | torch.device):
        return str(value)
    if torch is not None and isinstance(value, torch.Size):
        return [_json_safe(item, seen) for item in value]
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return {
            "kind": "numpy_array",
            "dtype": str(value.dtype),
            "shape": [int(dim) for dim in value.shape],
            "nbytes": int(value.nbytes),
        }
    if _is_tensor(value):
        return _field_info(value)
    if NonTensorStack is not None and isinstance(value, NonTensorStack):
        return {"kind": "non_tensor_stack", "length": len(value), "raw_estimated_bytes": _estimate_value_bytes(value)}

    value_id = id(value)
    if value_id in seen:
        return repr(value)
    if isinstance(value, Mapping):
        seen.add(value_id)
        mapping_result = {str(_json_safe(key, seen)): _json_safe(item, seen) for key, item in value.items()}
        seen.remove(value_id)
        return mapping_result
    if isinstance(value, list | tuple | set | frozenset):
        seen.add(value_id)
        sequence_result = [_json_safe(item, seen) for item in value]
        seen.remove(value_id)
        return sequence_result

    return repr(value)


def _safe_shape(tensor: Any) -> list[Any] | None:
    try:
        return _json_safe(tuple(tensor.shape))
    except Exception:
        return None


def _is_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _is_sparse_tensor(tensor: Any) -> bool:
    if getattr(tensor, "is_sparse", False):
        return True
    return str(getattr(tensor, "layout", "")).startswith("torch.sparse")


def _sanitize_partition_id(partition_id: str) -> str:
    sanitized = _DUMP_NAME_RE.sub("_", str(partition_id)).strip("_")
    return sanitized or "partition"


def _sanitize_component_id(component_id: str | None) -> str | None:
    if component_id is None:
        return None
    sanitized = _DUMP_NAME_RE.sub("_", str(component_id)).strip("_")
    return sanitized or "component"


def _bytes_len(value: bytes | bytearray | memoryview) -> int:
    if isinstance(value, memoryview):
        return value.nbytes
    return len(value)
