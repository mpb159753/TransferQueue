# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import itertools
import threading
from typing import Dict, Tuple, List, Any, Iterator

import numpy as np
import zmq

from tq_utils import setup_logger, assign_idx_for_prompt


class MemorySegment:
    """
    物理存储单元：持有 zmq.Frame 并管理引用计数
    """
    __slots__ = ('_frame', '_buffer', 'ref_count')

    def __init__(self, frame: zmq.Frame, initial_ref: int):
        self._frame = frame
        # 创建 memoryview，实现零拷贝访问 C++ 底层内存
        self._buffer = memoryview(frame)
        self.ref_count = initial_ref

    @property
    def buffer(self) -> memoryview:
        return self._buffer

    def release(self, count: int = 1) -> bool:
        """
        减少引用计数。
        返回 True 表示资源已被彻底释放（引用计数归零）。
        """
        self.ref_count -= count
        if self.ref_count <= 0:
            # 显式解除引用，允许 GC 回收底层的 zmq.Frame
            self._frame = None
            self._buffer = None
            return True
        return False


class ExperienceTable:
    """
    高性能经验回放表：支持列式存储、批量写入和零拷贝拼接读取。
    """

    def __init__(self, n_samples_per_prompt: int, experience_columns: List[str]):
        if n_samples_per_prompt <= 0:
            raise ValueError("n_samples_per_prompt must be a positive integer")
        self.n_samples_per_prompt = n_samples_per_prompt
        if not experience_columns:
            raise ValueError("experience_columns must be provided")
        self.experience_columns = experience_columns
        self.logger = setup_logger("ExperienceTable")
        self.logger.debug(f"ExperienceTable initialized successfully, n_samples_per_prompt={self.n_samples_per_prompt}")

        # Column Name -> Global ID -> (MemorySegment, byte_offset, byte_length)
        self.indices: Dict[str, Dict[int, Tuple[MemorySegment, int, int]]] = {}
        self.col_metas: Dict[str, Tuple[int, str]] = {}
        self._lock = threading.Lock()
        self.owned_groups = set()

    @staticmethod
    def _get_item_size(dtype_str: str) -> int:
        return np.dtype(dtype_str).itemsize

    def put_batch(self,
                  global_ids: List[int],
                  col_order: List[str],
                  col_inputs_meta: Dict[str, Any],
                  payload_frames: List[zmq.Frame],
                  is_prompt: bool = False):
        batch_size = len(global_ids)
        if batch_size == 0:
            return

        cols_info = col_inputs_meta["columns"]
        validated_metas = []

        if len(payload_frames) < len(col_order):
            missing_col = col_order[len(payload_frames)]
            raise ValueError(f"Missing frame for column {missing_col}")

        # 校验收到结果是否正确
        for i, col_name in enumerate(col_order):
            frame = payload_frames[i]
            col_info = cols_info[col_name]
            elem_lengths = col_info["lengths"]
            dtype_str = col_info["dtype"]

            if col_name in self.col_metas:
                item_size, _ = self.col_metas[col_name]
            else:
                item_size = np.dtype(dtype_str).itemsize
                self.col_metas[col_name] = (item_size, dtype_str)

            total_elements = sum(elem_lengths)
            expected_bytes = total_elements * item_size

            if expected_bytes != len(frame.bytes):
                raise ValueError(
                    f"Size Mismatch '{col_name}': Meta {expected_bytes} bytes != Frame {len(frame.bytes)} bytes"
                )

            if len(elem_lengths) != batch_size:
                raise ValueError(f"Length mismatch for col '{col_name}'")

            validated_metas.append((col_name, frame, item_size, elem_lengths, col_info["dtype"]))

        # 如果是 Prompt，一份物理内存被 N 个逻辑 ID 共享，引用计数 * N
        ref_multiplier = self.n_samples_per_prompt if is_prompt else 1
        initial_ref_count = batch_size * ref_multiplier
        with self._lock:
            for col_name, frame, item_size, elem_lengths, dtype_str in validated_metas:
                if col_name not in self.col_metas:
                    self.col_metas[col_name] = (item_size, dtype_str)

                if col_name not in self.indices:
                    self.indices[col_name] = {}

                col_index = self.indices[col_name]

                # 初始引用计数 = 样本数 * N
                segment = MemorySegment(frame, initial_ref=initial_ref_count)
                current_byte_offset = 0

                for gid, n_elems in zip(global_ids, elem_lengths):
                    byte_len = n_elems * item_size
                    entry = (segment, current_byte_offset, byte_len)

                    if is_prompt:
                        target_ids_iter = assign_idx_for_prompt(gid, self.n_samples_per_prompt)
                    else:
                        target_ids_iter = (gid,)

                    # 执行写入
                    for target_id in target_ids_iter:
                        if target_id in col_index:
                            # 覆盖旧数据，释放旧引⽤
                            old_entry = col_index[target_id]
                            old_segment = old_entry[0]
                            old_segment.release()

                        col_index[target_id] = entry

                    current_byte_offset += byte_len

    def get_batch(self, target_gids: List[int], target_cols: List[str]) -> Tuple[Dict[str, Any], List[bytes]]:
        result_meta = {
            "indexes": target_gids,
            "columns": {col: {"lengths": []} for col in target_cols},
            "order": target_cols
        }

        result_frames = []

        with self._lock:
            for col_name in target_cols:
                col_static_meta = self.col_metas.get(col_name)
                if not col_static_meta:
                    raise KeyError(f"Column {col_name} not found.")

                item_size, dtype = col_static_meta
                result_meta["columns"][col_name]["dtype"] = dtype

                col_index = self.indices.get(col_name)
                if col_index is None:
                    raise KeyError(f"Column index {col_name} is empty or missing.")

                total_bytes = 0
                batch_entries = []
                meta_lengths_append = result_meta["columns"][col_name]["lengths"].append

                for gid in target_gids:
                    entry = col_index.get(gid)
                    if entry is None:
                        raise KeyError(f"Global ID {gid} missing in column {col_name}.")
                    batch_entries.append(entry)

                    byte_len = entry[2]
                    total_bytes += byte_len
                    meta_lengths_append(byte_len // item_size)

                dest_buffer = bytearray(total_bytes)
                cursor = 0

                for seg, off, byte_len in batch_entries:
                    dest_buffer[cursor: cursor + byte_len] = seg.buffer[off: off + byte_len]
                    cursor += byte_len

                result_frames.append(dest_buffer)

        return result_meta, result_frames

    def prune(self, gids_to_remove: List[int]):
        """
        清理逻辑：遍历所有列，尝试移除指定的 GID。
        """
        with self._lock:
            for col_name, col_index in self.indices.items():
                for gid in gids_to_remove:
                    if gid in col_index:
                        entry = col_index.pop(gid)
                        segment = entry[0]
                        segment.release()
            for gid in gids_to_remove:
                group_id = gid // self.n_samples_per_prompt
                self.owned_groups.discard(group_id)

    def clear(self):
        """
        1. 清理整个 ExperienceTable
        2. 显式释放所有 MemorySegment 的引用计数 (从而释放 zmq.frame)
        3. 重置元数据
        """
        with self._lock:
            for col_index in self.indices.values():
                for entry in col_index.values():
                    # entry: (segment, offset, length)
                    segment = entry[0]
                    segment.release()

            self.indices.clear()
            self.col_metas.clear()
            self.owned_groups.clear()
