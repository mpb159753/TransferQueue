# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import logging
import struct
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import zmq

_NP_DTYPE_MAP = {
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "float16": np.dtype("float16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "int16": np.dtype("int16"),
    "int8": np.dtype("int8"),
    "uint8": np.dtype("uint8"),
    "bool": np.dtype("bool"),
    "single": np.dtype("float32"),
    "double": np.dtype("float64"),
    "long": np.dtype("int64"),
    "int": np.dtype("int32"),
}


def get_numpy_dtype(dtype_str: str) -> np.dtype:
    if t := _NP_DTYPE_MAP.get(dtype_str):
        return t

    try:
        return np.dtype(dtype_str)
    except Exception:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


def serialize_batch(batch_data: Union[np.ndarray, List[np.ndarray]],
                    batch_lens: Optional[np.ndarray] = None,
                    pad_mode: Optional[str] = None,
                    save_dtype: str = None) -> Tuple[np.ndarray, List[int], str]:
    # 取第一个元素判断类型
    sample = batch_data[0]
    source_dtype = sample.dtype

    if save_dtype is not None:
        target_dtype = get_numpy_dtype(save_dtype)
    else:
        target_dtype = source_dtype

    # 判断是否为 2D 矩阵
    is_2d_matrix = isinstance(batch_data, np.ndarray) and batch_data.ndim == 2

    # 输入是 Tensor 转换来的整块 Numpy，直接操作 Numpy
    if is_2d_matrix:
        rows, cols = batch_data.shape

        # 不需要 Unpad
        if batch_lens is None:
            # 尝试 Zero-copy View
            if source_dtype == target_dtype and batch_data.flags['C_CONTIGUOUS']:
                final_buffer = batch_data.ravel()
            else:
                final_buffer = batch_data.astype(target_dtype).ravel()

            # 长度固定为列长
            length_list = [int(cols)] * rows
        # 需要 Unpad
        else:
            batch_lens = np.asarray(batch_lens).reshape(-1)
            # 校验 pad 输入
            if len(batch_lens) != rows:
                raise ValueError(f"Batch lens length ({len(batch_lens)}) must match number of rows ({rows})")
            # 利用 Broadcasting 生成布尔掩码
            col_indices = np.arange(cols).reshape(1, -1)
            if pad_mode == 'right_pad':
                mask = col_indices < batch_lens[:, None]
            elif pad_mode == 'left_pad':
                mask = col_indices >= (cols - batch_lens[:, None])
            else:
                raise ValueError("pad_mode must be 'right_pad' or 'left_pad'")

            #  Masking 直接提取有效数据并 Flatten
            valid_data = batch_data[mask]

            if source_dtype != target_dtype:
                final_buffer = valid_data.astype(target_dtype)
            else:
                final_buffer = valid_data
            # Meta 计算
            length_list = [int(l) for l in batch_lens]
    # 输入为 List[Tensor]
    else:
        if batch_lens is None:
            # 等长数组无 unpad 或变长数组时，直接获取需要的长度
            effective_lens = [x.size for x in batch_data]
        else:
            # 需要 unpad
            effective_lens = batch_lens

        # 一次性分配内存
        total_elements = sum(effective_lens)
        final_buffer = np.empty(total_elements, dtype=target_dtype)

        length_list = []
        curr_offset = 0

        for i, arr in enumerate(batch_data):
            length = int(effective_lens[i])
            length_list.append(length)
            flat_src = arr.ravel()
            src_len = flat_src.size

            # 仅当源数据长度与目标长度不一致时处理
            if batch_lens is not None and src_len != length:
                if pad_mode == 'right_pad':
                    flat_src = flat_src[:length]
                elif pad_mode == 'left_pad':
                    flat_src = flat_src[-length:]
                else:
                    raise ValueError(f"Length mismatch ({src_len} vs {length}) but valid pad_mode not provided.")
            # 写入 Buffer
            final_buffer[curr_offset: curr_offset + length] = flat_src
            curr_offset += length

    return final_buffer, length_list, target_dtype.name


def deserialize_column_from_frame(
        frame: zmq.Frame,
        dtype_str: str,
        lengths: List[int],
        copy: bool = False
) -> List[torch.Tensor]:
    if not lengths:
        return []

    # 1. 准备 Dtype 与 容量计算
    np_dtype = np.dtype(dtype_str)

    torch_dtype = get_numpy_dtype(dtype_str)

    total_elements = sum(lengths)
    src_view = np.frombuffer(frame, dtype=np_dtype)
    if src_view.size != total_elements:
        raise ValueError(
            f"Data size mismatch. Meta sums to {total_elements}, Frame has {src_view.size}"
        )
    if copy:
        # 触发一次 copy, list 内 Tensor 变为非只读
        slab_np = src_view.copy()
        # 将 ZMQ Buffer 拷入 PyTorch 连续内存
        slab = torch.from_numpy(slab_np)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*not writable.*")
            slab = torch.from_numpy(src_view)

    # 切分 list
    return list(torch.split(slab, lengths))


def get_no_pad_length(
        prompts: torch.Tensor,
        pad_id: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Compute the actual (unpadded) length of each prompt in the batch.

    Args:
        prompts: Tensor of shape (batch_size, seq_len).
        pad_id: Optional padding token ID. If provided, lengths are computed
                up to the first occurrence of pad_id; otherwise full length.

    Returns:
        A list of 1D tensor containing the length (number of non-pad tokens) of the corresponding prompt.
    """
    lengths: List[torch.Tensor] = []
    batch_size, seq_len = prompts.size()

    for i in range(batch_size):
        tokens = prompts[i]
        if pad_id is not None:
            # find first pad token
            pads = torch.nonzero(tokens == pad_id, as_tuple=False)
            if pads.numel() > 0:
                length = pads[0].item()
            else:
                length = seq_len
        else:
            length = seq_len

        lengths.append(torch.tensor([length]))

    return lengths

def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免 Ray Worker 重用导致 Handler 堆积
    if not logger.handlers:
        # 显式输出到 stdout，Ray Log Monitor 会自动捕获这个流
        sh = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - (%(name)s) - %(levelname)s - %(message)s'
        )
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.propagate = False
    return logger


def assign_idx_for_prompt(prompt_id, n_samples):
    start = prompt_id * n_samples
    end = start + n_samples
    return list(range(start, end))