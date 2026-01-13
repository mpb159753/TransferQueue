import asyncio
import logging
import math
import os
import sys
import time
import gc
from pathlib import Path
import numpy as np

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict, NonTensorStack

# Datasets & Image processing
import io
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import (
    AsyncTransferQueueClient,
    SimpleStorageUnit,
    TransferQueueController,
    process_zmq_server_info,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 配置部分 ---
HEAD_NODE_IP = "192.168.100.1"
WORKER_NODE_IP = "192.168.100.1"

DATA_PATH = "/home/z30079621/data_v0.8_visual_toolbox_v2.parquet"
MODEL_PATH = "/home/z30079621/Qwen2.5-VL-3B-Instruct"

config_str = """
  global_batch_size: 2000
  seq_length: 8192
  field_num: 18
  num_global_batch: 20 
  num_data_storage_units: 8
  num_data_controllers: 1
"""
dict_conf = OmegaConf.create(config_str)

# 全局缓存
_cached_real_data = None

try:
    from verl.utils.torch_functional import postprocess_data as verl_F_postprocess_data
    from verl.utils.model import compute_position_id_with_mask
except ImportError:
    logger.warning("Verl module not found. Ensure it is installed if needed for data processing.")


# --- Dataset Class ---
class CustomRLHFDataset(Dataset):
    def __init__(
            self,
            dataframe,
            prompt_key,
            image_key,
            processor=None,
            tokenizer=None,
            max_prompt_length=2048,
            truncation="right",
            return_multi_modal_inputs=False,
            return_raw_chat=False,
            return_full_prompt=False,
            **kwargs
    ):
        self.dataframe = dataframe
        self.prompt_key = prompt_key
        self.image_key = image_key
        self.processor = processor
        self.tokenizer = tokenizer or (processor.tokenizer if processor else None)
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.return_multi_modal_inputs = return_multi_modal_inputs
        self.return_raw_chat = return_raw_chat
        self.return_full_prompt = return_full_prompt

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, row: dict) -> list:
        prompt = row.get(self.prompt_key)
        if isinstance(prompt, list):
            user_text = " ".join(str(p).strip() for p in prompt)
        elif isinstance(prompt, str):
            user_text = prompt.strip()
        else:
            user_text = ""

        image_data = row.get(self.image_key)
        has_image = image_data is not None and str(image_data).strip() != ""

        if has_image:
            content = [
                {"type": "image"},
                {"type": "text", "text": user_text}
            ]
        else:
            content = user_text

        return [{"role": "user", "content": content}]

    def __getitem__(self, item):
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        user_content = row_dict[self.prompt_key]
        row_dict[self.prompt_key] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You can call functions to assist with the user query. "
                    "Important: You must call only one function at a time. After each function call, "
                    "wait for the execution result before making the next function call if needed."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [Image.open(io.BytesIO(img["bytes"])) for img in row_dict_images]
                multi_modal_data["image"] = images

            model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            row_dict["multi_modal_data"] = multi_modal_data

            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]

        input_ids, attention_mask = verl_F_postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
            left_pad=True,
            truncation=self.truncation,
        )

        try:
            if self.processor is not None and "Qwen2VLImageProcessor" in str(type(self.processor.image_processor)):
                from verl.models.transformers.qwen2_vl import get_rope_index
                position_ids = [
                    get_rope_index(
                        self.processor,
                        input_ids=input_ids[0],
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                        attention_mask=attention_mask[0],
                    )
                ]
            else:
                position_ids = compute_position_id_with_mask(attention_mask)
        except Exception:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        if self.tokenizer:
            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        else:
            raw_prompt_ids = self.processor.tokenizer.encode(raw_prompt, add_special_tokens=False)

        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        index = row_dict.get("extra_info", {}).get("index", item)
        row_dict["index"] = index
        return row_dict


# --- Data Loading Utils ---
def _load_real_multi_modal_data(batch_size):
    global _cached_real_data
    if _cached_real_data is not None and len(_cached_real_data) >= batch_size:
        return _cached_real_data[:batch_size]

    parquet_path = DATA_PATH
    logger.info(f"Loading dataset from {parquet_path}")
    dataset = load_dataset("parquet", data_files=parquet_path, split="train")
    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df)} samples.")

    from transformers import AutoProcessor
    model_path = MODEL_PATH
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    custom_dataset = CustomRLHFDataset(
        dataframe=df,
        prompt_key="prompt",
        image_key="images",
        processor=processor,
        tokenizer=None,
        max_prompt_length=2048,
        truncation="right",
        return_multi_modal_inputs=True,
        return_raw_chat=True,
        return_full_prompt=True,
    )
    return custom_dataset


def create_complex_test_case(batch_size=None, seq_length=None, field_num=None):
    if None in (batch_size, seq_length, field_num):
        raise ValueError("batch_size, seq_length, field_num must be provided.")

    real_multi_modal_inputs = _load_real_multi_modal_data(batch_size)
    if len(real_multi_modal_inputs) < batch_size:
        logger.warning(f"Requested batch_size={batch_size}, but only {len(real_multi_modal_inputs)} samples available.")
        batch_size = len(real_multi_modal_inputs)

    col_names = real_multi_modal_inputs[0].keys()
    fields = {col: [] for col in col_names}

    # Load into memory
    for i in range(batch_size):
        for k, v in real_multi_modal_inputs[i].items():
            fields[k].append(v)

    # Calculate size
    def calculate_size(obj, seen=None):
        if seen is None: seen = set()
        obj_id = id(obj)
        if obj_id in seen: return 0
        seen.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, torch.Tensor):
            size += obj.element_size() * obj.numel()
        elif isinstance(obj, dict):
            for k, v in obj.items():
                size += calculate_size(k, seen) + calculate_size(v, seen)
        elif isinstance(obj, (list, tuple)):
            for item in obj: size += calculate_size(item, seen)
        return size

    total_size = calculate_size(fields) / 1024 ** 3
    logger.info(f"Estimated Total Data Size: {total_size:.4f} GB")

    processed_fields = {}
    for key, value_list in fields.items():
        if not value_list: continue
        first_item = value_list[0]

        if isinstance(first_item, torch.Tensor):
            shapes = [t.shape for t in value_list]
            ndims = {len(s) for s in shapes}

            # Case 1: 标准 Batch (Shape完全一致)
            if len(ndims) == 1 and all(s == shapes[0] for s in shapes):
                processed_fields[key] = torch.stack(value_list, dim=0)

            # Case 2: Nested Tensor (Shape 或 Ndim 不一致)
            else:
                try:
                    # 修复：计算最大维度
                    max_ndim = 0
                    for t in value_list:
                        if t.ndim > max_ndim:
                            max_ndim = t.ndim

                    # 确保目标维度至少为 1 (处理 scalar)
                    target_ndim = max(max_ndim, 1)

                    normalized_tensors = []
                    for t in value_list:
                        curr = t
                        # 循环升维直到匹配 target_ndim，解决 "dimension 1 vs 2" 错误
                        while curr.ndim < target_ndim:
                            curr = curr.unsqueeze(0)
                        normalized_tensors.append(curr)

                    processed_fields[key] = torch.nested.nested_tensor(normalized_tensors)

                except Exception as e:
                    # Fallback: 如果数据实在太乱（如 dtype 不一致），使用 NonTensorStack
                    logger.warning(f"Field '{key}' fallback to NonTensorStack due to: {str(e)}")
                    processed_fields[key] = NonTensorStack(*value_list)
        else:
            processed_fields[key] = NonTensorStack(*value_list)

    prompt_batch = TensorDict(
        processed_fields,
        batch_size=(batch_size,),
        device=None,
    )
    return prompt_batch, total_size


# --- TQ Tester with Profile Sync ---
class TQBandwidthTester:
    def __init__(self, config):
        self.config = config
        self.data_system_client = None
        self.data_system_storage_units = {}
        self.data_system_controller = None

    def sync_stage(self, stage_name):
        ready_flag = Path(f"{stage_name}_ready.flag")
        start_flag = Path(f"{stage_name}_start.flag")

        if start_flag.exists():
            start_flag.unlink()

        logger.info(f"[{stage_name}] Signal READY. Waiting for Profiler START...")
        ready_flag.touch()

        while not start_flag.exists():
            time.sleep(0.1)

        logger.info(f"[{stage_name}] START Signal received. Proceeding...")
        if ready_flag.exists():
            ready_flag.unlink()

    def _initialize_data_system(self):
        logger.info("Initializing Data System Actors...")
        total_storage_size = (self.config.global_batch_size * self.config.num_global_batch)
        self.data_system_storage_units = {}

        target_nodes = [HEAD_NODE_IP, WORKER_NODE_IP]

        for storage_unit_rank in range(self.config.num_data_storage_units):
            target_ip = target_nodes[storage_unit_rank % len(target_nodes)]

            storage_node = SimpleStorageUnit.options(
                num_cpus=4,
                resources={f"node:{target_ip}": 0.001},
                runtime_env={"env_vars": {"OMP_NUM_THREADS": "2"}},
            ).remote(
                storage_unit_size=math.ceil(total_storage_size / self.config.num_data_storage_units)
            )
            self.data_system_storage_units[storage_unit_rank] = storage_node

        self.data_system_controller = TransferQueueController.remote()

        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)
        self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        tq_config = OmegaConf.create({}, flags={"allow_objects": True})
        tq_config.controller_info = self.data_system_controller_info
        tq_config.storage_unit_infos = self.data_system_storage_unit_infos
        current_round_config = OmegaConf.merge(tq_config, self.config)

        self.data_system_client = AsyncTransferQueueClient(
            client_id='Trainer_Profile',
            controller_info=self.data_system_controller_info
        )
        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager",
                                                           config=current_round_config)
        logger.info(f"Data System Initialized.")

    def _teardown_data_system(self):
        logger.info("Tearing down Data System...")
        if self.data_system_storage_units:
            for unit in self.data_system_storage_units.values():
                ray.kill(unit)
            self.data_system_storage_units = {}

        if self.data_system_controller:
            ray.kill(self.data_system_controller)
            self.data_system_controller = None

        gc.collect()

    def run_profile_test(self):
        logger.info("Generating COMPLEX MULTI-MODAL data (ONCE)...")
        test_data, total_data_size_gb = create_complex_test_case(
            batch_size=self.config.global_batch_size,
            seq_length=self.config.seq_length,
            field_num=self.config.field_num
        )
        logger.info(f"Total Size: {total_data_size_gb:.4f} GB. Data generation done.")

        self._initialize_data_system()
        partition_id = "profile_partition"

        # --- STAGE 1: PUT ---
        self.sync_stage("put_phase")

        logger.info("Starting PUT...")
        start_put = time.time()
        asyncio.run(self.data_system_client.async_put(data=test_data, partition_id=partition_id))
        end_put = time.time()

        put_tp = (total_data_size_gb * 8) / (end_put - start_put)
        logger.info(f"PUT Done. TP: {put_tp:.4f} Gb/s")

        # --- STAGE 2: META ---
        self.sync_stage("put_done_meta_phase")

        logger.info("Getting Meta (Un-profiled)...")
        prompt_meta = asyncio.run(self.data_system_client.async_get_meta(
            data_fields=list(test_data.keys()),
            batch_size=test_data.size(0),
            partition_id=partition_id,
            task_name='generate_sequences',
        ))

        # --- STAGE 3: GET ---
        self.sync_stage("get_phase")

        logger.info("Starting GET...")
        start_get = time.time()
        asyncio.run(self.data_system_client.async_get_data(prompt_meta))
        end_get = time.time()

        get_tp = (total_data_size_gb * 8) / (end_get - start_get)
        logger.info(f"GET Done. TP: {get_tp:.4f} Gb/s")

        self._teardown_data_system()
        logger.info("Profile Test Completed.")


def main():
    ray.init(resources={"node:192.168.100.1": 100})
    logger.info("Starting Multi-Modal TQ Profile Script")
    tester = TQBandwidthTester(config=dict_conf)
    tester.run_profile_test()


if __name__ == "__main__":
    main()