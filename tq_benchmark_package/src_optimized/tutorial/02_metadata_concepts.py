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

"""
TransferQueue Tutorial 2: Columnar BatchMeta Concepts

This script introduces the columnar metadata system in TransferQueue.
BatchMeta uses a column-oriented structure for efficient batch operations:
- global_indexes: List of unique sample identifiers
- partition_ids: List of partition assignments
- fields: Dict of field properties (dtype, shape, production_status per sample)

Key Benefits:
- Vectorized operations (no per-sample loops)
- Reduced object creation (no SampleMeta/FieldMeta classes)
- Efficient slicing and concatenation
"""

import os
import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings(
    action="ignore",
    message=r"The PyTorch API of nested tensors is in prototype stage*",
    category=UserWarning,
    module=r"torch\.nested",
)

warnings.filterwarnings(
    action="ignore",
    message=r"Tip: In future versions of Ray, Ray will no longer override accelerator visible "
    r"devices env var if num_gpus=0 or num_gpus=None.*",
    category=FutureWarning,
    module=r"ray\._private\.worker",
)


import ray  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tensordict import TensorDict  # noqa: E402

# Add the parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import (  # noqa: E402
    SimpleStorageUnit,
    TransferQueueClient,
    TransferQueueController,
    process_zmq_server_info,
)
from transfer_queue.metadata import BatchMeta  # noqa: E402
from transfer_queue.utils.enum_utils import ProductionStatus  # noqa: E402

# Configure Ray
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEBUG"] = "1"


def demonstrate_columnar_batch_meta():
    """
    Demonstrate the columnar BatchMeta structure.
    """
    print("=" * 80)
    print("Columnar BatchMeta - Column-Oriented Metadata Structure")
    print("=" * 80)

    print("BatchMeta uses columnar storage for efficiency:")
    print("- global_indexes: [0, 1, 2, ...] - unique sample IDs")
    print("- partition_ids: ['p0', 'p0', ...] - partition assignments")
    print("- fields: {'field_name': {'dtype': [...], 'shape': [...], ...}}")

    # Example 1: Create a BatchMeta
    print("\n[Example 1] Creating a columnar BatchMeta...")
    batch_size = 5
    batch = BatchMeta(
        global_indexes=list(range(batch_size)),
        partition_ids=["train_0"] * batch_size,
        fields={
            "input_ids": {
                "dtype": [torch.int64] * batch_size,
                "shape": [(512,)] * batch_size,
                "production_status": [ProductionStatus.READY_FOR_CONSUME] * batch_size,
            },
            "attention_mask": {
                "dtype": [torch.int64] * batch_size,
                "shape": [(512,)] * batch_size,
                "production_status": [ProductionStatus.READY_FOR_CONSUME] * batch_size,
            },
        },
    )
    print(f"✓ Created batch with {len(batch)} samples")
    print(f"  Global indexes: {batch.global_indexes}")
    print(f"  Field names: {batch.field_names}")
    print(f"  Size: {batch.size}")

    # Example 2: Add extra_info
    print("\n[Example 2] Adding batch-level information...")
    batch.set_extra_info("epoch", 1)
    batch.set_extra_info("batch_idx", 0)
    print(f"✓ Extra info: {batch.get_all_extra_info()}")

    # Example 3: Chunk a batch
    print("\n[Example 3] Chunking a batch into parts...")
    chunks = batch.chunk(3)
    print(f"✓ Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} samples, indexes={chunk.global_indexes}")

    # Example 4: Select specific fields
    print("\n[Example 4] Selecting specific fields...")
    selected_batch = batch.select_fields(["input_ids"])
    print(f"✓ Selected fields: {selected_batch.field_names}")
    print(f"  Original fields: {batch.field_names}")

    # Example 5: Select specific samples
    print("\n[Example 5] Selecting specific samples...")
    selected_samples = batch.select_samples([0, 2, 4])
    print(f"✓ Selected samples at indexes: {selected_samples.global_indexes}")

    # Example 6: Reorder samples
    print("\n[Example 6] Reordering samples...")
    print(f"  Original order: {batch.global_indexes}")
    batch.reorder([4, 3, 2, 1, 0])
    print(f"  After reorder: {batch.global_indexes}")

    # Example 7: Concat batches
    print("\n[Example 7] Concatenating batches...")
    batch1 = BatchMeta(
        global_indexes=[0, 1, 2],
        partition_ids=["train_0"] * 3,
        fields={
            "input_ids": {
                "dtype": [torch.int64] * 3,
                "shape": [(512,)] * 3,
                "production_status": [1] * 3,
            },
        },
    )
    batch2 = BatchMeta(
        global_indexes=[3, 4, 5],
        partition_ids=["train_0"] * 3,
        fields={
            "input_ids": {
                "dtype": [torch.int64] * 3,
                "shape": [(512,)] * 3,
                "production_status": [1] * 3,
            },
        },
    )
    concatenated = BatchMeta.concat([batch1, batch2])
    print(f"✓ Concatenated {len(batch1)} + {len(batch2)} = {len(concatenated)} samples")
    print(f"  Global indexes: {concatenated.global_indexes}")

    # Example 8: Union batches (merging fields)
    print("\n[Example 8] Unioning batches (different fields, same samples)...")
    batch_with_input = BatchMeta(
        global_indexes=[0, 1, 2],
        partition_ids=["train_0"] * 3,
        fields={
            "input_ids": {
                "dtype": [torch.int64] * 3,
                "shape": [(512,)] * 3,
                "production_status": [1] * 3,
            },
        },
    )
    batch_with_output = BatchMeta(
        global_indexes=[0, 1, 2],
        partition_ids=["train_0"] * 3,
        fields={
            "responses": {
                "dtype": [torch.int64] * 3,
                "shape": [(128,)] * 3,
                "production_status": [1] * 3,
            },
        },
    )
    print(f"  Batch1 has fields: {batch_with_input.field_names}")
    print(f"  Batch2 has fields: {batch_with_output.field_names}")

    unioned_batch = batch_with_input.union(batch_with_output)
    print("✓ Union successful!")
    print(f"  Unioned fields: {unioned_batch.field_names}")

    print("\n" + "=" * 80)
    print("concat vs union:")
    print("  - concat: Combines batches with SAME structure into one larger batch")
    print("  - union: Merges fields from batches with IDENTICAL sample indexes")
    print("=" * 80)


def demonstrate_real_workflow():
    """
    Demonstrate a realistic workflow with actual TransferQueue interaction.
    """
    print("=" * 80)
    print("Real Workflow: Interacting with TransferQueue")
    print("=" * 80)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Setup TransferQueue
    config = OmegaConf.create(
        {
            "num_data_storage_units": 2,
        }
    )

    storage_units = {}
    for i in range(config["num_data_storage_units"]):
        storage_units[i] = SimpleStorageUnit.remote(storage_unit_size=100)

    controller = TransferQueueController.remote()
    controller_info = process_zmq_server_info(controller)
    storage_unit_infos = process_zmq_server_info(storage_units)

    client = TransferQueueClient(
        client_id="TutorialClient",
        controller_info=controller_info,
    )

    tq_config = OmegaConf.create({}, flags={"allow_objects": True})
    tq_config.controller_info = controller_info
    tq_config.storage_unit_infos = storage_unit_infos
    config = OmegaConf.merge(tq_config, config)

    client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=config)

    print("[Step 1] Putting data into TransferQueue...")
    input_ids = torch.randint(0, 1000, (8, 512))
    attention_mask = torch.ones(8, 512)

    data_batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        batch_size=8,
    )

    partition_id = "demo_partition"
    batch_meta = client.put(data=data_batch, partition_id=partition_id)
    print(f"✓ Put {data_batch.batch_size[0]} samples into partition '{partition_id}'")

    print("[Step 2] Getting metadata from TransferQueue...")
    batch_meta = client.get_meta(
        data_fields=["input_ids", "attention_mask"],
        batch_size=8,
        partition_id=partition_id,
        task_name="demo_task",
    )
    print("✓ Got BatchMeta from TransferQueue:")
    print(f"  Number of samples: {len(batch_meta)}")
    print(f"  Global indexes: {batch_meta.global_indexes}")
    print(f"  Field names: {batch_meta.field_names}")
    print(f"  Partition IDs: {batch_meta.partition_ids[:3]}...")

    print("[Step 3] Retrieve samples with specific fields...")
    selected_meta = batch_meta.select_fields(["input_ids"])
    print("✓ Selected 'input_ids' field only:")
    print(f"  New field names: {selected_meta.field_names}")
    retrieved_data = client.get_data(selected_meta)
    print(f"  Retrieved data keys: {list(retrieved_data.keys())}")

    print("[Step 4] Select specific samples...")
    partial_meta = batch_meta.select_samples([0, 2, 4, 6])
    print("✓ Selected samples at indices [0, 2, 4, 6]:")
    print(f"  New global indexes: {partial_meta.global_indexes}")
    retrieved_data = client.get_data(partial_meta)
    print(f"  Retrieved data batch size: {retrieved_data.batch_size}")

    print("[Step 5] Demonstrate chunk operation...")
    chunks = batch_meta.chunk(2)
    print(f"✓ Chunked into {len(chunks)} parts:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} samples, indexes={chunk.global_indexes}")

    # Cleanup
    client.clear_partition(partition_id=partition_id)
    client.close()
    ray.shutdown()
    print("✓ Partition cleared and resources cleaned up")


def main():
    """Main function to run the tutorial."""
    print("=" * 80)
    print(
        textwrap.dedent(
            """
        TransferQueue Tutorial 2: Columnar Metadata System

        This script introduces the columnar metadata system in TransferQueue.

        BatchMeta Structure:
        - global_indexes: [0, 1, 2, ...] - unique sample identifiers
        - partition_ids: ['p0', 'p0', ...] - partition assignments
        - fields: {'name': {'dtype': [...], 'shape': [...], 'production_status': [...]}}

        Key Operations:
        - chunk: Split batch into smaller parts
        - concat: Combine multiple batches
        - union: Merge fields from batches with same samples
        - select_fields: Keep only specified fields
        - select_samples: Keep only specified samples
        - reorder: Change sample order
        """
        )
    )
    print("=" * 80)

    try:
        demonstrate_columnar_batch_meta()
        demonstrate_real_workflow()

        print("=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("Key Takeaways:")
        print("1. BatchMeta uses columnar storage (no SampleMeta/FieldMeta)")
        print("2. Fields are stored as dicts: {'dtype': [...], 'shape': [...], ...}")
        print("3. Operations are vectorized for efficiency")
        print("4. concat combines batches; union merges fields of same samples")

        # Cleanup
        ray.shutdown()
        print("\n✓ Cleanup complete")

    except Exception as e:
        print(f"Error during tutorial: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
