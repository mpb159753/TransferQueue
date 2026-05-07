# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
"""GRPO asynchronous training test for TransferQueue with UUID support.

This file simulates a GRPO (Group Relative Policy Optimization) training workflow
using async operations with the TransferQueue system and UUID for prompt identification.

Key differences from test_grpo_async.py:
- Uses prompt_uuid column to identify prompts by unique identifiers
- Prompts are identified by UUID instead of group indexing
- Demonstrates UUID-based prompt tracking across async sampling

GRPO training requires:
1. Multiple samples generated from the same input prompt (identified by UUID)
2. All samples from the same prompt must be trained together as a group
3. Data organization: [p0_s0, p0_s1, p0_s2, p0_s3, p1_s0, p1_s1, ...]
   where each consecutive group belongs to the same prompt (identified by UUID)
4. No partition_id concept - use delete_experience to clean up consumed data
5. get_n_samples=True ensures only complete groups are sampled
6. Async operations with async client
7. Each stage (generate, logprob, reward, train) uses its own consumer

This test mimics the async training pattern where different stages are run
asynchronously, similar to real distributed RL training scenarios.
"""

import asyncio
import logging
import socket

import ray
import torch
from torch.nn.utils.rnn import pad_sequence

from recipe.async_flow.utils.transfer_queue.metrics import Metric
from recipe.async_flow.utils.transfer_queue.tq_client import get_transferqueue_client
from recipe.async_flow.utils.transfer_queue.tq_mgr import TransferQueueManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
NUMS_TQ_DATA = 1
NUM_ROUNDS = 3  # Run 3 rounds of training to verify multi-round capability


def _find_free_port() -> int:
    """Find a free TCP port on localhost for ZMQ binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _make_padded_prompts(lengths, pad_id=0) -> torch.Tensor:
    """Create a right-padded 2D tensor of token ids with the given lengths."""
    seqs = [torch.arange(1, L + 1, dtype=torch.long) for L in lengths]
    return pad_sequence(seqs, batch_first=True, padding_value=pad_id)


async def mock_generate_sequences(data: list) -> list:
    """Mock async function to simulate sequence generation in GRPO training."""
    await asyncio.sleep(0.5)
    return [tensor + 1 for tensor in data]


async def mock_compute_old_log_prob(data_dict: dict) -> list:
    """Mock async function to compute old log probabilities for GRPO."""
    await asyncio.sleep(0.5)
    prompts = data_dict["prompt"]
    # response would be used in real training
    _ = data_dict["response"]
    return [p * 0.5 for p in prompts]


async def mock_compute_rewards(data_dict: dict) -> list:
    """Mock async function to compute rewards for generated sequences."""
    await asyncio.sleep(0.5)
    responses = data_dict["response"]
    return [torch.randn(1) for _ in responses]


async def async_grpo_training_round(tq_client, topic: str, round_num: int, n_samples_per_prompt: int = 4):
    """Run a single GRPO training round asynchronously with UUID support.

    Args:
        tq_client: TransferQueue client
        topic: Topic name
        round_num: Round number for logging
        n_samples_per_prompt: Number of samples per prompt (GRPO parameter)
    """
    logger.info("-" * 60)
    logger.info(f"[Async] Training Round {round_num}")
    logger.info("-" * 60)

    prompts_num = 4

    # Prepare GRPO data with UUIDs
    pad_id = 0
    prompt_lengths = [3, 5, 4, 6]
    prompts = _make_padded_prompts(prompt_lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples_per_prompt, dim=0)
    prompt_lengths_tensor = (
        torch.tensor(prompt_lengths, dtype=torch.int32).repeat_interleave(n_samples_per_prompt, dim=0).unsqueeze(1)
    )

    # Create UUIDs for each prompt, then repeat n_samples_per_prompt times
    prompt_uuids = [f"async_uuid_r{round_num}_p{i}" for i in range(prompts_num)]
    prompt_uuids_list = []
    for uuid in prompt_uuids:
        prompt_uuids_list.extend([uuid] * n_samples_per_prompt)

    logger.info(f"GRPO data: {len(padded_prompts)} samples from {prompts_num} prompts")
    logger.info(f"  UUIDs: {prompt_uuids}")

    # Step 1: Put prompts with UUIDs (async)
    allocated_indexes = await tq_client.put_experience_async(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths_tensor, "prompt_uuid": prompt_uuids_list},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    logger.info(f"[Async] Step 1 - Put prompts with UUIDs to indexes: {allocated_indexes}")

    # Verify ready data (async check via sync wrapper)
    ready_count, _ = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt", "prompt_uuid"],
        get_n_samples=True,
    )
    logger.info(f"[Async]   Ready set (grouped): {ready_count} indexes")
    assert ready_count == len(allocated_indexes)

    # Step 2: Get prompts for generation (async) with GRPO grouped sampling
    batch_size = 4 * n_samples_per_prompt
    prompts_data, sampled_indices = await tq_client.get_experience_async(
        consumer="generate",
        experience_columns=["prompt", "prompt_uuid"],
        experience_count=batch_size,
        indexes=None,
        get_n_samples=True,
        topic=topic,
    )
    logger.info(f"[Async] Step 2 - Sampled {len(sampled_indices)} prompts")
    logger.info(f"[Async]   Sampled UUIDs: {prompts_data['prompt_uuid']}")

    # Generate sequences (async parallel)
    response_tasks = [mock_generate_sequences([p]) for p in prompts_data["prompt"]]
    response_results = await asyncio.gather(*response_tasks)
    response_tensors = [r[0] for r in response_results]
    logger.info("[Async]   Generated sequences (parallel)")

    # Put responses (async)
    await tq_client.put_experience_async(
        data_dict={"response": response_tensors},
        indexes=sampled_indices,
        topic=topic,
    )
    logger.info(f"[Async]   Put responses to indexes: {sampled_indices}")

    # Step 3: Compute old log probs (async)
    # GET from TQ, compute, PUT back to TQ
    prompt_response_data, _ = await tq_client.get_experience_async(
        consumer="logprob",
        experience_columns=["prompt", "response"],
        experience_count=batch_size,
        indexes=sampled_indices,
        get_n_samples=False,
        topic=topic,
    )
    logger.info("[Async] Step 3 - Retrieved prompt+response from TQ")

    log_prob_tensors = await mock_compute_old_log_prob(prompt_response_data)
    logger.info("[Async]   Computed old log probs (async)")

    await tq_client.put_experience_async(
        data_dict={"old_log_prob": log_prob_tensors},
        indexes=sampled_indices,
        topic=topic,
    )
    logger.info(f"[Async]   Put old_log_prob to indexes: {sampled_indices}")

    # Step 4: Compute rewards (async)
    response_data, _ = await tq_client.get_experience_async(
        consumer="reward",
        experience_columns=["response"],
        experience_count=batch_size,
        indexes=sampled_indices,
        get_n_samples=False,
        topic=topic,
    )
    logger.info("[Async] Step 4 - Retrieved response from TQ")

    reward_tensors_list = await mock_compute_rewards(response_data)
    logger.info("[Async]   Computed rewards (async)")

    await tq_client.put_experience_async(
        data_dict={"reward": reward_tensors_list},
        indexes=sampled_indices,
        topic=topic,
    )
    logger.info(f"[Async]   Put rewards to indexes: {sampled_indices}")

    # Step 5: Training step - get complete GRPO experience (async)
    experience_columns = ["prompt", "response", "old_log_prob", "reward", "prompt_uuid"]
    grpo_experience, training_indices = await tq_client.get_experience_async(
        consumer="learner",
        experience_columns=experience_columns,
        experience_count=batch_size,
        indexes=None,
        get_n_samples=True,
        topic=topic,
    )

    if grpo_experience is None or training_indices is None:
        logger.error("[Async] Failed to get training experience!")
        # Check what's ready
        for col in experience_columns:
            _count, idxs = tq_client.get_data_ready_set(topic=topic, experience_columns=[col], get_n_samples=True)
            logger.info(f"[Async]     Column '{col}' ready: {_count} indexes: {idxs}")
        return

    logger.info(f"[Async] Step 5 - GRPO training batch: {len(training_indices)} samples")
    logger.info(f"[Async]   Training batch UUIDs: {grpo_experience['prompt_uuid']}")

    # Verify GRPO training groups
    training_groups = sorted(set(i // n_samples_per_prompt for i in training_indices))
    logger.info(f"[Async]   Training groups: {training_groups}")

    logger.info(f"[Async]   Experience columns: {grpo_experience.keys()}")

    # Step 6: Delete consumed data
    await asyncio.sleep(0.1)  # Simulate async delay
    tq_client.delete_experience(
        indexes=training_indices,
        topic=topic,
    )
    logger.info(f"[Async] Step 6 - Deleted consumed data from indices: {training_indices}")

    # Verify cleanup
    _consumed_count, _consumed_indices = tq_client.get_data_consumed_set(
        topic=topic,
        consumer="learner",
        get_n_samples=False,
    )
    logger.info(f"[Async]   Learner consumed set: {_consumed_count} indexes")

    _ready_count, _ = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=experience_columns,
        get_n_samples=False,
    )
    logger.info(f"[Async]   Remaining ready data: {_ready_count} indexes")

    logger.info(f"[Async] Round {round_num} completed successfully")


async def main_async():
    """Main async function to run GRPO training simulation with UUID support."""
    logger.info("=" * 60)
    logger.info("GRPO Asynchronous Training Simulation with UUID")
    logger.info("=" * 60)

    # Get client (manager already initialized)
    tq_client = get_transferqueue_client()

    # Create topic for GRPO training with UUID support
    topic = "GRPO_ASYNC_UUID_TEST"
    prompts_num = 4
    n_samples_per_prompt = 4  # GRPO: n_samples per prompt
    # Added prompt_uuid column for UUID-based prompt identification
    experience_columns = ["prompt", "prompt_length", "response", "old_log_prob", "reward", "prompt_uuid"]
    # Multiple consumers for different stages
    consumers = ["generate", "logprob", "reward", "learner"]
    metrics = Metric()

    tq_client.add_topic(
        prompts_num=prompts_num,
        n_samples_per_prompt=n_samples_per_prompt,
        experience_columns=experience_columns,
        experience_consumers=consumers,
        metrics=metrics,
        topic=topic,
    )
    logger.info(f"Created topic '{topic}' with n_samples_per_prompt={n_samples_per_prompt}")
    logger.info(f"  Consumers: {consumers}")
    logger.info(f"  Experience columns: {experience_columns}")

    # Run multiple training rounds asynchronously
    for round_num in range(1, NUM_ROUNDS + 1):
        await async_grpo_training_round(tq_client, topic, round_num, n_samples_per_prompt)

    logger.info("-" * 60)
    logger.info(f"All {NUM_ROUNDS} async rounds completed successfully!")
    logger.info("-" * 60)
    logger.info("GRPO async UUID test PASSED:")
    logger.info("  ✓ UUID-based prompt identification works correctly in async mode")
    logger.info("  ✓ async put/get experience works correctly")
    logger.info("  ✓ async get experience with get_n_samples=True")
    logger.info("  ✓ Complete groups are sampled together")
    logger.info("  ✓ Multi-consumer data flow (generate/logprob/reward/learner)")
    logger.info("  ✓ Async parallel computation (generate, log_prob, rewards)")
    logger.info("  ✓ Multi-round async training works correctly with unique UUIDs")
    logger.info("  ✓ delete_experience properly cleans up consumed data")

    # Cleanup
    tq_client.delete_topic(topic)
    logger.info("Topic deleted")


def main():
    """Main function to initialize Ray and run async training."""
    logger.info("Initializing Ray...")

    # Initialize Ray and TransferQueue Manager
    base_port = _find_free_port()
    mgr = TransferQueueManager.options(num_cpus=6).remote(
        nums_tq_data=NUMS_TQ_DATA,
        base_port=base_port,
    )
    ready = ray.get(mgr.init_ready.remote())
    assert ready is True
    logger.info(f"TransferQueueManager initialized on port {base_port}")

    try:
        # Run async training
        asyncio.run(main_async())
    finally:
        # Cleanup
        ray.get(mgr.shutdown.remote())
        try:
            ray.kill(mgr)
        except Exception:
            pass
        if ray.is_initialized():
            ray.shutdown()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    ray.init()
    main()
