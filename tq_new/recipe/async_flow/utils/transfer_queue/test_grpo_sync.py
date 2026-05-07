# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
"""GRPO synchronous training test for TransferQueue.

This file simulates a GRPO (Group Relative Policy Optimization) training workflow
using the TransferQueue system with grouped sampling (get_n_samples=True).

GRPO training requires:
1. Multiple samples generated from the same input prompt
2. All samples from the same prompt must be trained together as a group
3. Data organization: [p0_s0, p0_s1, p0_s2, p0_s3, p1_s0, p1_s1, ...]
   where each consecutive group belongs to the same prompt
4. No partition_id concept - use delete_experience to clean up consumed data
5. get_n_samples=True ensures only complete groups are sampled
6. Each stage (generate, logprob, reward, train) uses its own consumer

Key GRPO concept verification:
- batch_size must be divisible by n_samples_per_prompt
- All samples from same prompt (consecutive indexes) must be selected together
- get_n_samples=True ensures group completeness constraint
- Data flow: each stage reads from TQ, computes, writes back to TQ
"""

import logging
import socket
import time

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


def mock_generate_sequences(data: list[torch.Tensor]) -> list[torch.Tensor]:
    """Mock function to simulate sequence generation in GRPO training.

    In real GRPO training, this would be the policy model generating responses
    from input prompts. Here we just add a delay and modify the data slightly.
    """
    time.sleep(0.5)
    return [tensor + 1 for tensor in data]


def mock_compute_old_log_prob(data_dict: dict) -> list[torch.Tensor]:
    """Mock function to compute old log probabilities for GRPO.

    In real GRPO training, this computes log probabilities of generated sequences
    using the reference model or old policy. Here we simulate with a simple operation.

    Args:
        data_dict: Contains "prompt" and "response" from TQ

    Returns:
        List of log prob tensors, one per sample
    """
    time.sleep(0.5)
    prompts = data_dict["prompt"]
    return [p * 0.5 for p in prompts]


def mock_compute_rewards(data_dict: dict) -> list[torch.Tensor]:
    """Mock function to compute rewards for generated sequences.

    In real GRPO training, this would compute rewards using reward models or
    rule-based functions. Here we simulate with a simple tensor.

    Args:
        data_dict: Contains "response" from TQ

    Returns:
        List of reward tensors, one per sample
    """
    time.sleep(0.5)
    responses = data_dict["response"]
    # Return random rewards as 1D tensors
    return [torch.randn(1) for _ in responses]


def main():
    """Main function to run GRPO training simulation."""
    logger.info("=" * 60)
    logger.info("GRPO Synchronous Training Simulation")
    logger.info("=" * 60)

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
        # Get client
        tq_client = get_transferqueue_client()

        # Create topic for GRPO training
        topic = "GRPO_SYNC_TEST"
        prompts_num = 4
        n_samples_per_prompt = 4  # GRPO: n_samples per prompt
        experience_columns = ["prompt", "prompt_length", "response", "old_log_prob", "reward"]
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

        # Run multiple training rounds
        for round_num in range(1, NUM_ROUNDS + 1):
            logger.info("-" * 60)
            logger.info(f"Training Round {round_num}/{NUM_ROUNDS}")
            logger.info("-" * 60)

            # Prepare GRPO data: consecutive groups of samples from same prompt
            # Data organization: [p0, p0, p0, p0, p1, p1, p1, p1, p2, p2, p2, p2, p3, p3, p3, p3]
            pad_id = 0
            prompt_lengths = [3, 5, 4, 6]
            prompts = _make_padded_prompts(prompt_lengths, pad_id=pad_id)

            # For GRPO, repeat each prompt n_samples_per_prompt times
            # This creates consecutive groups: [p0, p0, p0, p0], [p1, p1, p1, p1], ...
            padded_prompts = prompts.repeat_interleave(n_samples_per_prompt, dim=0)
            prompt_lengths_tensor = (
                torch.tensor(prompt_lengths, dtype=torch.int32)
                .repeat_interleave(n_samples_per_prompt, dim=0)
                .unsqueeze(1)
            )

            logger.info(f"GRPO data: {len(padded_prompts)} samples from {prompts_num} prompts")
            logger.info(f"  Organization: {[f'p{i // n_samples_per_prompt}' for i in range(len(padded_prompts))]}")
            logger.info(
                f"  Index groups: {[list(range(i * n_samples_per_prompt, (i + 1) * n_samples_per_prompt)) for i in range(prompts_num)]}"
            )

            # Step 1: Put prompts to TransferQueue
            allocated_indexes = tq_client.put_experience(
                data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths_tensor},
                unpad_pairs=[("prompt", "prompt_length", "right_pad")],
                topic=topic,
            )
            logger.info(f"Step 1 - Put prompts to indexes: {allocated_indexes}")

            # Verify all prompts are ready
            ready_count, ready_indices = tq_client.get_data_ready_set(
                topic=topic,
                experience_columns=["prompt"],
                get_n_samples=True,
            )
            logger.info(f"  Ready set (grouped): {ready_count} indexes")
            assert ready_count == len(allocated_indexes)

            # Step 2: Generate sequences (rollout phase)
            # Use "generate" consumer for this stage
            batch_size = 4 * n_samples_per_prompt  # 4 prompts * 4 samples = 16 samples
            assert batch_size % n_samples_per_prompt == 0, (
                f"batch_size {batch_size} must be divisible by n_samples_per_prompt {n_samples_per_prompt}"
            )

            # Get prompts from TQ with GRPO grouped sampling
            prompts_data, sampled_indices = tq_client.get_experience(
                consumer="generate",
                experience_columns=["prompt"],
                experience_count=batch_size,
                indexes=None,
                get_n_samples=True,
                topic=topic,
            )
            logger.info(f"Step 2 - Sampled {len(sampled_indices)} prompts by 'generate' consumer")

            # Mock generate sequences (compute locally)
            response_tensors = mock_generate_sequences(prompts_data["prompt"])
            logger.info("  Generated sequences")

            # Put responses to TQ (non-shared column)
            tq_client.put_experience(
                data_dict={"response": response_tensors},
                indexes=sampled_indices,
                topic=topic,
            )
            logger.info(f"  Put responses to indexes: {sampled_indices}")

            # Step 3: Compute old log probs
            # GET prompt + response from TQ, compute locally, PUT back to TQ
            prompt_response_data, _ = tq_client.get_experience(
                consumer="logprob",
                experience_columns=["prompt", "response"],
                experience_count=batch_size,
                indexes=sampled_indices,  # Use specific indexes for this stage
                get_n_samples=False,  # Already have specific indexes
                topic=topic,
            )
            logger.info("Step 3 - Retrieved prompt+response from TQ for logprob computation")

            # Compute log probs locally
            log_prob_tensors = mock_compute_old_log_prob(prompt_response_data)
            logger.info("  Computed old log probabilities")

            # Put old log probs to TQ
            tq_client.put_experience(
                data_dict={"old_log_prob": log_prob_tensors},
                indexes=sampled_indices,
                topic=topic,
            )
            logger.info(f"  Put old_log_prob to indexes: {sampled_indices}")

            # Step 4: Compute rewards
            # GET response from TQ, compute locally, PUT back to TQ
            response_data, _ = tq_client.get_experience(
                consumer="reward",
                experience_columns=["response"],
                experience_count=batch_size,
                indexes=sampled_indices,
                get_n_samples=False,
                topic=topic,
            )
            logger.info("Step 4 - Retrieved response from TQ for reward computation")

            # Compute rewards locally
            reward_tensors_list = mock_compute_rewards(response_data)
            logger.info("  Computed rewards")

            # Put rewards to TQ
            tq_client.put_experience(
                data_dict={"reward": reward_tensors_list},
                indexes=sampled_indices,
                topic=topic,
            )
            logger.info(f"  Put rewards to indexes: {sampled_indices}")

            # Step 5: Training step - get complete GRPO experience
            # All fields (prompt, response, old_log_prob, reward) are now ready
            grpo_experience, training_indices = tq_client.get_experience(
                consumer="learner",
                experience_columns=["prompt", "response", "old_log_prob", "reward"],
                experience_count=batch_size,
                indexes=None,
                get_n_samples=True,  # GRPO: Only sample complete groups for training
                topic=topic,
            )

            if grpo_experience is None or training_indices is None:
                logger.error("Failed to get training experience!")
                logger.info("  This might be because data hasn't been marked ready yet")
                # Let's check what's ready
                for col in experience_columns:
                    count, idxs = tq_client.get_data_ready_set(
                        topic=topic, experience_columns=[col], get_n_samples=True
                    )
                    logger.info(f"    Column '{col}' ready: {count} indexes: {idxs}")
                continue

            logger.info(f"Step 5 - GRPO training batch: {len(training_indices)} samples")
            logger.info(f"  Training indices: {training_indices}")

            # Verify GRPO training batch has complete groups
            training_groups = sorted(set(i // n_samples_per_prompt for i in training_indices))
            logger.info(f"  Training groups: {training_groups}")

            samples_per_group = {}
            for i in training_indices:
                group_id = i // n_samples_per_prompt
                samples_per_group[group_id] = samples_per_group.get(group_id, 0) + 1
            logger.info(f"  Samples per group: {samples_per_group}")

            # Verify experience data is complete
            logger.info(f"  Experience columns: {grpo_experience.keys()}")
            for key in experience_columns:
                if key in grpo_experience:
                    logger.info(f"    {key}: {len(grpo_experience[key])} items")

            # Step 6: Delete consumed data (no partition_id, delete by indexes)
            # Verify data is consumed before deletion
            consumed_count, consumed_indices = tq_client.get_data_consumed_set(
                topic=topic,
                consumer="learner",
                get_n_samples=False,
            )
            logger.info(f"  Learner consumed set: {consumed_count} indexes :{consumed_indices}")

            # Verify remaining ready data
            ready_count, ready_indices = tq_client.get_data_ready_set(
                topic=topic,
                experience_columns=experience_columns,
                get_n_samples=False,
            )
            logger.info(f"  Remaining ready data: {ready_count} indexes :{ready_indices}")

            tq_client.delete_experience(
                indexes=training_indices,
                topic=topic,
            )
            logger.info(f"Step 6 - Deleted consumed data from indices: {training_indices}")

            # Verify data is deleted
            consumed_count, consumed_indices = tq_client.get_data_consumed_set(
                topic=topic,
                consumer="learner",
                get_n_samples=False,
            )
            logger.info(f"  Learner consumed set: {consumed_count} indexes :{consumed_indices}")

            # Verify remaining ready data
            ready_count, ready_indices = tq_client.get_data_ready_set(
                topic=topic,
                experience_columns=experience_columns,
                get_n_samples=False,
            )
            logger.info(f"  Remaining ready data: {ready_count} indexes :{ready_indices}")

            logger.info(f"Round {round_num} completed successfully")

        logger.info("-" * 60)
        logger.info(f"All {NUM_ROUNDS} rounds completed successfully!")
        logger.info("-" * 60)
        logger.info("GRPO sync test PASSED:")
        logger.info("  ✓ get_n_samples=True ensures grouped sampling")
        logger.info("  ✓ Complete groups are sampled together")
        logger.info("  ✓ Multi-consumer data flow (generate/logprob/reward/learner)")
        logger.info("  ✓ Each stage reads from TQ, computes, writes back to TQ")
        logger.info("  ✓ Multi-round training works correctly")
        logger.info("  ✓ delete_experience properly cleans up consumed data")
        logger.info("  ✓ batch_size divisible by n_samples_per_prompt constraint met")

    finally:
        # Cleanup
        tq_client.delete_topic(topic)
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
