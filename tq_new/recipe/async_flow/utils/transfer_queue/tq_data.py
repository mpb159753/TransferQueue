# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import asyncio
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import ray
import zmq
import zmq.asyncio

from recipe.async_flow.utils.transfer_queue.tq_config import GROUP_SHARED_COLUMNS, ZMQ_HWM
from recipe.async_flow.utils.transfer_queue.tq_structures import ExperienceTable
from recipe.async_flow.utils.transfer_queue.tq_utils import ZMQClient, setup_logger


# @ray.remote
# @ray.remote(max_concurrency=100, num_cpus=10, name="TransferShard")
@ray.remote(max_concurrency=100, num_cpus=10)
class TransferQueueShard:
    def __init__(self, shard_id: int, manager_endpoint: str):
        """
        Initialize a data shard.
        - shard_id: ID of this shard (for logging).
        - port: TCP port to bind the ZMQ Router server on.
        """

        self.logger = setup_logger(f"Shard_{shard_id}")

        self.shard_id = shard_id
        # Store ExperienceTables for different topics
        self.tables: dict[str, Any] = {}

        # Set up ZMQ Router socket
        self.zmq_ctx = zmq.asyncio.Context()
        self.router = self.zmq_ctx.socket(zmq.ROUTER)

        # Configure HWM to prevent memory overflow from burst traffic
        self.router.setsockopt(zmq.RCVHWM, ZMQ_HWM)
        self.router.setsockopt(zmq.SNDHWM, ZMQ_HWM)

        # Disable LINGER to prevent blocking during Actor destruction
        self.router.setsockopt(zmq.LINGER, 0)

        # Bind port
        port = self.router.bind_to_random_port("tcp://0.0.0.0")

        self.endpoint = f"tcp://{ray.util.get_node_ip_address()}:{port}"
        self.logger.info(f"DATA Shard[{shard_id}]: Started at {self.endpoint}")
        self.manager = ray.get_actor("TransferQueueManager")

        # Start background loop
        self._running = True
        self.background_tasks: set[asyncio.Task] = set()
        self._loop_task = asyncio.get_event_loop().create_task(self._serve_loop())

        self._cleaner_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tq_cleaner")

        self.zmq_client = ZMQClient(
            self.zmq_ctx,
            endpoint=manager_endpoint,
        )
        self.logger.info(f"Shard {shard_id} started")

    def add_experience_table(self, topic: str, n_samples_per_prompt: int, experience_columns: list[str]) -> None:
        """
        Create an ExperienceTable for a topic (auto-scaling).
        Define logical distribution only; storage grows dynamically.
        """
        if topic in self.tables:
            raise ValueError(f"Topic '{topic}' already exists")
        self.tables[topic] = ExperienceTable(
            n_samples_per_prompt=n_samples_per_prompt,
            experience_columns=experience_columns,
        )

    def remove_experience_table(self, topic: str) -> None:
        """
        Remove the storage table for the given topic.
        """
        if topic in self.tables:
            table = self.tables.pop(topic)
            self._cleaner_pool.submit(self._background_clear, table, topic)
            self.logger.info(f"Shard: Scheduled cleanup for topic '{topic}'")
            return

    def _background_clear(self, table, topic_name):
        try:
            table.clear()
        except Exception as e:
            self.logger.error(f"Error cleaning topic {topic_name}: {e}")
        finally:
            del table

    def clear_experience_table(self, topic: str):
        if topic not in self.tables:
            raise ValueError(f"Unknown topic '{topic}'. Known topics: {list(self.tables.keys())}")

        old_table = self.tables[topic]

        new_table = ExperienceTable(
            n_samples_per_prompt=old_table.n_samples_per_prompt, experience_columns=old_table.experience_columns
        )

        self.tables[topic] = new_table

        # Destroy old table in background
        self._cleaner_pool.submit(self._background_clear, old_table, topic)

        self.logger.info(f"DATA Shard[{self.shard_id}]: Cleared topic '{topic}' (Async swap).")

    def prune_experience_table(self, topic: str, indexes: list[int] = None, check_n_samples: bool = False):
        """Prune stored experiences by index for a single topic in this shard."""
        if topic not in self.tables:
            raise ValueError(f"Unknown topic '{topic}. Known topics: {list(self.tables.keys())}'")
        if not indexes:
            raise ValueError("Indexes must be provided for prune_topic_by_indexes")

        tbl = self.tables[topic]

        if check_n_samples:
            # Check if all samples for a single prompt are pruned together.
            n_samples = tbl.n_samples_per_prompt
            groups = {}
            for idx in indexes:
                groups.setdefault(idx // n_samples, []).append(idx)

            incomplete = []
            for group_sub_indexes in groups.values():
                if len(set(group_sub_indexes)) != n_samples:
                    incomplete.extend(group_sub_indexes)
            if incomplete:
                raise ValueError(
                    f"Shard {self.shard_id} received indexes: {sorted(set(incomplete))}, which are incomplete for prompt groups"
                )
            absence = [group_id for group_id in groups.keys() if group_id not in tbl.owned_groups]
            if absence:
                raise ValueError(
                    f"Shard {self.shard_id} received group_ids: {sorted(set(absence))}, which are not in the shard"
                )

        tbl.prune(indexes)
        self.logger.info(f"DATA Shard[{self.shard_id}]: Pruned indexes{indexes} on topic '{topic}'.")

    async def _serve_loop(self):
        """
        Core event loop: continuously listen and dispatch tasks.
        """
        while self._running:
            try:
                # Use copy=False to reduce memory copying
                frames = await self.router.recv_multipart(copy=False)

                # Create concurrent task to process message without blocking receive loop
                task = asyncio.create_task(self._process_message(frames))
                self.background_tasks.add(task)
                task.add_done_callback(self._on_process_message_done)
            except zmq.ZMQError as e:
                self.logger.error(f"ZMQ Error in serve loop: {e}")
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.exception(f"Unexpected Error in serve loop: {e}")
                await asyncio.sleep(0.1)

    def _on_process_message_done(self, task: asyncio.Task):
        self.background_tasks.discard(task)
        if task.cancelled():
            self.logger.warning(f"Task {task.get_name()} cancelled")
        elif exc := task.exception():
            self.logger.error(f"Task {task.get_name()} exception: {exc}", exc_info=exc)

    async def _process_message(self, frames: list[zmq.Frame]):
        """
        Protocol parsing and routing.
        Frames structure: [Identity, Command, Header, Payload...]
        """
        # Basic protocol length validation
        if len(frames) < 3:
            self.logger.error(f"Malformed request, frames len: {len(frames)}")
            return

        identity = frames[0]
        command = frames[1].bytes

        try:
            if command == b"PUT":
                # PUT: [Identity, b'PUT', Header, Payload_1, Payload_2...]
                await self._handle_put(identity, frames[2], frames[3:])
            elif command == b"GET":
                # GET: [Identity, b'GET', Header]
                await self._handle_get(identity, frames[2])
            else:
                self.logger.warning(f"Unknown command: {command}")
                await self.router.send_multipart([identity, b"ERROR: Unknown command"])

        except Exception as e:
            # Catch all processing-level exceptions and return to client
            self.logger.exception(f"Error processing {command}")
            error_msg = f"ERROR: {str(e)}".encode()
            try:
                await self.router.send_multipart([identity, error_msg])
            except Exception as send_err:
                self.logger.error(f"Failed to send error response: {send_err}")

    async def _handle_put(self, identity: zmq.Frame, header_frame: zmq.Frame, payload_frames: list[zmq.Frame]):
        """
        Handle write requests.
        """
        header = pickle.loads(header_frame.bytes)

        topic = header.get("topic")
        if not topic:
            raise ValueError("Missing 'topic' in header")

        if topic not in self.tables:
            raise ValueError(f"Table is not established for topic:{topic}")

        table = self.tables[topic]
        table.put_batch(
            global_ids=header["indexes"],
            col_order=header["order"],
            col_inputs_meta=header,
            payload_frames=payload_frames,
        )
        reported_indexes = header["indexes"]
        self.logger.debug(f"Putting {len(reported_indexes)} indexes in {topic}")

        # 3. Notify manager of updated columns
        # Process columns separately: extend indexes for shared columns, keep original indexes for non-shared columns
        try:
            await self._update_data_status_by_column(topic, reported_indexes, header["order"], header["columns"])
        except Exception as e:
            self.logger.error(f"Shard [{self.shard_id}]: update_data_status failed: {e}")

        await self.router.send_multipart([identity, b"ACK"])

    async def _update_data_status_by_column(self, topic: str, indexes: list[int], col_order: list[str], col_info: dict):
        """
        Update data status by column: extend indexes for shared columns, keep original indexes for non-shared columns

        Args:
            topic: Topic name
            indexes: Original index list
            col_order: Column order
            col_info: Column info dictionary containing ref_multiplier
        """
        # Group by column: shared and non-shared columns need different indexes
        shared_columns = []
        non_shared_columns = []

        for col_name in col_order:
            ref_multiplier = col_info[col_name].get("ref_multiplier", 1)
            if ref_multiplier > 1 and col_name in GROUP_SHARED_COLUMNS:
                shared_columns.append(col_name)
            else:
                non_shared_columns.append(col_name)

        # Update non-shared column status (using original indexes)
        if non_shared_columns:
            try:
                await self.zmq_client.call(
                    "update_data_status",
                    topic=topic,
                    indexes=indexes,
                    columns=non_shared_columns,
                )
            except Exception as e:
                self.logger.error(f"Failed to update status for non-shared columns {non_shared_columns}: {e}")

        # Update shared column status (extend indexes to entire group)
        if shared_columns:
            n_samples = self.tables[topic].n_samples_per_prompt
            extended_indexes = set()

            for idx in indexes:
                group_id = idx // n_samples
                # Build mapping for all indexes in the entire group
                for offset in range(n_samples):
                    extended_indexes.add(group_id * n_samples + offset)

            extended_indexes = sorted(extended_indexes)

            try:
                await self.zmq_client.call(
                    "update_data_status",
                    topic=topic,
                    indexes=extended_indexes,
                    columns=shared_columns,
                )
            except Exception as e:
                self.logger.error(f"Failed to update status for shared columns {shared_columns}: {e}")

            self.logger.debug(
                f"Updated shared columns {shared_columns} status: indexes={indexes} -> extended={extended_indexes}"
            )

    async def _handle_get(self, identity: zmq.Frame, header_frame: zmq.Frame):
        """
        Handle read requests.
        """
        header = pickle.loads(header_frame.bytes)

        # Topic validation
        topic = header.get("topic")
        if topic not in self.tables:
            raise ValueError(f"Topic '{topic}' not found on shard {self.shard_id}")

        table = self.tables[topic]

        result_meta, result_payloads = table.get_batch(
            target_global_idxs=header["indexes"], target_cols=header["experience_columns"]
        )

        # Response format: [Identity, MetaBytes, Payload_1, Payload_2...]
        meta_bytes = pickle.dumps(result_meta)

        send_frames = [identity, meta_bytes]
        send_frames.extend(result_payloads)

        await self.router.send_multipart(send_frames)

    def get_endpoint(self):
        return self.endpoint

    def update_owned_groups(self, topic: str, group_ids: list[int]):
        if topic not in self.tables:
            raise ValueError(f"Unknown topic '{topic}'")
        self.tables[topic].owned_groups.update(group_ids)

    def reset_all(self):
        topics = list(self.tables.keys())

        for topic in topics:
            self.remove_experience_table(topic)

        self.logger.info(f"DATA Shard[{self.shard_id}]: Dropped ALL topics/tables ({len(topics)} topics).")

    async def async_shutdown(self):
        """
        Shutdown service:
        1. Stop receive loop
        2. Cancel pending async tasks
        3. Destroy ZMQ context and socket
        """
        self.logger.info(f"DATA Shard[{self.shard_id}]: Shutting down...")

        # Stop flag
        self._running = False

        # Cancel background loop task
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                # Wait for task cancellation to complete, ensuring no Socket access after resource destruction
                await self._loop_task
            except asyncio.CancelledError:
                self.logger.info("Serve loop cancelled successfully.")
            except Exception as e:
                self.logger.warning(f"Error during loop cancellation: {e}")

        # Clean up ZMQ resources
        # LINGER=0 ensures immediate release without waiting for unsent messages
        if not self.router.closed:
            self.router.close(linger=0)

        if not self.zmq_ctx.closed:
            self.zmq_ctx.destroy(linger=0)

        self.logger.info(f"DATA Shard[{self.shard_id}]: Shutdown complete.")
