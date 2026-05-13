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

"""Offline summary tool for TransferQueue replay JSONL events."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

METADATA_EVENTS = {
    "meta_insert",
    "meta_fetch",
    "meta_force_fetch",
    "custom_meta_set",
    "meta_clear",
    "partition_clear",
}
RAW_EVENTS = {"put_raw", "get_raw"}
WIRE_EVENTS = {"put_wire", "get_wire"}
KNOWN_EVENTS = METADATA_EVENTS | RAW_EVENTS | WIRE_EVENTS
DEFAULT_MATCH_WINDOW_MS = 5_000.0


@dataclass(frozen=True)
class ReplayEvent:
    event_type: str
    partition: str
    indexes: frozenset[str] | None
    timestamp_ms: float | None
    raw_bytes: int
    wire_frame_bytes: int
    compressed_tensor_bytes: int
    storage_unit: str


@dataclass
class PartitionStats:
    put_count: int = 0
    get_count: int = 0
    raw_bytes: int = 0
    put_raw: list[ReplayEvent] = field(default_factory=list)
    get_raw: list[ReplayEvent] = field(default_factory=list)
    put_wire: list[ReplayEvent] = field(default_factory=list)
    get_wire: list[ReplayEvent] = field(default_factory=list)


@dataclass
class StorageStats:
    put_wire_count: int = 0
    wire_frame_bytes: int = 0
    compressed_tensor_bytes: int = 0


@dataclass
class Analysis:
    events_file: Path
    processed_events: int
    malformed_lines: int
    unknown_events: int
    partitions: dict[str, PartitionStats]
    storage_units: dict[str, StorageStats]
    latencies: dict[str, list[float]]
    all_timestamps: list[float]
    raw_bytes: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, help="Directory containing events.jsonl")
    parser.add_argument("--events-file", type=Path, help="Direct path to a replay events JSONL file")
    parser.add_argument(
        "--match-window-ms",
        type=float,
        default=DEFAULT_MATCH_WINDOW_MS,
        help="Maximum timestamp distance used when matching raw and wire events",
    )
    return parser.parse_args(argv)


def resolve_events_file(args: argparse.Namespace) -> Path:
    if args.events_file is not None:
        return args.events_file
    if args.replay_dir is not None:
        return args.replay_dir / "events.jsonl"
    raise ValueError("Specify --replay-dir or --events-file")


def load_jsonl(events_file: Path) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    malformed = 0
    with events_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                malformed += 1
                print(
                    f"WARNING: skipped malformed JSONL line {line_number}: {exc.msg}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(event, dict):
                malformed += 1
                print(
                    f"WARNING: skipped malformed JSONL line {line_number}: expected object",
                    file=sys.stderr,
                )
                continue
            events.append(event)
    return events, malformed


def analyze_events(events_file: Path, events: list[dict[str, Any]], malformed_lines: int) -> Analysis:
    partitions: dict[str, PartitionStats] = {}
    storage_units: dict[str, StorageStats] = {}
    latencies: dict[str, list[float]] = {}
    all_timestamps: list[float] = []
    total_raw_bytes = 0
    unknown_events = 0

    for raw_event in events:
        event_type = _event_type(raw_event)
        if event_type not in KNOWN_EVENTS:
            unknown_events += 1
            continue

        latency_ms = _number(raw_event.get("elapsed_ms"))
        if latency_ms is not None:
            latencies.setdefault(event_type, []).append(latency_ms)

        timestamp_ms = _timestamp_ms(raw_event)
        if timestamp_ms is not None:
            all_timestamps.append(timestamp_ms)

        if event_type not in RAW_EVENTS | WIRE_EVENTS:
            continue

        event = _normalize_event(event_type, raw_event)
        partition = partitions.setdefault(event.partition, PartitionStats())

        if event_type == "put_raw":
            partition.put_count += 1
            partition.raw_bytes += event.raw_bytes
            total_raw_bytes += event.raw_bytes
            partition.put_raw.append(event)
        elif event_type == "get_raw":
            partition.get_count += 1
            partition.raw_bytes += event.raw_bytes
            total_raw_bytes += event.raw_bytes
            partition.get_raw.append(event)
        elif event_type == "put_wire":
            partition.put_wire.append(event)
            storage = storage_units.setdefault(event.storage_unit, StorageStats())
            storage.put_wire_count += 1
            storage.wire_frame_bytes += event.wire_frame_bytes
            storage.compressed_tensor_bytes += event.compressed_tensor_bytes
        elif event_type == "get_wire":
            partition.get_wire.append(event)

    return Analysis(
        events_file=events_file,
        processed_events=len(events),
        malformed_lines=malformed_lines,
        unknown_events=unknown_events,
        partitions=partitions,
        storage_units=storage_units,
        latencies=latencies,
        all_timestamps=all_timestamps,
        raw_bytes=total_raw_bytes,
    )


def render_analysis(analysis: Analysis, match_window_ms: float) -> str:
    lines = [
        "Replay Analysis Summary",
        f"Events file: {analysis.events_file}",
        f"Processed events: {analysis.processed_events}",
        f"Malformed lines skipped: {analysis.malformed_lines}",
        f"Unknown events ignored: {analysis.unknown_events}",
        "",
        "Throughput Summary",
        _throughput_line(analysis),
        "",
        "Partition Summary",
    ]

    if analysis.partitions:
        for partition_id in sorted(analysis.partitions):
            stats = analysis.partitions[partition_id]
            wire_ratio, compressed_ratio, basis = _partition_ratios(stats, match_window_ms)
            lines.append(
                f"partition={partition_id} put_count={stats.put_count} get_count={stats.get_count} "
                f"raw_bytes={stats.raw_bytes} wire_frame_ratio={wire_ratio} "
                f"compressed_ratio={compressed_ratio} ratio_basis={basis}"
            )
        total_puts = sum(stats.put_count for stats in analysis.partitions.values())
        total_gets = sum(stats.get_count for stats in analysis.partitions.values())
        lines.append(f"TOTAL put_count={total_puts} get_count={total_gets} raw_bytes={analysis.raw_bytes}")
    else:
        lines.append("N/A")

    lines.extend(["", "Latency Summary"])
    if analysis.latencies:
        for event_type in sorted(analysis.latencies):
            values = analysis.latencies[event_type]
            lines.append(
                f"{event_type} count={len(values)} avg_ms={_average(values):.2f} p99_ms={_percentile(values, 99):.2f}"
            )
    else:
        lines.append("N/A")

    lines.extend(["", "Storage Unit Distribution"])
    if analysis.storage_units:
        for storage_unit in sorted(analysis.storage_units):
            stats = analysis.storage_units[storage_unit]
            lines.append(
                f"storage_unit={storage_unit} put_wire_count={stats.put_wire_count} "
                f"wire_frame_bytes={stats.wire_frame_bytes} "
                f"compressed_tensor_bytes={stats.compressed_tensor_bytes}"
            )
    else:
        lines.append("N/A")

    return "\n".join(lines) + "\n"


def _normalize_event(event_type: str, event: dict[str, Any]) -> ReplayEvent:
    return ReplayEvent(
        event_type=event_type,
        partition=_partition_id(event),
        indexes=_indexes(event.get("indexes", event.get("indices", event.get("index")))),
        timestamp_ms=_timestamp_ms(event),
        raw_bytes=_raw_bytes(event),
        wire_frame_bytes=_int_bytes(event.get("wire_frame_bytes")),
        compressed_tensor_bytes=_int_bytes(event.get("compressed_tensor_bytes")),
        storage_unit=str(event.get("target_storage_unit") or event.get("storage_unit") or "unknown"),
    )


def _event_type(event: dict[str, Any]) -> str:
    return str(event.get("event") or event.get("event_type") or event.get("type") or "")


def _partition_ratios(stats: PartitionStats, match_window_ms: float) -> tuple[str, str, str]:
    put_match = _match_family(stats.put_raw, stats.put_wire, match_window_ms)
    get_match = _match_family(stats.get_raw, stats.get_wire, match_window_ms)
    has_wire = put_match["has_wire"] or get_match["has_wire"]
    if not has_wire:
        return "N/A", "N/A", "N/A"
    if put_match["partial"] or get_match["partial"]:
        return "N/A", "N/A", "partial"

    basis = "aggregate" if put_match["aggregate"] or get_match["aggregate"] else "matched"
    raw_bytes = put_match["raw_bytes"] + get_match["raw_bytes"]
    wire_bytes = put_match["wire_frame_bytes"] + get_match["wire_frame_bytes"]
    compressed_raw_bytes = put_match["compressed_raw_bytes"] + get_match["compressed_raw_bytes"]
    compressed_bytes = put_match["compressed_tensor_bytes"] + get_match["compressed_tensor_bytes"]
    return _ratio(raw_bytes, wire_bytes), _ratio(compressed_raw_bytes, compressed_bytes), basis


def _match_family(
    raw_events: list[ReplayEvent], wire_events: list[ReplayEvent], match_window_ms: float
) -> dict[str, Any]:
    if not wire_events:
        return {
            "has_wire": False,
            "aggregate": False,
            "partial": False,
            "raw_bytes": 0,
            "wire_frame_bytes": 0,
            "compressed_raw_bytes": 0,
            "compressed_tensor_bytes": 0,
        }

    raw_by_wire: list[int] = []
    for wire in wire_events:
        candidates = [
            raw_index
            for raw_index, raw in enumerate(raw_events)
            if _within_window(raw, wire, match_window_ms) and _wire_indexes_match_raw(raw, wire)
        ]
        if len(candidates) != 1:
            return _aggregate_family(raw_events, wire_events)
        raw_by_wire.append(candidates[0])

    wires_by_raw: dict[int, list[ReplayEvent]] = {}
    for raw_index, wire in zip(raw_by_wire, wire_events, strict=False):
        wires_by_raw.setdefault(raw_index, []).append(wire)

    for raw_index, wires in wires_by_raw.items():
        if _wire_indexes_overlap(wires):
            return _aggregate_family(raw_events, wire_events)
        if not _wire_indexes_cover_raw(raw_events[raw_index], wires):
            return _partial_family()

    matched_raw_bytes = sum(raw_events[raw_index].raw_bytes for raw_index in wires_by_raw)
    compressed_bytes = sum(wire.compressed_tensor_bytes for wire in wire_events)
    return {
        "has_wire": True,
        "aggregate": False,
        "partial": False,
        "raw_bytes": matched_raw_bytes,
        "wire_frame_bytes": sum(wire.wire_frame_bytes for wire in wire_events),
        "compressed_raw_bytes": matched_raw_bytes if compressed_bytes > 0 else 0,
        "compressed_tensor_bytes": compressed_bytes,
    }


def _aggregate_family(raw_events: list[ReplayEvent], wire_events: list[ReplayEvent]) -> dict[str, Any]:
    raw_bytes = sum(raw.raw_bytes for raw in raw_events)
    compressed_bytes = sum(wire.compressed_tensor_bytes for wire in wire_events)
    return {
        "has_wire": True,
        "aggregate": True,
        "partial": False,
        "raw_bytes": raw_bytes,
        "wire_frame_bytes": sum(wire.wire_frame_bytes for wire in wire_events),
        "compressed_raw_bytes": raw_bytes if compressed_bytes > 0 else 0,
        "compressed_tensor_bytes": compressed_bytes,
    }


def _partial_family() -> dict[str, Any]:
    return {
        "has_wire": True,
        "aggregate": False,
        "partial": True,
        "raw_bytes": 0,
        "wire_frame_bytes": 0,
        "compressed_raw_bytes": 0,
        "compressed_tensor_bytes": 0,
    }


def _wire_indexes_match_raw(raw: ReplayEvent, wire: ReplayEvent) -> bool:
    if raw.indexes is None or wire.indexes is None:
        return True
    return wire.indexes.issubset(raw.indexes)


def _wire_indexes_cover_raw(raw: ReplayEvent, wires: list[ReplayEvent]) -> bool:
    if raw.indexes is None:
        return True

    covered: set[str] = set()
    for wire in wires:
        if wire.indexes is None:
            return True
        covered.update(wire.indexes)
    return covered == raw.indexes


def _wire_indexes_overlap(wires: list[ReplayEvent]) -> bool:
    seen: set[str] = set()
    for wire in wires:
        if wire.indexes is None:
            continue
        if seen.intersection(wire.indexes):
            return True
        seen.update(wire.indexes)
    return False


def _within_window(raw: ReplayEvent, wire: ReplayEvent, match_window_ms: float) -> bool:
    if raw.timestamp_ms is None or wire.timestamp_ms is None:
        return True
    return abs(raw.timestamp_ms - wire.timestamp_ms) <= match_window_ms


def _throughput_line(analysis: Analysis) -> str:
    if len(analysis.all_timestamps) < 2:
        return "raw_mib_per_s=N/A event_rate_per_s=N/A"

    span_ms = max(analysis.all_timestamps) - min(analysis.all_timestamps)
    if span_ms <= 0:
        return "raw_mib_per_s=N/A event_rate_per_s=N/A"

    span_seconds = span_ms / 1000.0
    raw_mib_per_s = analysis.raw_bytes / (1024 * 1024) / span_seconds
    event_rate_per_s = analysis.processed_events / span_seconds
    return f"raw_mib_per_s={raw_mib_per_s:.2f} event_rate_per_s={event_rate_per_s:.2f}"


def _partition_id(event: dict[str, Any]) -> str:
    return str(event.get("pid", event.get("partition_id", event.get("partition", "unknown"))))


def _indexes(value: Any) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, list | tuple | set):
        return frozenset(_stable_value(item) for item in value)
    return frozenset({_stable_value(value)})


def _stable_value(value: Any) -> str:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _timestamp_ms(event: dict[str, Any]) -> float | None:
    for key in ("timestamp_ms", "ts_ms", "time_ms", "timestamp"):
        value = _number(event.get(key))
        if value is not None:
            return value
    ts_seconds = _number(event.get("ts"))
    if ts_seconds is not None:
        return ts_seconds * 1000.0
    return None


def _raw_bytes(event: dict[str, Any]) -> int:
    for key in ("raw_tensor_bytes", "raw_estimated_bytes"):
        byte_count = _int_bytes(event.get(key))
        if byte_count:
            return byte_count
    return 0


def _int_bytes(value: Any) -> int:
    number = _number(value)
    if number is None or number <= 0:
        return 0
    return int(number)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        if math.isfinite(number):
            return number
    return None


def _ratio(numerator: int, denominator: int) -> str:
    if numerator <= 0 or denominator <= 0:
        return "N/A"
    return f"{numerator / denominator:.2f}"


def _average(values: list[float]) -> float:
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        events_file = resolve_events_file(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not events_file.exists() or events_file.stat().st_size == 0:
        print(f"No replay events found at {events_file}", file=sys.stderr)
        return 2

    events, malformed = load_jsonl(events_file)
    if not events:
        print(f"No replay events found in {events_file}", file=sys.stderr)
        return 2

    analysis = analyze_events(events_file, events, malformed)
    print(render_analysis(analysis, args.match_window_ms), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
