from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

STAGE_PREFIXES = (
    "allocate_indexes",
    "route_targets",
    "serialize_request",
    "send_request",
    "storage_put",
    "status_update",
    "fetch_targets",
    "storage_get",
    "deserialize_response",
    "aggregate_results",
)


def extract_stage_name(raw_name: str | None) -> str | None:
    if not raw_name:
        return None

    normalized_name = str(raw_name).strip()
    for stage_name in STAGE_PREFIXES:
        if normalized_name.startswith(stage_name):
            return stage_name
    return None


def _round_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("workload_name"),
        int(entry.get("batch_size", 0)),
        int(entry.get("seq_length", 0)),
        int(entry.get("field_count", 0)),
        int(entry.get("num_shards_or_storage_units", 0)),
        int(entry.get("num_clients", 0)),
    )


def pair_common_core_results(entries: Iterable[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    paired: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for entry in entries:
        implementation = str(entry.get("implementation", "")).strip()
        if implementation not in {"tq_new", "simple_storage"}:
            continue
        paired.setdefault(_round_key(entry), {})[implementation] = entry

    return [
        {"tq_new": pair["tq_new"], "simple_storage": pair["simple_storage"]}
        for pair in paired.values()
        if "tq_new" in pair and "simple_storage" in pair
    ]


def summarize_viztracer_round(round_dir: str | Path) -> dict[str, Any]:
    resolved_round_dir = Path(round_dir)
    stage_breakdown_us: dict[str, float] = {}
    event_counts: dict[str, int] = {}
    component_summaries: list[dict[str, Any]] = []

    for trace_path in sorted(resolved_round_dir.glob("*.json")):
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        component_breakdown_us: dict[str, float] = {}
        component_event_counts: dict[str, int] = {}

        for event in payload.get("traceEvents", []):
            if event.get("ph") != "X":
                continue
            stage_name = extract_stage_name(event.get("name"))
            if stage_name is None:
                continue

            duration_us = float(event.get("dur", 0.0))
            stage_breakdown_us[stage_name] = stage_breakdown_us.get(stage_name, 0.0) + duration_us
            event_counts[stage_name] = event_counts.get(stage_name, 0) + 1
            component_breakdown_us[stage_name] = component_breakdown_us.get(stage_name, 0.0) + duration_us
            component_event_counts[stage_name] = component_event_counts.get(stage_name, 0) + 1

        component_summaries.append(
            {
                "component": trace_path.stem,
                "trace_path": str(trace_path),
                "stage_breakdown_ms": {
                    stage_name: duration_us / 1_000.0
                    for stage_name, duration_us in sorted(component_breakdown_us.items())
                },
                "event_counts_by_stage": dict(sorted(component_event_counts.items())),
            }
        )

    total_stage_us = sum(stage_breakdown_us.values())
    stage_breakdown_ms = {
        stage_name: duration_us / 1_000.0 for stage_name, duration_us in sorted(stage_breakdown_us.items())
    }
    stage_breakdown_pct = {
        stage_name: (duration_us / total_stage_us * 100.0) if total_stage_us > 0 else 0.0
        for stage_name, duration_us in sorted(stage_breakdown_us.items())
    }

    return {
        "round_dir": str(resolved_round_dir),
        "trace_files": len(component_summaries),
        "components": component_summaries,
        "stage_breakdown_ms": stage_breakdown_ms,
        "stage_breakdown_pct": stage_breakdown_pct,
        "event_counts_by_stage": dict(sorted(event_counts.items())),
    }
