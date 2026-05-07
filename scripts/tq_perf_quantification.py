from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transfer_queue.utils.perf_report import pair_common_core_results, summarize_viztracer_round

PYTHON = sys.executable
COMMON_CORE_MODE = "common-core"

RESULTS_ROOT = REPO_ROOT / "docs" / "perf" / "results"
TRACES_ROOT = REPO_ROOT / "docs" / "perf" / "traces"
REPORTS_ROOT = REPO_ROOT / "docs" / "perf" / "reports"


@dataclass(frozen=True)
class BaselineCase:
    config: str
    shards: int
    clients: int
    rounds: int
    matrix_label: str = "local"

    @property
    def slug(self) -> str:
        return f"{self.config}-{self.matrix_label}-{self.shards}shards-{self.clients}clients-r{self.rounds}"


@dataclass(frozen=True)
class TraceCase:
    config: str
    shards: int
    clients: int
    matrix_label: str = "local"

    @property
    def slug(self) -> str:
        return f"{self.config}-{self.matrix_label}-{self.shards}shards-{self.clients}clients-profile"


@dataclass(frozen=True)
class MicrobenchCase:
    put_config: str
    preset: str
    repeats: int
    warmup: int
    matrix_label: str = ""

    @property
    def slug(self) -> str:
        matrix_part = f"-{self.matrix_label}" if self.matrix_label else ""
        return f"microbench-{self.put_config}-{self.preset}{matrix_part}-r{self.repeats}-w{self.warmup}"


@dataclass(frozen=True)
class TqFeatureCase:
    label: str
    config: str
    mode: str
    rounds: int
    shards: int
    n_samples_per_prompt: int = 1
    shared_columns: tuple[str, ...] = ()
    extra_flags: tuple[str, ...] = ()
    matrix_label: str = ""

    @property
    def slug(self) -> str:
        return self.label

    @property
    def filename_slug(self) -> str:
        if not self.matrix_label:
            return self.label
        return f"{self.label}-{self.matrix_label}"


LOCAL_BASELINE_CASES = [
    BaselineCase(config=config, shards=1, clients=clients, rounds=3)
    for config in ("debug", "tiny")
    for clients in (1, 4)
] + [BaselineCase(config="small", shards=1, clients=clients, rounds=2) for clients in (1, 4)]

SERVER_BASELINE_CASES = (
    [
        BaselineCase(config=config, shards=shards, clients=clients, rounds=5, matrix_label="server")
        for config in ("debug", "tiny", "small")
        for shards in (1, 4, 8)
        for clients in (1, 4)
    ]
    + [
        BaselineCase(config="large", shards=shards, clients=clients, rounds=2, matrix_label="server")
        for shards in (1, 4, 8)
        for clients in (1, 4)
    ]
    + [BaselineCase(config="huge", shards=shards, clients=1, rounds=1, matrix_label="server") for shards in (1, 4, 8)]
)

LOCAL_TRACE_CASES = [
    TraceCase(config="debug", shards=1, clients=1),
    TraceCase(config="tiny", shards=1, clients=1),
]

SERVER_TRACE_CASES = LOCAL_TRACE_CASES + [TraceCase(config="small", shards=8, clients=1)]
SERVER_TRACE_CASES = [replace(case, matrix_label="server") for case in SERVER_TRACE_CASES]

LOCAL_MICROBENCH_CASES = [
    MicrobenchCase(put_config="tiny", preset="debug", repeats=9, warmup=2),
    MicrobenchCase(put_config="small", preset="debug", repeats=7, warmup=2),
]

SERVER_MICROBENCH_CASES = [
    MicrobenchCase(put_config="tiny", preset="debug", repeats=11, warmup=3, matrix_label="server"),
    MicrobenchCase(put_config="small", preset="debug", repeats=9, warmup=3, matrix_label="server"),
    MicrobenchCase(put_config="large", preset="debug", repeats=5, warmup=1, matrix_label="server"),
    MicrobenchCase(put_config="huge", preset="debug", repeats=3, warmup=1, matrix_label="server"),
]

LOCAL_TQ_FEATURE_CASES = [
    TqFeatureCase(
        label="mixed-uuid-tiny-baseline",
        config="tiny",
        mode="uuid",
        rounds=1,
        shards=1,
    ),
    TqFeatureCase(
        label="mixed-uuid-tiny-force-tensor-only",
        config="tiny",
        mode="uuid",
        rounds=1,
        shards=1,
        extra_flags=("--force-tensor-only-raw-path",),
    ),
    TqFeatureCase(
        label="mixed-uuid-tiny-bypass-uuid-allocation",
        config="tiny",
        mode="uuid",
        rounds=1,
        shards=1,
        extra_flags=("--bypass-uuid-allocation",),
    ),
    TqFeatureCase(
        label="common-core-tiny-preallocate-indexes",
        config="tiny",
        mode="dict",
        rounds=3,
        shards=1,
        extra_flags=("--preallocate-indexes",),
    ),
    TqFeatureCase(
        label="group-shared-tiny-baseline",
        config="tiny",
        mode="uuid",
        rounds=1,
        shards=1,
        n_samples_per_prompt=4,
        shared_columns=("prompt_uuid",),
    ),
    TqFeatureCase(
        label="group-shared-tiny-disable-written-checks",
        config="tiny",
        mode="uuid",
        rounds=1,
        shards=1,
        n_samples_per_prompt=4,
        shared_columns=("prompt_uuid",),
        extra_flags=("--disable-shared-column-written-checks",),
    ),
    TqFeatureCase(
        label="group-shared-tiny-disable-mark-writes",
        config="tiny",
        mode="uuid",
        rounds=1,
        shards=1,
        n_samples_per_prompt=4,
        shared_columns=("prompt_uuid",),
        extra_flags=("--disable-shared-column-mark-writes",),
    ),
]

SERVER_TQ_FEATURE_CASES = [replace(case, matrix_label="server") for case in LOCAL_TQ_FEATURE_CASES] + [
    TqFeatureCase(
        label="mixed-uuid-small-baseline",
        config="small",
        mode="uuid",
        rounds=3,
        shards=8,
        matrix_label="server",
    ),
    TqFeatureCase(
        label="group-shared-small-baseline",
        config="small",
        mode="uuid",
        rounds=3,
        shards=8,
        n_samples_per_prompt=4,
        shared_columns=("prompt_uuid",),
        matrix_label="server",
    ),
]


def _ensure_output_dirs() -> None:
    for path in (
        RESULTS_ROOT / "simple_storage",
        RESULTS_ROOT / "tq_new",
        RESULTS_ROOT / "tq_new" / "feature_tax",
        RESULTS_ROOT / "tq_new" / "ablations",
        RESULTS_ROOT / "microbench",
        TRACES_ROOT / "simple_storage",
        TRACES_ROOT / "tq_new",
        REPORTS_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _run_command(
    *,
    cmd: list[str],
    manifest: list[dict[str, Any]],
    force: bool,
    output_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    if output_path is not None and output_path.exists() and not force:
        manifest.append({"cmd": cmd, "skipped": True, "output_path": str(output_path)})
        return

    subprocess.run(
        cmd,
        check=True,
        cwd=REPO_ROOT,
        env=(os.environ | env) if env is not None else None,
    )
    manifest.append({"cmd": cmd, "skipped": False, "output_path": str(output_path) if output_path else None})


def _common_core_result_path(implementation: str, case: BaselineCase) -> Path:
    return RESULTS_ROOT / implementation / f"common-core-{case.slug}.json"


def _trace_output_dir(implementation: str, case: TraceCase) -> Path:
    return TRACES_ROOT / implementation / case.slug


def _trace_result_path(implementation: str, case: TraceCase) -> Path:
    return RESULTS_ROOT / implementation / f"common-core-{case.slug}.json"


def _microbench_output_path(case: MicrobenchCase) -> Path:
    return RESULTS_ROOT / "microbench" / f"{case.slug}.json"


def _feature_output_path(case: TqFeatureCase) -> Path:
    folder = "ablations" if case.label.startswith("common-core") else "feature_tax"
    return RESULTS_ROOT / "tq_new" / folder / f"{case.filename_slug}.json"


def _run_baselines(cases: Iterable[BaselineCase], manifest: list[dict[str, Any]], force: bool) -> list[Path]:
    output_paths: list[Path] = []
    for case in cases:
        simple_output = _common_core_result_path("simple_storage", case)
        tq_output = _common_core_result_path("tq_new", case)
        _run_command(
            cmd=[
                PYTHON,
                "scripts/performance_test.py",
                "tq-normal",
                "--comparison-mode",
                COMMON_CORE_MODE,
                "--config",
                case.config,
                "--rounds",
                str(case.rounds),
                "--shards",
                str(case.shards),
                "--num-clients",
                str(case.clients),
                "--output",
                str(simple_output),
            ],
            manifest=manifest,
            force=force,
            output_path=simple_output,
        )
        _run_command(
            cmd=[
                PYTHON,
                "tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py",
                "--comparison-mode",
                COMMON_CORE_MODE,
                "--config",
                case.config,
                "--mode",
                "dict",
                "--rounds",
                str(case.rounds),
                "--shards",
                str(case.shards),
                "--num-clients",
                str(case.clients),
                "--output",
                str(tq_output),
            ],
            manifest=manifest,
            force=force,
            output_path=tq_output,
        )
        output_paths.extend([simple_output, tq_output])
    return output_paths


def _run_traces(cases: Iterable[TraceCase], manifest: list[dict[str, Any]], force: bool) -> dict[str, list[Path]]:
    trace_outputs: dict[str, list[Path]] = {"simple_storage": [], "tq_new": []}
    for case in cases:
        simple_trace_dir = _trace_output_dir("simple_storage", case)
        tq_trace_dir = _trace_output_dir("tq_new", case)
        simple_result = _trace_result_path("simple_storage", case)
        tq_result = _trace_result_path("tq_new", case)
        _run_command(
            cmd=[
                PYTHON,
                "scripts/performance_test.py",
                "tq-normal",
                "--comparison-mode",
                COMMON_CORE_MODE,
                "--config",
                case.config,
                "--rounds",
                "1",
                "--shards",
                str(case.shards),
                "--num-clients",
                str(case.clients),
                "--profile",
                "--profile-output-dir",
                str(simple_trace_dir.resolve()),
                "--output",
                str(simple_result),
            ],
            manifest=manifest,
            force=force,
            output_path=simple_result,
        )
        _run_command(
            cmd=[
                PYTHON,
                "tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py",
                "--comparison-mode",
                COMMON_CORE_MODE,
                "--config",
                case.config,
                "--mode",
                "dict",
                "--rounds",
                "1",
                "--shards",
                str(case.shards),
                "--num-clients",
                str(case.clients),
                "--profile",
                "--profile-output-dir",
                str(tq_trace_dir.resolve()),
                "--output",
                str(tq_result),
            ],
            manifest=manifest,
            force=force,
            output_path=tq_result,
        )
        trace_outputs["simple_storage"].append(simple_trace_dir)
        trace_outputs["tq_new"].append(tq_trace_dir)
    return trace_outputs


def _run_microbench(cases: Iterable[MicrobenchCase], manifest: list[dict[str, Any]], force: bool) -> list[Path]:
    outputs: list[Path] = []
    for case in cases:
        output_path = _microbench_output_path(case)
        _run_command(
            cmd=[
                PYTHON,
                "scripts/perf_microbench.py",
                "--put-config",
                case.put_config,
                "--preset",
                case.preset,
                "--repeats",
                str(case.repeats),
                "--warmup",
                str(case.warmup),
                "--output",
                str(output_path),
            ],
            manifest=manifest,
            force=force,
            output_path=output_path,
        )
        outputs.append(output_path)
    return outputs


def _run_tq_features(cases: Iterable[TqFeatureCase], manifest: list[dict[str, Any]], force: bool) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for case in cases:
        output_path = _feature_output_path(case)
        cmd = [
            PYTHON,
            "tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py",
            "--config",
            case.config,
            "--mode",
            case.mode,
            "--rounds",
            str(case.rounds),
            "--shards",
            str(case.shards),
            "--num-clients",
            "1",
            "--output",
            str(output_path),
            "--n-samples-per-prompt",
            str(case.n_samples_per_prompt),
        ]
        if case.label.startswith("common-core-"):
            cmd.extend(["--comparison-mode", COMMON_CORE_MODE])
        for shared_column in case.shared_columns:
            cmd.extend(["--shared-columns", shared_column])
        cmd.extend(case.extra_flags)
        _run_command(cmd=cmd, manifest=manifest, force=force, output_path=output_path)
        outputs[case.label] = output_path
    return outputs


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_round_trip_gap_ms(pair: dict[str, Any]) -> float:
    tq_entry = pair["tq_new"]
    simple_entry = pair["simple_storage"]
    tq_per_round_ms = (float(tq_entry["round_trip_seconds"]) / int(tq_entry["rounds"])) * 1000.0
    simple_per_round_ms = (float(simple_entry["round_trip_seconds"]) / int(simple_entry["rounds"])) * 1000.0
    return tq_per_round_ms - simple_per_round_ms


def _harmonic_throughput_proxy(put_mean: float, get_mean: float) -> float:
    if put_mean <= 0 or get_mean <= 0:
        return 0.0
    return 2.0 / ((1.0 / put_mean) + (1.0 / get_mean))


def _load_legacy_tq_result(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    by_operation = {entry["operation"]: entry for entry in payload}
    put_mean = float(by_operation["PUT"]["stats_gbps"]["mean"])
    get_mean = float(by_operation["GET"]["stats_gbps"]["mean"])
    first_entry = payload[0]
    return {
        "path": str(path),
        "label": path.stem,
        "put_mean_gbps": put_mean,
        "get_mean_gbps": get_mean,
        "round_trip_proxy_gbps": _harmonic_throughput_proxy(put_mean, get_mean),
        "n_samples_per_prompt": int(first_entry.get("n_samples_per_prompt", 1)),
        "shared_columns": list(first_entry.get("shared_columns", [])),
        "ablation_label": first_entry.get("ablation", {}).get("label", "baseline"),
    }


def _baseline_rows(pairs: list[dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        tq_entry = pair["tq_new"]
        simple_entry = pair["simple_storage"]
        rows.append(
            {
                "workload_name": tq_entry["workload_name"],
                "config": tq_entry["workload_name"].replace("common-core-", "", 1),
                "shards": int(tq_entry["num_shards_or_storage_units"]),
                "clients": int(tq_entry["num_clients"]),
                "tq_round_trip_gbps": float(tq_entry["round_trip_gbps"]),
                "simple_round_trip_gbps": float(simple_entry["round_trip_gbps"]),
                "tq_round_trip_ms_per_round": float(tq_entry["round_trip_seconds"]) / int(tq_entry["rounds"]) * 1000.0,
                "simple_round_trip_ms_per_round": float(simple_entry["round_trip_seconds"])
                / int(simple_entry["rounds"])
                * 1000.0,
                "gap_ms_per_round": _mean_round_trip_gap_ms(pair),
                "tq_vs_simple_ratio": (
                    float(tq_entry["round_trip_gbps"]) / float(simple_entry["round_trip_gbps"])
                    if float(simple_entry["round_trip_gbps"]) > 0
                    else None
                ),
                "tq_path": str(tq_entry.get("_source_path", "")),
                "simple_path": str(simple_entry.get("_source_path", "")),
            }
        )
    return sorted(rows, key=lambda row: (row["config"], row["shards"], row["clients"]))


def _trace_rows(trace_cases: Iterable[TraceCase]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in trace_cases:
        tq_summary = summarize_viztracer_round(_trace_output_dir("tq_new", case) / "round_01")
        simple_summary = summarize_viztracer_round(_trace_output_dir("simple_storage", case) / "round_01")
        all_stages = sorted(set(tq_summary["stage_breakdown_ms"]) | set(simple_summary["stage_breakdown_ms"]))
        stage_delta_ms = {
            stage: float(tq_summary["stage_breakdown_ms"].get(stage, 0.0))
            - float(simple_summary["stage_breakdown_ms"].get(stage, 0.0))
            for stage in all_stages
        }
        rows.append(
            {
                "config": case.config,
                "shards": case.shards,
                "clients": case.clients,
                "tq_path": str((_trace_output_dir("tq_new", case) / "round_01").resolve()),
                "simple_path": str((_trace_output_dir("simple_storage", case) / "round_01").resolve()),
                "tq_summary": tq_summary,
                "simple_summary": simple_summary,
                "stage_delta_ms": stage_delta_ms,
            }
        )
    return rows


def _microbench_rows(microbench_paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in microbench_paths:
        payload = _load_json(path)
        for comparison in payload.get("comparisons", []):
            row = dict(comparison)
            row["path"] = str(path)
            rows.append(row)
    return rows


def _representative_baseline_row(baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = next(
        (row for row in baseline_rows if row["config"] == "tiny" and row["shards"] == 4 and row["clients"] == 1),
        None,
    )
    if preferred is not None:
        return preferred
    return max(baseline_rows, key=lambda row: (row["gap_ms_per_round"], row["shards"], row["clients"]))


def _representative_trace_row(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = next(
        (row for row in trace_rows if row["config"] == "tiny" and row["shards"] == 4 and row["clients"] == 1),
        None,
    )
    if preferred is not None:
        return preferred
    preferred = next(
        (row for row in trace_rows if row["config"] == "tiny" and row["shards"] == 1 and row["clients"] == 1),
        None,
    )
    if preferred is not None:
        return preferred
    return trace_rows[0]


def _attribution_payload(
    *,
    baseline_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    microbench_rows: list[dict[str, Any]],
    feature_rows: dict[str, dict[str, Any]],
    server_command: str,
) -> dict[str, Any]:
    representative_baseline = _representative_baseline_row(baseline_rows)
    representative_trace = _representative_trace_row(trace_rows)
    common_core_gap_ms = max(0.0, float(representative_baseline["gap_ms_per_round"]))

    control_plane_ms = sum(
        max(0.0, float(representative_trace["stage_delta_ms"].get(stage, 0.0)))
        for stage in ("allocate_indexes", "route_targets", "send_request")
    )
    bookkeeping_ms = sum(
        max(0.0, float(representative_trace["stage_delta_ms"].get(stage, 0.0)))
        for stage in ("fetch_targets", "status_update")
    )
    write_path_ms = sum(
        max(0.0, float(representative_trace["stage_delta_ms"].get(stage, 0.0)))
        for stage in ("storage_put", "aggregate_results")
    )

    microbench_by_key = {row["comparison_key"]: row for row in microbench_rows}
    selection_ratio = float(microbench_by_key.get("selection", {}).get("median_time_ratio_tq_over_simple", 1.0) or 1.0)
    serialization_ratio = float(
        microbench_by_key.get("serialization", {}).get("median_time_ratio_tq_over_simple", 1.0) or 1.0
    )
    storage_put_ratio = float(
        microbench_by_key.get("storage_put", {}).get("median_time_ratio_tq_over_simple", 1.0) or 1.0
    )

    mixed_baseline = feature_rows.get("mixed-uuid-tiny-baseline")
    mixed_force_raw = feature_rows.get("mixed-uuid-tiny-force-tensor-only")
    mixed_bypass_uuid = feature_rows.get("mixed-uuid-tiny-bypass-uuid-allocation")
    group_baseline = feature_rows.get("group-shared-tiny-baseline")
    group_no_written = feature_rows.get("group-shared-tiny-disable-written-checks")
    group_no_mark = feature_rows.get("group-shared-tiny-disable-mark-writes")

    def _throughput_tax_pct(baseline: dict[str, Any] | None, candidate: dict[str, Any] | None) -> float:
        if baseline is None or candidate is None:
            return 0.0
        baseline_proxy = float(baseline["round_trip_proxy_gbps"])
        candidate_proxy = float(candidate["round_trip_proxy_gbps"])
        if baseline_proxy <= 0 or candidate_proxy <= 0:
            return 0.0
        return max(0.0, ((candidate_proxy / baseline_proxy) - 1.0) * 100.0)

    preallocate_tax_pct = 0.0
    preallocate_row = feature_rows.get("common-core-tiny-preallocate-indexes")
    common_core_baseline = next(
        (row for row in baseline_rows if row["config"] == "tiny" and row["shards"] == 4 and row["clients"] == 1),
        representative_baseline,
    )
    if common_core_baseline is not None and preallocate_row is not None:
        baseline_tq_gbps = float(common_core_baseline["tq_round_trip_gbps"])
        candidate_gbps = float(preallocate_row["round_trip_gbps"])
        if baseline_tq_gbps > 0 and candidate_gbps > 0:
            preallocate_tax_pct = max(0.0, ((candidate_gbps / baseline_tq_gbps) - 1.0) * 100.0)

    bottlenecks = [
        {
            "name": "Control-plane extra RPC / routing",
            "kind": "common_core_gap",
            "estimated_gap_ms": control_plane_ms,
            "estimated_gap_share_pct": (control_plane_ms / common_core_gap_ms * 100.0)
            if common_core_gap_ms > 0
            else 0.0,
            "evidence": {
                "baseline": str(common_core_baseline["tq_path"]) if common_core_baseline else None,
                "trace": representative_trace["tq_path"],
                "microbench": None,
                "ablation": str(preallocate_row["source_path"]) if preallocate_row else None,
            },
        },
        {
            "name": "Python set/dict sampling bookkeeping",
            "kind": "common_core_gap",
            "estimated_gap_ms": bookkeeping_ms,
            "estimated_gap_share_pct": (bookkeeping_ms / common_core_gap_ms * 100.0) if common_core_gap_ms > 0 else 0.0,
            "evidence": {
                "baseline": str(common_core_baseline["tq_path"]) if common_core_baseline else None,
                "trace": representative_trace["tq_path"],
                "microbench": microbench_by_key.get("selection", {}).get("path"),
                "ablation": str(group_no_written["path"]) if group_no_written else None,
            },
            "supporting_metric": {"selection_time_ratio_tq_over_simple": selection_ratio},
        },
        {
            "name": "Write-side flatten/concat/copy",
            "kind": "common_core_gap",
            "estimated_gap_ms": write_path_ms,
            "estimated_gap_share_pct": (write_path_ms / common_core_gap_ms * 100.0) if common_core_gap_ms > 0 else 0.0,
            "evidence": {
                "baseline": str(common_core_baseline["tq_path"]) if common_core_baseline else None,
                "trace": representative_trace["tq_path"],
                "microbench": microbench_by_key.get("serialization", {}).get("path"),
                "ablation": str(preallocate_row["source_path"]) if preallocate_row else None,
            },
            "supporting_metric": {
                "serialization_time_ratio_tq_over_simple": serialization_ratio,
                "storage_put_time_ratio_tq_over_simple": storage_put_ratio,
            },
        },
        {
            "name": "Pickle / mixed-object path",
            "kind": "feature_tax",
            "throughput_tax_pct": _throughput_tax_pct(mixed_baseline, mixed_force_raw),
            "evidence": {
                "baseline": mixed_baseline["path"] if mixed_baseline else None,
                "trace": None,
                "microbench": None,
                "ablation": mixed_force_raw["path"] if mixed_force_raw else None,
            },
        },
        {
            "name": "UUID allocation path",
            "kind": "feature_tax",
            "throughput_tax_pct": _throughput_tax_pct(mixed_baseline, mixed_bypass_uuid),
            "evidence": {
                "baseline": mixed_baseline["path"] if mixed_baseline else None,
                "trace": None,
                "microbench": None,
                "ablation": mixed_bypass_uuid["path"] if mixed_bypass_uuid else None,
            },
        },
        {
            "name": "Shared-column written checks",
            "kind": "feature_tax",
            "throughput_tax_pct": _throughput_tax_pct(group_baseline, group_no_written),
            "evidence": {
                "baseline": group_baseline["path"] if group_baseline else None,
                "trace": None,
                "microbench": None,
                "ablation": group_no_written["path"] if group_no_written else None,
            },
        },
        {
            "name": "Shared-column mark writes",
            "kind": "feature_tax",
            "throughput_tax_pct": _throughput_tax_pct(group_baseline, group_no_mark),
            "evidence": {
                "baseline": group_baseline["path"] if group_baseline else None,
                "trace": None,
                "microbench": None,
                "ablation": group_no_mark["path"] if group_no_mark else None,
            },
        },
    ]

    def _sort_key(item: dict[str, Any]) -> float:
        if item["kind"] == "common_core_gap":
            return float(item.get("estimated_gap_share_pct", 0.0))
        return float(item.get("throughput_tax_pct", 0.0))

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "representative_common_core_case": representative_baseline,
        "representative_trace_case": {
            "config": representative_trace["config"],
            "shards": representative_trace["shards"],
            "clients": representative_trace["clients"],
            "stage_delta_ms": representative_trace["stage_delta_ms"],
            "tq_trace_dir": representative_trace["tq_path"],
            "simple_trace_dir": representative_trace["simple_path"],
        },
        "ranked_bottlenecks": sorted(bottlenecks, key=_sort_key, reverse=True),
        "open_questions": [
            "当前 benchmark 路径没有覆盖 version recording，因此本地还无法量化 version-tax。",
            "mixed 和 grouped workload 目前只有 tq_new 的特性税测量，还没有语义完全对齐的 SimpleStorage 对照。",
            "grouped/shared 的 tiny 试跑已经能产出吞吐数字，但本地校验仍然报 fetched data=None，所以这些数字在 get-path 语义修复前只能作为方向性证据。",
            "考虑到本地机器只有 5 GB 可用内存，这一轮 local matrix 采用了保守覆盖；更高置信度的 small/8-shard 和 huge 规模结果建议在服务器矩阵上补齐。",
        ],
        "server_handoff_command": server_command,
    }


def _format_markdown_report(
    *,
    baseline_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    microbench_rows: list[dict[str, Any]],
    feature_rows: dict[str, dict[str, Any]],
    attribution: dict[str, Any],
    manifest: list[dict[str, Any]],
    matrix_name: str,
) -> str:
    lines = [
        "# TQ vs SimpleStorage 性能总结",
        "",
        f"- 生成时间: `{time.strftime('%Y-%m-%d %H:%M:%S %z')}`",
        f"- 实验矩阵: `{matrix_name}`",
        "",
        "## 主要瓶颈",
        "",
    ]
    for bottleneck in attribution["ranked_bottlenecks"]:
        if bottleneck["kind"] == "common_core_gap":
            lines.append(
                f"- `{bottleneck['name']}`: 约占代表性 common-core 性能差距的 "
                f"`{bottleneck['estimated_gap_share_pct']:.1f}%` "
                f"(`{bottleneck['estimated_gap_ms']:.2f} ms/round`)."
            )
        else:
            lines.append(
                f"- `{bottleneck['name']}`: 相比其优化 ablation，约有 "
                f"`{bottleneck['throughput_tax_pct']:.1f}%` 的吞吐损耗。"
            )
    lines.extend(["", "## Common-Core 基线", ""])
    lines.append("| config | shards | clients | tq rt Gbps | simple rt Gbps | gap ms/round | tq/simple |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in baseline_rows:
        lines.append(
            f"| {row['config']} | {row['shards']} | {row['clients']} | {row['tq_round_trip_gbps']:.3f} | "
            f"{row['simple_round_trip_gbps']:.3f} | {row['gap_ms_per_round']:.2f} | {row['tq_vs_simple_ratio']:.3f} |"
        )

    lines.extend(["", "## Trace 分解", ""])
    for row in trace_rows:
        lines.append(
            f"- `{row['config']}` `shards={row['shards']}` `clients={row['clients']}`: "
            f"tq_new 正向增量最大的阶段 = "
            + ", ".join(
                f"`{stage}={delta_ms:.2f}ms`"
                for stage, delta_ms in sorted(
                    row["stage_delta_ms"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if delta_ms > 0
            )
        )

    lines.extend(["", "## 微基准", ""])
    for row in sorted(microbench_rows, key=lambda item: item["comparison_key"]):
        lines.append(
            f"- `{row['comparison_key']}`: 更快实现=`{row['faster_implementation']}` "
            f"tq/simple=`{row['median_time_ratio_tq_over_simple']:.3f}` "
            f"来源=[{Path(row['path']).name}]({Path(row['path']).as_posix()})"
        )

    lines.extend(["", "## TQ 特性税", ""])
    for label, row in sorted(feature_rows.items()):
        if "put_mean_gbps" in row:
            lines.append(
                f"- `{label}`: put=`{row['put_mean_gbps']:.3f} Gbps`, get=`{row['get_mean_gbps']:.3f} Gbps`, "
                f"rt-proxy=`{row['round_trip_proxy_gbps']:.3f} Gbps`, "
                f"`n_samples_per_prompt={row['n_samples_per_prompt']}` 共享列=`{','.join(row['shared_columns']) or '<none>'}`"
            )
        else:
            lines.append(f"- `{label}`: round-trip=`{row['round_trip_gbps']:.3f} Gbps`")

    lines.extend(["", "## 开放问题", ""])
    for item in attribution["open_questions"]:
        lines.append(f"- {item}")

    lines.extend(["", "## 执行命令", ""])
    for item in manifest:
        prefix = "(已存在，跳过) " if item["skipped"] else ""
        lines.append(f"- {prefix}`{' '.join(item['cmd'])}`")
    lines.append("")
    return "\n".join(lines)


def _annotate_source_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in paths:
        for entry in _load_json(path):
            annotated = dict(entry)
            annotated["_source_path"] = str(path)
            entries.append(annotated)
    return entries


def run_suite(matrix_name: str, force: bool) -> dict[str, Any]:
    _ensure_output_dirs()
    manifest: list[dict[str, Any]] = []
    if matrix_name == "server":
        baseline_cases = SERVER_BASELINE_CASES
        trace_cases = SERVER_TRACE_CASES
        microbench_cases = SERVER_MICROBENCH_CASES
        feature_cases = SERVER_TQ_FEATURE_CASES
    else:
        baseline_cases = LOCAL_BASELINE_CASES
        trace_cases = LOCAL_TRACE_CASES
        microbench_cases = LOCAL_MICROBENCH_CASES
        feature_cases = LOCAL_TQ_FEATURE_CASES

    baseline_paths = _run_baselines(baseline_cases, manifest, force)
    _run_traces(trace_cases, manifest, force)
    microbench_paths = _run_microbench(microbench_cases, manifest, force)
    feature_paths = _run_tq_features(feature_cases, manifest, force)

    baseline_pairs = pair_common_core_results(_annotate_source_paths(baseline_paths))
    baseline_rows = _baseline_rows(baseline_pairs)
    trace_rows = _trace_rows(trace_cases)
    microbench_rows = _microbench_rows(microbench_paths)
    feature_rows: dict[str, dict[str, Any]] = {}
    for label, path in feature_paths.items():
        if label == "common-core-tiny-preallocate-indexes":
            payload = _load_json(path)[0]
            feature_rows[label] = {
                "path": str(path),
                "source_path": str(path),
                "round_trip_gbps": float(payload["round_trip_gbps"]),
                "label": label,
            }
            continue
        feature_rows[label] = _load_legacy_tq_result(path)
        feature_rows[label]["path"] = str(path)

    server_command = f"{PYTHON} {REPO_ROOT / 'scripts' / 'tq_perf_quantification.py'} --matrix server"
    attribution = _attribution_payload(
        baseline_rows=baseline_rows,
        trace_rows=trace_rows,
        microbench_rows=microbench_rows,
        feature_rows=feature_rows,
        server_command=server_command,
    )

    summary_path = REPORTS_ROOT / "tq_vs_simple_storage_summary.md"
    attribution_path = REPORTS_ROOT / "tq_vs_simple_storage_attribution.json"
    summary_path.write_text(
        _format_markdown_report(
            baseline_rows=baseline_rows,
            trace_rows=trace_rows,
            microbench_rows=microbench_rows,
            feature_rows=feature_rows,
            attribution=attribution,
            manifest=manifest,
            matrix_name=matrix_name,
        ),
        encoding="utf-8",
    )
    attribution_path.write_text(json.dumps(attribution, indent=2), encoding="utf-8")

    return {
        "summary_path": str(summary_path),
        "attribution_path": str(attribution_path),
        "baseline_paths": [str(path) for path in baseline_paths],
        "microbench_paths": [str(path) for path in microbench_paths],
        "feature_paths": [str(path) for path in feature_paths.values()],
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local-safe tq_new vs SimpleStorage perf quantification suite.")
    parser.add_argument("--matrix", choices=["local", "server"], default="local")
    parser.add_argument(
        "--force", action="store_true", help="Re-run commands even when the target output already exists."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_suite(matrix_name=args.matrix, force=args.force)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
