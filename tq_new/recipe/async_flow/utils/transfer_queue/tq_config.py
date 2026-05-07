# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
"""TransferQueue global configuration constants and ablation helpers."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable, Mapping

DEFAULT_TOPIC = "experience"
TQ_GROUP_SHARED_COLUMNS_ENV_VAR = "TQ_GROUP_SHARED_COLUMNS"
NUM_SAMPLE_PER_SEGMENT = 1024
MAX_CONCURRENT_GETS = 100
ZMQ_HWM = 2000
PADDED_COLUMNS = {"prompt", "response"}
# PADDED_COLUMNS = {}

TQ_NEW_ABLATIONS_ENV_VAR = "TQ_NEW_ABLATIONS"


def _parse_group_shared_column_tokens(raw_value: str | Iterable[str] | None) -> set[str]:
    if raw_value is None:
        return set()

    if isinstance(raw_value, str):
        candidates = raw_value.split(",")
    else:
        candidates = []
        for item in raw_value:
            candidates.extend(str(item).split(","))

    return {token.strip() for token in candidates if token and token.strip()}


def format_group_shared_columns(columns: Iterable[str] | None) -> str:
    return ",".join(sorted(_parse_group_shared_column_tokens(columns)))


def parse_group_shared_columns(
    args: argparse.Namespace | None = None,
    env: Mapping[str, str] | None = None,
) -> set[str]:
    env = env or os.environ
    enabled_columns = _parse_group_shared_column_tokens(env.get(TQ_GROUP_SHARED_COLUMNS_ENV_VAR))

    if args is not None:
        enabled_columns.update(_parse_group_shared_column_tokens(getattr(args, "shared_columns", None)))

    return enabled_columns


GROUP_SHARED_COLUMNS = parse_group_shared_columns()


@dataclass(frozen=True, slots=True)
class TransferQueueAblationConfig:
    disable_shared_column_written_checks: bool = False
    disable_shared_column_mark_writes: bool = False
    disable_version_recording: bool = False
    bypass_uuid_allocation: bool = False
    preallocate_indexes: bool = False
    force_tensor_only_raw_path: bool = False

    def active_flags(self) -> list[str]:
        return [field.name for field in fields(self) if getattr(self, field.name)]

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)

    def label(self) -> str:
        active = self.active_flags()
        return ",".join(active) if active else "baseline"

    def to_env_value(self) -> str:
        return ",".join(self.active_flags())

    def to_metadata(self) -> dict[str, Any]:
        return {
            "label": self.label(),
            "active_flags": self.active_flags(),
            "flags": self.as_dict(),
            "env_var": TQ_NEW_ABLATIONS_ENV_VAR,
        }


ABLATION_FLAG_HELP = {
    "disable_shared_column_written_checks": "Skip manager written-status checks and always resend shared columns.",
    "disable_shared_column_mark_writes": "Skip manager-side mark-written updates after sending shared columns.",
    "disable_version_recording": "Skip manager-side version recording after successful puts.",
    "bypass_uuid_allocation": "Allocate sequential indexes even when *_uuid/_uid columns are present.",
    "preallocate_indexes": "Allocate put indexes before the timed PUT section in benchmark runs.",
    "force_tensor_only_raw_path": "Force benchmark workloads onto tensor-only raw serialization paths.",
}


def _ablation_field_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(TransferQueueAblationConfig))


ABLATION_FIELD_NAMES = _ablation_field_names()


def normalize_ablation_config(
    ablation: TransferQueueAblationConfig | Mapping[str, Any] | None = None,
) -> TransferQueueAblationConfig:
    if ablation is None:
        return TransferQueueAblationConfig()
    if isinstance(ablation, TransferQueueAblationConfig):
        return ablation
    if isinstance(ablation, Mapping):
        flag_mapping = ablation.get("flags") if isinstance(ablation.get("flags"), Mapping) else ablation
        return TransferQueueAblationConfig(
            **{name: bool(flag_mapping.get(name, False)) for name in ABLATION_FIELD_NAMES}
        )
    raise TypeError(f"Unsupported ablation config type: {type(ablation)!r}")


def _parse_ablation_tokens(raw_value: str | Iterable[str] | None) -> set[str]:
    if raw_value is None:
        return set()

    if isinstance(raw_value, str):
        candidates = raw_value.split(",")
    else:
        candidates = []
        for item in raw_value:
            candidates.extend(str(item).split(","))

    normalized_tokens = {token.strip().replace("-", "_") for token in candidates if token and token.strip()}
    unknown_tokens = normalized_tokens - set(ABLATION_FIELD_NAMES)
    if unknown_tokens:
        raise ValueError(
            f"Unknown ablation flag(s): {sorted(unknown_tokens)}. Known flags: {list(ABLATION_FIELD_NAMES)}"
        )
    return normalized_tokens


def parse_ablation_config(
    args: argparse.Namespace | None = None,
    env: Mapping[str, str] | None = None,
) -> TransferQueueAblationConfig:
    env = env or os.environ
    enabled_flags = _parse_ablation_tokens(env.get(TQ_NEW_ABLATIONS_ENV_VAR))

    if args is not None:
        enabled_flags.update(_parse_ablation_tokens(getattr(args, "ablation", None)))
        for flag_name in ABLATION_FIELD_NAMES:
            if bool(getattr(args, flag_name, False)):
                enabled_flags.add(flag_name)

    return TransferQueueAblationConfig(**{flag_name: flag_name in enabled_flags for flag_name in ABLATION_FIELD_NAMES})


def add_ablation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ablation",
        action="append",
        default=None,
        metavar="FLAG[,FLAG...]",
        help=(
            "Comma-separated ablation flag names. "
            f"Supported flags: {', '.join(ABLATION_FIELD_NAMES)}. "
            f"Also configurable via ${TQ_NEW_ABLATIONS_ENV_VAR}."
        ),
    )
    for flag_name in ABLATION_FIELD_NAMES:
        parser.add_argument(
            f"--{flag_name.replace('_', '-')}",
            action="store_true",
            help=ABLATION_FLAG_HELP[flag_name],
        )


def add_shared_column_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shared-columns",
        action="append",
        default=None,
        metavar="COLUMN[,COLUMN...]",
        help=(
            "Comma-separated shared column names for grouped tq_new workloads. "
            f"Also configurable via ${TQ_GROUP_SHARED_COLUMNS_ENV_VAR}."
        ),
    )
