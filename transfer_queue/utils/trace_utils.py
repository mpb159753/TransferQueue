from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

try:
    from viztracer import VizTracer
    from viztracer import get_tracer as _get_registered_tracer
except ImportError:  # pragma: no cover - exercised via runtime guard instead
    VizTracer = None

    def _get_registered_tracer() -> Any | None:
        return None


def _format_marker_name(name: str, **fields: Any) -> str:
    if not fields:
        return name

    formatted_fields = ", ".join(f"{key}={value}" for key, value in fields.items())
    return f"{name} [{formatted_fields}]"


class TraceMarker:
    @staticmethod
    def scope(name: str, tracer: Any | None = None, **fields: Any) -> Any:
        active_tracer = tracer if tracer is not None else _get_registered_tracer()
        if active_tracer is None or not hasattr(active_tracer, "log_event"):
            return nullcontext()
        return active_tracer.log_event(_format_marker_name(name, **fields))


class VizTracerProfileSession:
    def __init__(
        self,
        output_dir: str | Path,
        component_name: str,
        enabled: bool = False,
        tracer_entries: int = 1_000_000,
        min_duration_us: int = 0,
        log_async: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.component_name = component_name
        self.enabled = enabled
        self.tracer_entries = tracer_entries
        self.min_duration_us = min_duration_us
        self.log_async = log_async
        self._active_tracer: Any | None = None
        self._active_output_path: Path | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_output_path(self, round_number: int) -> Path:
        return self.output_dir / f"round_{round_number:02d}" / f"{self.component_name}.json"

    def start(self, round_number: int) -> Path | None:
        if not self.enabled:
            return None
        if VizTracer is None:
            raise RuntimeError("viztracer is required when profiling is enabled")
        if self._active_tracer is not None:
            raise RuntimeError("A VizTracer session is already active")

        output_path = self.build_output_path(round_number)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tracer = VizTracer(
            output_file=str(output_path),
            tracer_entries=self.tracer_entries,
            min_duration=self.min_duration_us,
            log_async=self.log_async,
            register_global=True,
        )
        tracer.start()
        self._active_tracer = tracer
        self._active_output_path = output_path
        return output_path

    def stop(self) -> Path | None:
        if self._active_tracer is None:
            return None

        tracer = self._active_tracer
        output_path = self._active_output_path
        self._active_tracer = None
        self._active_output_path = None

        tracer.stop()
        if output_path is not None:
            tracer.save(output_file=str(output_path))
        else:  # pragma: no cover - defensive branch
            tracer.save()
        return output_path

    def marker(self, name: str, **fields: Any) -> Any:
        return TraceMarker.scope(name, tracer=self._active_tracer, **fields)
