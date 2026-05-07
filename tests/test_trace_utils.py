from pathlib import Path

from transfer_queue.utils.trace_utils import TraceMarker, VizTracerProfileSession


def test_trace_marker_scope_is_noop_without_registered_tracer():
    marker_scope = TraceMarker.scope("allocate_indexes", stage="put")

    assert hasattr(marker_scope, "__enter__")
    assert hasattr(marker_scope, "__exit__")

    with marker_scope:
        pass


def test_profile_session_builds_round_output_path(tmp_path: Path):
    session = VizTracerProfileSession(
        output_dir=tmp_path,
        component_name="client",
        enabled=False,
    )

    assert session.build_output_path(3) == tmp_path / "round_03" / "client.json"


def test_profile_session_is_noop_when_disabled(tmp_path: Path):
    session = VizTracerProfileSession(
        output_dir=tmp_path,
        component_name="client",
        enabled=False,
    )

    assert session.start(1) is None
    assert session.stop() is None
