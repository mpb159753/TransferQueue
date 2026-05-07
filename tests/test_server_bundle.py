from pathlib import Path

from transfer_queue.utils.server_bundle import collect_server_bundle_paths


def test_collect_server_bundle_paths_includes_required_runtime_files() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    relative_paths = {path.as_posix() for path in collect_server_bundle_paths(repo_root)}

    assert "scripts/tq_perf_quantification.py" in relative_paths
    assert "scripts/performance_test.py" in relative_paths
    assert "scripts/perf_microbench.py" in relative_paths
    assert "transfer_queue/client.py" in relative_paths
    assert "tq_new/recipe/async_flow/utils/transfer_queue/put_benchmark.py" in relative_paths
    assert "requirements.txt" in relative_paths
    assert "pyproject.toml" in relative_paths
    assert "run_server.sh" in relative_paths
    assert "SERVER_RUN.md" in relative_paths


def test_collect_server_bundle_paths_excludes_non_runtime_content() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    relative_paths = {path.as_posix() for path in collect_server_bundle_paths(repo_root)}

    assert all(not path.startswith("docs/") for path in relative_paths)
    assert all(not path.startswith("tests/") for path in relative_paths)
    assert all(not path.startswith("venv/") for path in relative_paths)
    assert all("__pycache__" not in path for path in relative_paths)
    assert all(not path.endswith(".pyc") for path in relative_paths)
    assert "tq_new/recipe/async_flow/utils/transfer_queue/test_transferqueue.py" not in relative_paths
    assert "tq_new/recipe/async_flow/utils/transfer_queue/1.md" not in relative_paths
