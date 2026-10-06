from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sparselab.data import preparation_benchmark as benchmark
from sparselab.data.preparation_benchmark import (
    BenchmarkDeclaration,
    run_preparation_benchmark,
)
from sparselab.workspace_preflight import StorageCheck


def _declaration(workspace: str, **changes: object) -> dict[str, object]:
    return {
        "format": "sparselab-preparation-benchmark-v1",
        "workspace": workspace,
        "sizes_bytes": [32768],
        "seed": 1701,
        "batch_documents": [1, 16],
        "batch_source_bytes": 1024 * 1024,
        "threads": [1],
        "worker_timeout_seconds": 120,
        **changes,
    }


def _write(path: Path, document: dict[str, object]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_declaration_rejects_non_strict_and_duplicate_coordinates() -> None:
    with pytest.raises(ValueError, match="strict integer"):
        BenchmarkDeclaration.model_validate(
            _declaration("workspace", sizes_bytes=["32768"])
        )
    with pytest.raises(ValueError, match="unique"):
        BenchmarkDeclaration.model_validate(
            _declaration("workspace", batch_documents=[1, 1])
        )


def test_workspace_admission_rejects_occupied_traversal_missing_parent_and_link(
    tmp_path: Path,
) -> None:
    (tmp_path / "occupied").mkdir()
    with pytest.raises(FileExistsError):
        run_preparation_benchmark(
            _write(tmp_path / "occupied.yaml", _declaration("occupied"))
        )
    with pytest.raises(ValueError, match="traversal"):
        run_preparation_benchmark(
            _write(tmp_path / "escaped.yaml", _declaration("nested/../workspace"))
        )
    with pytest.raises(ValueError, match="parent"):
        run_preparation_benchmark(
            _write(tmp_path / "missing.yaml", _declaration("missing/workspace"))
        )
    (tmp_path / "linked").symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        run_preparation_benchmark(
            _write(tmp_path / "link.yaml", _declaration("linked"))
        )


def test_storage_shortage_is_admission_failure_before_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(tmp_path / "benchmark.yaml", _declaration("workspace"))
    insufficient = StorageCheck(
        str(tmp_path), str(tmp_path), 0, 0, 1, 1, 1, 1, "insufficient"
    )
    monkeypatch.setattr(
        benchmark, "check_storage", lambda *args, **kwargs: insufficient
    )
    with pytest.raises(OSError, match="storage preflight"):
        run_preparation_benchmark(source)
    assert not (tmp_path / "workspace").exists()


def test_case_ceiling_rejects_before_workspace_claim(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "cases.yaml",
        _declaration(
            "workspace",
            sizes_bytes=list(range(1, 34)),
            batch_documents=[1],
            threads=[1],
        ),
    )
    with pytest.raises(ValueError, match="exceeds 32 cases"):
        run_preparation_benchmark(source)
    assert not (tmp_path / "workspace").exists()


def test_real_bounded_workers_preserve_per_size_identity(tmp_path: Path) -> None:
    report = run_preparation_benchmark(
        _write(tmp_path / "benchmark.yaml", _declaration("workspace"))
    )
    assert report["status"] == "completed"
    assert report["exact_scientific_parity_per_size"] == {"32768": True}
    assert all(row["status"] == "completed" for row in report["cases"])
    assert report["cases"][0]["arrays_sha256"] == report["cases"][1]["arrays_sha256"]
    assert json.loads((tmp_path / "workspace" / "report.json").read_text()) == report


def test_real_worker_timeout_retains_negative_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(
        tmp_path / "timeout.yaml",
        _declaration("timeout", batch_documents=[1], worker_timeout_seconds=0.01),
    )
    original = benchmark.subprocess.run
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda command, **kwargs: original(
            [sys.executable, "-c", "import time; time.sleep(60)"], **kwargs
        ),
    )
    report = run_preparation_benchmark(source)
    row = report["cases"][0]
    assert report["status"] == "failed" and row["status"] == "timed_out"
    assert row["preparation_seconds"] is None
    assert Path(row["log_path"]).is_dir()


def test_real_worker_failure_retains_null_metrics_and_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(
        tmp_path / "failure.yaml", _declaration("failure", batch_documents=[1])
    )
    original = benchmark.subprocess.run
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda command, **kwargs: original(
            [sys.executable, "-c", "raise RuntimeError('controlled worker failure')"],
            **kwargs,
        ),
    )
    report = run_preparation_benchmark(source)
    row = report["cases"][0]
    assert row["status"] == "failed" and row["arrays_sha256"] is None
    assert Path(row["log_path"], "stderr.log").is_file()


def test_identity_mismatch_from_real_workers_is_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(tmp_path / "mismatch.yaml", _declaration("mismatch"))
    original, calls = benchmark.subprocess.run, 0

    def alter_second_result(*args: object, **kwargs: object) -> object:
        nonlocal calls
        completed = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            result = Path(json.loads(Path(args[0][-1]).read_text())["result"])
            document = json.loads(result.read_text())
            document["cache_identity"] = {"real_worker_tuple": "mismatch"}
            result.write_text(json.dumps(document), encoding="utf-8")
        return completed

    monkeypatch.setattr(benchmark.subprocess, "run", alter_second_result)
    report = run_preparation_benchmark(source)
    assert report["status"] == "failed"
    assert report["exact_scientific_parity_per_size"] == {"32768": False}


def test_setup_failure_after_claim_retains_every_unstarted_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(tmp_path / "setup.yaml", _declaration("setup"))
    monkeypatch.setattr(
        benchmark,
        "_corpus",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("setup failure")),
    )
    report = run_preparation_benchmark(source)
    assert report["status"] == "failed"
    assert {row["reason"] for row in report["cases"]} == {"setup_failed"}
    assert (tmp_path / "setup" / "report.json").is_file()


@pytest.mark.parametrize(
    "changes",
    [
        {"sizes_bytes": [True]},
        {"sizes_bytes": [0]},
        {"sizes_bytes": [33554433]},
        {"sizes_bytes": [33554432, 33554431, 2]},
        {"batch_documents": [0]},
        {"batch_documents": [257]},
        {"batch_source_bytes": True},
        {"batch_source_bytes": 8388609},
        {"seed": -1},
        {"seed": 4294967296},
        {"threads": [0]},
        {"threads": [999999]},
        {"worker_timeout_seconds": float("inf")},
        {"worker_timeout_seconds": True},
    ],
)
def test_schema_bounds_reject_before_output(tmp_path, changes):
    source = _write(tmp_path / "input.json", _declaration("workspace", **changes))
    with pytest.raises(ValueError):
        run_preparation_benchmark(source)
    assert not (tmp_path / "workspace").exists()


def test_concurrent_claim_never_owns_competing_workspace(tmp_path, monkeypatch):
    source = _write(tmp_path / "input.json", _declaration("workspace"))
    workspace = tmp_path / "workspace"
    original = Path.mkdir

    def competing_claim(path, *args, **kwargs):
        if path == workspace:
            original(path)
            (path / "competitor").write_text("owned by competitor")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", competing_claim)
    with pytest.raises(FileExistsError):
        run_preparation_benchmark(source)
    assert (workspace / "competitor").read_text() == "owned by competitor"
    assert not (workspace / "report.json").exists()


def test_symlinked_ancestor_rejects_before_claim(tmp_path):
    parent = tmp_path / "real"
    parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    source = _write(tmp_path / "input.json", _declaration(str(alias / "workspace")))
    with pytest.raises(ValueError, match="symlink"):
        run_preparation_benchmark(source)
    assert not (parent / "workspace").exists()
