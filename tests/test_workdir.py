from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from sparselab.workdir import WORK_DIR_ENV, ensure_work_dir, resolve_work_dir
from sparselab.workers import transport
from sparselab.workers.models import WorkerDefinition

_ROOT = Path(__file__).resolve().parents[1]


def test_default_work_dir_uses_nearest_project_without_creating_it(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    nested = project / "src" / "sparselab"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(nested)
    monkeypatch.delenv(WORK_DIR_ENV, raising=False)

    work_dir = resolve_work_dir()

    assert work_dir == project / "sparselab-work"
    assert not work_dir.exists()


def test_configured_relative_work_dir_redirects_tempfile_allocations(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(WORK_DIR_ENV, "relative-scratch")
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, "previous-temp")
    monkeypatch.setattr(tempfile, "tempdir", None)

    work_dir = ensure_work_dir()

    assert work_dir == tmp_path / "relative-scratch"
    assert work_dir.is_dir()
    assert os.environ[WORK_DIR_ENV] == str(work_dir)
    assert all(os.environ[name] == str(work_dir) for name in ("TMPDIR", "TEMP", "TMP"))
    with tempfile.TemporaryDirectory(prefix="sparselab-test-") as temp_dir:
        assert Path(temp_dir).parent == work_dir


def test_worker_rpc_fallback_allocates_under_work_dir_and_cleans_up(
    tmp_path: Path, monkeypatch
) -> None:
    work_dir = tmp_path / "scratch"
    monkeypatch.setenv(WORK_DIR_ENV, str(work_dir))
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, "previous-temp")
    monkeypatch.setattr(tempfile, "tempdir", None)
    allocated_dirs: list[Path] = []
    original_mkdtemp = tempfile.mkdtemp

    def record_mkdtemp(*, prefix: str, dir: str | Path | None = None) -> str:
        assert dir is not None
        allocated_dirs.append(Path(dir))
        return original_mkdtemp(prefix=prefix, dir=dir)

    def unavailable_endpoint(*args, **kwargs):
        raise FileNotFoundError("endpoint unavailable")

    worker = WorkerDefinition(
        worker_id="test-worker",
        name="test-worker",
        transport="local",
        python=Path(sys.executable),
        root=tmp_path,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    monkeypatch.setattr(transport.tempfile, "mkdtemp", record_mkdtemp)
    monkeypatch.setattr(transport, "_endpoint_argv", lambda _: ["unused-endpoint"])
    monkeypatch.setattr(transport.subprocess, "Popen", unavailable_endpoint)

    with pytest.raises(FileNotFoundError, match="endpoint unavailable"):
        transport.call_worker(worker, "status", {"attempt_id": None})

    assert allocated_dirs == [work_dir]
    assert list(work_dir.iterdir()) == []


def test_research_scaffold_cli_uses_explicit_work_dir(tmp_path: Path) -> None:
    work_dir = tmp_path / "scratch"
    output = tmp_path / "scaffold"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sparselab.cli.main import main; main()",
            "--work-dir",
            str(work_dir),
            "research",
            "scaffold",
            "engram-ffn-substitution-v1",
            "--output",
            str(output),
            "--scale",
            "smoke",
            "--data",
            "offline",
            "--backend",
            "cpu",
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.stdout.strip() == str(output)
    assert (output / "study.yaml").is_file()
    assert work_dir.is_dir()
    assert list(work_dir.iterdir()) == []
