from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from sparselab.workdir import (
    CHECKOUT_STORAGE_REASON,
    WORK_DIR_ENV,
    ensure_work_dir,
    record_storage_observation,
    resolve_work_dir,
    storage_checks,
    warn_storage_checks,
)
from sparselab.workers import transport
from sparselab.workers.models import WorkerDefinition

_ROOT = Path(__file__).resolve().parents[1]


def test_default_work_dir_uses_external_xdg_or_home_without_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    nested = project / "src" / "sparselab"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv(WORK_DIR_ENV, raising=False)
    assert resolve_work_dir(cwd=nested) == tmp_path / "data" / "sparselab"
    assert not (tmp_path / "data").exists()
    monkeypatch.setenv("XDG_DATA_HOME", "relative-data")
    assert resolve_work_dir(cwd=nested) == home / ".local" / "share" / "sparselab"
    monkeypatch.setenv("XDG_DATA_HOME", " ")
    assert resolve_work_dir(cwd=nested) == home / ".local" / "share" / "sparselab"
    assert not home.exists()


def test_explicit_path_precedes_environment_and_retains_relative_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv(WORK_DIR_ENV, "legacy/sparselab-work")
    assert resolve_work_dir(cwd=tmp_path) == tmp_path / "legacy" / "sparselab-work"
    assert resolve_work_dir("chosen", cwd=tmp_path) == tmp_path / "chosen"
    assert (
        resolve_work_dir(tmp_path / "absolute", cwd=tmp_path) == tmp_path / "absolute"
    )
    assert not (tmp_path / "chosen").exists()
    assert not (tmp_path / "legacy").exists()


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
    assert (work_dir / "scratch").is_dir()
    assert os.environ[WORK_DIR_ENV] == str(work_dir)
    assert all(
        os.environ[name] == str(work_dir / "scratch")
        for name in ("TMPDIR", "TEMP", "TMP")
    )
    with tempfile.TemporaryDirectory(prefix="sparselab-test-") as temp_dir:
        assert Path(temp_dir).parent == work_dir / "scratch"


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

    assert allocated_dirs == [work_dir / "scratch"]
    assert list(work_dir.iterdir()) == [work_dir / "scratch"]
    assert list((work_dir / "scratch").iterdir()) == []


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
    assert (work_dir / "scratch").is_dir()
    assert list(work_dir.iterdir()) == [work_dir / "scratch"]


def test_checkout_storage_checks_find_git_ancestor_without_creating_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    nested = checkout / "not-created" / "persistent"
    outside = tmp_path / "persistent"
    monkeypatch.setenv(WORK_DIR_ENV, str(nested))
    checks = storage_checks(resolve_work_dir())
    assert checks == [
        {
            "reason_code": CHECKOUT_STORAGE_REASON,
            "kind": "work_root",
            "path": str(nested),
            "git_root": str(checkout),
        }
    ]
    assert warn_storage_checks(checks) == checks
    assert CHECKOUT_STORAGE_REASON in capsys.readouterr().err
    assert (
        storage_checks(checkout / "not-created" / "output", kind="output")[0]["kind"]
        == "output"
    )
    assert storage_checks(outside) == []
    assert not nested.exists()
    assert not outside.exists()


def test_two_checkouts_share_external_root_without_checkout_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "persistent-state"
    monkeypatch.setenv(WORK_DIR_ENV, str(root))
    for name in ("first", "second"):
        checkout = tmp_path / name
        checkout.mkdir()
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        assert resolve_work_dir(cwd=checkout) == root
        assert storage_checks(resolve_work_dir(cwd=checkout)) == []
    assert not root.exists()


def test_implicit_runs_follow_selected_root_and_explicit_destination_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.cli.main import build_parser

    environment = tmp_path / "environment-state"
    explicit = tmp_path / "flag-state"
    monkeypatch.setenv(WORK_DIR_ENV, str(environment))
    assert build_parser().parse_args(["eval", "run"]).runs_dir == str(
        environment / "runs"
    )
    parser = build_parser(explicit)
    assert parser.parse_args(
        ["--work-dir", str(explicit), "eval", "run"]
    ).runs_dir == str(explicit / "runs")
    selected = parser.parse_args(["eval", "run", "--runs-dir", "legacy/runs"])
    assert selected.runs_dir == "legacy/runs"
    assert not environment.exists()
    assert not explicit.exists()


def test_storage_observation_replay_keeps_original_bytes_and_detects_tamper(
    tmp_path: Path,
) -> None:
    import json

    from sparselab.campaign.state import read_canonical

    root = tmp_path / "persistent-state"
    checks = [
        {
            "reason_code": CHECKOUT_STORAGE_REASON,
            "kind": "output",
            "path": str(tmp_path / "output"),
            "git_root": str(tmp_path),
        }
    ]
    receipt = record_storage_observation(
        root, checks, "train", target=tmp_path / "output"
    )
    before = receipt.read_bytes()
    assert read_canonical(receipt)["storage_checks"] == checks
    assert (
        record_storage_observation(root, checks, "train", target=tmp_path / "output")
        == receipt
    )
    assert receipt.read_bytes() == before
    value = json.loads(before)
    value["observed_at_utc"] = "2000-01-01T00:00:00Z"
    receipt.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="conflicting storage observation"):
        record_storage_observation(root, checks, "train", target=tmp_path / "output")
