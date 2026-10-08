"""Deterministic live-tree accounting for the Card 05 monitor."""

from __future__ import annotations

import errno
import os
import runpy
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SAMPLER = (
    Path(__file__).resolve().parents[1] / "tools/kernel-memory-lab/sample-task-root.py"
)


def _sample():
    return runpy.run_path(str(SAMPLER))["sample_tree"]


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "task"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "nested/data.bin").write_bytes(b"data" * 1024)
    (root / "nested/second-link").hardlink_to(root / "nested/data.bin")
    (root / "experiments.sqlite3-wal").write_bytes(b"W" * 4096)
    (root / "experiments.sqlite3-shm").write_bytes(b"S" * 1024)
    return root


def test_stable_sample_matches_apparent_du_and_counts_live_sqlite_files(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path)
    counted_bytes, counted_inodes = _sample()(root)
    du_bytes = int(
        subprocess.check_output(["du", "-sbx", str(root)], text=True).split()[0]
    )
    assert counted_bytes == du_bytes
    assert counted_inodes == 6  # root, nested, data, link, WAL and SHM
    assert counted_bytes >= 4096 + 1024


@pytest.mark.parametrize("vanish_at", ["stat", "scandir"])
def test_disappearance_restarts_and_counts_live_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vanish_at: str
) -> None:
    root = _fixture(tmp_path)
    sample = _sample()
    globals_ = sample.__globals__
    real_stat, real_scandir = os.stat, os.scandir
    vanished = False

    def stat_once(path, *args, **kwargs):
        nonlocal vanished
        if (
            vanish_at == "stat"
            and Path(path).name == "experiments.sqlite3-wal"
            and not vanished
        ):
            vanished = True
            raise FileNotFoundError(errno.ENOENT, "WAL rotated", str(path))
        return real_stat(path, *args, **kwargs)

    def scandir_once(path):
        nonlocal vanished
        if vanish_at == "scandir" and Path(path).name == "nested" and not vanished:
            vanished = True
            raise FileNotFoundError(errno.ENOENT, "directory rotated", str(path))
        return real_scandir(path)

    monkeypatch.setitem(
        globals_, "os", SimpleNamespace(stat=stat_once, scandir=scandir_once)
    )
    actual = sample(root)
    assert vanished
    assert actual == _sample()(root)
    assert actual[1] == 6


@pytest.mark.parametrize("code", [errno.EACCES, errno.EIO])
def test_genuine_access_or_io_error_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    root = _fixture(tmp_path)
    sample = _sample()
    real_stat = os.stat

    def fail_stat(path, *args, **kwargs):
        if Path(path).name == "experiments.sqlite3-wal":
            raise OSError(code, "genuine read failure", str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setitem(
        sample.__globals__, "os", SimpleNamespace(stat=fail_stat, scandir=os.scandir)
    )
    with pytest.raises(OSError) as caught:
        sample(root)
    assert caught.value.errno == code


def test_continuous_disappearance_exhausts_deadline_without_stale_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture(tmp_path)
    sample = _sample()
    real_stat = os.stat

    def missing_stat(path, *args, **kwargs):
        if Path(path).name == "experiments.sqlite3-wal":
            raise FileNotFoundError(errno.ENOENT, "keeps rotating", str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setitem(
        sample.__globals__, "os", SimpleNamespace(stat=missing_stat, scandir=os.scandir)
    )
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        sample(root, seconds=0.05)
    assert time.monotonic() - start < 1
