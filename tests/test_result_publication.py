"""Tiny, model-free faults at the actual study/capability result callsites."""

import hashlib
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from test_study import _write_study

from sparselab.evaluation import capabilities
from sparselab.experiments import study
from sparselab.training.manifest import canonical_json


@pytest.fixture(params=["study", "capability"])
def publication(request, tmp_path, monkeypatch):
    if request.param == "study":
        planned = study.plan_study(
            _write_study(tmp_path, [{"label": "dense", "set": {}}], [], seeds=False)
        )
        receipt = study._study_receipt_template(planned)
        receipt["runs"][0]["run_id"] = "synthetic-run"
        receipt_path = tmp_path / "receipt.json"
        receipt_path.write_bytes(canonical_json(receipt) + b"\n")
        monkeypatch.setattr(
            study, "_evaluate_trial", lambda *args: {"status": "synthetic", "text": "é"}
        )

        def publish():
            return study.collect_study(planned, receipt_path, runs_dir=tmp_path)[1]

        path = publish()
        payload = json.loads(path.read_bytes())
        digest = payload.pop("report_sha256")
        assert digest == hashlib.sha256(canonical_json(payload)).hexdigest()
        expected = canonical_json({**payload, "report_sha256": digest}) + b"\n"
        assert path.name == f"architecture-{planned.study_sha256[:12]}-{digest}.json"
    else:
        result = {"card": "é", "checkpoint_sha256": "synthetic", "score": 0.5}

        def publish():
            return capabilities.write_capability_result(tmp_path, result)

        path = publish()
        digest = hashlib.sha256(
            json.dumps(
                result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        expected = (
            json.dumps(
                {**result, "result_digest": digest},
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        checkpoint = hashlib.sha256(b'"synthetic"').hexdigest()[:12]
        assert path.name == f"capability-{checkpoint}-{digest}.json"
    assert path.read_bytes() == expected
    path.unlink()  # Only this test's disposable synthetic baseline.
    return publish, path, expected


@pytest.mark.parametrize("boundary", ["short", "flush", "fsync", "directory"])
def test_publication_failure_never_leaves_partial_final(
    publication, monkeypatch, boundary
):
    publish, path, expected = publication
    fdopen = os.fdopen
    path_open = Path.open
    fsync = os.fsync

    class FaultyHandle:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def write(self, data):
            return self.handle.write(data[:2] if boundary == "short" else data)

        def flush(self):
            if boundary == "flush":
                raise OSError("injected flush")
            self.handle.flush()

        def fileno(self):
            return self.handle.fileno()

    if boundary in {"short", "flush"}:
        monkeypatch.setattr(
            os, "fdopen", lambda *a, **kw: FaultyHandle(fdopen(*a, **kw))
        )

        def open_handle(self, mode="r", *args, **kwargs):
            handle = path_open(self, mode, *args, **kwargs)
            return FaultyHandle(handle) if mode in {"xb", "w"} else handle

        monkeypatch.setattr(Path, "open", open_handle)
    else:

        def fail_sync(fd):
            directory = stat.S_ISDIR(os.fstat(fd).st_mode)
            if directory == (boundary == "directory"):
                raise OSError(f"injected {boundary}")
            fsync(fd)

        monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError):
        publish()
    if boundary == "directory":
        assert path.read_bytes() == expected
    else:
        assert not path.exists()
    assert not list(path.parent.glob(".*.tmp"))


def test_identical_replay_preserves_bytes_and_inode(publication):
    publish, path, expected = publication
    assert publish() == path
    inode = path.stat().st_ino
    assert publish() == path
    assert path.read_bytes() == expected
    assert path.stat().st_ino == inode


def test_existing_collision_is_never_overwritten(publication):
    publish, path, _ = publication
    path.write_bytes(b"competing evidence")
    with pytest.raises(ValueError, match="conflict"):
        publish()
    assert path.read_bytes() == b"competing evidence"


@pytest.mark.parametrize("identical", [False, True])
def test_competitor_arriving_at_publication_is_preserved(
    publication, monkeypatch, identical
):
    publish, path, expected = publication
    competitor = expected if identical else b"competing evidence"

    def collide(source, destination):
        assert Path(source).read_bytes() == expected
        assert not path.exists()
        path.write_bytes(competitor)
        raise FileExistsError("injected competing publisher")

    monkeypatch.setattr(os, "link", collide)
    if identical:
        assert publish() == path
    else:
        with pytest.raises(ValueError, match="conflict"):
            publish()
    assert path.read_bytes() == competitor
    assert not list(path.parent.glob(".*.tmp"))


def test_concurrent_identical_publication(publication, monkeypatch):
    publish, path, expected = publication
    barrier = Barrier(2)
    link = os.link

    def synchronize(source, destination):
        assert Path(source).read_bytes() == expected
        barrier.wait(timeout=10)
        link(source, destination)

    monkeypatch.setattr(os, "link", synchronize)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: publish(), range(2)))
    assert results == [path, path]
    assert path.read_bytes() == expected
    assert not list(path.parent.glob(".*.tmp"))


def test_new_parent_sync_failure_prevents_publication(publication, monkeypatch):
    publish, path, _ = publication
    path.parent.rmdir()  # Empty synthetic result directory from the fixture.
    fsync = os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected parent directory fsync")
        fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory)
    with pytest.raises(OSError, match="parent directory"):
        publish()
    assert not path.exists()
    assert not list(path.parent.glob(".*.tmp"))


def test_replay_after_uncertain_directory_durability(publication, monkeypatch):
    publish, path, expected = publication
    fsync = os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected directory fsync")
        fsync(fd)

    with monkeypatch.context() as fault:
        fault.setattr(os, "fsync", fail_directory)
        with pytest.raises(OSError, match="directory fsync"):
            publish()
    inode = path.stat().st_ino
    assert path.read_bytes() == expected
    assert publish() == path
    assert path.stat().st_ino == inode
    assert path.read_bytes() == expected


def test_capability_exists_write_race_never_overwrites(tmp_path, monkeypatch):
    path_open = Path.open
    link = os.link
    competitor = b"complete competing evidence"
    targets = []

    def compete(path):
        targets.append(path)
        with path_open(path, "wb") as handle:
            handle.write(competitor)

    def racing_open(path, mode="r", *args, **kwargs):
        if mode == "w" and path.name.startswith("capability-"):
            compete(path)
        return path_open(path, mode, *args, **kwargs)

    def racing_link(source, destination):
        compete(destination)
        link(source, destination)

    monkeypatch.setattr(Path, "open", racing_open)
    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(ValueError, match="conflict"):
        capabilities.write_capability_result(tmp_path, {"card": "synthetic"})
    assert len(targets) == 1
    assert targets[0].read_bytes() == competitor
