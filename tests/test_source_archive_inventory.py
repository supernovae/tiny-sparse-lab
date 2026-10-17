"""Neutral archive fixtures cover producer lifetime without copying a study."""

import hashlib
import io
import subprocess
import sys
import tarfile

import pytest

from sparselab.experiments import source_compatibility as compatibility
from sparselab.training.manifest import canonical_json


def archive_bytes(members):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, body in members.items():
            member = tarfile.TarInfo("src/sparselab/" + name)
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
    # More than pipe capacity: producer cannot finish until tar's unread tail
    # is consumed. This is legal trailing tar padding, not another member.
    return output.getvalue() + bytes(128 * 1024)


@pytest.fixture
def producer(tmp_path, monkeypatch):
    processes = []
    original = subprocess.Popen

    def bind(payload, returncode=0):
        path = tmp_path / "archive.tar"
        path.write_bytes(payload)

        def launch(argv, **kwargs):
            assert argv == [
                "git",
                "-C",
                str(tmp_path),
                "archive",
                "fixture",
                "src/sparselab",
            ]
            if sys.platform == "linux":
                kwargs["pipesize"] = 4096
            process = original(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    (
                        "import shutil,signal,sys; "
                        "signal.signal(signal.SIGPIPE,signal.SIG_DFL); "
                        "f=open(sys.argv[1],'rb'); "
                        "shutil.copyfileobj(f,sys.stdout.buffer,4096); "
                        "sys.stdout.buffer.flush(); sys.exit(int(sys.argv[2]))"
                    ),
                    str(path),
                    str(returncode),
                ],
                **kwargs,
            )
            processes.append(process)
            return process

        monkeypatch.setattr(compatibility.subprocess, "Popen", launch)
        return processes

    yield bind
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGPIPE producer fixture")
def test_inventory_drains_padding_and_preserves_canonical_identity(tmp_path, producer):
    members = {
        "z.py": b"z = 1\n",
        "a.json": b"{}\n",
        "ignored.txt": b"not inventoried",
        "research/resources/lifecycle.json": b"{}",
        "__pycache__/ignored.py": b"ignored",
    }
    processes = producer(archive_bytes(members))
    files = [
        (name, hashlib.sha256(members[name]).hexdigest()) for name in ("a.json", "z.py")
    ]
    expected = canonical_json(
        {
            "algorithm": "package-path-sha256-v2",
            "files": files,
            "sha256": hashlib.sha256(canonical_json(files)).hexdigest(),
        }
    )
    assert compatibility._committed_identity_bytes(tmp_path, "fixture") == expected
    assert processes[0].returncode == 0
    assert processes[0].stdout.closed


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGPIPE producer fixture")
@pytest.mark.parametrize("members,returncode", [({"a.py": b"pass\n"}, 7), ({}, 0)])
def test_inventory_rejects_producer_failure_and_empty_archive(
    tmp_path, producer, members, returncode
):
    processes = producer(archive_bytes(members), returncode)
    with pytest.raises(ValueError, match="package inventory unavailable"):
        compatibility._committed_identity_bytes(tmp_path, "fixture")
    assert processes[0].returncode == returncode
    assert processes[0].stdout.closed


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGPIPE producer fixture")
def test_invalid_archive_closes_and_reaps_producer(tmp_path, producer):
    processes = producer(b"invalid tar" + bytes(128 * 1024))
    with pytest.raises(tarfile.ReadError):
        compatibility._committed_identity_bytes(tmp_path, "fixture")
    assert processes[0].returncode is not None
    assert processes[0].stdout.closed
