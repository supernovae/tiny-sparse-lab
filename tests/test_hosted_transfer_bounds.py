from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sparselab.hosted import transport
from sparselab.hosted.transport import (
    ColabFileTransport,
    TransferManifest,
    assembly_python,
)
from sparselab.runtime_env_subprocess import run_bounded


def test_manifest_rejects_impossible_size_or_shard_count_before_transfer() -> None:
    with pytest.raises(ValueError):
        TransferManifest(
            "/content/request",
            "a" * 64,
            transport._MAX_PARTS * transport._CHUNK_BYTES + 1,
            1,
        )
    with pytest.raises(ValueError):
        TransferManifest("/content/request", "a" * 64, 1, 2)


def test_upload_rejects_oversized_source_before_hashing_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "too-large"
    with source.open("wb") as stream:
        stream.truncate(transport._MAX_PARTS * transport._CHUNK_BYTES + 1)
    client = ColabFileTransport("fixture")
    monkeypatch.setattr(
        client,
        "_digest",
        lambda path: (_ for _ in ()).throw(AssertionError("hashed oversized source")),
    )
    monkeypatch.setattr(
        client,
        "_run",
        lambda command: (_ for _ in ()).throw(
            AssertionError("uploaded oversized source")
        ),
    )

    with pytest.raises(ValueError):
        client.upload(source, "/content/request")


def test_upload_uses_external_scratch_when_source_parent_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_parent = tmp_path / "prepared-input"
    source_parent.mkdir()
    source = source_parent / "bundle"
    source.write_bytes(b"source bytes")
    scratch = tmp_path / "external-scratch"
    scratch.mkdir()
    source_parent.chmod(0o500)
    observed_chunks: list[Path] = []
    client = ColabFileTransport("fixture")

    def uploaded(command: list[str]) -> None:
        chunk = Path(command[-2])
        observed_chunks.append(chunk)
        assert chunk.parent.parent == scratch
        assert chunk.read_bytes() == b"source bytes"

    monkeypatch.setattr(transport, "ensure_scratch_dir", lambda: scratch)
    monkeypatch.setattr(client, "_run", uploaded)
    try:
        manifest = client.upload(source, "/content/request")
    finally:
        source_parent.chmod(0o700)

    assert manifest.sha256 == hashlib.sha256(b"source bytes").hexdigest()
    assert len(observed_chunks) == 1
    assert not observed_chunks[0].exists()


def test_assembly_refuses_wrong_shard_size_and_preserves_existing_destination(
    tmp_path: Path,
) -> None:
    target = tmp_path / "request"
    target.write_bytes(b"already-published")
    (tmp_path / "request.part-00000000").write_bytes(b"short")
    manifest = TransferManifest(
        str(target), hashlib.sha256(b"shorter").hexdigest(), 7, 1
    )

    result = subprocess.run(
        [sys.executable, "-c", assembly_python(manifest)],
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert target.read_bytes() == b"already-published"
    assert not (tmp_path / ".request.assembling").exists()


def test_assembly_hash_mismatch_never_publishes_destination(tmp_path: Path) -> None:
    target = tmp_path / "request"
    payload = b"untrusted"
    (tmp_path / "request.part-00000000").write_bytes(payload)
    manifest = TransferManifest(str(target), "0" * 64, len(payload), 1)

    result = subprocess.run(
        [sys.executable, "-c", assembly_python(manifest)],
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert not target.exists()
    assert not (tmp_path / ".request.assembling").exists()


def test_assembly_deadline_prevents_publication(tmp_path: Path) -> None:
    target = tmp_path / "request"
    payload = b"deadline"
    (tmp_path / "request.part-00000000").write_bytes(payload)
    manifest = TransferManifest(
        str(target), hashlib.sha256(payload).hexdigest(), len(payload), 1
    )

    with pytest.raises(TimeoutError):
        exec(assembly_python(manifest), {"_sparselab_deadline": 0.0})  # noqa: S102 — fixed versioned program

    assert not target.exists()
    assert not (tmp_path / ".request.assembling").exists()


def test_bounded_subprocess_delivers_large_stdin_under_backpressure() -> None:
    payload = os.urandom(512 * 1024)
    result = run_bounded(
        [
            sys.executable,
            "-c",
            (
                "import sys, time\n"
                "total = 0\n"
                "while data := sys.stdin.buffer.read(4096):\n total += len(data); time.sleep(.0001)\n"
                "print(total)"
            ),
        ],
        input=payload,
        timeout=10,
    )
    assert result.stdout == f"{len(payload)}\n".encode()


def test_bounded_subprocess_never_replaces_preexisting_stdout_destination(
    tmp_path: Path,
) -> None:
    output = tmp_path / "response"
    output.write_bytes(b"already published")

    with pytest.raises(FileExistsError):
        run_bounded(
            [sys.executable, "-c", "import sys; sys.stdout.write('new response')"],
            timeout=5,
            stdout_path=output,
            stdout_limit=1024,
        )

    assert output.read_bytes() == b"already published"


def test_bounded_subprocess_enforces_stdout_file_cap_and_cleans_partial_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "response"

    with pytest.raises(ValueError):
        run_bounded(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 1025)"],
            timeout=5,
            stdout_path=output,
            stdout_limit=1024,
        )

    assert not output.exists()


def test_bounded_subprocess_deadline_and_cwd_are_enforced(tmp_path: Path) -> None:
    result = run_bounded(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        timeout=5,
        cwd=tmp_path,
    )
    assert result.stdout == f"{tmp_path}\n".encode()

    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            timeout=0.01,
        )
