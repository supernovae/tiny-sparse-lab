from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sparselab.hosted.cli import _ssh_command
from sparselab.hosted.kernel import observer_program
from sparselab.hosted.models import HostedTarget, SetupRequest
from sparselab.hosted.transport import (
    ColabFileTransport,
    TransferManifest,
    assembly_python,
)


def test_hosted_target_requires_exactly_one_endpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        HostedTarget.model_validate({})
    with pytest.raises(ValueError, match="exactly one"):
        HostedTarget.model_validate({"ssh": "host", "colab_session": "session"})
    selected = HostedTarget.model_validate(
        {"colab_session": "session", "root": "/content/work"}
    )
    assert selected.colab_session == "session"


def test_setup_refuses_relative_operational_paths(tmp_path: Path) -> None:
    bundle = tmp_path / "source.bundle"
    bundle.write_bytes(b"bundle")
    with pytest.raises(ValueError, match="runtime_root must be absolute"):
        SetupRequest.model_validate(
            {
                "ssh": "host",
                "root": "/work",
                "source_bundle": str(bundle),
                "source_commit": "a" * 40,
                "runtime_root": "runtime",
            }
        )


def test_assembly_program_publishes_only_verified_bytes(tmp_path: Path) -> None:
    target = tmp_path / "request"
    payload = b"verified request"
    (tmp_path / "request.part-00000000").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    exec(assembly_python(TransferManifest(str(target), digest, len(payload), 1)), {})  # noqa: S102 — fixed versioned program
    assert target.read_bytes() == payload


def test_assembly_program_rejects_wrong_digest_without_destination(
    tmp_path: Path,
) -> None:
    target = tmp_path / "request"
    (tmp_path / "request.part-00000000").write_bytes(b"altered")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            assembly_python(TransferManifest(str(target), "0" * 64, 7, 1)),
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not target.exists()


def test_exact_control_upload_is_capped(tmp_path: Path) -> None:
    source = tmp_path / "too-large"
    source.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    transport = ColabFileTransport("fixture")
    with pytest.raises(ValueError):
        transport.upload_exact(source, "/content/cancel.json")


def test_ssh_program_executes_without_shell_interpretation() -> None:
    target = HostedTarget.model_validate({"ssh": "user@example"})
    command = _ssh_command(target, "print('quoted; safely')")
    result = subprocess.run(
        ["/bin/sh", "-c", command[-1]],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "quoted; safely\n"


def test_kernel_observer_survives_later_cell_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Events:
        def __init__(self) -> None:
            self.callbacks: dict[str, object] = {}

        def register(self, name: str, callback: object) -> None:
            self.callbacks[name] = callback

    class Shell:
        def __init__(self) -> None:
            self.events = Events()

    original = Path.read_text

    def fake_proc(path: Path, *args, **kwargs):
        if str(path) == "/proc/sys/kernel/random/boot_id":
            return "fixture-boot"
        if str(path) == "/proc/self/stat":
            return f"{os.getpid()} (kernel) S " + "0 " * 18 + "123456 0"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_proc)
    namespace = {"get_ipython": lambda: shell}
    shell = Shell()
    exec(  # noqa: S102 — exercising fixed IPython event registration
        observer_program(str(tmp_path), "a" * 64),
        namespace,
    )
    published = json.loads((tmp_path / "kernel-status.json").read_text())
    assert published["kernel_status_version"] == 1
    assert published["instance_id"] == "a" * 64
    assert published["state"] == "BUSY"
    assert isinstance(published["process_start"], int)
    namespace.update(p={}, root="unrelated", publish=None, time=None)
    shell.events.callbacks["post_run_cell"](None)  # type: ignore[index,operator]
    assert json.loads((tmp_path / "kernel-status.json").read_text())["state"] == "IDLE"
    shell.events.callbacks["pre_run_cell"](None)
    updated = json.loads((tmp_path / "kernel-status.json").read_text())
    assert updated["state"] == "BUSY"
    assert updated["instance_id"] == "a" * 64
