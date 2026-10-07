"""Provider-simulated Colab CLI execution through the native worker stack.

The fake below is only the public Contents/foreground CLI boundary.  It runs the
selected interpreter, agent RPC, bundle materializer, trainer, relay, and
collection paths unchanged; it is deliberately not CUDA acceptance evidence.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.config.models import RunConfig
from sparselab.training.manifest import canonical_json
from sparselab.workers.colab import download_hosted_status, read_delivery
from sparselab.workers.controller import Controller
from sparselab.workers.leases import boot_identity
from sparselab.workers.models import ColabEndpoint, WorkerDefinition
from sparselab.workers.relay_collection import collect_relay
from sparselab.workers.relay_models import RelayLocation, RelayProfile


class _PublicColabCli:
    """A serialized Contents/``exec`` provider without replacing native work."""

    def __init__(self, root: Path, *, foreground_delay: float = 0) -> None:
        self.root = root
        self.foreground_delay = foreground_delay
        self.exec_calls = 0
        self.foreground_calls = 0
        self.processes: list[subprocess.Popen[bytes]] = []
        self._popen = subprocess.Popen

    @staticmethod
    def _result(returncode: int = 0, stderr: bytes = b"") -> SimpleNamespace:
        return SimpleNamespace(returncode=returncode, stdout=b"", stderr=stderr)

    def _command(self, argv: list[str]) -> tuple[str, list[str]]:
        assert argv[0] == "colab"
        auth = argv.index("--auth")
        args = argv[auth + 3 :]
        if args[:1] == ["-s"]:
            args = args[1:]
        return argv[auth + 2], args

    def run_bounded(
        self, argv: list[str], *, timeout: float, output_limit: int, **_kwargs: object
    ) -> SimpleNamespace:
        command, args = self._command(argv)
        if command == "upload":
            _session, source, destination = args[0], Path(args[1]), Path(args[2])
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            destination.write_bytes(source.read_bytes())
            return self._result()
        if command == "download":
            _session, source, destination = args[0], args[1], Path(args[2])
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if source.startswith("/proc/"):
                if source.endswith("/stat"):
                    destination.write_text(
                        "1 (kernel) R " + " ".join(["0"] * 18 + ["17"]) + " 0\n"
                    )
                else:
                    destination.write_text("provider-simulated-boot\n")
            else:
                destination.write_bytes(Path(source).read_bytes())
            return self._result()
        if command == "rm":
            Path(args[1]).unlink(missing_ok=True)
            return self._result()
        if command == "exec":
            self.exec_calls += 1
            script = Path(args[args.index("-f") + 1])
            completed = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            return self._result(completed.returncode, completed.stderr[-output_limit:])
        raise AssertionError(f"unexpected public Colab CLI command: {argv!r}")

    def foreground_popen(
        self, argv: list[str], **kwargs: object
    ) -> subprocess.Popen[bytes]:
        command, args = self._command(argv)
        assert command == "exec"
        self.foreground_calls += 1
        script = Path(args[args.index("-f") + 1])
        command = [sys.executable, str(script)]
        if self.foreground_delay:
            command = [
                sys.executable,
                "-c",
                "import runpy, sys, time; time.sleep(float(sys.argv[1])); runpy.run_path(sys.argv[2], run_name='__main__')",
                str(self.foreground_delay),
                str(script),
            ]
        process = self._popen(
            command,
            stdin=kwargs["stdin"],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
            start_new_session=bool(kwargs["start_new_session"]),
        )
        self.processes.append(process)
        return process

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Contents operations and idle control exec share this one public provider.
        monkeypatch.setattr("sparselab.hosted.transport.run_bounded", self.run_bounded)
        monkeypatch.setattr(
            "sparselab.runtime_env_subprocess.run_bounded", self.run_bounded
        )
        # Do not mutate the process-global subprocess module: the fake itself uses it.
        monkeypatch.setattr(
            "sparselab.workers.colab.subprocess",
            SimpleNamespace(
                Popen=self.foreground_popen,
                DEVNULL=subprocess.DEVNULL,
                SubprocessError=subprocess.SubprocessError,
            ),
        )


def _tiny_config(root: Path) -> RunConfig:
    from test_training import config

    value = config(root).model_dump(mode="json")
    value["training"].update(max_steps=2, max_tokens=64, deterministic=True)
    value["optimizer"]["warmup_steps"] = 0
    value["checkpoint"].update(every_steps=1, keep_periodic=True)
    value["evaluation"].update(every_steps=1, max_batches=1)
    value["staging"].update(smoke_steps=1, warmup_steps=1)
    return RunConfig.model_validate(value)


def _write_kernel_observer(root: Path, instance_id: str) -> None:
    root.mkdir(parents=True, mode=0o700)
    (root / "kernel-status.json").write_bytes(
        canonical_json(
            {
                "kernel_status_version": 1,
                "instance_id": instance_id,
                "boot_id": "provider-simulated-boot",
                "pid": 1,
                "process_start": 17,
                "state": "IDLE",
                "observed_at": time.time(),
            }
        )
    )
    # The provider simulation is allowed to synthesize Linux observer facts on Darwin;
    # native worker status still binds this ID to the actual local boot.
    (root / "hosted-instance.json").write_bytes(
        canonical_json(
            {
                "instance_id": instance_id,
                "boot_id": boot_identity(),
            }
        )
    )


def test_provider_simulated_colab_cli_executes_native_bundle_prepare_and_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider simulation only: it is not a CUDA/GPU correctness claim."""
    monkeypatch.setenv("SPARSELAB_WORKER_LEASE_DIR", str(tmp_path / "leases"))
    remote_root = (tmp_path / "colab-vm" / "worker").absolute()
    instance_id = "provider-sim-boot"
    _write_kernel_observer(remote_root, instance_id)
    provider = _PublicColabCli(remote_root)
    provider.install(monkeypatch)

    durable = RelayLocation(kind="file", root=str((tmp_path / "relay").absolute()))
    profile = RelayProfile(
        namespace="colab-native-execution", controller=durable, worker=durable
    )
    worker = WorkerDefinition(
        worker_id="provider-sim-worker",
        name="provider-sim-worker",
        transport="colab",
        host=None,
        colab=ColabEndpoint(session="provider-sim", instance_id=instance_id),
        relay=profile.binding(),
        python=Path(sys.executable).absolute(),
        root=remote_root,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    controller = Controller(tmp_path / "controller")
    controller.register(worker)
    submission = controller.submit(
        _tiny_config(tmp_path / "inputs"), worker=worker.name
    )
    attempt = controller.store.attempt_by_run(submission.run_id)
    assert attempt is not None
    selected, reason = controller._eligible_worker(attempt)
    assert reason is None
    assert selected is not None

    # This performs relay-backed bulk install_bundle twice, then direct sensitive
    # prepare (spec plus relay key), and only then writes the real PREPARED receipt.
    controller._launch(attempt, selected)
    delivery = read_delivery(controller.root, submission.attempt_id)
    assert delivery is not None and delivery["state"] == "ISSUED"
    receipt = json.loads(
        (remote_root / "attempts" / submission.attempt_id / "receipt.json").read_bytes()
    )
    assert receipt["state"] in {"PREPARED", "RUNNING", "COMPLETE"}
    assert (
        remote_root / "attempts" / submission.attempt_id / "relay-key.bin"
    ).stat().st_size == 32
    assert provider.foreground_calls == 1

    # An uncertain issued delivery is reconciled, never sent to a second kernel.
    controller._launch(controller.store.attempt_by_run(submission.run_id), selected)
    assert provider.foreground_calls == 1
    before_passive = provider.exec_calls
    status = download_hosted_status(worker, tmp_path / "passive-status.json")
    assert status["receipt"] is not None
    assert provider.exec_calls == before_passive

    assert provider.processes[0].wait(timeout=60) == 0
    collected = collect_relay(controller, submission.run_id, profile, flush=False)
    assert collected["status"] == "COMPLETE"
    assert collected["ingestion_status"] == "COMPLETE"
    assert collected["durable_checkpoint"]["step"] == 2
    completed = controller.store.attempt_by_run(submission.run_id)
    assert completed is not None
    assert completed["terminal_receipt"]["state"] == "COMPLETE"
    assert (
        controller.root / submission.run_id / "checkpoints" / "latest.json"
    ).is_file()


def test_provider_simulated_busy_cancellation_uses_contents_not_second_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real foreground executor acknowledges the uploaded cancellation intent."""
    monkeypatch.setenv("SPARSELAB_WORKER_LEASE_DIR", str(tmp_path / "leases"))
    remote_root = (tmp_path / "colab-vm" / "worker").absolute()
    instance_id = "provider-sim-cancel"
    _write_kernel_observer(remote_root, instance_id)
    provider = _PublicColabCli(remote_root, foreground_delay=1)
    provider.install(monkeypatch)
    durable = RelayLocation(kind="file", root=str((tmp_path / "relay").absolute()))
    profile = RelayProfile(
        namespace="colab-native-cancellation", controller=durable, worker=durable
    )
    worker = WorkerDefinition(
        worker_id="provider-sim-cancel-worker",
        name="provider-sim-cancel-worker",
        transport="colab",
        host=None,
        colab=ColabEndpoint(session="provider-sim-cancel", instance_id=instance_id),
        relay=profile.binding(),
        python=Path(sys.executable).absolute(),
        root=remote_root,
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    controller = Controller(tmp_path / "controller")
    controller.register(worker)
    submission = controller.submit(
        _tiny_config(tmp_path / "inputs"), worker=worker.name
    )
    attempt = controller.store.attempt_by_run(submission.run_id)
    assert attempt is not None
    selected, reason = controller._eligible_worker(attempt)
    assert reason is None and selected is not None
    controller._launch(attempt, selected)
    before_passive = provider.exec_calls
    assert controller.cancel(submission.run_id)["status"] == "CANCEL_REQUESTED"
    assert provider.exec_calls == before_passive
    assert provider.foreground_calls == 1
    assert provider.processes[0].wait(timeout=60) == 0
    collected = collect_relay(controller, submission.run_id, profile, flush=False)
    assert collected["status"] == "CANCELLED"
    assert collected["terminal_receipt"]["cancellation_acknowledged"] is True
