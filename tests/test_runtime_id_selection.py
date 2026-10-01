"""Consumer-visible logical runtime selection and immutable experiment identity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from test_experiment_evidence import _lock
from test_training import config as training_config

from sparselab.cli.main import _prepare_runtime_command, build_parser, main
from sparselab.experiments.binding import open_runtime_binding
from sparselab.experiments.lock import _identities
from sparselab.runtime_environments import RuntimeEntry, register_runtime


@pytest.fixture
def registered_cpu(tmp_path, monkeypatch):
    config_home = tmp_path / "config-home"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data-home"))
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    entry = RuntimeEntry(
        python=Path(sys.executable).absolute(),
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    register_runtime("cpu-test", entry)
    return entry


def _command_args(tmp_path, command):
    config = tmp_path / "run.yaml"
    value = training_config(tmp_path)
    config.write_text(yaml.safe_dump(value.model_dump(mode="json")))
    runs = tmp_path / "runs"
    run = runs / "sample"
    run.mkdir(parents=True)
    (run / "resolved_config.yaml").write_text(value.model_dump_json())
    commands = {
        "train": ["train", str(config)],
        "stage": ["stage", str(config), "--output", str(tmp_path / "stage")],
        "eval": ["eval", "sample", "--runs-dir", str(runs)],
        "generate": [
            "generate",
            "sample",
            "--prompt",
            "hello",
            "--runs-dir",
            str(runs),
        ],
        "chat": [
            "chat",
            "sample",
            "--message",
            "hello",
            "--runs-dir",
            str(runs),
        ],
        "run": ["run", str(config)],
        "experiment bind": ["experiment", "bind", str(tmp_path / "lock")],
        "experiment run": ["experiment", "run", str(tmp_path / "lock")],
        "evaluation suite run": [
            "evaluation",
            "suite",
            "run",
            str(tmp_path / "suite"),
            "sample",
            "--checkpoint",
            "latest.json",
            "--runs-dir",
            str(runs),
        ],
        "research snapshot": ["research", "snapshot", str(tmp_path / "plan")],
    }
    return commands[command]


@pytest.mark.parametrize(
    "command",
    [
        "train",
        "stage",
        "eval",
        "generate",
        "chat",
        "run",
        "experiment bind",
        "experiment run",
        "evaluation suite run",
        "research snapshot",
    ],
)
def test_runtime_id_excludes_file_profile_at_parser(tmp_path, command):
    args = _command_args(tmp_path, command)
    with pytest.raises(SystemExit) as rejected:
        build_parser(tmp_path / "work").parse_args(
            [
                *args,
                "--runtime",
                "cpu-test",
                "--runtime-profile",
                str(tmp_path / "profile"),
            ]
        )
    assert rejected.value.code == 2


@pytest.mark.parametrize(
    ("command", "conflict"),
    [
        ("run", "--worker"),
        ("experiment bind", "--worker"),
        ("experiment run", "--worker"),
        ("experiment run", "--binding"),
        ("evaluation suite run", "--worker"),
    ],
)
def test_runtime_id_excludes_worker_and_binding(tmp_path, command, conflict):
    args = _command_args(tmp_path, command)
    value = str(tmp_path / "binding") if conflict == "--binding" else "worker-id"
    with pytest.raises(SystemExit) as rejected:
        build_parser(tmp_path / "work").parse_args(
            [*args, "--runtime", "cpu-test", conflict, value]
        )
    assert rejected.value.code == 2


@pytest.mark.parametrize(
    "command",
    [
        "train",
        "stage",
        "eval",
        "generate",
        "chat",
        "run",
        "experiment bind",
        "experiment run",
        "evaluation suite run",
        "research snapshot",
    ],
)
def test_unknown_runtime_id_fails_preflight_before_workdir(
    tmp_path, monkeypatch, command
):
    args = _command_args(tmp_path, command)
    work = tmp_path / "not-created"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-config"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["sparselab", "--work-dir", str(work), *args, "--runtime", "missing"],
    )
    with pytest.raises(SystemExit, match="unknown runtime id"):
        main()
    assert not work.exists()


def test_stage_inspect_resolves_id_passively_and_preserves_file_profile(
    tmp_path, monkeypatch, registered_cpu
):
    args = _command_args(tmp_path, "stage")
    inspect = [*args, "--through", "inspect"]
    work = tmp_path / "absent-work"
    monkeypatch.setattr(
        sys,
        "argv",
        ["sparselab", "--work-dir", str(work), *inspect, "--runtime", "missing"],
    )
    with pytest.raises(SystemExit, match="unknown runtime id"):
        main()
    assert not work.exists()
    assert not (tmp_path / "stage").exists()

    import sparselab.cli.main as cli

    monkeypatch.setattr(
        cli,
        "probe_runtime_profile",
        lambda _: pytest.fail("inspect probed runtime"),
    )
    monkeypatch.setattr(
        cli,
        "_ensure_runtime_interpreter",
        lambda *a: pytest.fail("inspect re-executed"),
    )
    parsed = build_parser(work).parse_args([*inspect, "--runtime", "cpu-test"])
    _prepare_runtime_command(parsed)
    assert not work.exists()
    assert not (tmp_path / "stage").exists()

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "runtime_profile_version": 1,
                "id": "cpu-file",
                "python": str(registered_cpu.python),
                "engine": "pytorch",
                "backend": "cpu",
                "device_index": 0,
                "requirements": {},
            }
        )
    )
    _prepare_runtime_command(
        build_parser(work).parse_args(
            [*inspect, "--runtime-profile", str(profile_path)]
        )
    )
    profile_path.write_text("runtime_profile_version: 2\n")
    with pytest.raises(ValueError):
        _prepare_runtime_command(
            build_parser(work).parse_args(
                [*inspect, "--runtime-profile", str(profile_path)]
            )
        )
    assert not work.exists()


@pytest.mark.parametrize(
    "command", ["train", "stage", "eval", "generate", "chat", "run"]
)
def test_registered_cpu_id_freshly_authorizes_direct_execution(
    tmp_path, registered_cpu, command
):
    args = _command_args(tmp_path, command)
    work = tmp_path / "not-created"
    selected = build_parser(work).parse_args([*args, "--runtime", "cpu-test"])
    _prepare_runtime_command(selected)
    assert selected.runtime_profile_loaded.id == "cpu-test"
    assert selected.runtime_profile_loaded.python == registered_cpu.python
    assert selected.runtime_authorization is not None
    assert selected.runtime_authorization.as_dict()["probe"]["backend"] == "cpu"
    assert not work.exists()


@pytest.mark.parametrize(
    "command", ["train", "stage", "eval", "generate", "chat", "run"]
)
def test_cpu_id_cannot_authorize_accelerator_config_before_workdir(
    tmp_path, monkeypatch, registered_cpu, command
):
    args = _command_args(tmp_path, command)
    config_path = (
        Path(args[1])
        if command in {"train", "stage", "run"}
        else tmp_path / "runs/sample/resolved_config.yaml"
    )
    value = training_config(tmp_path).model_dump(mode="json")
    value["runtime"]["backend"] = "rocm"
    config_path.write_text(json.dumps(value))
    work = tmp_path / "new-work"
    monkeypatch.setattr(
        sys,
        "argv",
        ["sparselab", "--work-dir", str(work), *args, "--runtime", "cpu-test"],
    )
    with pytest.raises(SystemExit, match="backend|rocm|cpu"):
        main()
    assert not work.exists()
    assert not (tmp_path / "stage").exists()


def test_cli_bind_id_reopens_receipt_without_changing_science_under_relocated_registry(
    tmp_path, monkeypatch, capsys, registered_cpu
):
    import sparselab.experiments.lock as locking

    lock = _lock()
    original = lock.model_dump(mode="json")
    monkeypatch.setattr(locking, "open_lock", lambda _: lock)
    path = tmp_path / "locked-plan.json"
    work = tmp_path / "work"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "--work-dir",
            str(work),
            "experiment",
            "bind",
            str(path),
            "--cell",
            "main:single",
            "--runtime",
            "cpu-test",
            "--json",
        ],
    )
    main()
    result = json.loads(capsys.readouterr().out)
    receipt = Path(result["bindings"][0]["binding"])
    opened = open_runtime_binding(receipt, lock, lock.cells[0])
    assert opened["source_kind"] == "profile"
    assert opened["descriptor"]["id"] == "cpu-test"
    assert (
        Path(opened["descriptor"]["python"]).resolve()
        == registered_cpu.python.resolve()
    )
    assert opened["tested_runtime"]["backend"] == "cpu"
    assert "forward_backward_optimizer" in opened["tested_runtime"]["tested_features"]
    assert opened["scientific_sha256"] == lock.scientific_sha256
    assert opened["plan_sha256"] == lock.plan_sha256

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "relocated-config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "relocated-runtime-root"))
    register_runtime("cpu-test", registered_cpu)
    relocated = open_runtime_binding(receipt, lock, lock.cells[0])
    assert relocated["binding_sha256"] == opened["binding_sha256"]
    assert lock.model_dump(mode="json") == original
    assert _identities(original) == (lock.scientific_sha256, lock.plan_sha256)
