"""Parser and adapter contracts only: no tokenizer fitting or model execution."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.cli.main import build_parser, main
from sparselab.config.loading import load_config
from sparselab.workers import cli

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/runtime_smoke_cpu.yaml"


def test_public_launch_routes_and_removed_command(tmp_path):
    parser = build_parser(tmp_path / "work")
    choices = next(
        action.choices
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert "run" not in choices
    for family, verbs in {
        "experiment": {"submit", "run", "resume", "cancel", "collect"},
        "study": {"submit"},
        "attempt": {"run", "run-phase"},
    }.items():
        nested = next(
            action.choices
            for action in choices[family]._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        assert verbs <= nested.keys()
    assert {"try", "train", "stage"} <= choices.keys()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["run", str(CONFIG)])
    assert error.value.code == 2
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("other", ["--worker", "--runtime-profile"])
def test_runtime_selection_is_exclusive(other):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(
            ["experiment", "submit", str(CONFIG), "--runtime", "cpu", other, "other"]
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        [str(CONFIG), "--matrix", "absent.yaml"],
        [str(CONFIG), "--dry-run"],
        ["--matrix", "absent.yaml", "--wait"],
        ["--matrix", "absent.yaml", "--runtime", "unknown"],
        ["--matrix", "absent.yaml", "--runtime-profile", "absent.yaml"],
        ["--matrix", "absent.yaml", "--resource-envelope", "absent.yaml"],
        ["--matrix", "absent.yaml", "--transfer-timeout", "5"],
        ["--matrix", "absent.yaml", "--transfer-timeout", "1800"],
        ["--matrix", "absent.yaml", "--tokenizer-batch-documents", "256"],
        ["--matrix", "absent.yaml", "--tokenizer-batch-source-bytes", "1048576"],
        ["--matrix", "absent.yaml", "--tokenizer-batch-documents", "1"],
        ["--matrix", "absent.yaml", "--tokenizer-batch-source-bytes", "1"],
    ],
)
def test_invalid_submission_fails_before_workdir(tmp_path, monkeypatch, arguments):
    work = tmp_path / "absent"
    monkeypatch.setattr(
        sys,
        "argv",
        ["sparselab", "--work-dir", str(work), "experiment", "submit", *arguments],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not work.exists()


@pytest.mark.parametrize("wait", [False, True])
def test_unknown_runtime_fails_before_workdir(tmp_path, monkeypatch, wait):
    work = tmp_path / "absent"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "--work-dir",
            str(work),
            "experiment",
            "submit",
            str(CONFIG),
            "--runtime",
            "missing",
            *(["--wait"] if wait else []),
        ],
    )
    with pytest.raises(SystemExit, match="unknown runtime id"):
        main()
    assert not work.exists()


@pytest.mark.parametrize("wait", [False, True])
@pytest.mark.parametrize("worker", [None, "owned-worker"])
def test_single_config_dispatch_preserves_options_and_wait_semantics(
    tmp_path, monkeypatch, capsys, wait, worker
):
    observed = {}
    submission = SimpleNamespace(
        run_id="run-id", model_dump=lambda **_: {"run_id": "run-id"}
    )

    class Controller:
        root = tmp_path / "controller"

        def register(self, definition):
            observed["registered"] = definition

        def submit(self, config, **kwargs):
            observed["submission"] = kwargs
            return submission

    controller = Controller()
    monkeypatch.setattr(
        cli,
        "_controller",
        lambda root, **kwargs: observed.update(controller_options=kwargs) or controller,
    )
    monkeypatch.setattr(
        cli,
        "_wait_for_run",
        lambda owner, run_id: (
            observed.update(waited=(owner, run_id))
            or {"status": "COMPLETE", "ingestion_status": "COMPLETE"}
        ),
    )
    args = build_parser().parse_args(
        [
            "experiment",
            "submit",
            str(CONFIG),
            "--store",
            str(controller.root),
            "--stage-bundle",
            "stage",
            "--promote",
            "parent",
            "--transfer-timeout",
            "17",
            "--tokenizer-batch-documents",
            "3",
            "--tokenizer-batch-source-bytes",
            "1000",
            *(["--wait"] if wait else []),
            *(["--worker", worker] if worker else []),
        ]
    )
    args.runtime_authorization = None
    envelope = object()
    args.resource_envelope_value = envelope
    cli._experiment_submit(args)
    options = observed["submission"]
    assert options["stage_bundle"] == Path("stage")
    assert options["promote"] == Path("parent")
    assert options["resource_envelope"] is envelope
    assert options["tokenizer_batch_documents"] == 3
    assert options["tokenizer_batch_source_bytes"] == 1000
    assert observed["controller_options"] == {"transfer_timeout": 17}
    assert ("registered" in observed) == (wait and worker is None)
    assert options["worker"] == (
        observed["registered"].name if wait and worker is None else worker
    )
    assert ("waited" in observed) == wait
    payload = json.loads(capsys.readouterr().out)
    assert payload == (
        {
            "submission": {"run_id": "run-id"},
            "result": {"status": "COMPLETE", "ingestion_status": "COMPLETE"},
        }
        if wait
        else {"run_id": "run-id"}
    )


def test_local_accelerator_requires_authorization_before_controller(
    tmp_path, monkeypatch
):
    config = load_config(CONFIG)
    args = build_parser().parse_args(["experiment", "submit", str(CONFIG), "--wait"])
    args.runtime_authorization = None
    monkeypatch.setattr(
        "sparselab.staging.inspect_runtime", lambda _: SimpleNamespace(backend="cuda")
    )
    monkeypatch.setattr(cli, "_controller", lambda *a, **k: pytest.fail("opened store"))
    with pytest.raises(ValueError, match="requires --runtime-profile"):
        cli._submit_config(args, config)


def test_wait_drives_only_its_owned_controller_lock(monkeypatch):
    observed = []
    controller = SimpleNamespace(
        _process_lock=lambda **_: nullcontext(True),
        tick=lambda: observed.append("tick"),
        list_experiments=lambda: [
            {"run_id": "one", "status": "COMPLETE", "ingestion_status": "COMPLETE"}
        ],
    )
    assert cli._wait_for_run(controller, "one")["status"] == "COMPLETE"
    assert observed == ["tick"]


@pytest.mark.parametrize("status", ["FAILED", "CANCELLED", "INTERRUPTED", "UNKNOWN"])
def test_wait_forwards_non_success(status, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_wait_for_run", lambda *_: {"status": status})
    submission = SimpleNamespace(run_id="one", model_dump=lambda **_: {"run_id": "one"})
    with pytest.raises(SystemExit) as error:
        cli._wait_for_submission(object(), submission)
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["result"]["status"] == status


def test_disconnect_does_not_cancel_or_resubmit(tmp_path, monkeypatch, capsys):
    controller = SimpleNamespace(root=tmp_path / "store with spaces")
    submission = SimpleNamespace(run_id="one")

    def interrupted(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_wait_for_run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli._wait_for_submission(controller, submission)
    message = capsys.readouterr().err
    assert "no cancellation requested" in message
    assert "sparselab experiment collect one" in message
    assert "sparselab experiment cancel one" in message
    assert f"--store '{controller.root}'" in message


def test_colab_unknown_is_handed_off_without_resubmission(capsys):
    controller = SimpleNamespace(
        _process_lock=lambda **_: nullcontext(False),
        list_experiments=lambda: [
            {"run_id": "one", "status": "UNKNOWN", "ingestion_status": "PENDING"}
        ],
        _worker_for_attempt=lambda _: SimpleNamespace(transport="colab"),
    )
    assert cli._wait_for_run(controller, "one")["status"] == "UNKNOWN"
    assert "do not resubmit" in capsys.readouterr().err


@pytest.mark.parametrize("wait", [False, True])
def test_selected_runtime_authorizes_config_and_registers_same_interpreter(
    tmp_path, monkeypatch, wait
):
    import sparselab.cli.main as main_cli
    from sparselab.runtime_profile import RuntimeProfile

    profile = RuntimeProfile(
        runtime_profile_version=1,
        id="selected",
        python=Path(sys.executable).absolute(),
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    observed = {}
    authorization = object()
    monkeypatch.setattr(
        "sparselab.runtime_environments.profile_for_id", lambda _: profile
    )
    monkeypatch.setattr(
        main_cli,
        "_ensure_runtime_interpreter",
        lambda python, digest, probe: observed.update(python=python),
    )
    monkeypatch.setattr(
        main_cli,
        "authorize_profile",
        lambda selected, config: (
            observed.update(profile=selected, config=config) or authorization
        ),
    )
    args = build_parser().parse_args(
        [
            "experiment",
            "submit",
            str(CONFIG),
            "--runtime",
            "selected",
            *(["--wait"] if wait else []),
        ]
    )
    main_cli._prepare_runtime_command(args)
    assert args.runtime_authorization is authorization
    assert observed["profile"] is profile
    assert observed["config"] == load_config(CONFIG)
    assert observed["python"] == profile.python
    args.resource_envelope_value = None
    controller = SimpleNamespace(
        root=tmp_path / "controller",
        register=lambda worker: observed.update(worker=worker),
        submit=lambda config, **kwargs: observed.update(submission=kwargs),
    )
    monkeypatch.setattr(cli, "_controller", lambda *a, **k: controller)
    cli._submit_config(args, observed["config"])
    assert observed["worker"].python == profile.python
    assert observed["submission"]["worker"] == observed["worker"].name


def test_registered_worker_submission_skips_local_authorization(monkeypatch):
    import sparselab.cli.main as main_cli

    monkeypatch.setattr(
        main_cli, "require_authorization", lambda *a: pytest.fail("local authorization")
    )
    args = build_parser().parse_args(
        [
            "experiment",
            "submit",
            "unavailable-here.yaml",
            "--wait",
            "--worker",
            "remote",
        ]
    )
    main_cli._prepare_runtime_command(args)
    assert args.runtime_authorization is None


def test_submission_resource_envelope_is_checked_at_store_before_creation(
    tmp_path, monkeypatch
):
    import sparselab.cli.main as main_cli
    from sparselab.resource_envelope import ResourceEnvelope

    store = tmp_path / "controller"
    work = tmp_path / "work"
    envelope = ResourceEnvelope(resource_envelope_version=1)
    monkeypatch.setattr(main_cli, "load_resource_envelope", lambda _: envelope)

    def reject(selected, *, workspace, **kwargs):
        assert selected is envelope
        assert workspace == store
        raise ValueError("resource envelope refused")

    monkeypatch.setattr(main_cli, "check_envelope", reject)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "--work-dir",
            str(work),
            "experiment",
            "submit",
            str(CONFIG),
            "--worker",
            "owned",
            "--wait",
            "--store",
            str(store),
            "--resource-envelope",
            "envelope.json",
        ],
    )
    with pytest.raises(SystemExit, match="resource envelope refused"):
        main()
    assert not store.exists()
    assert not work.exists()


@pytest.mark.parametrize("wait", [False, True])
def test_omitted_single_config_controls_keep_native_defaults(
    tmp_path, monkeypatch, wait
):
    observed = {}
    controller = SimpleNamespace(
        root=tmp_path / "controller",
        submit=lambda config, **kwargs: observed.update(submission=kwargs),
    )
    monkeypatch.setattr(
        cli,
        "_controller",
        lambda root, **kwargs: observed.update(controller_options=kwargs) or controller,
    )
    args = build_parser().parse_args(
        [
            "experiment",
            "submit",
            str(CONFIG),
            "--worker",
            "owned",
            *(["--wait"] if wait else []),
        ]
    )
    assert args.transfer_timeout is None
    assert args.tokenizer_batch_documents is None
    assert args.tokenizer_batch_source_bytes is None
    cli.validate_submit_args(args)
    args.resource_envelope_value = None
    cli._submit_config(args, load_config(CONFIG))
    assert observed["controller_options"] == {"transfer_timeout": 1800}
    assert observed["submission"]["tokenizer_batch_documents"] == 256
    assert observed["submission"]["tokenizer_batch_source_bytes"] == 1048576
