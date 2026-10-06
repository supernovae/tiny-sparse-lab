"""CLI observation envelope admission and publication behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from test_training import config as training_config

from sparselab.phase_observations import (
    observation_session,
    validate_observations_destination,
)


def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    from sparselab.cli.main import main

    monkeypatch.setattr(sys, "argv", ["sparselab", *argv])
    main()


def _config_file(config: Any, root: Path) -> Path:
    path = root / "config.json"
    path.write_text(config.model_dump_json())
    return path


def _prepare_args(config: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        command="data",
        data_command="prepare",
        config=str(config),
        observations_output=output,
        runtime=None,
        runtime_profile_loaded=None,
    )


def test_prepare_observation_envelope_is_operational_and_atomic(tmp_path: Path) -> None:
    config = training_config(tmp_path)
    source = _config_file(config, tmp_path)
    destination = tmp_path / "observations.json"
    args = _prepare_args(source, destination)

    validate_observations_destination(args)
    with observation_session(args) as observer:
        assert observer is not None
        with observer.phase("native_work"):
            pass

    report = json.loads(destination.read_text())
    assert report["format"] == "sparselab-phase-observations-v1"
    assert report["operation"] == "data.prepare"
    assert report["status"] == "completed"
    assert report["input"]["path"] == str(source.absolute())
    assert {row["phase"] for row in report["records"]} == {
        "native_work",
        "data.prepare",
    }
    assert report["records"][-1]["cpu_io_coverage"] == (
        "surviving_processes_at_both_endpoints_only"
    )
    assert not hasattr(args, "observer")


def test_native_prepare_without_option_never_allocates_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = training_config(tmp_path)
    source = _config_file(config, tmp_path)
    import sparselab.phase_observations as observations

    def forbidden() -> None:
        raise AssertionError("observer allocated without --observations-output")

    monkeypatch.setattr(observations, "BottleneckObserver", forbidden)
    _run_main(
        monkeypatch,
        "--work-dir",
        str(tmp_path / "work"),
        "data",
        "prepare",
        str(source),
    )
    assert capsys.readouterr().out.strip()


def test_native_prepare_and_stage_observations_preserve_artifact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise real native artifact verification, preparation, and staging."""
    config = training_config(tmp_path)
    source = _config_file(config, tmp_path)
    work = tmp_path / "work"
    observations = tmp_path / "observations"
    observations.mkdir()

    _run_main(monkeypatch, "--work-dir", str(work), "data", "prepare", str(source))
    prepared_root = Path(capsys.readouterr().out.strip())
    prepared_manifest = (prepared_root / "manifest.json").read_bytes()

    _run_main(
        monkeypatch,
        "--work-dir",
        str(work),
        "data",
        "prepare",
        str(source),
        "--observations-output",
        str(observations / "prepare.json"),
    )
    assert Path(capsys.readouterr().out.strip()) == prepared_root
    assert (prepared_root / "manifest.json").read_bytes() == prepared_manifest
    prepare_report = json.loads((observations / "prepare.json").read_text())
    assert prepare_report["status"] == "completed"
    assert prepare_report["runtime"]["reason"] == "preparation_executes_on_host"
    assert any(row["phase"] == "data.prepare" for row in prepare_report["records"])

    plain_stage = tmp_path / "stage-plain"
    observed_stage = tmp_path / "stage-observed"
    _run_main(
        monkeypatch,
        "--work-dir",
        str(work),
        "stage",
        str(source),
        "--through",
        "validate",
        "--output",
        str(plain_stage),
    )
    capsys.readouterr()
    _run_main(
        monkeypatch,
        "--work-dir",
        str(work),
        "stage",
        str(source),
        "--through",
        "validate",
        "--output",
        str(observed_stage),
        "--observations-output",
        str(observations / "stage.json"),
    )
    capsys.readouterr()
    plain_bundle = json.loads((plain_stage / "bundle.json").read_text())
    observed_bundle = json.loads((observed_stage / "bundle.json").read_text())
    assert (
        plain_bundle["config_sha256"],
        plain_bundle["source_identity_sha256"],
        plain_bundle["inputs_sha256"],
    ) == (
        observed_bundle["config_sha256"],
        observed_bundle["source_identity_sha256"],
        observed_bundle["inputs_sha256"],
    )
    stage_report = json.loads((observations / "stage.json").read_text())
    assert stage_report["status"] == "completed"
    assert stage_report["runtime"]["observation_scope"] == "host_process_tree"
    assert stage_report["runtime"]["resolved"] is not None


def test_invalid_observation_destinations_fail_before_work_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = training_config(tmp_path)
    source = _config_file(config, tmp_path)
    work = tmp_path / "new-work"
    for destination in (
        config.dataset.cache_dir / "report.json",
        config.tokenizer.path,
        tmp_path / "not-json.txt",
    ):
        with pytest.raises(SystemExit):
            _run_main(
                monkeypatch,
                "--work-dir",
                str(work),
                "data",
                "prepare",
                str(source),
                "--observations-output",
                str(destination),
            )
        assert not work.exists()
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing.json")
    with pytest.raises(SystemExit):
        _run_main(
            monkeypatch,
            "--work-dir",
            str(work),
            "data",
            "prepare",
            str(source),
            "--observations-output",
            str(dangling),
        )
    assert not work.exists()


def test_observation_publication_race_preserves_native_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = training_config(tmp_path)
    source = _config_file(config, tmp_path)
    output = tmp_path / "report.json"
    import sparselab.phase_observations as observations

    original = observations._exclusive_bytes

    def competing(path: Path, content: bytes) -> None:
        path.write_bytes(b"other-writer\n")
        original(path, content)

    monkeypatch.setattr(observations, "_exclusive_bytes", competing)
    _run_main(
        monkeypatch,
        "--work-dir",
        str(tmp_path / "work"),
        "data",
        "prepare",
        str(source),
        "--observations-output",
        str(output),
    )
    captured = capsys.readouterr()
    assert captured.out.strip()
    assert output.read_bytes() == b"other-writer\n"
    assert "warning: could not publish phase observations" in captured.err


def test_invalid_observation_destinations_fail_before_session(tmp_path: Path) -> None:
    config = training_config(tmp_path)
    source = _config_file(config, tmp_path)
    for destination in (
        config.dataset.cache_dir / "report.json",
        config.tokenizer.path,
        tmp_path / "not-json.txt",
    ):
        args = _prepare_args(source, destination)
        with pytest.raises((ValueError, FileExistsError)):
            validate_observations_destination(args)
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        validate_observations_destination(_prepare_args(source, dangling))


@pytest.mark.parametrize(
    ("failure", "status"),
    [(RuntimeError("workflow failed"), "failed"), (KeyboardInterrupt(), "interrupted")],
)
def test_session_retains_failure_status(
    tmp_path: Path, failure: BaseException, status: str
) -> None:
    config = training_config(tmp_path)
    source = _config_file(config, tmp_path)
    args = _prepare_args(source, tmp_path / f"{status}.json")
    validate_observations_destination(args)
    with pytest.raises(type(failure)), observation_session(args):
        raise failure
    assert json.loads(Path(args.observations_output).read_text())["status"] == status


def test_unavailable_counter_reasons_are_field_addressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = training_config(tmp_path)
    source = _config_file(config, tmp_path)
    args = _prepare_args(source, tmp_path / "unavailable.json")
    validate_observations_destination(args)
    import sparselab.bottleneck_observations as bottleneck
    import sparselab.phase_observations as observations

    def unavailable_snapshot() -> None:
        raise OSError("probe unavailable")

    def unavailable_accelerator() -> None:
        raise OSError("device probe unavailable")

    monkeypatch.setattr(bottleneck, "_snapshot", unavailable_snapshot)
    monkeypatch.setattr(
        observations,
        "BottleneckObserver",
        lambda: bottleneck.BottleneckObserver(
            accelerator_probe=unavailable_accelerator
        ),
    )
    with observation_session(args):
        pass
    report = json.loads(Path(args.observations_output).read_text())
    reasons = report["unavailable_reasons"]
    assert (
        reasons["records[0].host_available_ram_bytes"]
        == "counter_unavailable_or_incomplete"
    )
    assert (
        reasons["records[0].accelerator_utilization_percent"]
        == "counter_unavailable_or_incomplete"
    )


def test_publication_failure_does_not_replace_workflow_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = training_config(tmp_path)
    source = _config_file(config, tmp_path)
    args = _prepare_args(source, tmp_path / "observations.json")
    validate_observations_destination(args)
    import sparselab.phase_observations as observations

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise OSError("output raced")

    monkeypatch.setattr(observations, "_exclusive_bytes", unavailable)
    with (
        pytest.raises(RuntimeError, match="workflow failed"),
        observation_session(args),
    ):
        raise RuntimeError("workflow failed")
    assert "warning: could not publish phase observations" in capsys.readouterr().err


def test_protected_inventory_alias_resolves_before_containment(tmp_path):
    config = training_config(tmp_path)
    actual = tmp_path / "actual-cache"
    actual.mkdir()
    alias = tmp_path / "cache-alias"
    alias.symlink_to(actual, target_is_directory=True)
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"cache_dir": alias})}
    )
    source = _config_file(config, tmp_path)
    args = _prepare_args(source, actual / "observations.json")
    with pytest.raises(ValueError, match="inventoried"):
        validate_observations_destination(args)


@pytest.mark.parametrize("selection", ["runtime", "runtime-profile"])
def test_native_stage_observations_preserve_cpu_runtime_authorization(
    tmp_path, monkeypatch, selection
):
    from sparselab.runtime_environments import RuntimeEntry, register_runtime
    from sparselab.runtime_profile import RuntimeProfile

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
    config = training_config(tmp_path)
    source = _config_file(config, tmp_path)
    profile = RuntimeProfile(
        runtime_profile_version=1,
        id="observed-cpu",
        python=Path(sys.executable).absolute(),
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    if selection == "runtime":
        register_runtime(
            profile.id,
            RuntimeEntry(
                python=profile.python,
                engine=profile.engine,
                backend=profile.backend,
                device_index=profile.device_index,
            ),
        )
        selector = profile.id
    else:
        path = tmp_path / "profile.json"
        path.write_text(profile.model_dump_json())
        selector = str(path)
    output = tmp_path / "stage"
    observations = tmp_path / "observations.json"
    _run_main(
        monkeypatch,
        "--work-dir",
        str(tmp_path / "work"),
        "stage",
        str(source),
        "--through",
        "validate",
        "--output",
        str(output),
        f"--{selection}",
        selector,
        "--observations-output",
        str(observations),
    )
    report = json.loads(observations.read_text())
    assert report["runtime"]["requested"]["selection"]["id"] == "observed-cpu"
    assert report["runtime"]["resolved"]["backend"] == "cpu"
    stage_report = json.loads((output / "stage.json").read_text())
    assert stage_report["runtime_authorization"] is not None
    assert report["runtime"]["observation_scope"] == "host_process_tree"


@pytest.mark.parametrize("boundary", ["setup", "collection"])
def test_optional_observer_failures_preserve_native_preparation(
    tmp_path, monkeypatch, capsys, caplog, boundary
):
    import sparselab.bottleneck_observations as counters
    import sparselab.phase_observations as observations

    config = training_config(tmp_path)
    source = _config_file(config, tmp_path)
    work = tmp_path / "work"
    _run_main(monkeypatch, "--work-dir", str(work), "data", "prepare", str(source))
    prepared = Path(capsys.readouterr().out.strip())
    original = (prepared / "manifest.json").read_bytes()

    def unavailable(*args, **kwargs):
        raise RuntimeError("injected optional observation failure")

    if boundary == "setup":
        monkeypatch.setattr(observations, "BottleneckObserver", unavailable)
    else:
        monkeypatch.setattr(counters, "_classify", unavailable)
    destination = tmp_path / "observations.json"
    _run_main(
        monkeypatch,
        "--work-dir",
        str(work),
        "data",
        "prepare",
        str(source),
        "--observations-output",
        str(destination),
    )
    captured = capsys.readouterr()
    assert Path(captured.out.strip()) == prepared
    assert (prepared / "manifest.json").read_bytes() == original
    report = json.loads(destination.read_text())
    assert report["status"] == "completed"
    assert report["records"] == []
    assert (
        report["unavailable_reasons"]["records"] == "counter_unavailable_or_incomplete"
    )
    assert "unavailable" in captured.err + caplog.text


def test_failed_stage_retains_selected_profile_without_claiming_resolved_execution(
    tmp_path, monkeypatch
):
    import sparselab.cli.main as cli_module
    from sparselab.runtime_profile import RuntimeProfile

    config = training_config(tmp_path)
    source = _config_file(config, tmp_path)
    profile = RuntimeProfile(
        runtime_profile_version=1,
        id="selected-cpu",
        python=Path(sys.executable).absolute(),
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(profile.model_dump_json())
    destination = tmp_path / "observations.json"

    def failed_stage(*args, **kwargs):
        raise RuntimeError("stage execution failed")

    monkeypatch.setattr(cli_module, "stage", failed_stage)
    with pytest.raises(RuntimeError, match="stage execution failed"):
        _run_main(
            monkeypatch,
            "--work-dir",
            str(tmp_path / "work"),
            "stage",
            str(source),
            "--through",
            "validate",
            "--output",
            str(tmp_path / "stage"),
            "--runtime-profile",
            str(profile_path),
            "--observations-output",
            str(destination),
        )
    report = json.loads(destination.read_text())
    assert report["status"] == "failed"
    assert report["runtime"]["requested"]["id"] == profile.id
    assert report["runtime"]["resolved"] is None
    assert report["runtime"]["reason"] == "execution_runtime_not_resolved"


@pytest.mark.parametrize(
    "target", ["tokenizer_reference", "inline_plan_runs", "file_plan_runs"]
)
def test_campaign_observation_admission_protects_declared_native_inputs(
    tmp_path, monkeypatch, capsys, target
):
    import yaml
    from test_experiment_direct_inputs import inputs

    from sparselab.experiments.direct_inputs import bind_direct_inputs
    from sparselab.experiments.plan import base_run_config

    config, prepared, run, template = inputs(tmp_path, monkeypatch)
    source_plan = tmp_path / "experiment.yaml"
    plan = bind_direct_inputs(run, template, prepared, source_plan)
    stages = [
        {
            "id": "tokenizer",
            "kind": "tokenizer_reference",
            "scope": "tokenizer",
            "artifact": plan.artifacts["tokenizer"].model_dump(mode="json"),
        }
    ]
    if target == "tokenizer_reference":
        parent = config.tokenizer.path.parent
    else:
        stages.extend(
            [
                {
                    "id": "prepared",
                    "kind": "artifact_reference",
                    "scope": "model",
                    "artifact": plan.artifacts["packed"].model_dump(mode="json"),
                },
                {
                    "id": "plan",
                    "kind": "experiment_plan",
                    "scope": "model",
                    "mode": "lock",
                    "source": source_plan.name,
                    "tokenizer": "tokenizer",
                    "prepared": "prepared",
                    "requires": ["tokenizer", "prepared"],
                },
            ]
        )
        if target == "file_plan_runs":
            plan = plan.model_copy(update={"base_run": run.name})
            source_plan.write_text(yaml.safe_dump(plan.model_dump(mode="json")))
        parent = base_run_config(plan, source_plan).logging.root_dir
        parent.mkdir(exist_ok=True)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        yaml.safe_dump(
            {
                "campaign_version": 1,
                "id": "observed-inputs",
                "stages": stages,
            }
        )
    )
    work = tmp_path / "campaign-work"
    capsys.readouterr()
    with pytest.raises(SystemExit) as failure:
        _run_main(
            monkeypatch,
            "--work-dir",
            str(work),
            "campaign",
            "status",
            str(campaign),
            "--observations-output",
            str(parent / "observations.json"),
            "--json",
        )
    assert failure.value.code == 2
    assert "inventoried" in json.loads(capsys.readouterr().out)["error"]["reason"]
    assert not work.exists()
    assert not (parent / "observations.json").exists()


def test_campaign_admission_protects_unmaterialized_corpus_namespace(
    tmp_path, monkeypatch, capsys
):
    import yaml

    work = tmp_path / "work"
    output = work / "corpora" / "declared-corpus" / "releases"
    output.mkdir(parents=True)
    source = tmp_path / "campaign.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "campaign_version": 1,
                "id": "declared-corpus",
                "stages": [
                    {
                        "id": "corpus",
                        "kind": "corpus_release",
                        "scope": "corpus",
                        "project": "not-yet-materialized/project.yaml",
                    }
                ],
            }
        )
    )
    with pytest.raises(SystemExit) as failure:
        _run_main(
            monkeypatch,
            "--work-dir",
            str(work),
            "campaign",
            "status",
            str(source),
            "--observations-output",
            str(output / "observations.json"),
            "--json",
        )
    assert failure.value.code == 2
    assert "inventoried" in json.loads(capsys.readouterr().out)["error"]["reason"]
    assert not (work / "campaigns").exists()
    assert not (output / "observations.json").exists()
