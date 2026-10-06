"""Verified ExperimentPlan derivation retains the native lock authority."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from test_experiment_direct_inputs import inputs
from test_experiment_variant_tokenizer import authored

from sparselab.cli.main import main
from sparselab.experiments.direct_inputs import bind_direct_inputs
from sparselab.experiments.export_config import export_effective_config
from sparselab.experiments.lock import open_lock, publish_lock, resolve_plan
from sparselab.experiments.plan import load_plan
from sparselab.training.manifest import sha256_file

# Register the shared real Forge preparation fixture under its local name.
forge_authored = authored


def _bound_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Create an authored direct-input declaration with real prepared bytes."""
    _, prepared_root, run, template = inputs(tmp_path, monkeypatch)
    source = tmp_path / "bound.yaml"
    bind_direct_inputs(run, template, prepared_root, source)
    return source, prepared_root


def _sidecar(output: Path) -> Path:
    return output.with_name(output.name + ".derivation.json")


def _derive_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: Path,
    output: Path,
    *settings: str,
) -> dict[str, object]:
    argv = ["sparselab", "experiment", "derive", str(source)]
    for setting in settings:
        argv.extend(["--set", setting])
    argv.extend(["--output", str(output), "--json"])
    monkeypatch.setattr(sys, "argv", argv)
    main()
    return json.loads(capsys.readouterr().out)


def test_derive_resolves_real_inputs_then_lock_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _ = _bound_source(tmp_path, monkeypatch)
    output = tmp_path / "derived" / "plan.yaml"
    output.parent.mkdir()

    response = _derive_cli(
        monkeypatch,
        capsys,
        source,
        output,
        'id="derived-direct-input"',
        "retention.keep_periodic=false",
        "base_run.optimizer.peak=0.0005",
    )

    assert response["format"] == "sparselab-experiment-command-v1"
    assert response["command"] == "derive"
    assert response["status"] == "ok"
    assert response["receipt"] == str(_sidecar(output))
    receipt = json.loads(_sidecar(output).read_text(encoding="utf-8"))
    assert response["derivation"] == receipt
    assert receipt["kind"] == "experiment_plan"
    assert receipt["validation"] == {
        "scope": "resolved_experiment",
        "executable": False,
        "runtime_verified": False,
    }
    assert receipt["origin"]["file_sha256"] == sha256_file(source)
    assert receipt["output"]["file_sha256"] == sha256_file(output)
    assert receipt["declared_delta"] == {
        "id": "derived-direct-input",
        "retention.keep_periodic": False,
        "base_run.optimizer.peak": 0.0005,
    }

    plan = load_plan(output)
    assert plan.id == "derived-direct-input"
    assert not plan.retention.keep_periodic
    assert plan.base_run.optimizer.peak == 0.0005
    resolved = resolve_plan(plan, output)
    lock_path = publish_lock(resolved, tmp_path / "locks")
    locked = open_lock(lock_path)
    effective = tmp_path / "effective.yaml"
    assert (
        export_effective_config(lock_path, "pretrain:single", effective)
        == locked.cells[0].config_sha256
    )
    assert locked.cells[0].config.optimizer.peak == 0.0005
    assert locked.cells[0].config.tokenizer.path == Path(
        plan.artifacts["tokenizer"].path
    )
    assert locked.artifacts["packed"]["sha256"] == plan.artifacts["packed"].sha256
    prepared_manifest = json.loads(
        (Path(plan.artifacts["packed"].path) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan.artifacts["packed"].sha256 == prepared_manifest["manifest_sha256"]
    assert plan.artifacts["packed"].sha256 != sha256_file(
        Path(plan.artifacts["packed"].path) / "manifest.json"
    )
    assert "plan_sha256" not in receipt
    monkeypatch.setattr(
        sys,
        "argv",
        ["sparselab", "experiment", "lock", str(output), "--json"],
    )
    main()
    public_lock = Path(json.loads(capsys.readouterr().out)["lock"])
    cli_effective = tmp_path / "cli-effective.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "experiment",
            "export-config",
            str(public_lock),
            "--cell",
            "pretrain:single",
            "--output",
            str(cli_effective),
            "--json",
        ],
    )
    main()
    exported = json.loads(capsys.readouterr().out)
    assert exported["config_sha256"] == open_lock(public_lock).cells[0].config_sha256


def test_derive_materializes_relative_base_without_fabricating_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A referenced base supports nested edits and records materialization separately."""
    from sparselab.derivation import derive_experiment

    source, _ = _bound_source(tmp_path, monkeypatch)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["base_run"] = "run.yaml"
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    output = tmp_path / "relative-base-derived.yaml"

    receipt = derive_experiment(source, output, {"base_run.optimizer.peak": 0.0004})
    plan = load_plan(output)
    assert plan.base_run.optimizer.peak == 0.0004
    materialization = next(
        record for record in receipt["path_rebindings"] if record["path"] == "base_run"
    )
    assert materialization["source_value"] == "run.yaml"
    assert materialization["resolved_target"] == str((tmp_path / "run.yaml").resolve())
    assert resolve_plan(plan, output).cells[0].config.optimizer.peak == 0.0004
    assert {record["path"] for record in receipt["origin_declarations"]} == {
        str(source.resolve()),
        str((tmp_path / "run.yaml").resolve()),
    }


def test_derive_relocates_a_real_evaluation_suite_only_to_same_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declaration reference is rewritten only when destination resolution agrees."""
    from sparselab.derivation import derive_experiment
    from sparselab.recovery.provenance import declaration_reference

    source, _ = _bound_source(tmp_path, monkeypatch)
    destination = tmp_path / "derived"
    destination.mkdir()
    suite = destination / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "evaluation_suite_version": 1,
                "id": "synthetic-heldout",
                "evaluations": [
                    {"id": "loss", "role": "descriptive", "kind": "heldout_lm"}
                ],
            }
        ),
        encoding="utf-8",
    )
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["evaluation_suite"] = "derived/suite.yaml"
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    output = destination / "plan.yaml"

    receipt = derive_experiment(source, output, {"retention.keep_best": False})
    plan = load_plan(output)
    assert plan.evaluation_suite == "suite.yaml"
    assert declaration_reference(output, plan.evaluation_suite) == suite.resolve()
    assert any(
        record["path"] == "evaluation_suite"
        and record["resolved_target"] == str(suite.resolve())
        for record in receipt["path_rebindings"]
    )


def test_derive_rejects_suite_relocation_outside_destination_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.derivation import derive_experiment

    source, _ = _bound_source(tmp_path, monkeypatch)
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "evaluation_suite_version: 1\nid: synthetic-heldout\nevaluations: []\n",
        encoding="utf-8",
    )
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["evaluation_suite"] = "suite.yaml"
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    destination = tmp_path / "isolated"
    destination.mkdir()
    output = destination / "plan.yaml"

    with pytest.raises(ValueError, match="evaluation_suite.*cannot relocate"):
        derive_experiment(source, output, {"retention.keep_best": False})
    assert not output.exists()
    assert not _sidecar(output).exists()


def test_derive_preserves_repository_root_reference_and_rejects_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repository fallback is retained only when it resolves the original target."""
    from sparselab.derivation import derive_experiment
    from sparselab.recovery.provenance import declaration_reference

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source, _ = _bound_source(tmp_path, monkeypatch)
    declarations = tmp_path / "declarations"
    declarations.mkdir()
    source = source.replace(declarations / "bound.yaml")
    shared = tmp_path / "shared"
    shared.mkdir()
    suite = shared / "suite.yaml"
    suite.write_text(
        "evaluation_suite_version: 1\nid: root-suite\n"
        "evaluations: [{id: loss, role: descriptive, kind: heldout_lm}]\n",
        encoding="utf-8",
    )
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["evaluation_suite"] = "shared/suite.yaml"
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    output_dir = tmp_path / "published"
    output_dir.mkdir()
    output = output_dir / "plan.yaml"

    derive_experiment(source, output, {"retention.keep_best": False})
    plan = load_plan(output)
    assert plan.evaluation_suite == "shared/suite.yaml"
    assert declaration_reference(output, plan.evaluation_suite) == suite.resolve()

    shadow = output_dir / "shared"
    shadow.mkdir()
    (shadow / "suite.yaml").write_text(
        "evaluation_suite_version: 1\nid: shadow\n"
        "evaluations: [{id: loss, role: descriptive, kind: heldout_lm}]\n",
        encoding="utf-8",
    )
    rejected = output_dir / "shadowed.yaml"
    with pytest.raises(ValueError, match="evaluation_suite.*cannot relocate"):
        derive_experiment(source, rejected, {"retention.keep_best": False})
    assert not rejected.exists()
    assert not _sidecar(rejected).exists()


def test_derive_requires_matching_real_variant_preparation(
    forge_authored: tuple[Path, Path, dict], tmp_path: Path
) -> None:
    """Prepared corpus-variant evidence remains bound to the authored plan ID."""
    from sparselab.derivation import derive_experiment
    from sparselab.experiments.plan import ExperimentPlan
    from sparselab.experiments.prepare import prepare_plan

    source, workspace, raw = forge_authored
    plan = ExperimentPlan.model_validate(raw)
    preparation = {
        "format": "experiment-preparation-v1",
        "id": plan.id,
        "variants": prepare_plan(plan, source, workspace)["variants"],
    }
    prepared_path = tmp_path / "preparation.json"
    prepared_path.write_text(json.dumps(preparation), encoding="utf-8")
    output = source.with_name("derived-variant-plan.yaml")
    receipt = derive_experiment(
        source,
        output,
        {"retention.keep_best": False},
        prepared=prepared_path,
    )
    assert receipt["preparation"]["file_sha256"] == sha256_file(prepared_path)
    assert (
        resolve_plan(
            load_plan(output),
            output,
            prepared=preparation,
        ).id
        == plan.id
    )
    from sparselab.recovery.provenance import declaration_reference

    assert declaration_reference(
        output, load_plan(output).corpus_variants[0].project
    ) == declaration_reference(source, plan.corpus_variants[0].project)

    wrong = {**preparation, "id": "wrong-preparation-id"}
    wrong_path = tmp_path / "wrong-preparation.json"
    wrong_path.write_text(json.dumps(wrong), encoding="utf-8")
    rejected = source.with_name("wrong-prepared-plan.yaml")
    with pytest.raises(ValueError):
        derive_experiment(
            source,
            rejected,
            {"retention.keep_best": False},
            prepared=wrong_path,
        )
    assert not rejected.exists()
    assert not _sidecar(rejected).exists()

    renamed = source.with_name("renamed-prepared-plan.yaml")
    with pytest.raises(ValueError):
        derive_experiment(
            source, renamed, {"id": "new-plan-id"}, prepared=prepared_path
        )
    assert not renamed.exists()
    assert not _sidecar(renamed).exists()
    assert json.loads(prepared_path.read_text()) == preparation


def test_derive_anchors_axis_and_phase_path_patches_to_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.derivation import derive_experiment

    source, _ = _bound_source(tmp_path, monkeypatch)
    output = tmp_path / "paths-derived.yaml"
    derive_experiment(
        source,
        output,
        {
            "axes": [
                {
                    "name": "destination",
                    "choices": [
                        {
                            "label": "one",
                            "set": {"logging.root_dir": "axis-runs"},
                        }
                    ],
                }
            ],
            "phases": [
                {
                    "id": "pretrain",
                    "transition": "fresh",
                    "set": {"logging.root_dir": "phase-runs"},
                }
            ],
        },
    )
    plan = load_plan(output)
    assert plan.axes[0].choices[0].set["logging.root_dir"] == str(
        (source.parent / "axis-runs").resolve()
    )
    assert plan.phases[0].set["logging.root_dir"] == str(
        (source.parent / "phase-runs").resolve()
    )


def test_derive_rejects_invalid_full_declaration_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.derivation import derive_experiment

    source, _ = _bound_source(tmp_path, monkeypatch)
    invalid = tmp_path / "invalid.yaml"
    with pytest.raises(ValueError):
        derive_experiment(
            source,
            invalid,
            {
                "phases": [
                    {
                        "id": "bad",
                        "transition": "fresh",
                        "set": {"not.a.real.run_config_field": 1},
                    }
                ]
            },
        )
    assert not invalid.exists()
    assert not _sidecar(invalid).exists()

    comparison = tmp_path / "comparison.yaml"
    with pytest.raises(ValueError):
        derive_experiment(
            source,
            comparison,
            {
                "axes": [
                    {
                        "name": "batch",
                        "choices": [
                            {
                                "label": "baseline",
                                "set": {"training.micro_batch_size": 1},
                            },
                            {
                                "label": "variant",
                                "set": {"training.micro_batch_size": 2},
                            },
                        ],
                    }
                ],
                "comparisons": [
                    {
                        "id": "invalid",
                        "baseline": {"batch": "baseline"},
                        "variant": {"batch": "variant"},
                        "interventions": ["model.ffn_dim"],
                        "phases": ["pretrain"],
                    }
                ],
            },
        )
    assert not comparison.exists()
    assert not _sidecar(comparison).exists()


def test_derive_reauthenticates_tampered_prepared_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.derivation import derive_experiment

    source, prepared_root = _bound_source(tmp_path, monkeypatch)
    packed = prepared_root / "train.npy"
    packed.write_bytes(packed.read_bytes() + b"tampered")
    output = tmp_path / "tampered.yaml"

    with pytest.raises(ValueError):
        derive_experiment(source, output, {"retention.keep_best": False})
    assert not output.exists()
    assert not _sidecar(output).exists()


def test_derive_cli_expected_errors_are_enveloped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, _ = _bound_source(tmp_path, monkeypatch)
    output = tmp_path / "failed.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "experiment",
            "derive",
            str(source),
            "--set",
            "training.seq_len=999999",
            "--output",
            str(output),
            "--json",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    response = json.loads(capsys.readouterr().out)
    assert response["format"] == "sparselab-experiment-command-v1"
    assert response["command"] == "derive"
    assert response["status"] == "error"
    assert not output.exists()
    assert not _sidecar(output).exists()


@pytest.mark.parametrize(
    "settings",
    [
        ["retention.keep_best=not-json"],
        ["retention.keep_best=false", "retention.keep_best=true"],
        ['retention={"keep_best":false}', "retention.keep_best=true"],
    ],
)
def test_derive_cli_rejects_invalid_assignments_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings: list[str],
) -> None:
    source, _ = _bound_source(tmp_path, monkeypatch)
    output = tmp_path / "invalid-assignment.yaml"
    argv = ["sparselab", "experiment", "derive", str(source)]
    for setting in settings:
        argv.extend(["--set", setting])
    argv.extend(["--output", str(output), "--json"])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "error"
    assert response["command"] == "derive"
    assert not output.exists()
    assert not _sidecar(output).exists()


@pytest.mark.parametrize("artifact", ["tokenizer", "packed"])
def test_derive_rejects_wrong_declared_artifact_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    """The candidate is cold-verified rather than trusting declaration metadata."""
    from sparselab.derivation import derive_experiment

    source, _ = _bound_source(tmp_path, monkeypatch)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["artifacts"][artifact]["sha256"] = "0" * 64
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    output = tmp_path / f"wrong-{artifact}.yaml"

    with pytest.raises(ValueError):
        derive_experiment(source, output, {"retention.keep_terminal": False})
    assert not output.exists()
    assert not _sidecar(output).exists()


def test_derive_preserves_valid_extension_and_rejects_invalid_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continuation compatibility is checked by the native resolver, not inferred."""
    from sparselab.config.models import RunConfig
    from sparselab.derivation import derive_experiment
    from sparselab.training.checkpoints import CheckpointManager
    from sparselab.training.trainer import train

    config, prepared_root, run, template = inputs(tmp_path, monkeypatch)
    raw_config = config.model_dump(mode="json")
    raw_config["training"].update(max_steps=4, max_tokens=128)
    raw_config["optimizer"]["decay_steps"] = 4
    parent = RunConfig.model_validate(raw_config)
    run.write_text(yaml.safe_dump(parent.model_dump(mode="json")), encoding="utf-8")
    run_id = train(parent, run_id="derivation-parent")
    manager = CheckpointManager(parent.logging.root_dir / run_id)
    generation = manager._resolve(manager.root / "latest.json")
    checkpoint = manager.load(generation)

    template_document = yaml.safe_load(template.read_text(encoding="utf-8"))
    template_document["artifacts"] = {
        "parent": {
            "kind": "checkpoint",
            "version": 2,
            "producer": "sparselab",
            "identifier": generation.name,
            "sha256": checkpoint.checkpoint_sha256,
            "path": generation.relative_to(tmp_path).as_posix(),
            "state": "full",
        }
    }
    template_document["phases"] = [
        {
            "id": "continue",
            "transition": "extend_budget",
            "checkpoint": "parent",
            "set": {
                "training.max_steps": 8,
                "training.max_tokens": 256,
                "optimizer.decay_steps": 4,
            },
        }
    ]
    template.write_text(yaml.safe_dump(template_document), encoding="utf-8")
    source = tmp_path / "extension.yaml"
    bind_direct_inputs(run, template, prepared_root, source)

    valid = tmp_path / "extension-derived.yaml"
    derive_experiment(source, valid, {"retention.keep_previous": False})
    assert resolve_plan(load_plan(valid), valid).cells[0].config.training.max_steps == 8

    invalid = tmp_path / "extension-invalid.yaml"
    with pytest.raises(ValueError):
        derive_experiment(
            source,
            invalid,
            {
                "phases": [
                    {
                        "id": "continue",
                        "transition": "extend_budget",
                        "checkpoint": "parent",
                        "set": {
                            "training.max_steps": 8,
                            "training.max_tokens": 256,
                            "optimizer.decay_steps": 8,
                        },
                    }
                ]
            },
        )
    assert not invalid.exists()
    assert not _sidecar(invalid).exists()

    resume = tmp_path / "resume-invalid.yaml"
    with pytest.raises(ValueError):
        derive_experiment(
            source,
            resume,
            {
                "phases": [
                    {
                        "id": "continue",
                        "transition": "resume",
                        "checkpoint": "parent",
                        "set": {
                            "training.max_steps": 8,
                            "training.max_tokens": 256,
                            "optimizer.decay_steps": 4,
                        },
                    }
                ]
            },
        )
    assert not resume.exists()
    assert not _sidecar(resume).exists()

    partial_update = tmp_path / "partial-update.yaml"
    partial_phases = template_document["phases"]
    partial_phases[0]["set"]["training.max_tokens"] = 257
    with pytest.raises(ValueError, match="whole-update"):
        derive_experiment(source, partial_update, {"phases": partial_phases})
    assert not partial_update.exists()
    assert not _sidecar(partial_update).exists()
