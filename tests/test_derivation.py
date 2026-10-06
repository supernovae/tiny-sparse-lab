from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from sparselab.cli.main import main
from sparselab.config.loading import load_config
from sparselab.derivation import derive_config, parse_assignments
from sparselab.training.manifest import sha256_file


@pytest.fixture
def source_config(tmp_path: Path) -> Path:
    """A relocatable v2 config whose path values exercise source anchoring."""
    source = tmp_path / "source" / "run.yaml"
    source.parent.mkdir()
    raw = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1] / "configs" / "runtime_smoke_cpu.yaml"
        ).read_text(encoding="utf-8")
    )
    assert isinstance(raw, dict)
    raw["tokenizer"] = {"path": "assets/tokenizer.json"}
    raw["dataset"]["cache_dir"] = "cache"
    raw["logging"]["root_dir"] = "runs"
    raw["training"].update(
        trainable_parameters=["blocks.0.attention.q_proj.weight"],
        portability_manifest_path="manifests/portable.json",
    )
    source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return source


def _sidecar(output: Path) -> Path:
    return output.with_name(output.name + ".derivation.json")


def _run_config_derive(monkeypatch, capsys, argv: list[str]) -> dict[str, object]:
    monkeypatch.setattr(sys, "argv", ["sparselab", "config", "derive", *argv])
    main()
    return json.loads(capsys.readouterr().out)


def test_config_derivation_relocates_paths_and_records_exact_bytes(
    source_config: Path, tmp_path: Path
) -> None:
    output = tmp_path / "published" / "derived.yaml"
    output.parent.mkdir()

    receipt = derive_config(
        source_config,
        output,
        {
            "optimizer.peak": 0.001,
            "optimizer.decay_steps": 12,
            "tokenizer": {"path": "replacement/tokenizer.json"},
        },
    )

    loaded = load_config(output)
    assert loaded.optimizer.peak == 0.001
    assert loaded.optimizer.decay_steps == 12
    assert (
        loaded.tokenizer.path
        == (source_config.parent / "replacement/tokenizer.json").resolve()
    )
    assert loaded.dataset.cache_dir == (source_config.parent / "cache").resolve()
    assert loaded.logging.root_dir == (source_config.parent / "runs").resolve()
    assert (
        loaded.training.portability_manifest_path
        == (source_config.parent / "manifests/portable.json").resolve()
    )
    assert receipt["format"] == "sparselab-derivation-v1"
    assert receipt["kind"] == "run_config"
    assert receipt["origin"]["path"] == str(source_config.resolve())
    assert receipt["origin"]["file_sha256"] == sha256_file(source_config)
    assert (
        receipt["output"]["file_sha256"]
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert receipt["declared_delta"]["optimizer.decay_steps"] == 12
    assert receipt["validation"] == {
        "scope": "typed_config",
        "executable": False,
        "runtime_verified": False,
    }
    assert _sidecar(output).exists()
    assert json.loads(_sidecar(output).read_text(encoding="utf-8")) == receipt


def test_config_derivation_noop_is_not_an_observed_scientific_delta(
    source_config: Path, tmp_path: Path
) -> None:
    output = tmp_path / "derived.yaml"
    receipt = derive_config(source_config, output, {"optimizer.peak": 0.003})

    assert "optimizer.peak" not in receipt["delta"]
    assert receipt["declared_delta"] == {"optimizer.peak": 0.003}


@pytest.mark.parametrize(
    "values",
    [
        [],
        ["optimizer.peak"],
        ["=1"],
        ["optimizer.peak=not-json"],
        ["optimizer.peak=NaN"],
        ["optimizer.peak=1e999"],
        ["optimizer.peak=1", "optimizer.peak=2"],
        ["optimizer.peak=Infinity"],
        ['tokenizer={"path":"a","path":"b"}'],
    ],
)
def test_assignment_parser_rejects_ambiguous_or_nonfinite_values(
    values: list[str],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_assignments(values)


def test_config_derivation_rejects_invalid_schema_without_publication(
    source_config: Path, tmp_path: Path
) -> None:
    output = tmp_path / "invalid.yaml"
    with pytest.raises(ValueError, match="seq_len"):
        derive_config(source_config, output, {"training.seq_len": 999999})
    assert not output.exists()
    assert not _sidecar(output).exists()


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("- not-a-mapping\n", "mapping"),
        ("schema_version: 1\n", "migrated"),
        (
            "schema_version: 2\nname: first\nname: second\n",
            "duplicate",
        ),
    ],
)
def test_config_derivation_rejects_invalid_source_documents(
    tmp_path: Path, content: str, match: str
) -> None:
    source = tmp_path / "invalid.yaml"
    output = tmp_path / "derived.yaml"
    source.write_text(content, encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=match):
        derive_config(source, output, {"optimizer.peak": 0.001})
    assert not output.exists()
    assert not _sidecar(output).exists()


def test_config_derivation_rejects_linked_destination_components(
    source_config: Path, tmp_path: Path
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(target, target_is_directory=True)
    output = linked_parent / "derived.yaml"

    with pytest.raises((OSError, ValueError), match="symlink"):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert not (target / "derived.yaml").exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"unknown": 1},
        {"optimizer.unknown": 1},
        {"optimizer": {}, "optimizer.peak": 0.001},
    ],
)
def test_config_derivation_rejects_unknown_and_overlapping_paths(
    source_config: Path, tmp_path: Path, settings: dict[str, object]
) -> None:
    output = tmp_path / "invalid.yaml"
    with pytest.raises(ValueError):
        derive_config(source_config, output, settings)
    assert not output.exists()
    assert not _sidecar(output).exists()


def test_config_derivation_publication_is_exclusive(
    source_config: Path, tmp_path: Path
) -> None:
    output = tmp_path / "derived.yaml"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises((FileExistsError, ValueError)):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert output.read_text(encoding="utf-8") == "existing"

    output.unlink()
    _sidecar(output).write_text("existing receipt", encoding="utf-8")
    with pytest.raises((FileExistsError, ValueError)):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert not output.exists()
    assert _sidecar(output).read_text(encoding="utf-8") == "existing receipt"

    _sidecar(output).unlink()
    output.symlink_to(tmp_path / "missing-target.yaml")
    with pytest.raises((FileExistsError, OSError, ValueError)):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert output.is_symlink()


def test_config_derivation_removes_only_its_receipt_after_payload_failure(
    source_config: Path, tmp_path: Path, monkeypatch
) -> None:
    from sparselab import derivation

    output = tmp_path / "derived.yaml"
    original = derivation._exclusive_bytes
    calls = 0

    def fail_payload(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected payload publication failure")
        original(path, content)

    monkeypatch.setattr(derivation, "_exclusive_bytes", fail_payload)
    with pytest.raises(OSError, match="injected"):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert not output.exists()
    assert not _sidecar(output).exists()


def test_config_derivation_rehashes_source_before_publication(
    source_config: Path, tmp_path: Path, monkeypatch
) -> None:
    from sparselab import derivation

    output = tmp_path / "derived.yaml"
    original = derivation.sha256_file
    source_reads = 0

    def mutate_between_hashes(path: Path) -> str:
        nonlocal source_reads
        digest = original(path)
        if path.resolve() == source_config.resolve():
            source_reads += 1
            if source_reads == 1:
                source_config.write_text(
                    source_config.read_text(encoding="utf-8").replace(
                        "seed: 7", "seed: 8"
                    ),
                    encoding="utf-8",
                )
        return digest

    monkeypatch.setattr(derivation, "sha256_file", mutate_between_hashes)
    with pytest.raises(ValueError, match="changed"):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert not output.exists()
    assert not _sidecar(output).exists()


def test_config_derive_cli_emits_receipt_and_expected_errors(
    source_config: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "cli.yaml"
    response = _run_config_derive(
        monkeypatch,
        capsys,
        [
            str(source_config),
            "--set",
            "optimizer.peak=0.001",
            "--output",
            str(output),
            "--json",
        ],
    )
    assert response["status"] == "ok"
    assert response["receipt"] == str(_sidecar(output))
    assert response["validation"]["scope"] == "typed_config"

    failed = tmp_path / "failed.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "config",
            "derive",
            str(source_config),
            "--set",
            "training.seq_len=999999",
            "--output",
            str(failed),
            "--json",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert isinstance(payload["error"], str)


@pytest.mark.parametrize(
    "dataset",
    [
        {
            "source": "snapshot",
            "revision": "a" * 64,
            "license": "test",
            "train_path": "data/train.jsonl",
            "validation_path": "data/validation.jsonl",
            "source_manifest_path": "data/source.json",
        },
        {
            "source": "local_chat",
            "revision": "a" * 64,
            "license": "test",
            "train_path": "data/train.jsonl",
            "validation_path": "data/validation.jsonl",
            "allocation_manifest_path": "data/allocation.json",
            "corpus_release_path": "forge/release",
            "corpus_export_path": "forge/export",
        },
    ],
)
def test_config_relocation_preserves_all_typed_input_targets(
    source_config, tmp_path, dataset
):
    raw = yaml.safe_load(source_config.read_text())
    raw["dataset"].update(dataset)
    raw["model"].update(
        memory="portable",
        memory_table_size=64,
        memory_ngram_size=2,
        memory_dim=8,
        memory_package_path="memory/package",
    )
    source_config.write_text(yaml.safe_dump(raw))
    output = tmp_path / "relocated.yaml"
    receipt = derive_config(source_config, output, {"optimizer.peak": 0.001})
    loaded = load_config(output)
    for field, authored in dataset.items():
        if field.endswith("_path"):
            assert (
                getattr(loaded.dataset, field)
                == (source_config.parent / authored).resolve()
            )
    assert (
        loaded.model.memory_package_path
        == (source_config.parent / "memory/package").resolve()
    )
    assert receipt["delta"] == {"optimizer.peak": {"base": 0.003, "variant": 0.001}}
    rebound = {row["path"]: row for row in receipt["path_rebindings"]}
    assert rebound["model.memory_package_path"]["source_value"] == "memory/package"
    assert rebound["model.memory_package_path"]["resolved_target"] == str(
        loaded.model.memory_package_path
    )


@pytest.mark.parametrize(
    "settings", [{}, {"optimizer.peak": float("inf")}, {"optimizer.peak": float("nan")}]
)
def test_api_rejects_empty_and_nonfinite_settings(source_config, tmp_path, settings):
    output = tmp_path / "invalid.yaml"
    with pytest.raises(ValueError):
        derive_config(source_config, output, settings)
    assert not output.exists()
    assert not _sidecar(output).exists()


@pytest.mark.parametrize("destination", ["invalid.json", "missing/derived.yaml"])
def test_config_requires_yaml_and_existing_parent(source_config, tmp_path, destination):
    output = tmp_path / destination
    with pytest.raises(ValueError):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert not output.exists()
    assert not output.parent.joinpath(output.name + ".derivation.json").exists()


def test_competing_publication_never_overwrites_competitor(
    source_config, tmp_path, monkeypatch
):
    from sparselab import derivation

    output = tmp_path / "derived.yaml"
    original = derivation._exclusive_bytes

    def race(path, content):
        if path == output:
            output.write_bytes(b"competing writer")
        original(path, content)

    monkeypatch.setattr(derivation, "_exclusive_bytes", race)
    with pytest.raises(FileExistsError):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert output.read_bytes() == b"competing writer"
    assert not _sidecar(output).exists()


def test_dangling_receipt_is_not_replaced(source_config, tmp_path):
    output = tmp_path / "derived.yaml"
    _sidecar(output).symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        derive_config(source_config, output, {"optimizer.peak": 0.001})
    assert _sidecar(output).is_symlink()
    assert not output.exists()


def test_config_cli_text_reports_published_handoff(
    source_config, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "text.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "config",
            "derive",
            str(source_config),
            "--set",
            "optimizer.peak=0.001",
            "--output",
            str(output),
        ],
    )
    main()
    captured = capsys.readouterr()
    assert str(output) in captured.out
    assert str(_sidecar(output)) in captured.out
    assert "typed_config" in captured.out
    assert load_config(output).optimizer.peak == 0.001
    assert json.loads(_sidecar(output).read_text())["delta"]["optimizer.peak"] == {
        "base": 0.003,
        "variant": 0.001,
    }


def _real_bound_plan(tmp_path, monkeypatch):
    from test_experiment_direct_inputs import inputs

    from sparselab.experiments.direct_inputs import bind_direct_inputs

    config, prepared, run, template = inputs(tmp_path, monkeypatch)
    source = tmp_path / "bound.yaml"
    bind_direct_inputs(run, template, prepared, source)
    return config, source, run


def test_inline_base_relative_paths_relocate_without_intervention(
    tmp_path, monkeypatch
):
    from sparselab.derivation import derive_experiment
    from sparselab.experiments.plan import load_plan

    config, source, _ = _real_bound_plan(tmp_path, monkeypatch)
    raw = yaml.safe_load(source.read_text())
    raw["base_run"]["tokenizer"]["path"] = config.tokenizer.path.relative_to(
        tmp_path
    ).as_posix()
    raw["base_run"]["dataset"]["cache_dir"] = config.dataset.cache_dir.relative_to(
        tmp_path
    ).as_posix()
    source.write_text(yaml.safe_dump(raw))
    output = tmp_path / "moved" / "plan.yaml"
    output.parent.mkdir()
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    receipt = derive_experiment(source, output, {"retention.keep_periodic": False})
    plan = load_plan(output)
    assert plan.base_run.tokenizer.path == config.tokenizer.path
    assert plan.base_run.dataset.cache_dir == config.dataset.cache_dir
    assert receipt["delta"] == {
        "retention.keep_periodic": {"base": True, "variant": False}
    }
    assert any(
        row["path"] == "base_run.tokenizer.path" for row in receipt["path_rebindings"]
    )
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after.keys() - before.keys() == {output, _sidecar(output)}
    assert all(after[path] == content for path, content in before.items())


def test_experiment_rehashes_referenced_config_closure_before_publication(
    tmp_path, monkeypatch
):
    from sparselab import derivation

    _, source, run = _real_bound_plan(tmp_path, monkeypatch)
    raw = yaml.safe_load(source.read_text())
    raw["base_run"] = run.name
    source.write_text(yaml.safe_dump(raw))
    original = derivation.sha256_file
    hashes = 0

    def mutate_before_final_hash(path):
        nonlocal hashes
        if path == run:
            hashes += 1
            if hashes == 2:
                run.write_text(run.read_text() + "\n# changed during derivation\n")
        return original(path)

    monkeypatch.setattr(derivation, "sha256_file", mutate_before_final_hash)
    output = tmp_path / "invalid.yaml"
    with pytest.raises(ValueError, match="source changed"):
        derivation.derive_experiment(source, output, {"retention.keep_periodic": False})
    assert not output.exists()
    assert not _sidecar(output).exists()
    assert not list(tmp_path.glob(".derivation-validation-*"))


def test_experiment_rejects_incompatible_weight_promotion(tmp_path, monkeypatch):
    from sparselab.derivation import derive_experiment

    config, source, _ = _real_bound_plan(tmp_path, monkeypatch)
    phases = [
        {"id": "pretrain", "transition": "fresh"},
        {
            "id": "promoted",
            "transition": "promote",
            "parent": "pretrain",
            "selector": "terminal",
            "at_step": config.training.max_steps,
            "set": {"model.hidden_dim": config.model.hidden_dim * 2},
        },
    ]
    output = tmp_path / "invalid.yaml"
    with pytest.raises(ValueError, match="promotion changes architecture"):
        derive_experiment(source, output, {"phases": phases})
    assert not output.exists()
    assert not _sidecar(output).exists()


def test_experiment_reauthenticates_tokenizer_bytes(tmp_path, monkeypatch):
    from sparselab.derivation import derive_experiment

    config, source, _ = _real_bound_plan(tmp_path, monkeypatch)
    config.tokenizer.path.write_bytes(config.tokenizer.path.read_bytes() + b"tampered")
    output = tmp_path / "invalid.yaml"
    with pytest.raises(ValueError):
        derive_experiment(source, output, {"retention.keep_periodic": False})
    assert not output.exists()
    assert not _sidecar(output).exists()


@pytest.mark.parametrize("kind", ["capability_card", "prompt_set"])
def test_case_artifacts_preserve_both_consumers_or_reject_relocation(
    tmp_path, monkeypatch, kind
):
    import subprocess

    from sparselab.derivation import derive_experiment
    from sparselab.evaluation.capabilities import (
        capability_card,
        capability_card_payload,
    )
    from sparselab.experiments.lock import resolve_plan
    from sparselab.experiments.plan import load_plan
    from sparselab.recovery.provenance import declaration_reference

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    config, source, _ = _real_bound_plan(tmp_path, monkeypatch)
    evidence = tmp_path / "cases.json"
    if kind == "capability_card":
        card = capability_card("chat-alias-retention-v1")
        payload = capability_card_payload(card)
        identifier, version = card.name, card.version
    else:
        payload = {
            "format": "dense_lm_decoding_prompts_v1",
            "split": "test",
            "prompts": [{"id": "one", "text": "Once", "category": "story"}],
        }
        identifier, version = evidence.stem, 1
    evidence.write_text(json.dumps(payload))
    raw = yaml.safe_load(source.read_text())
    raw["artifacts"]["cases.json"] = {
        "kind": kind,
        "version": version,
        "producer": "fixture",
        "identifier": identifier,
        "path": "cases.json",
        "sha256": sha256_file(evidence),
    }
    raw["artifacts"]["terminal"] = {
        "kind": "checkpoint",
        "version": 1,
        "producer": "pretrain",
        "identifier": "terminal",
        "from_phase": "pretrain",
        "selector": "terminal",
        "state": "full",
    }
    raw["evaluations"] = [
        {
            "id": "cases",
            "checkpoint": "terminal",
            "data": "packed",
            "cases": "cases.json",
            "metric": "exact",
            "denominator": "cases",
            "endpoint": config.training.max_steps,
            "role": "gate",
        }
    ]
    source.write_text(yaml.safe_dump(raw))
    output = tmp_path / "same-root.yaml"
    receipt = derive_experiment(source, output, {"retention.keep_periodic": False})
    plan = load_plan(output)
    locked = resolve_plan(plan, output)
    assert locked.artifacts["cases.json"]["sha256"] == sha256_file(evidence)
    assert declaration_reference(output, plan.evaluations[0].cases) == evidence
    assert receipt["delta"] == {
        "retention.keep_periodic": {"base": True, "variant": False}
    }
    moved = tmp_path / "moved" / "plan.yaml"
    moved.parent.mkdir()
    # Repository fallback would satisfy the declaration walker only. Artifact
    # authentication is local-relative, so this move must not silently pass.
    with pytest.raises(ValueError, match="artifacts.cases.json.path.*cannot relocate"):
        derive_experiment(source, moved, {"retention.keep_periodic": False})
    assert not moved.exists()
    assert not _sidecar(moved).exists()


def test_unbound_experiment_cannot_publish_a_verified_handoff(tmp_path, monkeypatch):
    from test_experiment_direct_inputs import inputs

    from sparselab.derivation import derive_experiment

    _, _, _, template = inputs(tmp_path, monkeypatch)
    output = tmp_path / "unbound.yaml"
    with pytest.raises(ValueError):
        derive_experiment(template, output, {"retention.keep_periodic": False})
    assert not output.exists()
    assert not _sidecar(output).exists()
