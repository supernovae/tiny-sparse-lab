"""Recovery declarations cannot turn missing human decisions into executable work."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sparselab.recovery.engine import (
    inspect_manifest,
    plan_manifest,
    reconstruct_manifest,
)
from sparselab.recovery.evidence import export_evidence
from sparselab.recovery.manifest import RecoveryManifest


def _copy_implementation(repository: Path) -> None:
    """Fixture input commits pin producer bytes, not just scientific declarations."""
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "src",
        repository / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(Path(__file__).resolve().parents[1] / name, repository / name)


def _manifest(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "recovery_version": 1,
        "id": "tiny-recovery",
        "source_commit": "a" * 40,
        "steps": [
            {
                "id": "choice",
                "kind": "external_required",
                "role": "model_choice",
                "reason": "No model decision has been authored.",
            }
        ],
    }
    return {**value, **overrides}


@pytest.mark.parametrize(
    "steps",
    [
        [
            {
                "id": "x",
                "kind": "corpus_export",
                "corpus": "missing",
                "base_run": "run.yaml",
                "view": "lm",
                "vocab_size": 260,
            }
        ],
        [
            {
                "id": "x",
                "kind": "optional_cache",
                "role": "cache",
                "reason": "optional",
            },
            {
                "id": "x",
                "kind": "optional_cache",
                "role": "cache",
                "reason": "optional",
            },
        ],
        [{"id": "x", "kind": "corpus_release", "project": "../outside.yaml"}],
        [{"id": "x", "kind": "checkpoint", "run": "../../escape", "cell": "cell"}],
        [
            {
                "id": "x",
                "kind": "corpus_release",
                "project": "corpus.yaml",
                "expected_release_sha256": "short",
            }
        ],
    ],
)
def test_rejects_invalid_scientific_graph(steps: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError):
        RecoveryManifest.model_validate(_manifest(steps=steps))


def test_published_schema_matches_strict_manifest_model() -> None:
    schema = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "recovery-manifest-v1.schema.json"
    )
    assert (
        json.loads(schema.read_text(encoding="utf-8"))
        == RecoveryManifest.model_json_schema()
    )


def test_unavailable_decision_remains_read_only(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "inputs",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    declaration = repository / "recovery.json"
    declaration.write_text(
        json.dumps(_manifest(source_commit=commit)), encoding="utf-8"
    )
    work_root = tmp_path / "persistent-state"
    inspected = inspect_manifest(declaration, work_root)
    assert inspected["steps"][0]["classification"] == "MISSING_EXTERNAL"
    assert plan_manifest(declaration, work_root)["commands"] == []
    assert not work_root.exists()
    with pytest.raises(ValueError, match="UNTRACKED_DECLARATION"):
        reconstruct_manifest(declaration, work_root)
    assert not work_root.exists()
    subprocess.run(["git", "-C", str(repository), "add", "recovery.json"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-q",
            "-m",
            "recipe",
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="MISSING_EXTERNAL"):
        reconstruct_manifest(declaration, work_root)
    assert not work_root.exists()


def test_different_rebuilt_release_keeps_new_identity(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "examples" / "tiny-campaign",
        repository / "corpus",
    )
    _copy_implementation(repository)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    identity = ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.test"]
    subprocess.run(
        ["git", "-C", str(repository), *identity, "commit", "-qm", "inputs"], check=True
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    declaration = repository / "recovery.json"
    declaration.write_text(
        json.dumps(
            _manifest(
                source_commit=commit,
                steps=[
                    {
                        "id": "release",
                        "kind": "corpus_release",
                        "project": "corpus/corpus.yaml",
                        "expected_release_sha256": "0" * 64,
                    },
                    {
                        "id": "gate",
                        "kind": "external_required",
                        "role": "model",
                        "reason": "not authored",
                    },
                ],
            )
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repository), "add", "recovery.json"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), *identity, "commit", "-qm", "recipe"], check=True
    )
    root = tmp_path / "persistent-state"
    planned = plan_manifest(declaration, root)
    assert planned["commands"][0]["output"].endswith("/" + "0" * 64)
    assert "corpus build" in planned["commands"][0]["command"]
    result = reconstruct_manifest(declaration, root)
    outcome = result["outcomes"][0]
    assert outcome["status"] == "EXPECTED_DIGEST_MISMATCH"
    assert outcome["actual_sha256"] != "0" * 64
    assert Path(outcome["path"]).name == outcome["actual_sha256"]
    assert not Path(planned["commands"][0]["output"]).exists()
    (repository / "corpus" / "texts" / "tiny_docs-0.txt").unlink()
    assert (
        inspect_manifest(declaration, root)["steps"][0]["classification"]
        == "MISSING_EXTERNAL"
    )


def test_evidence_replay_preserves_timestamp_and_refuses_conflict(
    tmp_path: Path,
) -> None:
    observation = tmp_path / "probe.json"
    output = tmp_path / "reference.json"
    observation.write_text(
        json.dumps({"format_version": 1, "ok": True}), encoding="utf-8"
    )
    first = export_evidence(
        "runtime_probe",
        observation,
        output,
        source_commit="a" * 40,
        declaration_hashes=[],
    )
    original = output.read_bytes()
    assert (
        export_evidence(
            "runtime_probe",
            observation,
            output,
            source_commit="a" * 40,
            declaration_hashes=[],
        )
        == first
    )
    assert output.read_bytes() == original
    observation.write_text(
        json.dumps({"format_version": 1, "ok": False}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="conflicting evidence publication"):
        export_evidence(
            "runtime_probe",
            observation,
            output,
            source_commit="a" * 40,
            declaration_hashes=[],
        )
    assert output.read_bytes() == original
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        export_evidence(
            "tokenizer_selection",
            observation,
            tmp_path / "selection.json",
            source_commit="a" * 40,
            declaration_hashes=[],
        )
    assert not (tmp_path / "selection.json").exists()


@pytest.fixture
def tiny_inputs(tmp_path: Path) -> tuple[Path, str]:
    import yaml

    repository = tmp_path / "repo"
    repository.mkdir()
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "examples/tiny-campaign",
        repository / "corpus",
    )
    splits = yaml.safe_load((repository / "corpus/splits.yaml").read_text())
    splits["assignments"]["tiny_docs_family"] = "validation"
    (repository / "corpus/splits.yaml").write_text(yaml.safe_dump(splits))
    shutil.copy(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml",
        repository / "base.yaml",
    )
    _copy_implementation(repository)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "inputs",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    return repository, commit


def _publish_recipe(repository: Path, recipe: dict[str, object]) -> Path:
    declaration = repository / "recovery.json"
    declaration.write_text(json.dumps(recipe))
    subprocess.run(["git", "-C", str(repository), "add", "recovery.json"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "recipe",
        ],
        check=True,
    )
    return declaration


def test_unsealed_output_stops_before_dependent_commands(
    tiny_inputs, tmp_path: Path
) -> None:
    repository, commit = tiny_inputs
    declaration = _publish_recipe(
        repository,
        _manifest(
            source_commit=commit,
            steps=[
                {
                    "id": "release",
                    "kind": "corpus_release",
                    "project": "corpus/corpus.yaml",
                },
                {
                    "id": "export",
                    "kind": "corpus_export",
                    "corpus": "release",
                    "base_run": "base.yaml",
                    "view": "lm",
                    "vocab_size": 300,
                },
                {
                    "id": "model-choice",
                    "kind": "external_required",
                    "role": "family",
                    "reason": "Data-only fixture has no reviewed model-family declaration.",
                },
            ],
        ),
    )
    root = tmp_path / "state"
    planned = plan_manifest(declaration, root)
    assert [row["step"] for row in planned["commands"]] == ["release"]
    assert planned["commands"][0]["output"] is None
    assert planned["commands"][0]["requirements"]["storage_estimate_bytes"] > 0
    assert not root.exists()
    result = reconstruct_manifest(declaration, root)
    assert [(row["id"], row["status"]) for row in result["outcomes"]] == [
        ("release", "UNSEALED_RESULT")
    ]
    produced = Path(result["outcomes"][0]["path"])
    assert produced.name == result["outcomes"][0]["actual_sha256"]
    assert not (produced.parent.parent / "exports").exists()


def test_standalone_preparation_recovers_domain_identity_not_manifest_filename(
    tiny_inputs, tmp_path: Path
) -> None:
    from sparselab.config.loading import load_config, load_tokenizer_config
    from sparselab.corpus.acquisition import acquire
    from sparselab.corpus.export import export_release
    from sparselab.corpus.pipeline import build
    from sparselab.corpus.project import load_project
    from sparselab.corpus.release import freeze
    from sparselab.data.packing import prepare_data
    from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
    from sparselab.training.manifest import sha256_file

    repository, commit = tiny_inputs
    root = tmp_path / "state"
    project = load_project(repository / "corpus/corpus.yaml")
    acquire(project, root)
    release = freeze(build(project, root, offline=True), root)
    export = export_release(release, "lm", repository / "base.yaml", 300, root)
    tokenizer = train_tokenizer(load_tokenizer_config(export / "tokenizer.yaml"))
    tokenizer_sha = sha256_file(tokenizer)
    prepared = prepare_data(load_config(export / "run.yaml"), load_tokenizer(tokenizer))
    manifest_sha = json.loads((prepared.root / "manifest.json").read_text())[
        "manifest_sha256"
    ]
    assert prepared.root.name != manifest_sha
    declaration = _publish_recipe(
        repository,
        _manifest(
            source_commit=commit,
            steps=[
                {
                    "id": "release",
                    "kind": "corpus_release",
                    "project": "corpus/corpus.yaml",
                    "expected_release_sha256": release.name,
                },
                {
                    "id": "export",
                    "kind": "corpus_export",
                    "corpus": "release",
                    "base_run": "base.yaml",
                    "view": "lm",
                    "vocab_size": 300,
                    "expected_export_sha256": export.name,
                },
                {
                    "id": "tokenizer",
                    "kind": "tokenizer_train",
                    "export": "export",
                    "expected_tokenizer_sha256": tokenizer_sha,
                },
                {
                    "id": "prepared",
                    "kind": "prepared_data",
                    "export": "export",
                    "tokenizer": "tokenizer",
                    "expected_manifest_sha256": manifest_sha,
                },
                {
                    "id": "model-choice",
                    "kind": "external_required",
                    "role": "family",
                    "reason": "Data-only fixture has no reviewed model-family declaration.",
                },
            ],
        ),
    )
    shutil.rmtree(root)
    planned = plan_manifest(declaration, root)
    final_command = planned["commands"][-1]
    assert final_command["step"] == "prepared"
    assert final_command["output"] is None
    assert final_command["requires"] == ["export", "tokenizer"]
    assert not root.exists()
    result = reconstruct_manifest(declaration, root)
    recovered = next(row for row in result["steps"] if row["id"] == "prepared")
    assert recovered["classification"] == "PRESENT"
    assert recovered["actual_sha256"] == manifest_sha
    assert Path(recovered["path"]).name == prepared.root.name


def _corpus_recipe(commit: str, **expectations: str) -> dict[str, object]:
    return _manifest(
        source_commit=commit,
        steps=[
            {
                "id": "release",
                "kind": "corpus_release",
                "project": "corpus/corpus.yaml",
                **expectations,
            },
            {
                "id": "model-choice",
                "kind": "external_required",
                "role": "family",
                "reason": "Data-only fixture.",
            },
        ],
    )


def test_formatting_difference_requires_explicit_replay_before_acquisition(
    tiny_inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _ = tiny_inputs
    implementation = repository / "src/sparselab/corpus/acquisition.py"
    implementation.write_bytes(
        implementation.read_bytes() + b"\n# Historical formatting.\n"
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "Historical formatting",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    declaration = _publish_recipe(
        repository, _corpus_recipe(commit, expected_release_sha256="0" * 64)
    )
    root = tmp_path / "uncreated"

    def forbidden_acquisition(*args, **kwargs):
        pytest.fail("implementation preflight must run before any acquisition")

    monkeypatch.setattr("sparselab.corpus.acquisition.acquire", forbidden_acquisition)
    inspected = inspect_manifest(declaration, root)
    assert (
        inspected["steps"][0]["classification"]
        == "PINNED_IMPLEMENTATION_REPLAY_REQUIRED"
    )
    planned = plan_manifest(declaration, root)
    assert planned["commands"][0]["requires_explicit_review"] is True
    assert "--replay-pinned-implementation" in planned["commands"][0]["command"]
    with pytest.raises(ValueError, match="PINNED_IMPLEMENTATION_REPLAY_REQUIRED"):
        reconstruct_manifest(declaration, root, allow_network=True)
    assert not root.exists()


def test_missing_implementation_commit_is_reported_without_creating_state(
    tiny_inputs, tmp_path: Path
) -> None:
    repository, _ = tiny_inputs
    declaration = _publish_recipe(
        repository, _corpus_recipe("f" * 40, expected_release_sha256="0" * 64)
    )
    root = tmp_path / "uncreated"
    inspected = inspect_manifest(declaration, root)
    assert inspected["steps"][0]["classification"] == "MISSING_IMPLEMENTATION"
    assert plan_manifest(declaration, root)["commands"] == []
    with pytest.raises(ValueError, match="MISSING_IMPLEMENTATION"):
        reconstruct_manifest(declaration, root, replay_pinned_implementation=True)
    assert not root.exists()


def test_build_expectation_stops_before_release_publication(
    tiny_inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, commit = tiny_inputs
    declaration = _publish_recipe(
        repository, _corpus_recipe(commit, expected_build_sha256="0" * 64)
    )
    root = tmp_path / "state"

    def forbidden_freeze(*args, **kwargs):
        pytest.fail("divergent build must not be frozen")

    monkeypatch.setattr("sparselab.corpus.release.freeze", forbidden_freeze)
    with pytest.raises(ValueError, match="EXPECTED_DIGEST_MISMATCH: build"):
        reconstruct_manifest(declaration, root)
    assert not list(root.glob("corpora/*/releases"))
