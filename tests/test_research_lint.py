"""Tracked tiny fixtures exercise offline lint without external scientific bytes."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sparselab.research.lint import lint_research

_LEGACY_NAMES = (
    "experiments/research/kernel-memory-lab/results/2026-10-07-card03-review-index.jsonl",
    "experiments/research/kernel-memory-lab/results/evidence/card05-v2-language-final-scores.jsonl",
)


def _write(root: Path, name: str, content: str | dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content) if isinstance(content, dict) else content)
    return path


def _track(root: Path) -> None:
    subprocess.run(
        ["git", "-C", str(root), "add", "-A"], check=True, capture_output=True
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    campaign = "experiments/research/demo"
    protocol = _write(
        tmp_path, f"{campaign}/protocol.md", "Frozen question and acceptance gates.\n"
    )
    config = _write(tmp_path, "configs/demo.yaml", "seed: 42\n")
    _write(
        tmp_path,
        f"{campaign}/preregistration.json",
        {
            "frozen_sha256": {
                f"{campaign}/protocol.md": _digest(protocol),
                "configs/demo.yaml": _digest(config),
            },
        },
    )
    _write(
        tmp_path,
        f"{campaign}/checkpoint-reference.json",
        {
            "format": "scientific-evidence-reference-v1",
            "kind": "checkpoint",
            "sha256": "a" * 64,
            "identifier": "step_1",
            "source_commit": "b" * 40,
            "declaration_hashes": [
                {"path": f"{campaign}/protocol.md", "sha256": _digest(protocol)}
            ],
            "external_location": "/unavailable/checkpoint",
            "verification_scope": "verified_artifact",
            "verified_at": "2026-10-05T00:00:00Z",
        },
    )
    _track(tmp_path)
    return tmp_path


def _errors(root: Path) -> str:
    report = lint_research(root)
    assert report["valid"] is False
    return json.dumps(report["errors"])


@pytest.fixture
def legacy_repository(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    checkout = Path(__file__).resolve().parents[1]
    for name in _LEGACY_NAMES:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkout / name, destination)
    _track(tmp_path)
    return tmp_path


def test_exact_pinned_legacy_records_are_retained(legacy_repository: Path) -> None:
    report = lint_research(legacy_repository)
    assert report["valid"], report["errors"]
    assert report["checked_files"] == len(_LEGACY_NAMES)


@pytest.mark.parametrize("name", _LEGACY_NAMES)
def test_changed_legacy_bytes_are_rejected(legacy_repository: Path, name: str) -> None:
    path = legacy_repository / name
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    assert "pinned legacy research record digest mismatch" in _errors(legacy_repository)


@pytest.mark.parametrize("name", _LEGACY_NAMES)
def test_same_legacy_bytes_at_different_path_are_rejected(
    legacy_repository: Path, name: str
) -> None:
    original = legacy_repository / name
    alternate = original.with_name("unapproved-" + original.name)
    original.rename(alternate)
    _track(legacy_repository)
    errors = _errors(legacy_repository)
    assert alternate.relative_to(legacy_repository).as_posix() in errors


@pytest.mark.parametrize("name", _LEGACY_NAMES)
def test_additional_unapproved_legacy_artifact_is_rejected(
    legacy_repository: Path, name: str
) -> None:
    original = legacy_repository / name
    extra = original.with_name("additional-" + original.name)
    shutil.copyfile(original, extra)
    _track(legacy_repository)
    errors = _errors(legacy_repository)
    assert extra.relative_to(legacy_repository).as_posix() in errors


def test_valid_offline_research_records_and_untracked_outputs(repository: Path) -> None:
    _write(repository, "experiments/research/demo/runs/untracked.log", "not checked in")
    report = lint_research(repository)
    assert report["valid"] is True
    assert report["checked_files"] == 3
    assert "external payloads not verified" in report["scope"]


@pytest.mark.parametrize("target", ["protocol.md", "../../../configs/demo.yaml"])
def test_missing_or_changed_frozen_references(repository: Path, target: str) -> None:
    path = repository / "experiments/research/demo" / target
    path.write_text("Changed after freeze")
    assert "digest mismatch" in _errors(repository)
    path.unlink()
    assert "missing tracked reference" in _errors(repository)


def test_untracked_reference_is_missing(repository: Path) -> None:
    path = _write(repository, "experiments/research/demo/untracked.md", "review")
    _write(
        repository,
        "experiments/research/demo/preregistration.json",
        {
            "frozen_sha256": {"untracked.md": _digest(path)},
        },
    )
    assert "missing tracked reference" in _errors(repository)


@pytest.mark.parametrize(
    "name",
    [
        "checkpoints/state.json",
        "cache/key.json",
        "datasets/train.txt",
        "logs/train.txt",
        "runs/result.json",
        "outputs/stats.json",
        "state.pt",
        "state.safetensors",
        "tokens.npy",
        "train.parquet",
        "run.sqlite",
        "train.log",
        "metrics.jsonl",
        "events.out.tfevents.123",
        "nested/__pycache__/code.pyc",
    ],
)
def test_forbidden_tracked_output(repository: Path, name: str) -> None:
    _write(repository, f"experiments/research/demo/{name}", "{}")
    _track(repository)
    assert "forbidden mutable output" in _errors(repository)


@pytest.mark.parametrize(
    "name",
    [
        "../outside.md",
        "/unavailable/protocol.md",
        "a/../../secret",
        "a\\b",
        "./protocol.md",
    ],
)
def test_unsafe_binding(repository: Path, name: str) -> None:
    _write(
        repository,
        "experiments/research/demo/preregistration.json",
        {"frozen_sha256": {name: "a" * 64}},
    )
    assert "unsafe repository reference" in _errors(repository)


def test_symlink_reference_never_reads_external_target(
    repository: Path, tmp_path: Path
) -> None:
    target = repository / "configs/demo.yaml"
    target.unlink()
    target.symlink_to("/unavailable/private-config.yaml")
    assert "symlink" in _errors(repository)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "bad"),
        ("declaration_hashes", []),
        ("source_commit", "main"),
        ("kind", "unknown"),
        ("verification_scope", "observation"),
        ("external_location", ""),
        ("format", "scientific-evidence-reference-v2"),
        ("format", None),
    ],
)
def test_malformed_evidence_reference(
    repository: Path, field: str, value: object
) -> None:
    path = repository / "experiments/research/demo/checkpoint-reference.json"
    raw = json.loads(path.read_text())
    raw[field] = value
    path.write_text(json.dumps(raw))
    assert _errors(repository)


def test_evidence_declaration_mismatch(repository: Path) -> None:
    path = repository / "experiments/research/demo/checkpoint-reference.json"
    raw = json.loads(path.read_text())
    raw["declaration_hashes"][0]["sha256"] = "c" * 64
    path.write_text(json.dumps(raw))
    assert "reference digest mismatch" in _errors(repository)


def test_explicit_scientific_bindings(repository: Path) -> None:
    path = repository / "experiments/research/demo/preregistration.json"
    path.write_text(
        json.dumps(
            {"bindings": {"protocol": {"path": "protocol.md", "sha256": "a" * 64}}}
        )
    )
    assert "reference digest mismatch" in _errors(repository)


def test_plan_schema_and_config_closure(repository: Path) -> None:
    path = _write(
        repository,
        "experiments/research/demo/plan.yaml",
        "plan_version: 1\nid: demo\nbase_run: missing.yaml\n",
    )
    _track(repository)
    assert "missing tracked reference" in _errors(repository)
    path.write_text("plan_version: 999\nid: demo\nbase_run: missing.yaml\n")
    assert "invalid experiment plan" in _errors(repository)


def test_historical_implementation_inventory_is_not_a_current_binding(
    repository: Path,
) -> None:
    _write(
        repository,
        "experiments/research/demo/historical.json",
        {
            "implementation": [{"path": "src/old.py", "sha256": "c" * 64}],
            "archive_inventory": [
                {"path": "declarations/old.json", "sha256": "d" * 64}
            ],
        },
    )
    _track(repository)
    assert lint_research(repository)["valid"]


@pytest.mark.parametrize("name", ["train.jsonl", "evidence/train.jsonl"])
def test_arbitrary_dataset_in_evidence_directory_rejected(
    repository: Path, name: str
) -> None:
    _write(
        repository,
        f"experiments/research/demo/{name}",
        '{"text":"training document"}\n',
    )
    _track(repository)
    assert "JSONL payloads belong outside active research records" in _errors(
        repository
    )


def test_large_record_rejected(repository: Path) -> None:
    _write(repository, "experiments/research/demo/huge.md", "x" * (1024 * 1024 + 1))
    _track(repository)
    assert "1 MiB" in _errors(repository)


def test_duplicate_json_keys_rejected(repository: Path) -> None:
    _write(
        repository,
        "experiments/research/demo/preregistration.json",
        '{"frozen_sha256":{},"frozen_sha256":{}}',
    )
    assert "duplicate" in _errors(repository)


def test_actual_tracked_research_records() -> None:
    root = Path(__file__).resolve().parents[1]
    report = lint_research(root)
    assert report["valid"], report["errors"]
    assert report["checked_files"] > 100


def test_cli_failure_exit_code(repository: Path) -> None:
    _write(repository, "experiments/research/demo/protocol.md", "changed")
    import argparse

    from sparselab.research.lint import _handle

    with pytest.raises(SystemExit, match="1"):
        _handle(argparse.Namespace(repository=str(repository), json=True))


@pytest.mark.parametrize("fractional", [False, True])
def test_local_corpus_payloads_are_not_required(
    repository: Path, fractional: bool
) -> None:
    import shutil

    sample = Path(__file__).resolve().parents[1] / "corpora/devmind-sample-v0"
    target = repository / "corpora/demo"
    shutil.copytree(sample, target)
    shutil.rmtree(target / "sources/fixtures")
    if fractional:
        import yaml

        release = target / "release.yaml"
        raw = yaml.safe_load(release.read_text())
        raw["fraction"] = {
            "generated_share": 0.1,
            "train_tokens": 10,
            "tokenizer_path": "unavailable-tokenizer.json",
            "tokenizer_sha256": "a" * 64,
        }
        release.write_text(yaml.safe_dump(raw))
        from sparselab.corpus.project import load_project

        with pytest.raises(ValueError, match="fraction tokenizer"):
            load_project(target / "corpus.yaml")
    _write(
        repository,
        "experiments/research/demo/recovery.yaml",
        """recovery_version: 1
id: demo
source_commit: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
steps:
  - id: release
    kind: corpus_release
    project: corpora/demo/corpus.yaml
  - id: checkpoint
    kind: external_required
    role: checkpoint
    reason: unavailable outside fixture
""",
    )
    _track(repository)
    report = lint_research(repository)
    assert report["valid"], report["errors"]
    (target / "sources/docs.yaml").unlink()
    assert "missing tracked reference" in _errors(repository)


@pytest.mark.parametrize(
    "name,content",
    [
        ("train.json", '[{"text":"training document","label":1}]'),
        ("train.csv", "text,label\ntraining document,1"),
        ("train.tsv", "text\tlabel\ntraining document\t1"),
        ("evidence/review-blind.json", '[{"text":"training document","label":1}]'),
    ],
)
def test_dataset_disguised_as_record_rejected(
    repository: Path, name: str, content: str
) -> None:
    _write(repository, f"experiments/research/demo/{name}", content)
    _track(repository)
    assert _errors(repository)


@pytest.mark.parametrize("step", ["tokenizer_train", "corpus_export"])
def test_recovery_typed_config_outside_research_is_validated(
    repository: Path, step: str
) -> None:
    _write(repository, "configs/malformed.yaml", "not_a_config: true\n")
    steps = (
        (
            {
                "id": "tokenizer",
                "kind": "tokenizer_train",
                "config": "configs/malformed.yaml",
            },
        )
        if step == "tokenizer_train"
        else (
            {
                "id": "release",
                "kind": "external_required",
                "role": "corpus_release",
                "reason": "fixture",
            },
        )
    )
    if step == "corpus_export":
        # An experiment's base_run has the same RunConfig validation contract.
        _write(
            repository,
            "experiments/research/demo/plan.yaml",
            "plan_version: 1\nid: invalid\nbase_run: configs/malformed.yaml\n",
        )
    else:
        _write(
            repository,
            "experiments/research/demo/recovery.yaml",
            {
                "recovery_version": 1,
                "id": "demo",
                "source_commit": "b" * 40,
                "steps": [
                    *steps,
                    {
                        "id": "checkpoint",
                        "kind": "external_required",
                        "role": "checkpoint",
                        "reason": "fixture",
                    },
                ],
            },
        )
    _track(repository)
    assert "extra" in _errors(repository)


def test_referenced_suite_outside_research_is_validated(repository: Path) -> None:
    _write(
        repository,
        "configs/bad-suite.yaml",
        "evaluation_suite_version: 99\nid: broken\nevaluations: []\n",
    )
    _write(
        repository,
        "experiments/research/demo/plan.yaml",
        "plan_version: 1\nid: demo\nbase_run: configs/demo.yaml\nevaluation_suite: configs/bad-suite.yaml\n",
    )
    _track(repository)
    assert "validation error" in _errors(repository)


def test_recovery_run_config_outside_research_is_validated(repository: Path) -> None:
    tokenizer = Path(__file__).resolve().parents[1] / "configs/tokenizer_smoke.yaml"
    _write(repository, "configs/tokenizer.yaml", tokenizer.read_text())
    _write(repository, "configs/malformed.yaml", "not_a_config: true\n")
    _write(
        repository,
        "experiments/research/demo/recovery.yaml",
        {
            "recovery_version": 1,
            "id": "demo",
            "source_commit": "b" * 40,
            "steps": [
                {
                    "id": "tokenizer",
                    "kind": "tokenizer_train",
                    "config": "configs/tokenizer.yaml",
                },
                {
                    "id": "prepared",
                    "kind": "prepared_data",
                    "tokenizer": "tokenizer",
                    "config": "configs/malformed.yaml",
                },
                {
                    "id": "checkpoint",
                    "kind": "external_required",
                    "role": "checkpoint",
                    "reason": "fixture",
                },
            ],
        },
    )
    _track(repository)
    assert "extra" in _errors(repository)


@pytest.mark.parametrize(
    "defect",
    [None, "protocol", "corpus_project", "corpus_schema", "version", "traversal"],
)
def test_producer_declaration_references(repository: Path, defect: str | None) -> None:
    import shutil

    sample = Path(__file__).resolve().parents[1] / "corpora/devmind-sample-v0"
    corpus = repository / "corpora/demo"
    shutil.copytree(sample, corpus)
    shutil.rmtree(corpus / "sources/fixtures")
    raw = {
        "producer_record_version": 1,
        "protocol": "experiments/research/demo/protocol.md",
        "corpus_project": "corpora/demo/corpus.yaml",
        # Historical source/runtime evidence is neither today's source nor live input.
        "producer_commit": "a" * 40,
        "local_dense_readiness": {
            "path": "/unavailable/readiness.json",
            "sha256": "b" * 64,
        },
    }
    if defect in {"protocol", "corpus_project"}:
        raw[defect] = "missing.yaml"
    elif defect == "corpus_schema":
        (corpus / "release.yaml").write_text("not_a_release: true\n")
    elif defect == "version":
        raw["producer_record_version"] = 2
    elif defect == "traversal":
        raw["protocol"] = "../outside.md"
    _write(repository, "experiments/research/demo/producer.json", raw)
    _track(repository)
    report = lint_research(repository)
    assert report["valid"] == (defect is None), report


@pytest.fixture
def archived_repository(repository: Path) -> Path:
    archive = "experiments/research/history/demo/preregistration.json"
    record = _write(
        repository,
        archive,
        {"frozen_sha256": {"old/missing/config.yaml": "a" * 64}},
    )
    _write(
        repository,
        "experiments/research/history/path-map.json",
        {
            "format": "research-history-path-map-v1",
            "source_commit": "b" * 40,
            "entries": [
                {
                    "original_path": "experiments/research/demo/old.json",
                    "archive_path": archive,
                    "action": "relocated_unchanged",
                    "sha256": _digest(record),
                    "bytes": record.stat().st_size,
                },
                {
                    "original_path": "experiments/research/demo/old-runner.py",
                    "archive_path": None,
                    "action": "deleted_obsolete_executable",
                    "sha256": "c" * 64,
                    "bytes": 123,
                },
            ],
        },
    )
    _write(repository, "experiments/research/history/README.md", "Historical evidence")
    _track(repository)
    return repository


def test_archived_records_preserve_bytes_without_replaying_old_references(
    archived_repository: Path,
) -> None:
    report = lint_research(archived_repository)
    assert report["valid"], report["errors"]


@pytest.mark.parametrize("change", ["changed", "missing", "symlink", "unindexed"])
def test_archived_records_fail_closed(archived_repository: Path, change: str) -> None:
    record = (
        archived_repository / "experiments/research/history/demo/preregistration.json"
    )
    if change == "changed":
        record.write_text(record.read_text().replace("config", "CONFIG"))
    elif change == "missing":
        record.unlink()
    elif change == "symlink":
        record.unlink()
        record.symlink_to("/unavailable/private.json")
    else:
        record.with_name("unindexed.json").write_text("{}")
        _track(archived_repository)
    assert _errors(archived_repository)


@pytest.mark.parametrize(
    "field,value",
    [
        ("original_path", "../outside.py"),
        ("original_path", None),
        ("archive_path", "/unavailable/record.json"),
        ("archive_path", "experiments/research/history/../outside.json"),
        ("archive_path", "experiments/research/history\\record.json"),
        ("archive_path", "experiments/research/active.json"),
        ("archive_path", "experiments/research/history/README.md"),
        ("sha256", "bad"),
        ("bytes", -1),
        ("bytes", True),
        ("bytes", 0),
        ("action", "unrecognized"),
    ],
)
def test_history_map_rejects_unsafe_or_changed_bindings(
    archived_repository: Path, field: str, value: object
) -> None:
    path = archived_repository / "experiments/research/history/path-map.json"
    raw = json.loads(path.read_text())
    raw["entries"][0][field] = value
    path.write_text(json.dumps(raw))
    assert _errors(archived_repository)


@pytest.mark.parametrize(
    "change", ["duplicate", "missing_map", "source_commit", "format"]
)
def test_history_map_requires_unambiguous_inventory(
    archived_repository: Path, change: str
) -> None:
    path = archived_repository / "experiments/research/history/path-map.json"
    raw = json.loads(path.read_text())
    if change == "missing_map":
        path.unlink()
    else:
        if change == "duplicate":
            raw["entries"].append(raw["entries"][0])
        else:
            raw[change] = "invalid"
        path.write_text(json.dumps(raw))
    assert _errors(archived_repository)


@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_explicit_data_templates_defer_rendered_types(
    repository: Path, suffix: str
) -> None:
    _write(
        repository,
        f"experiments/research/demo/run.template.{suffix}",
        {
            "schema_version": 2,
            "model": {},
            "dataset": {"train_max_documents": "${VERIFIED_DOCUMENTS}"},
            "training": {},
        },
    )
    _track(repository)
    report = lint_research(repository)
    assert report["valid"], report["errors"]


@pytest.mark.parametrize(
    "name", ["run.yaml", "run.template.yaml.bak.yaml", "run.template.yaml"]
)
def test_templates_do_not_bypass_concrete_config_validation(
    repository: Path, name: str
) -> None:
    _write(
        repository,
        f"experiments/research/demo/{name}",
        {
            "model": {},
            "dataset": {
                "train_max_documents": "${DOCUMENTS}"
                if name != "run.template.yaml"
                else -1
            },
            "training": {},
        },
    )
    _track(repository)
    assert _errors(repository)


@pytest.mark.parametrize(
    "payload",
    [
        "count: ${lowercase}\n",
        "count: ${MISSING_CLOSE\n",
        "count: ${HAS-DASH}\n",
        "count: ${COUNT}\ncount: 2\n",
        "- ${COUNT}\n",
        "count: ${COUNT}\nvalue: .inf\n",
    ],
)
def test_data_template_structure_and_slot_syntax_fail_closed(
    repository: Path, payload: str
) -> None:
    _write(repository, "experiments/research/demo/run.template.yaml", payload)
    _track(repository)
    assert _errors(repository)
