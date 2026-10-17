"""Neutral metadata-only authoring fixtures: no corpus or model execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from sparselab.research import catalog
from sparselab.research.scaffold import scaffold_research


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def closure(tmp_path: Path) -> Path:
    root = tmp_path / "resources"
    root.mkdir()
    base = {
        "schema_version": 2,
        "name": "neutral-base",
        "seed": 5,
        "model": {
            "vocab_size": 260,
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "ffn_dim": 32,
            "max_seq_len": 16,
        },
        "tokenizer": {"path": "artifacts/tokenizer/tokenizer.json"},
        "dataset": {
            "source": "chat_recall",
            "cache_dir": "artifacts/data",
            "train_max_documents": 2,
            "validation_max_documents": 2,
            "train_max_tokens": 64,
            "validation_max_tokens": 64,
            "synthetic_seed": 3,
        },
        "training": {
            "seq_len": 8,
            "micro_batch_size": 1,
            "gradient_accumulation": 1,
            "max_steps": 8,
            "max_tokens": 64,
        },
        "optimizer": {"name": "adamw", "warmup_steps": 2},
        "logging": {"root_dir": "runs"},
    }
    from sparselab.config.models import RunConfig

    base = RunConfig.model_validate(base).model_dump(mode="json")
    (root / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    profile = {
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "ffn_dim": 32,
        "seq_len": 8,
        "max_seq_len": 16,
        "micro_batch_size": 1,
        "gradient_accumulation": 1,
        "max_steps": 8,
        "max_tokens": 64,
        "validation_steps": 4,
        "memory_table_size": 17,
        "memory_dim": 4,
    }
    scales = ("smoke", "nano", "micro", "tiny")
    _json(
        root / "profiles.json",
        {
            "format": "sparselab-research-profiles",
            "version": 1,
            "scales": {scale: profile for scale in scales},
        },
    )
    dataset = {
        "source": "chat_recall",
        "revision": None,
        "dataset_seed": 3,
        "train_max_documents": 2,
        "validation_max_documents": 2,
        "train_max_tokens": 64,
        "validation_max_tokens": 64,
        "vocab_size": 260,
        "min_frequency": 1,
        "tokenizer_max_documents": 2,
        "tokenizer_train_max_tokens": 64,
        "tokenizer_validation_max_documents": 2,
        "tokenizer_validation_max_tokens": 64,
        "license": "CC0-1.0",
        "source_url": "https://example.org/neutral",
        "notes": [],
        "card_applicability": {},
    }
    _json(
        root / "datasets.json",
        {
            "format": "sparselab-research-datasets",
            "version": 1,
            "datasets": {"neutral": dataset},
        },
    )
    entry = {
        "format": "sparselab-research-entry",
        "version": 1,
        "id": "neutral-study",
        "title": "Neutral comparison",
        "question": "Does the plan bind two widths?",
        "mechanisms": [],
        "hypothesis": "Widths differ",
        "failure_interpretation": "Invalid plan",
        "controls": [],
        "independent_variables": [],
        "dependent_metrics": [],
        "confounders": [],
        "minimum_useful_scale": "smoke",
        "recommended_scale": "smoke",
        "hardware_notes": "Metadata only",
        "runtime_class": "fixture",
        "evaluation_policy": {
            "periodic_validation_steps": {},
            "endpoint_cards": [],
            "endpoint_policy": "first_configured_limit",
            "milestones_supported": False,
            "primary_thresholds": [],
        },
        "cards": [
            {
                "reference": "cards/neutral.json",
                "role": "development",
                "category": "fixture",
                "applicability": {},
            }
        ],
        "papers": [],
        "implementation_links": [],
        "docs": [],
        "can_establish": [],
        "cannot_establish": [],
        "prerequisites": [],
        "recipe": "recipes/neutral-study.json",
    }
    _json(root / "catalog/neutral-study.json", entry)
    recipe = {
        "base_set": {},
        "axes": {
            "seed": [{"label": "s5", "set": {"seed": 5}}],
            "width": [
                {"label": "small", "set": {"model.ffn_dim": 32}},
                {"label": "wide", "set": {"model.ffn_dim": 48}},
            ],
        },
        "comparisons": [
            {
                "id": "width-effect",
                "vary": "custom",
                "vary_fields": ["model.ffn_dim"],
                "baseline": {"width": "small"},
                "variant": {"width": "wide"},
            }
        ],
    }
    _json(
        root / "recipes/neutral-study.json",
        {
            "format": "sparselab-research-recipe",
            "version": 1,
            "id": "neutral-study",
            "designs": {"default": {scale: recipe for scale in scales}},
        },
    )
    # Capability implementation is generic Lab code, independent of research resources.
    from sparselab.evaluation.capabilities import (
        capability_card,
        capability_card_payload,
    )

    _json(
        root / "cards/neutral.json",
        capability_card_payload(capability_card("chat-alias-recall-v1")),
    )
    _json(
        root / "lifecycle.json",
        {
            "format": "sparselab-research-lifecycle",
            "version": 1,
            "entries": [],
            "evidence": [],
            "baselines": [],
            "findings": [],
            "promotions": [],
        },
    )
    return root


def test_external_closure_without_packaged_resources(
    closure: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(catalog, "_RESOURCE_ROOT", tmp_path / "absent-package")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    output = scaffold_research(
        "neutral-study",
        tmp_path / "result",
        root=closure,
        scale="smoke",
        data="neutral",
    )
    record = json.loads((output / "research.json").read_text())
    from sparselab.experiments.reporting import _validate_research

    _validate_research(output / "research.json", output / "study.yaml")
    assert len(record["coordinates"]) == 2
    matrix = yaml.safe_load((output / "matrix.yaml").read_text())
    assert matrix["axes"]["seed"] == [{"label": "s5", "set": {"seed": 5}}]
    inventory = {item["path"]: item["sha256"] for item in record["inputs"]}
    for relative in (
        "research_sources/base.yaml",
        "research_sources/profiles.json",
        "research_sources/datasets.json",
        "cards/neutral.json",
    ):
        assert (
            inventory[relative]
            == hashlib.sha256((output / relative).read_bytes()).hexdigest()
        )
    with pytest.raises(FileExistsError):
        scaffold_research(
            "neutral-study", output, root=closure, scale="smoke", data="neutral"
        )


@pytest.mark.parametrize(
    "missing",
    [
        "base.yaml",
        "profiles.json",
        "datasets.json",
        "recipes/neutral-study.json",
        "cards/neutral.json",
    ],
)
def test_external_missing_resource_never_falls_back(
    closure: Path, tmp_path: Path, missing: str
) -> None:
    (closure / missing).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        scaffold_research(
            "neutral-study",
            tmp_path / "result",
            root=closure,
            scale="smoke",
            data="neutral",
        )
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("linked", ["base.yaml", "profiles.json", "recipes", "cards"])
def test_resource_symlink_refused(closure: Path, tmp_path: Path, linked: str) -> None:
    path = closure / linked
    moved = tmp_path / "moved"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ValueError, match="symlink"):
        scaffold_research(
            "neutral-study",
            tmp_path / "result",
            root=closure,
            scale="smoke",
            data="neutral",
        )
    assert not (tmp_path / "result").exists()


def test_explicit_seed_required(closure: Path, tmp_path: Path) -> None:
    path = closure / "recipes/neutral-study.json"
    value = json.loads(path.read_text())
    del value["designs"]["default"]["smoke"]["axes"]["seed"]
    _json(path, value)
    with pytest.raises(ValueError, match="explicitly declare.*seed"):
        scaffold_research(
            "neutral-study",
            tmp_path / "result",
            root=closure,
            scale="smoke",
            data="neutral",
        )


def test_unrelated_help_without_research_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.cli.main import build_parser

    monkeypatch.setattr(catalog, "_RESOURCE_ROOT", tmp_path / "absent-package")
    parser = build_parser(work_dir=tmp_path / "work")
    with pytest.raises(SystemExit) as outcome:
        parser.parse_args(["train", "--help"])
    assert outcome.value.code == 0


def test_cli_external_scaffold(
    closure: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.cli.main import build_parser

    monkeypatch.setattr(catalog, "_RESOURCE_ROOT", tmp_path / "absent-package")
    parser = build_parser(work_dir=tmp_path / "work")
    args = parser.parse_args(
        [
            "research",
            "scaffold",
            "neutral-study",
            "--resource-root",
            str(closure),
            "--scale",
            "smoke",
            "--data",
            "neutral",
            "--output",
            str(tmp_path / "result"),
        ]
    )
    args.handler(args)
    assert (tmp_path / "result/research.json").is_file()


def test_source_declaration_pins_are_validated(closure: Path, tmp_path: Path) -> None:
    path = closure / "datasets.json"
    value = json.loads(path.read_text())
    dataset = value["datasets"]["neutral"]
    dataset.update(
        source="snapshot", source_declaration="sources/neutral.json", revision="a" * 40
    )
    _json(path, value)
    source = {
        "schema_version": 1,
        "kind": "huggingface",
        "repo_id": "fixture/neutral",
        "revision": "a" * 40,
        "config": "default",
        "splits": {"train": "train", "validation": "validation"},
        "text_field": "text",
        "attribution": "Synthetic fixture",
        "license": "CC0-1.0",
        "selection": {"mode": "bounded", "documents": {"train": 2, "validation": 2}},
        "resources": {
            "max_source_records": 4,
            "max_text_bytes": 1024,
            "max_record_bytes": 256,
            "max_work_bytes": 4096,
            "min_free_bytes": 0,
        },
    }
    _json(closure / "sources/neutral.json", source)
    assert catalog.load_datasets(root=closure).datasets["neutral"].revision == "a" * 40
    output = scaffold_research(
        "neutral-study",
        tmp_path / "snapshot-result",
        root=closure,
        scale="smoke",
        data="neutral",
    )
    from sparselab.experiments.reporting import _validate_research

    _validate_research(output / "research.json", output / "study.yaml")
    assert (output / "research_sources/sources/neutral.json").read_bytes() == (
        closure / "sources/neutral.json"
    ).read_bytes()
    source["revision"] = "b" * 40
    _json(closure / "sources/neutral.json", source)
    with pytest.raises(ValueError, match="revisions differ"):
        catalog.load_datasets(root=closure)


def test_external_lesson_sources(
    closure: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.research.scaffold import scaffold_lesson

    _json(
        closure / "lessons/neutral-lesson.json",
        {
            "format": "sparselab-mechanism-lesson",
            "version": 1,
            "id": "neutral-lesson",
            "title": "Neutral lesson",
            "mechanism_kind": "model",
            "summary": "Metadata fixture",
            "benefit_scope": [],
            "steps": [],
            "patches_by_scale": {
                scale: {} for scale in ("smoke", "nano", "micro", "tiny")
            },
            "prerequisites": [],
            "try_changes": [],
            "limits": [],
            "papers": [],
            "docs": [],
        },
    )
    monkeypatch.setattr(catalog, "_RESOURCE_ROOT", tmp_path / "absent-package")
    output = scaffold_lesson(
        "neutral-lesson", tmp_path / "lesson", root=closure, data="neutral"
    )
    payload = json.loads((output / "lesson.json").read_text())
    assert {item["path"] for item in payload["inputs"]} == {
        "research_sources/lesson.json",
        "research_sources/base.yaml",
        "research_sources/profiles.json",
        "research_sources/datasets.json",
    }


def test_external_path_escape_refused(closure: Path, tmp_path: Path) -> None:
    entry_path = closure / "catalog/neutral-study.json"
    entry = json.loads(entry_path.read_text())
    entry["cards"][0]["reference"] = "../outside.json"
    _json(entry_path, entry)
    with pytest.raises(ValueError, match="relative|unsafe|path"):
        scaffold_research(
            "neutral-study",
            tmp_path / "escape",
            root=closure,
            scale="smoke",
            data="neutral",
        )
    assert not (tmp_path / "escape").exists()


def test_local_entry_and_symlinked_root(closure: Path, tmp_path: Path) -> None:
    entry = json.loads((closure / "catalog/neutral-study.json").read_text())
    _json(closure / "neutral-study.json", entry)
    output = scaffold_research(
        closure / "neutral-study.json",
        tmp_path / "local",
        root=closure,
        scale="smoke",
        data="neutral",
    )
    assert (output / "research.json").exists()
    linked = tmp_path / "linked"
    linked.symlink_to(closure, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        scaffold_research(
            "neutral-study",
            tmp_path / "refused",
            root=linked,
            scale="smoke",
            data="neutral",
        )


@pytest.mark.parametrize("mode", ["symlink", "missing", "unlisted"])
def test_reporting_refuses_unsafe_dataset_source(
    closure: Path, tmp_path: Path, mode: str
) -> None:
    from sparselab.experiments.reporting import _validate_research
    from sparselab.training.manifest import canonical_json

    output = scaffold_research(
        "neutral-study",
        tmp_path / "report",
        root=closure,
        scale="smoke",
        data="neutral",
    )
    source = output / "research_sources/datasets.json"
    if mode == "symlink":
        source.unlink()
        source.symlink_to(closure / "datasets.json")
    elif mode == "missing":
        source.unlink()
    else:
        receipt_path = output / "research.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["inputs"] = [
            item
            for item in receipt["inputs"]
            if item["path"] != "research_sources/datasets.json"
        ]
        receipt.pop("research_sha256")
        receipt["research_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
        _json(receipt_path, receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        _validate_research(output / "research.json", output / "study.yaml")
