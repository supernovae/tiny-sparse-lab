"""Exercise all Corpus Forge BPE candidates against a real frozen local release."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sparselab.corpus.acquisition import acquire
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project
from sparselab.corpus.release import freeze
from sparselab.corpus.tokenizer_bakeoff import GROUPS, bakeoff


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project_dir = tmp_path / "project"
    sources = project_dir / "sources"
    sources.mkdir(parents=True)
    references = []
    assignments = {}
    for split in ("train", "validation"):
        for group in GROUPS:
            identifier = f"{group}_{split}"
            source_path = sources / f"{identifier}.yaml"
            text_path = sources / f"{identifier}.txt"
            words = (
                f"{group}_{index:05x}_{(index * 7919 + GROUPS.index(group) * 104729) % 1048576:05x}"
                for index in range(24000 if split == "train" else 24)
            )
            passage = " ".join(words)
            text_path.write_text(
                f"{group} {split}: {passage} {passage}\n"
                if split == "train"
                else f"{group} {split}: {passage}\n",
                encoding="utf-8",
            )
            source_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "id": identifier,
                        "kind": "local",
                        "canonical_uri": f"fixture:{identifier}",
                        "revision": "v1",
                        "license": "MIT",
                        "redistribution": "redistributable",
                        "domains": ["fixture"],
                        "document_kinds": [group],
                        "source_family": identifier,
                        "acquisition": {
                            "files": [
                                {
                                    "path": f"sources/{identifier}.txt",
                                    "name": f"{identifier}.txt",
                                }
                            ],
                            "max_bytes": 2000000,
                        },
                    }
                )
            )
            references.append(f"sources/{identifier}.yaml")
            assignments[identifier] = split
    (project_dir / "transforms").mkdir()
    (project_dir / "transforms/lm.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "lm",
                "version": "1",
                "kind": "lm_text",
                "parameters": {},
                "inputs": [],
            }
        )
    )
    (project_dir / "splits.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "unit": "source_document_family",
                "family_key": "source_family",
                "assignments": assignments,
            }
        )
    )
    (project_dir / "release.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "mixture": {"fixture": 1.0},
                "lm": {"selected": True, "training_splits": ["train", "validation"]},
                "chat": {"selected": False, "training_splits": []},
            }
        )
    )
    (project_dir / "corpus.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "tokenizer-bakeoff-fixture",
                "sources": sorted(references),
                "transforms": ["transforms/lm.yaml"],
                "splits": "splits.yaml",
                "release": "release.yaml",
            }
        )
    )
    project = load_project(project_dir / "corpus.yaml")
    workspace = tmp_path / "workspace"
    acquire(project, workspace)
    release = freeze(build(project, workspace, offline=True), workspace)
    declaration = project_dir / "tokenizer-bakeoff.yaml"
    declaration.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "release_path": str(release),
                "vocab_sizes": [16384, 24576, 32768],
                "max_fit_bytes": 268435456,
                "eval_split": "validation",
                "eval_max_docs_per_group": 200,
                "groups": list(GROUPS),
                "near_best_ratio": 0.98,
            }
        )
    )
    return declaration, workspace


def test_real_frozen_release_bakeoff_candidates_and_tamper(tmp_path: Path) -> None:
    declaration, workspace = _project(tmp_path)
    output = workspace / "tokenizer-bakeoff"
    report_path = bakeoff(declaration, output, work_root=workspace)
    report = json.loads(report_path.read_text())
    assert [row["vocab_size"] for row in report["candidates"]] == [16384, 24576, 32768]
    assert all(
        report["sample_receipt"]["fit"][group]["document_ids"] for group in GROUPS
    )
    assert all(
        report["sample_receipt"]["heldout"][group]["document_ids"] for group in GROUPS
    )
    assert {
        group: row["documents"]
        for group, row in report["candidates"][0]["per_kind"].items()
    } == dict.fromkeys(GROUPS, 1)
    assert report["selected_tokenizer"] in {
        candidate["tokenizer_path"] for candidate in report["candidates"]
    }
    assert bakeoff(declaration, output, work_root=workspace) == report_path
    manifest = output / "candidates/16384/tokenizer_manifest.json"
    manifest.write_text(
        manifest.read_text().replace('"release_id":"', '"release_id":"tampered-')
    )
    with pytest.raises(ValueError, match="manifest|mismatch|identity"):
        bakeoff(declaration, output, work_root=workspace)
