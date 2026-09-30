"""LM-only v3 publication must bind evidence even after a manifest is rehashed."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from sparselab.corpus.acquisition import acquire
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import Project, load_project
from sparselab.corpus.release import freeze, verify_release
from sparselab.training.manifest import canonical_json, sha256_file

PROJECT = Path(__file__).resolve().parents[1] / "corpora/devmind-sample-v0/corpus.yaml"


@pytest.fixture(scope="module")
def lm_release(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("lm-v3")
    spec = load_project(PROJECT).model_dump(mode="json")
    for source in spec["sources"]:
        source["schema_version"] = 3
        source.pop("redistribution")
        source["explicit_training_restriction"] = "none_found"
        source["rights"] = {
            "training_eligibility": "eligible",
            "redistribution_mode": "metadata_reconstruction_only",
            "spdx_expression": source["license"],
            "license_references": ["https://example.org/notice"],
            "notices": ["Keep upstream copyright and license notices"],
        }
    spec["transforms"] = [
        transform for transform in spec["transforms"] if transform["kind"] == "lm_text"
    ]
    spec["release"]["schema_version"] = 3
    spec["release"]["training_use_policy"] = "allowed_unless_explicitly_prohibited"
    spec["release"]["publication_mode"] = "metadata_reconstruction_only"
    spec["release"]["chat"]["selected"] = False
    spec["release"]["chat"]["training_splits"] = []
    project = Project.model_validate(spec)
    acquire(project, root)
    result = freeze(build(project, root, offline=True), root)
    assert verify_release(result)["release_id"] == result.name
    return result


def _repin(release: Path, tmp_path: Path, relative: str, mutate) -> Path:
    staged = release.parent / (".tampered-" + tmp_path.name)
    shutil.copytree(release, staged)
    path = staged / relative
    if relative.endswith(".json"):
        report = json.loads(path.read_text())
        mutate(report)
        path.write_bytes(canonical_json(report) + b"\n")
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows)
        path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    manifest_path = staged / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][relative] = {
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }
    manifest["release_id"] = hashlib.sha256(
        canonical_json({key: value for key, value in manifest.items() if key != "release_id"})
    ).hexdigest()
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")
    destination = release.parent / manifest["release_id"]
    staged.rename(destination)
    return destination


@pytest.mark.parametrize(
    ("file", "mutate", "error"),
    [
        ("lm/train.jsonl", lambda rows: rows[0].update(text="tampered training text"), "LM"),
        (
            "lineage.jsonl",
            lambda rows: rows[0].update(rendered_sha256="0" * 64),
            "lineage",
        ),
        (
            "license-report.json",
            lambda report: next(
                row for row in report["files"] if row["role"] == "document"
            )["rights"].update(training_eligibility="ineligible"),
            "rights",
        ),
    ],
)
def test_rehashed_release_rejects_tampered_evidence(
    lm_release: Path, tmp_path: Path, file: str, mutate, error: str
) -> None:
    changed = _repin(lm_release, tmp_path, file, mutate)
    with pytest.raises(ValueError, match=error):
        verify_release(changed)
