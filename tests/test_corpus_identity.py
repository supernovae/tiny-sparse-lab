"""Compact corpus metadata remains checkable after all original bytes disappear."""

from __future__ import annotations

import copy
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

from sparselab.archive import create_archive, verify_archive
from sparselab.corpus.acquisition import _digest as digest
from sparselab.corpus.acquisition import acquire
from sparselab.corpus.identity import explain_corpus_identity, verify_corpus_identity
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project
from sparselab.corpus.release import freeze
from sparselab.training.manifest import canonical_json


@pytest.fixture(scope="module")
def corpus_archive(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus-identity")
    repo, state = root / "repo", root / "state"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "examples/tiny-campaign", repo / "recipe"
    )
    rejected_path = repo / "recipe/sources/tiny_test.yaml"
    rejected = yaml.safe_load(rejected_path.read_text())
    rejected.update(redistribution="rejected", rejection_reason="Unavailable rights")
    rejected_path.write_text(yaml.safe_dump(rejected))
    project = load_project(repo / "recipe/corpus.yaml")
    acquire(project, state)
    release = freeze(build(project, state, offline=True), state)
    explanation = explain_corpus_identity(release, project)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    def commit():
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "Pin declaration",
            ],
            check=True,
        )

    commit()
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    source = repo / "recovery.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "recovery_version": 1,
                "id": "identity-test",
                "source_commit": head,
                "runtime_requirement": {
                    "engine": "pytorch",
                    "backend": "cpu",
                    "device_index": 0,
                    "requirements": {},
                },
                "steps": [
                    {
                        "kind": "corpus_release",
                        "id": "release",
                        "project": "recipe/corpus.yaml",
                        "expected_release_sha256": release.name,
                    },
                    {
                        "kind": "external_required",
                        "id": "training",
                        "role": "checkpoint",
                        "reason": "Metadata-only recovery fixture",
                    },
                ],
            }
        )
    )
    commit()
    archive = root / "thin.tar"
    created = create_archive(source, "thin", archive, work_root=state)
    # Remove both acquisition inputs and every generated source/release file.
    shutil.rmtree(repo)
    shutil.rmtree(state)
    relocated = root / "relocated" / "metadata.tar"
    relocated.parent.mkdir()
    archive.rename(relocated)
    return relocated, created, explanation


def test_metadata_round_trip_and_relocation_without_source_bytes(corpus_archive):
    archive, created, explanation = corpus_archive
    verified = verify_archive(archive)
    assert (
        verified["corpus_identities"]
        == created["corpus_identities"]
        == {"release": explanation}
    )
    result = verify_corpus_identity(json.loads(json.dumps(explanation)))
    assert result["project_id"] == "tiny-campaign"
    assert result["source_snapshots"] == {
        row["source_id"]: row["sha256"] for row in explanation["release"]["snapshots"]
    }
    assert result["verification_scope"] == "declared_digest_payloads_only"
    assert "tiny_test" in explanation["source_declarations"]
    assert (
        explanation["release"]["build_identity"]["lock"]["tiny_test"]["snapshot_sha256"]
        is None
    )
    assert "tiny_test" not in result["source_snapshots"]
    assert result["source_contents_verified"] is False
    assert result["release_contents_verified"] is False
    assert verified["unresolved_references"]
    assert all("snapshot" not in entry["role"] for entry in created["entries"])
    assert not any("/texts/" in entry["path"] for entry in created["entries"])
    assert all(
        "path" not in entry
        for source in explanation["source_declarations"].values()
        for entry in source["acquisition"]["files"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "declaration",
        "snapshot",
        "build",
        "release",
        "mapping",
        "scope",
        "project",
        "missing_snapshot",
        "missing_declaration",
        "algorithm",
    ],
)
def test_identity_tampering_rejected_even_with_rewritten_outer_index(
    corpus_archive, tmp_path, change
):
    archive, created, _ = corpus_archive
    index = copy.deepcopy(created)
    explanation = index["corpus_identities"]["release"]
    release = explanation["release"]
    source_id = next(iter(explanation["snapshot_payloads"]))
    if change == "declaration":
        explanation["source_declarations"][source_id]["revision"] = "tampered"
    elif change == "snapshot":
        explanation["snapshot_payloads"][source_id]["files"][0]["sha256"] = "0" * 64
    elif change == "algorithm":
        explanation["snapshot_payloads"][source_id]["adapter"]["version"] = "tampered"
    elif change == "build":
        release["build_identity"]["implementation_sha256"] = "0" * 64
        # Keep release internally valid to reach the original build digest check.
        release["release_id"] = digest(
            {k: v for k, v in release.items() if k != "release_id"}
        )
    elif change == "release":
        release["files"][next(iter(release["files"]))]["sha256"] = "0" * 64
    elif change == "mapping":
        release["snapshots"][0]["sha256"] = "0" * 64
        release["release_id"] = digest(
            {k: v for k, v in release.items() if k != "release_id"}
        )
    elif change == "scope":
        explanation["verification_scope"] = "source_contents_verified"
    elif change == "project":
        release["corpus_id"] = "another-project"
        release["release_id"] = digest(
            {k: v for k, v in release.items() if k != "release_id"}
        )
    elif change == "missing_snapshot":
        del explanation["snapshot_payloads"][source_id]
    else:
        del explanation["source_declarations"][source_id]
    with pytest.raises(ValueError):
        verify_corpus_identity(explanation)
    destination = tmp_path / "tampered.tar"
    with tarfile.open(archive) as original, tarfile.open(destination, "w") as output:
        for member in original:
            raw = (
                canonical_json(index) + b"\n"
                if member.name == "archive-index.json"
                else original.extractfile(member).read()
            )
            member.size = len(raw)
            output.addfile(member, io.BytesIO(raw))
    with pytest.raises(ValueError):
        verify_archive(destination)


def test_missing_explanation_rejected_and_legacy_v1_still_readable(
    corpus_archive, tmp_path
):
    archive, created, _ = corpus_archive
    for legacy in (False, True):
        index = copy.deepcopy(created)
        index["corpus_identities"] = {}
        if legacy:
            index["format"] = "archive-index-v1"
            del index["corpus_identities"]
        destination = tmp_path / f"legacy-{legacy}.tar"
        with (
            tarfile.open(archive) as original,
            tarfile.open(destination, "w") as output,
        ):
            for member in original:
                raw = (
                    canonical_json(index) + b"\n"
                    if member.name == "archive-index.json"
                    else original.extractfile(member).read()
                )
                member.size = len(raw)
                output.addfile(member, io.BytesIO(raw))
        if legacy:
            assert verify_archive(destination)["format"] == "archive-index-v1"
        else:
            with pytest.raises(ValueError, match="explanation inventory"):
                verify_archive(destination)


def test_self_consistent_explanation_still_bound_to_archived_release(
    corpus_archive, tmp_path
):
    archive, created, _ = corpus_archive
    index = copy.deepcopy(created)
    explanation = index["corpus_identities"]["release"]
    release = explanation["release"]
    release["stages"] = []
    release["release_id"] = digest(
        {k: v for k, v in release.items() if k != "release_id"}
    )
    verify_corpus_identity(explanation)
    destination = tmp_path / "replacement.tar"
    with tarfile.open(archive) as original, tarfile.open(destination, "w") as output:
        for member in original:
            raw = (
                canonical_json(index) + b"\n"
                if member.name == "archive-index.json"
                else original.extractfile(member).read()
            )
            member.size = len(raw)
            output.addfile(member, io.BytesIO(raw))
    with pytest.raises(ValueError, match="release binding mismatch"):
        verify_archive(destination)


def test_wikimedia_identity_includes_retrieval_payload(corpus_archive):
    # Model the already-declared historical digest recipe; no retrieval is done.
    explanation = copy.deepcopy(corpus_archive[2])
    source_id = next(iter(explanation["snapshot_payloads"]))
    declaration = explanation["source_declarations"][source_id]
    declaration["kind"] = "wikimedia_dump"
    payload = explanation["snapshot_payloads"][source_id]
    payload["adapter"]["id"] = "wikimedia_dump"
    payload["declaration_sha256"] = digest(declaration)
    payload["retrieval"] = {"pages_scanned": 4, "pages_selected": 2}
    release = explanation["release"]
    build = release["build_identity"]
    build["lock"][source_id] = {
        "declaration_sha256": digest(declaration),
        "snapshot_sha256": digest(payload),
    }
    for row in release["snapshots"]:
        if row["source_id"] == source_id:
            row["sha256"] = digest(payload)
    release["build_id"] = digest(build)
    release["release_id"] = digest(
        {k: v for k, v in release.items() if k != "release_id"}
    )
    verify_corpus_identity(explanation)
    payload["retrieval"]["pages_selected"] = 3
    with pytest.raises(ValueError, match="snapshot identity payload"):
        verify_corpus_identity(explanation)
    del payload["retrieval"]
    with pytest.raises(ValueError, match="snapshot identity payload"):
        verify_corpus_identity(explanation)
