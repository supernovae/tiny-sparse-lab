"""Exercise the full-verifier and explicitly committed cold-proof trust boundaries."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE

from sparselab.corpus import token_denominator_identity as identity
from sparselab.corpus.release import _verification_operation, verify_release
from sparselab.training.manifest import sha256_file

pytest_plugins = ("test_corpus_release_streaming",)


def _tokenizer(root: Path, vocab: int) -> Path:
    root.mkdir(parents=True)
    path = root / "tokenizer.json"
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.add_special_tokens(["<unk>"])
    tokenizer.add_tokens([f"word_{index}" for index in range(vocab - 1)])
    tokenizer.save(str(path))
    return path


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()


@pytest.fixture
def _reviewed_fixture(
    lm_release: Path, tmp_path: Path
) -> tuple[Path, Path, Path, Path, Path, str, dict[str, str]]:
    """Real frozen release/tokenizer; synthetic minimal report and committed proof."""
    verify_release(lm_release)
    root = tmp_path
    selected = root / "bakeoff/candidates/32768"
    tokenizer = _tokenizer(selected, 32768)
    manifest_sha = sha256_file(lm_release / "manifest.json")
    document = json.loads((lm_release / "manifest.json").read_bytes())["files"][
        "documents.jsonl"
    ]
    binding = {
        "release_id": lm_release.name,
        "release_manifest_sha256": manifest_sha,
    }
    winner_manifest = {
        "sha256": sha256_file(tokenizer),
        "source": "local_text",
        "revision": "fixture",
        "vocab_size": 32768,
        "corpus_forge_bakeoff": binding,
    }
    winner_path = selected / "tokenizer_manifest.json"
    winner_path.write_text(json.dumps(winner_manifest))
    winner_sha = sha256_file(winner_path)
    report = {
        "identity": {
            "release_id": lm_release.name,
            "release_path": str(lm_release),
            "declaration": {"release_path": str(lm_release)},
        },
        "release_binding": binding,
        "selected_vocab_size": 32768,
        "selected_tokenizer": str(tokenizer),
        "candidates": [
            {
                "vocab_size": 32768,
                "weighted_bytes_per_token": 1.0,
                "tokenizer_sha256": sha256_file(tokenizer),
                "manifest_sha256": winner_sha,
                "manifest_path": str(winner_path),
                "manifest": winner_manifest,
            }
        ],
    }
    report_path = root / "bakeoff/report.json"
    report_path.write_text(json.dumps(report))
    report_sha = sha256_file(report_path)
    repository = root / "evidence-repo"
    evidence_dir = repository / identity._PREFIX
    evidence_dir.mkdir(parents=True)
    primary = evidence_dir / "primary-verification.json"
    primary.write_text(
        json.dumps(
            {
                "status": "VERIFIED_PRIMARY",
                "release_id": lm_release.name,
                "release_path": str(lm_release),
                "metadata": {"release-manifest.json": {"sha256": manifest_sha}},
            }
        )
    )
    selection = evidence_dir / "tokenizer-bakeoff-verification.json"
    selection.write_text(
        json.dumps(
            {
                "status": "VERIFIED_V5_BAKEOFF_AND_SELECTED_TOKENIZER",
                "release_id": lm_release.name,
                "report": {"path": str(report_path), "sha256": report_sha},
                "selected_tokenizer": {
                    "path": str(tokenizer),
                    "sha256": sha256_file(tokenizer),
                },
                "tokenizer_artifact": {
                    "path": str(tokenizer),
                    "sha256": sha256_file(tokenizer),
                },
                "selected_vocab_size": 32768,
                "candidates": [
                    {
                        "vocab_size": 32768,
                        "tokenizer_sha256": sha256_file(tokenizer),
                        "manifest_sha256": winner_sha,
                    }
                ],
            }
        )
    )
    (evidence_dir / "crash-recovery.md").write_text(
        "### Survival and cold authentication\n"
        f"verify_release(primary_release, expected_id=release_id) 493.71 seconds {lm_release.name}\n"
        f"{manifest_sha} {sha256_file(tokenizer)} {winner_sha} {report_sha}\n"
    )
    _git(repository, "init", "-q")
    _git(
        repository,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.org",
        "add",
        ".",
    )
    _git(
        repository,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.org",
        "commit",
        "-q",
        "-m",
        "Review frozen cold evidence",
    )
    commit = _git(repository, "rev-parse", "HEAD")
    constants = {
        "_RELEASE_ID": lm_release.name,
        "_MANIFEST_SHA": manifest_sha,
        "_DOCUMENT_SHA": document["sha256"],
        "_DOCUMENT_SIZE": document["size"],
        "_TOKENIZER_SHA": sha256_file(tokenizer),
        "_TOKENIZER_MANIFEST_SHA": winner_sha,
        "_REPORT_SHA": report_sha,
    }
    return lm_release, tokenizer, primary, selection, repository, commit, constants


@pytest.fixture
def reviewed_paths(
    _reviewed_fixture: tuple[Path, Path, Path, Path, Path, str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, Path, str]:
    release, tokenizer, primary, selection, repo, commit, constants = _reviewed_fixture
    monkeypatch.setattr(identity, "_REPO", repo)
    monkeypatch.setattr(identity, "_APPROVED", commit)
    for key, value in constants.items():
        monkeypatch.setattr(identity, key, value)
    return release, tokenizer, primary, selection, repo, commit


def test_normal_path_fully_authenticates_inputs(
    lm_release: Path, tmp_path: Path
) -> None:
    tokenizer = _tokenizer(tmp_path / "tokenizer", 2)
    (tokenizer.parent / "tokenizer_manifest.json").write_text(
        json.dumps(
            {
                "sha256": sha256_file(tokenizer),
                "source": "synthetic",
                "revision": None,
                "vocab_size": 2,
            }
        )
    )
    with _verification_operation():
        verify_release(
            lm_release
        )  # authentic upstream proof reused inside this operation
        manifest, bindings = identity._authenticate_inputs(lm_release, tokenizer)
    assert manifest["release_id"] == lm_release.name
    assert bindings == {
        "release_manifest_sha256": sha256_file(lm_release / "manifest.json"),
        "tokenizer_sha256": sha256_file(tokenizer),
        "tokenizer_manifest_sha256": sha256_file(
            tokenizer.with_name("tokenizer_manifest.json")
        ),
        "evidence": None,
    }
    tokenizer.write_bytes(tokenizer.read_bytes() + b" ")
    with pytest.raises(ValueError, match="provenance or digest"):
        identity._authenticate_inputs(lm_release, tokenizer)
    copied = tmp_path / "releases" / lm_release.name
    shutil.copytree(lm_release, copied)
    tampered = json.loads((copied / "manifest.json").read_bytes())
    tampered["release_id"] = "0" * 64
    (copied / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="directory identity"):
        identity._authenticate_inputs(copied, tokenizer)


def test_normal_operation_rejects_changed_manifest(
    lm_release: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "releases" / lm_release.name
    shutil.copytree(lm_release, copied)
    shutil.copytree(lm_release.parent.parent / "snapshots", tmp_path / "snapshots")
    tokenizer = _tokenizer(tmp_path / "tokenizer", 2)
    (tokenizer.parent / "tokenizer_manifest.json").write_text(
        json.dumps(
            {
                "sha256": sha256_file(tokenizer),
                "source": "synthetic",
                "revision": None,
                "vocab_size": 2,
            }
        )
    )
    with _verification_operation():
        verify_release(copied)
        with (copied / "manifest.json").open("ab") as stream:
            stream.write(b"\n")
        with pytest.raises(ValueError, match="changed within verification operation"):
            identity._authenticate_inputs(copied, tokenizer)


def test_unsafe_paths_and_partial_evidence(tmp_path: Path) -> None:
    directory = tmp_path / "actual"
    directory.mkdir()
    (tmp_path / "linked").symlink_to(directory)
    with pytest.raises(ValueError, match="symlink"):
        identity._safe_path(tmp_path / "linked" / "not-yet-created")
    with pytest.raises(ValueError, match="parent traversal"):
        identity._safe_path(tmp_path / "actual" / ".." / "other")
    with pytest.raises(ValueError, match="both tracked evidence paths"):
        identity._authenticate_inputs(tmp_path, tmp_path, evidence_commit="0" * 40)


def test_reviewed_cold_record_accepts_only_bound_artifacts(
    reviewed_paths: tuple[Path, Path, Path, Path, Path, str],
) -> None:
    release, tokenizer, primary, selection, repo, commit = reviewed_paths
    manifest, bindings = identity._authenticate_inputs(
        release,
        tokenizer,
        evidence_commit=commit,
        release_evidence=primary,
        selection_evidence=selection,
    )
    assert manifest["release_id"] == release.name
    assert bindings["release_manifest_sha256"] == sha256_file(release / "manifest.json")
    assert bindings["tokenizer_sha256"] == sha256_file(tokenizer)
    assert bindings["tokenizer_manifest_sha256"] == sha256_file(
        tokenizer.with_name("tokenizer_manifest.json")
    )
    assert bindings["evidence"]["commit"] == commit
    assert bindings["evidence"]["release_evidence_blob_sha"] == _git(
        repo, "rev-parse", f"{commit}:{identity._PRIMARY}"
    )
    assert (
        bindings["evidence"]["cold_record_sha256"]
        == hashlib.sha256(
            subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{commit}:{identity._COLD}"]
            )
        ).hexdigest()
    )


def test_reuse_rejects_changed_report(
    reviewed_paths: tuple[Path, Path, Path, Path, Path, str],
) -> None:
    release, tokenizer, primary, selection, _repo, commit = reviewed_paths
    report = tokenizer.parent.parent.parent / "report.json"
    report.write_bytes(report.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bakeoff report changed"):
        identity._authenticate_inputs(
            release,
            tokenizer,
            evidence_commit=commit,
            release_evidence=primary,
            selection_evidence=selection,
        )


def test_reuse_rejects_changed_committed_evidence(
    reviewed_paths: tuple[Path, Path, Path, Path, Path, str],
) -> None:
    release, tokenizer, primary, selection, repo, commit = reviewed_paths
    primary.write_bytes(primary.read_bytes() + b" ")
    with pytest.raises(ValueError, match="reviewed committed record"):
        identity._authenticate_inputs(
            release,
            tokenizer,
            evidence_commit=commit,
            release_evidence=primary,
            selection_evidence=selection,
        )
    cold = repo / identity._COLD
    cold.write_bytes(cold.read_bytes() + b" ")
    _git(repo, "add", identity._COLD)
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.org",
        "commit",
        "-q",
        "-m",
        "Alter reviewed cold record",
    )
    descendant = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="cold verification record changed"):
        identity._authenticate_inputs(
            release,
            tokenizer,
            evidence_commit=descendant,
            release_evidence=primary,
            selection_evidence=selection,
        )


@pytest.mark.parametrize("bad_commit", ["0" * 40, "deadbeef", "8d74147"])
def test_reuse_rejects_arbitrary_commit(
    reviewed_paths: tuple[Path, Path, Path, Path, Path, str], bad_commit: str
) -> None:
    release, tokenizer, primary, selection, _repo, _commit = reviewed_paths
    with pytest.raises(ValueError, match="committed|full committed"):
        identity._authenticate_inputs(
            release,
            tokenizer,
            evidence_commit=bad_commit,
            release_evidence=primary,
            selection_evidence=selection,
        )


def test_reuse_rejects_untracked_or_mismatched_evidence(
    reviewed_paths: tuple[Path, Path, Path, Path, Path, str], tmp_path: Path
) -> None:
    release, tokenizer, primary, selection, _repo, commit = reviewed_paths
    untracked = tmp_path / "primary-verification.json"
    untracked.write_bytes(primary.read_bytes())
    with pytest.raises(ValueError, match="tracked path"):
        identity._authenticate_inputs(
            release,
            tokenizer,
            evidence_commit=commit,
            release_evidence=untracked,
            selection_evidence=selection,
        )
    with pytest.raises(ValueError, match="v5 evidence release"):
        identity._authenticate_inputs(
            tmp_path / release.name,
            tokenizer,
            evidence_commit=commit,
            release_evidence=primary,
            selection_evidence=selection,
        )
    with pytest.raises(ValueError, match="v5 evidence release"):
        identity._authenticate_inputs(
            release,
            tmp_path / "tokenizer.json",
            evidence_commit=commit,
            release_evidence=primary,
            selection_evidence=selection,
        )
