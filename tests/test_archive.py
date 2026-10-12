"""Thin archives make unresolved bytes explicit; TAR verification never extracts."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

from sparselab.archive import create_archive, verify_archive


@pytest.fixture(scope="module")
def partial_recipe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo = tmp_path_factory.mktemp("archive-partial") / "repo"
    repo.mkdir()
    declaration = repo / "experiments/research/fixture"
    declaration.mkdir(parents=True)
    (repo / "README.md").write_text("Archive fixture; no runtime inputs.\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
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
            "Pin fixture repository",
        ],
        check=True,
    )
    pinned = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    recipe = {
        "recovery_version": 1,
        "id": "partial-fixture",
        "source_commit": pinned,
        "steps": [
            {
                "id": "reviewed-input",
                "kind": "external_required",
                "role": "tokenizer_selection",
                "reason": "No reviewed tokenizer choice is declared in this fixture.",
            }
        ],
    }
    source = declaration / "recovery.yaml"
    source.write_text(yaml.safe_dump(recipe))
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "add",
            "experiments/research/fixture/recovery.yaml",
        ],
        check=True,
    )
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
            "Declare partial recovery",
        ],
        check=True,
    )
    return source


def test_partial_thin_is_honest_and_portable_refuses(
    tmp_path: Path, partial_recipe: Path
) -> None:
    output = tmp_path / "thin.tar"
    state = tmp_path / "absent-state"
    result = create_archive(partial_recipe, "thin", output, work_root=state)
    assert result["family_lineage"] == "NOT_DECLARED"
    assert any(row["kind"] == "external_required" for row in result["external"])
    verified = verify_archive(output)
    assert verified["unresolved_references"] == result["external"]
    assert not state.exists()
    with pytest.raises(FileExistsError):
        create_archive(partial_recipe, "thin", output, work_root=state)
    with pytest.raises(ValueError, match="portable"):
        create_archive(
            partial_recipe, "portable", tmp_path / "portable.tar", work_root=state
        )
    assert not (tmp_path / "portable.tar").exists()


def test_unsafe_archive_members_and_mutation_are_rejected(
    tmp_path: Path, partial_recipe: Path
) -> None:
    source = tmp_path / "unsafe.tar"
    with tarfile.open(source, "w") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"X"))
    with pytest.raises(ValueError, match="unsafe"):
        verify_archive(source)
    source = tmp_path / "symlink.tar"
    with tarfile.open(source, "w") as tar:
        info = tarfile.TarInfo("archive-index.json")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(ValueError, match="unsafe"):
        verify_archive(source)
    source = tmp_path / "good.tar"
    create_archive(partial_recipe, "thin", source, work_root=tmp_path / "state")
    with source.open("r+b") as handle:
        handle.seek(1024)
        original = handle.read(1)
        handle.seek(1024)
        handle.write(bytes([original[0] ^ 1]))
    with pytest.raises((ValueError, tarfile.TarError)):
        verify_archive(source)


def test_declared_recovery_receipt_is_authenticated_in_thin_archive(
    tmp_path: Path,
    partial_recipe: Path,
) -> None:
    from sparselab.campaign.state import digest
    from sparselab.recovery.engine import verify_recovery_receipt
    from sparselab.training.manifest import canonical_json, sha256_file

    repo = tmp_path / "repo"
    shutil.copytree(partial_recipe.parents[3], repo)
    source = repo / partial_recipe.relative_to(partial_recipe.parents[3])
    recipe = yaml.safe_load(source.read_text())
    body = {
        "format": "sparselab-recovery-receipt-v1",
        "manifest_sha256": sha256_file(source),
        "source_commit": recipe["source_commit"],
        "declaration_hashes": [],
        "provenance": {"status": "CLEAN_AND_COMMITTED"},
        "allow_uncommitted_declaration": False,
        "work_root": str(tmp_path / "absent-state"),
        "storage_warnings": [],
        "outcomes": [],
    }
    scientific = {
        "manifest_sha256": body["manifest_sha256"],
        "source_commit": body["source_commit"],
        "declaration_hashes": [],
        "declaration_status": "CLEAN_AND_COMMITTED",
        "allow_uncommitted_declaration": False,
        "outcomes": [],
    }
    record = {
        **body,
        "receipt_sha256": digest("sparselab-recovery-receipt-v1", scientific),
        "issued_at_utc": "2026-01-01T00:00:00Z",
    }
    record["record_sha256"] = hashlib.sha256(canonical_json(record)).hexdigest()
    receipt = source.parent / "recovery-receipt.json"
    receipt.write_bytes(canonical_json(record) + b"\n")
    assert (
        verify_recovery_receipt(receipt)["receipt_sha256"] == record["receipt_sha256"]
    )
    recipe["evidence"] = ["recovery-receipt.json"]
    source.write_text(yaml.safe_dump(recipe))
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
            "Declare compact recovery outcome",
        ],
        check=True,
    )
    output = tmp_path / "receipt.tar"
    index = create_archive(source, "thin", output, work_root=tmp_path / "absent-state")
    assert any(row["role"] == "compact_evidence" for row in index["entries"])
    assert verify_archive(output)["verified_members"] == len(index["entries"])


def test_portable_verifies_real_release_export_tokenizer_and_prepared_bytes(
    tmp_path: Path,
) -> None:
    from sparselab.config.loading import load_config, load_tokenizer_config
    from sparselab.corpus.acquisition import acquire
    from sparselab.corpus.export import export_release
    from sparselab.corpus.pipeline import build
    from sparselab.corpus.project import load_project
    from sparselab.corpus.release import freeze
    from sparselab.data.packing import prepare_data
    from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
    from sparselab.evaluation.readiness import (
        assess_readiness,
        issue_review,
        verify_readiness_result,
    )
    from sparselab.evaluation.suite import run_suite, verify_evaluation_index
    from sparselab.family.cli import show
    from sparselab.training.manifest import architecture_sha256, sha256_file
    from sparselab.training.mlx_checkpoints import strict_json
    from sparselab.training.trainer import train

    original = Path(__file__).resolve().parents[1]
    repo, state = tmp_path / "repo", tmp_path / "persistent-state"
    repo.mkdir()
    shutil.copytree(original / "examples/tiny-campaign", repo / "recipe")
    splits = yaml.safe_load((repo / "recipe/splits.yaml").read_text())
    splits["assignments"]["tiny_docs_family"] = "validation"
    (repo / "recipe/splits.yaml").write_text(yaml.safe_dump(splits))
    config = yaml.safe_load((original / "configs/runtime_smoke_cpu.yaml").read_text())
    config["name"] = "archive-tiny"
    config["tokenizer"]["path"] = str(state / "placeholder.json")
    config["dataset"]["cache_dir"] = str(state / "cache")
    config["logging"]["root_dir"] = str(state / "runs")
    config["training"].update(max_steps=2, max_tokens=32, gradient_accumulation=1)
    config["optimizer"]["warmup_steps"] = 1
    config["checkpoint"]["every_steps"] = 1
    config["evaluation"]["every_steps"] = 1
    (repo / "base.yaml").write_text(yaml.safe_dump(config))
    (repo / "suite.yaml").write_text(
        yaml.safe_dump(
            {
                "evaluation_suite_version": 1,
                "id": "tiny",
                "evaluations": [{"id": "loss", "role": "gate", "kind": "heldout_lm"}],
            }
        )
    )
    (repo / "policy.yaml").write_text(
        yaml.safe_dump(
            {
                "readiness_version": 1,
                "id": "tiny",
                "require_verified_checkpoint": True,
                "required_gate_ids": ["loss"],
                "min_completed_evaluations": 1,
                "max_heldout_loss": None,
                "require_human_review": True,
            }
        )
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
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
            "Pin local corpus and base declaration",
        ],
        check=True,
    )
    project = load_project(repo / "recipe/corpus.yaml")
    acquire(project, state)
    release = freeze(build(project, state, offline=True), state)
    exported = export_release(release, "lm", repo / "base.yaml", 300, state)
    tokenizer = train_tokenizer(load_tokenizer_config(exported / "tokenizer.yaml"))
    run = load_config(exported / "run.yaml")
    prepared = prepare_data(run, load_tokenizer(tokenizer))
    train(run, run_id="archive-model")
    generation = max((state / "runs/archive-model/checkpoints").glob("step_*"))
    checkpoint_sha = strict_json(generation / "manifest.json")["sha256"]
    index_path = run_suite(
        repo / "suite.yaml",
        "archive-model",
        generation.name,
        state / "runs",
        backend="cpu",
    )
    index = verify_evaluation_index(index_path)
    review_path = repo / "review.json"
    issue_review(
        index_path,
        "archive-reviewer",
        "approve",
        "Reviewed checkpoint-bound heldout result",
        review_path,
    )
    readiness_path = assess_readiness(repo / "policy.yaml", index_path, review_path)
    readiness = verify_readiness_result(readiness_path)
    family = {
        "family_version": 1,
        "id": "tiny",
        "nodes": [
            {
                "id": "observed",
                "parent": None,
                "parent_checkpoint_sha256": None,
                "corpus": {"id": "tiny-campaign", "sha256": release.name},
                "tokenizer": {"id": "selection", "sha256": sha256_file(tokenizer)},
                "plan": {"id": "future", "sha256": "a" * 64},
                "architecture_sha256": architecture_sha256(run.model_dump(mode="json")),
                "objective": "next_token",
                "budget": {"max_steps": 2, "max_tokens": 32},
                "checkpoint": {"sha256": checkpoint_sha, "path": str(generation)},
                "evaluation_index": {
                    "sha256": index["index_sha256"],
                    "path": str(index_path),
                },
                "readiness_result": {
                    "sha256": readiness["result_sha256"],
                    "path": str(readiness_path),
                },
            }
        ],
    }
    (repo / "family.json").write_text(json.dumps(family))
    family["nodes"][0]["corpus"]["sha256"] = "b" * 64
    (repo / "family.json").write_text(json.dumps(family))
    with pytest.raises(ValueError, match="corpus release"):
        show(repo / "family.json", work_root=state)
    family["nodes"][0]["corpus"]["sha256"] = release.name
    (repo / "family.json").write_text(json.dumps(family))
    subprocess.run(["git", "-C", str(repo), "add", "family.json"], check=True)
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
            "Declare planned family",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    source = repo / "recovery.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "recovery_version": 1,
                "id": "archive-tiny",
                "source_commit": commit,
                "runtime_requirement": {
                    "engine": "pytorch",
                    "backend": "cpu",
                    "device_index": 0,
                    "requirements": {},
                },
                "evaluation_suite": "suite.yaml",
                "readiness_policy": "policy.yaml",
                "family": "family.json",
                "steps": [
                    {
                        "kind": "corpus_release",
                        "id": "release",
                        "project": "recipe/corpus.yaml",
                        "expected_release_sha256": release.name,
                    },
                    {
                        "kind": "corpus_export",
                        "id": "export",
                        "corpus": "release",
                        "base_run": "base.yaml",
                        "view": "lm",
                        "vocab_size": 300,
                        "expected_export_sha256": exported.name,
                    },
                    {
                        "kind": "tokenizer_train",
                        "id": "tokenizer",
                        "export": "export",
                        "expected_tokenizer_sha256": sha256_file(tokenizer),
                    },
                    {
                        "kind": "prepared_data",
                        "id": "prepared",
                        "export": "export",
                        "tokenizer": "tokenizer",
                        "expected_manifest_sha256": strict_json(
                            prepared.root / "manifest.json"
                        )["manifest_sha256"],
                    },
                ],
            }
        )
    )
    subprocess.run(["git", "-C", str(repo), "add", "recovery.yaml"], check=True)
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
            "Pin archive recovery expectations",
        ],
        check=True,
    )
    thin = create_archive(source, "thin", tmp_path / "thin-local.tar", work_root=state)
    assert any(entry["kind"] == "local_source" for entry in thin["external"])
    assert any(
        entry["kind"] == "checkpoint"
        and entry["state"] == "PRESENT_EXTERNAL"
        and entry["sha256"] == checkpoint_sha
        for entry in thin["external"]
    )
    assert not any("/texts/" in entry["path"] for entry in thin["entries"])
    assert (
        verify_archive(tmp_path / "thin-local.tar")["unresolved_references"]
        == thin["external"]
    )
    output = tmp_path / "portable.tar"
    created = create_archive(source, "portable", output, work_root=state)
    assert created["external"] == []
    assert any(item["role"] == "source_snapshot" for item in created["entries"])
    assert any(item["role"] == "prepared_data" for item in created["entries"])
    assert any(item["role"] == "checkpoint" for item in created["entries"])
    assert any(item["role"] == "evaluation_result" for item in created["entries"])
    assert any(item["role"] == "evaluation_suite" for item in created["entries"])
    assert any(item["role"] == "readiness_policy" for item in created["entries"])
    assert any(item["role"] == "model_review" for item in created["entries"])
    assert any(item["role"] == "run_manifest" for item in created["entries"])
    assert verify_archive(output)["verified_members"] == len(created["entries"])
    # A canonical rewritten outer index cannot omit scientific prerequisites.
    from sparselab.training.manifest import canonical_json

    def repack(
        destination: Path, changed: dict, replacement: dict[str, bytes] | None = None
    ) -> None:
        with (
            tarfile.open(output) as original_tar,
            tarfile.open(destination, "w") as rewritten,
        ):
            index_bytes = canonical_json(changed) + b"\n"
            header = tarfile.TarInfo("archive-index.json")
            header.size = len(index_bytes)
            rewritten.addfile(header, io.BytesIO(index_bytes))
            for entry in changed["entries"]:
                member = original_tar.extractfile(entry["path"])
                assert member is not None
                payload = (replacement or {}).get(entry["path"], member.read())
                header = tarfile.TarInfo(entry["path"])
                header.size = entry["size"]
                rewritten.addfile(header, io.BytesIO(payload))

    for label, stripped, error in (
        (
            "snapshot",
            {
                entry["path"]
                for entry in created["entries"]
                if "/snapshots/" in entry["path"]
            },
            "snapshot",
        ),
        (
            "run-manifest",
            {
                entry["path"]
                for entry in created["entries"]
                if entry["role"] == "run_manifest"
            },
            "run_manifest",
        ),
        (
            "evaluation-result",
            {
                entry["path"]
                for entry in created["entries"]
                if entry["role"] == "evaluation_result"
            },
            "evaluation dependency",
        ),
        (
            "readiness-policy",
            {
                entry["path"]
                for entry in created["entries"]
                if entry["role"] == "readiness_policy"
            },
            "evaluation dependency",
        ),
        (
            "human-review",
            {
                entry["path"]
                for entry in created["entries"]
                if entry["role"] == "model_review"
            },
            "evaluation dependency",
        ),
        (
            "corpus-release",
            {
                item["path"]
                for item in created["entries"]
                if item["path"].startswith("artifacts/release/")
            },
            "corpus_release",
        ),
        (
            "tokenizer",
            {
                item["path"]
                for item in created["entries"]
                if item["path"].startswith("artifacts/tokenizer/")
            },
            "tokenizer_train",
        ),
        (
            "prepared",
            {
                item["path"]
                for item in created["entries"]
                if item["path"].startswith("artifacts/prepared/")
            },
            "prepared_data",
        ),
    ):
        assert stripped
        missing = tmp_path / f"missing-{label}.tar"
        replaced = {
            **created,
            "entries": [
                entry for entry in created["entries"] if entry["path"] not in stripped
            ],
        }
        repack(missing, replaced)
        with pytest.raises(ValueError, match=error):
            verify_archive(missing)

    unresolved = {
        **created,
        "external": [
            {
                "kind": "checkpoint",
                "id": "fake",
                "sha256": checkpoint_sha,
                "location": "lost",
                "state": "MISSING_NONRECONSTRUCTABLE",
            }
        ],
    }
    unresolved_path = tmp_path / "portable-with-external.tar"
    repack(unresolved_path, unresolved)
    with pytest.raises(ValueError, match="portable archive cannot have unresolved"):
        verify_archive(unresolved_path)

    weight = next(
        item
        for item in created["entries"]
        if item["role"] == "checkpoint" and not item["path"].endswith("/manifest.json")
    )
    with tarfile.open(output) as original_tar:
        member = original_tar.extractfile(weight["path"])
        assert member is not None
        original_bytes = member.read()
    tampered_bytes = bytes([original_bytes[0] ^ 1]) + original_bytes[1:]
    for relabel in (False, True):
        changed = {
            **created,
            "entries": [
                {
                    **item,
                    **(
                        {"sha256": hashlib.sha256(tampered_bytes).hexdigest()}
                        if item["path"] == weight["path"]
                        else {}
                    ),
                    **(
                        {"role": "run_manifest"}
                        if relabel
                        and item["path"].startswith("artifacts/checkpoint/")
                        and item["path"].endswith("/manifest.json")
                        else {}
                    ),
                }
                for item in created["entries"]
            ],
        }
        tampered = tmp_path / f"repacked-weights-{relabel}.tar"
        repack(tampered, changed, {weight["path"]: tampered_bytes})
        with pytest.raises(ValueError, match="relabel|domain member identity mismatch"):
            verify_archive(tampered)


def test_lock_archive_requires_authenticated_availability_even_with_updated_index(
    tmp_path: Path,
) -> None:
    import hashlib

    from sparselab.training.manifest import canonical_json

    lock_name = "evidence/lock/" + "a" * 64 + ".json"
    binding_name = lock_name.removesuffix(".json") + ".availability.json"
    lock = canonical_json({"plan_sha256": "a" * 64}) + b"\n"
    invalid_binding = (
        canonical_json(
            {
                "plan_sha256": "a" * 64,
                "availability": {},
                "sha256": "b" * 64,
            }
        )
        + b"\n"
    )
    for filename, members in (
        ("without-sidecar.tar", {lock_name: ("experiment_lock", lock)}),
        (
            "changed-sidecar.tar",
            {
                lock_name: ("experiment_lock", lock),
                binding_name: ("lock_availability", invalid_binding),
            },
        ),
    ):
        path = tmp_path / filename
        entries = [
            {
                "path": name,
                "size": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
                "role": role,
            }
            for name, (role, value) in sorted(members.items())
        ]
        index = {
            "format": "archive-index-v1",
            "mode": "thin",
            "family_lineage": "NOT_DECLARED",
            "entries": entries,
            "external": [],
        }
        with tarfile.open(path, "w") as tar:
            contents = {
                "archive-index.json": canonical_json(index) + b"\n",
                **{name: value for name, (_, value) in sorted(members.items())},
            }
            for name, value in contents.items():
                info = tarfile.TarInfo(name)
                info.size = len(value)
                tar.addfile(info, io.BytesIO(value))
        with pytest.raises(ValueError):
            verify_archive(path)


@pytest.mark.parametrize("change", ["same-size", "shorter", "longer"])
def test_archive_validates_streamed_source(
    tmp_path: Path, partial_recipe: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    # Mutate a disposable copy after all pre-stream checks, just before TAR
    # consumes the open source. This also reaches the old second-hash gap.
    repo = tmp_path / "repo"
    shutil.copytree(partial_recipe.parents[3], repo)
    source = repo / partial_recipe.relative_to(partial_recipe.parents[3])
    original_addfile = tarfile.TarFile.addfile
    mutations = []

    def addfile(tar, info, fileobj=None):
        if info.name == f"declarations/{source.relative_to(repo).as_posix()}":
            data = source.read_bytes()
            changed = {
                "same-size": bytes([data[0] ^ 1]) + data[1:],
                "shorter": data[:-1],
                "longer": data + b"X",
            }[change]
            source.write_bytes(changed)
            mutations.append(info.name)
        return original_addfile(tar, info, fileobj)

    monkeypatch.setattr(tarfile.TarFile, "addfile", addfile)
    output = tmp_path / "result.tar"
    with pytest.raises((ValueError, OSError), match="changed|unexpected end"):
        create_archive(source, "thin", output, work_root=tmp_path / "state")
    assert len(mutations) == 1
    assert not output.exists()
    assert not list(tmp_path.glob(".result.tar.*.tmp"))


@pytest.mark.parametrize(
    "point",
    [
        "member",
        "tar-complete",
        "file-sync",
        "link-before",
        "link-after",
        "dir-open",
        "dir-sync",
    ],
)
def test_archive_faults_preserve_only_complete_publication(
    tmp_path: Path, partial_recipe: Path, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    import os
    import stat

    output = tmp_path / "result.tar"
    original_addfile = tarfile.TarFile.addfile
    original_close = tarfile.TarFile.close
    original_sync = os.fsync
    original_open = os.open
    original_link = os.link

    def addfile(tar, *args, **kwargs):
        original_addfile(tar, *args, **kwargs)
        if point == "member":
            raise KeyboardInterrupt("injected member")

    def close(tar):
        writing = tar.mode == "w" and not tar.closed
        original_close(tar)
        if point == "tar-complete" and writing:
            raise KeyboardInterrupt("injected tar-complete")

    def open_directory(path, flags, *args, **kwargs):
        if point == "dir-open" and flags & os.O_DIRECTORY:
            raise OSError("injected directory open")
        return original_open(path, flags, *args, **kwargs)

    def sync(fd):
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if point == ("dir-sync" if directory else "file-sync"):
            raise OSError("injected sync")
        return original_sync(fd)

    def link(source, destination):
        assert Path(source).parent == output.parent
        assert Path(source) != output
        # Publication must see a complete, independently verifiable archive.
        verify_archive(Path(source))
        if point == "link-before":
            raise OSError("injected link-before")
        original_link(source, destination)
        if point == "link-after":
            raise KeyboardInterrupt("injected link-after")

    with monkeypatch.context() as patch:
        patch.setattr(tarfile.TarFile, "addfile", addfile)
        patch.setattr(tarfile.TarFile, "close", close)
        patch.setattr(os, "fsync", sync)
        patch.setattr(os, "open", open_directory)
        patch.setattr(os, "link", link)
        with pytest.raises((OSError, KeyboardInterrupt), match="injected") as error:
            create_archive(partial_recipe, "thin", output, work_root=tmp_path / "state")
    if point in {"dir-open", "dir-sync"}:
        assert f"complete archive published at {output}" in str(error.value)
        assert "directory entry may not be durable" in str(error.value)
        assert isinstance(error.value.__cause__, OSError)
    if point in {"link-after", "dir-open", "dir-sync"}:
        assert verify_archive(output)["verified_members"] > 0
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".result.tar.*.tmp"))


@pytest.mark.parametrize("competitor", ["existing", "racing", "dangling-symlink"])
def test_archive_never_clobbers_destination(
    tmp_path: Path,
    partial_recipe: Path,
    monkeypatch: pytest.MonkeyPatch,
    competitor: str,
) -> None:
    import os

    output = tmp_path / "result.tar"
    original_link = os.link
    sentinel = b"competing evidence"
    if competitor == "existing":
        output.write_bytes(sentinel)
    elif competitor == "dangling-symlink":
        output.symlink_to(tmp_path / "absent")
    else:

        def link(source, destination):
            output.write_bytes(sentinel)
            original_link(source, destination)

        monkeypatch.setattr(os, "link", link)
    if competitor != "racing":

        def unexpected_tar(*args, **kwargs):
            pytest.fail("existing destination must be refused before TAR construction")

        monkeypatch.setattr(tarfile.TarFile, "addfile", unexpected_tar)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_archive(partial_recipe, "thin", output, work_root=tmp_path / "state")
    if competitor == "dangling-symlink":
        assert output.is_symlink()
        assert output.readlink() == tmp_path / "absent"
        assert not (tmp_path / "absent").exists()
    else:
        assert output.read_bytes() == sentinel
    assert not list(tmp_path.glob(".result.tar.*.tmp"))


def test_archive_preserves_tar_bytes(tmp_path: Path, partial_recipe: Path) -> None:
    from sparselab.training.manifest import canonical_json

    output = tmp_path / "result.tar"
    index = create_archive(partial_recipe, "thin", output, work_root=tmp_path / "state")
    # Rebuild using the original writer's format, order and metadata.
    expected = io.BytesIO()
    with (
        tarfile.open(output) as actual,
        tarfile.open(fileobj=expected, mode="w|") as tar,
    ):
        contents = [("archive-index.json", canonical_json(index) + b"\n")]
        for entry in index["entries"]:
            member = actual.extractfile(entry["path"])
            assert member is not None
            contents.append((entry["path"], member.read()))
        for name, data in contents:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    assert output.read_bytes() == expected.getvalue()
    ordinary = tmp_path / "ordinary.tar"
    with ordinary.open("xb") as stream:
        stream.write(expected.getvalue())
    assert output.stat().st_mode == ordinary.stat().st_mode
    assert verify_archive(output)["verified_members"] == len(index["entries"])


@pytest.mark.parametrize("point", ["member", "before-link", "after-link"])
def test_archive_process_exit_leaves_final_absent_or_complete(
    tmp_path: Path, partial_recipe: Path, point: str
) -> None:
    import sys

    output = tmp_path / "result.tar"
    script = r"""
import os
import sys
import tarfile
from pathlib import Path
from sparselab.archive import create_archive
source, output, point = sys.argv[1:]
original_addfile = tarfile.TarFile.addfile
original_link = os.link
def addfile(tar, *args, **kwargs):
    original_addfile(tar, *args, **kwargs)
    if point == "member":
        os._exit(47)
def link(source, destination):
    if point == "before-link":
        os._exit(47)
    original_link(source, destination)
    os._exit(47)
tarfile.TarFile.addfile = addfile
os.link = link
create_archive(Path(source), "thin", Path(output), work_root=Path(output).parent / "state")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(partial_recipe), str(output), point],
        timeout=20,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 47, result.stderr
    if point == "after-link":
        assert verify_archive(output)["verified_members"] > 0
    else:
        assert not output.exists()
    # Uncatchable termination may leave a stage, never a partial final name.
    assert len(list(tmp_path.glob(".result.tar.*.tmp"))) == 1


@pytest.mark.parametrize("mask", [0o022, 0o002, 0o077])
def test_archive_preserves_caller_umask(
    tmp_path: Path, partial_recipe: Path, mask: int
) -> None:
    import sys

    # Set umask only in an isolated child; production must never change it.
    script = r"""
import os
import stat
import sys
from pathlib import Path
from sparselab.archive import create_archive, verify_archive
source, destination, mask = sys.argv[1:]
mask = int(mask)
os.umask(mask)
def forbidden_umask(*args):
    raise AssertionError("archive creation must not change process-wide umask")
os.umask = forbidden_umask
output = Path(destination)
create_archive(Path(source), "thin", output, work_root=output.parent / "state")
assert stat.S_IMODE(output.stat().st_mode) == 0o666 & ~mask
assert verify_archive(output)["verified_members"] > 0
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(partial_recipe),
            str(tmp_path / "result.tar"),
            str(mask),
        ],
        check=True,
        timeout=20,
        capture_output=True,
        text=True,
    )


def test_archive_stage_collision_preserves_other_writer(
    tmp_path: Path, partial_recipe: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab import archive

    monkeypatch.setattr(archive.secrets, "token_hex", lambda size: "collision")
    output = tmp_path / "result.tar"
    stage = tmp_path / ".result.tar.collision.tmp"
    stage.write_bytes(b"another writer's stage")
    with pytest.raises(FileExistsError):
        create_archive(partial_recipe, "thin", output, work_root=tmp_path / "state")
    assert stage.read_bytes() == b"another writer's stage"
    assert not output.exists()
