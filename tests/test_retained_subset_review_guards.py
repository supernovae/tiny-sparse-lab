"""Removal-sensitive retained-subset refusal checks from PR204 review."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_corpus_retained_subset import fixture

from sparselab.corpus import retained_subset
from sparselab.corpus.acquisition import (
    _digest,
    acquire,
    declaration_sha256,
    verify_snapshot,
)
from sparselab.corpus.project import Project, SourceDeclaration
from sparselab.training.manifest import canonical_json, sha256_file


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("max_rows", 30001, "bounded row/byte limits"),
        ("max_bytes", 100 * 1024**2 + 1, "bounded row/byte limits"),
        ("snapshot_path", "relative", "must be absolute"),
        ("snapshot_sha256", "a" * 63, "full SHA256 pins"),
        ("manifest_sha256", "z" * 64, "full SHA256 pins"),
        ("profile_id", " ", "must not be blank"),
        ("seed", True, "valid integer"),
        ("seed", 1.0, "valid integer"),
        ("seed", "1", "valid integer"),
        ("max_record_bytes", 262145, "less than or equal"),
        ("max_record_bytes", True, "valid integer"),
        ("max_record_bytes", 0, "greater than 0"),
        ("max_record_bytes", -1, "greater than 0"),
        ("max_record_bytes", 1.0, "valid integer"),
        ("max_record_bytes", float("nan"), "valid integer"),
        ("max_record_bytes", float("inf"), "valid integer"),
        ("shards", None, "one shard"),
    ],
)
def test_declaration_refusals(tmp_path, monkeypatch, field, value, reason):
    project, _ = fixture(tmp_path, monkeypatch)
    spec = project.model_dump(mode="json")
    acquisition = spec["sources"][0]["acquisition"]
    if field == "shards":
        second = copy.deepcopy(acquisition["bounded_shards"][0])
        second["path"] = "default/second.jsonl"
        acquisition["bounded_shards"].append(second)
    elif field in {"max_rows", "max_bytes"}:
        acquisition[field] = value
    else:
        acquisition["retained_subset"][field] = value
    with pytest.raises(ValueError, match=reason):
        Project.model_validate(spec)


@pytest.mark.parametrize("fault", ["mixed", "transport", "effects", "bytes"])
def test_direct_project_gate_before_work_creation(tmp_path, monkeypatch, fault):
    project, _ = fixture(tmp_path, monkeypatch)
    spec = project.model_dump(mode="json")
    if fault in {"mixed", "bytes"}:
        second = copy.deepcopy(spec["sources"][0])
        second["id"] = "second"
        if fault == "mixed":
            second["acquisition"].pop("retained_subset")
        else:
            spec["sources"][0]["acquisition"]["max_bytes"] = 60 * 1024**2
            second["acquisition"]["max_bytes"] = 60 * 1024**2
        spec["sources"].append(second)
        spec["config"]["sources"].append("sources/second.yaml")
        spec["config"]["sources"].sort()
    elif fault == "transport":
        spec["config"]["transport_budget"] = {
            "attempt_id": "fixture",
            "max_source_body_bytes": 1000,
            "max_metadata_body_bytes": 1000,
            "max_transfers": 10,
            "max_retries_per_shard": 1,
            "max_wall_seconds": 100,
            "max_disk_bytes": 10000,
        }
    else:
        spec["config"]["source_effects"] = [
            {
                "source_id": "one",
                "effect": "acquire",
                "declaration_sha256": declaration_sha256(project.sources[0]),
            }
        ]
    work = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="aggregate selection caps"):
        acquire(Project.model_validate(spec), work)
    assert not work.exists()


def reseal_parent(project, parent, mutate):
    """Construct an independently pinned malformed fixture, retaining the origin."""
    manifest = json.loads((parent / "manifest.json").read_text())
    body = (parent / "files" / manifest["files"][0]["path"]).read_bytes()
    manifest, body = mutate(manifest, body)
    file = manifest["files"][0]
    file.update(size=len(body), sha256=hashlib.sha256(body).hexdigest())
    manifest["declaration_sha256"] = declaration_sha256(
        SourceDeclaration.model_validate(manifest["declaration"])
    )
    manifest["snapshot_sha256"] = _digest(
        {k: manifest[k] for k in ("declaration_sha256", "adapter", "files")}
    )
    target = parent.parent / "resealed" / manifest["snapshot_sha256"]
    (target / "files" / file["path"]).parent.mkdir(parents=True)
    (target / "files" / file["path"]).write_bytes(body)
    (target / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    spec = project.model_dump(mode="json")
    child = spec["sources"][0]
    # Keep child policy consistent when the test specifically changes a parent bound.
    if manifest["declaration"]["kind"] == "huggingface_dataset":
        caps = {
            k: child["acquisition"][k]
            for k in ("max_rows", "max_bytes", "retained_subset")
        }
        child.update(copy.deepcopy(manifest["declaration"]))
        child["acquisition"].update(caps)
    child["acquisition"]["retained_subset"].update(
        snapshot_path=str(target),
        snapshot_sha256=manifest["snapshot_sha256"],
        manifest_sha256=sha256_file(target / "manifest.json"),
    )
    return Project.model_validate(spec), target


def refused(project, work, reason):
    with pytest.raises(ValueError, match=reason):
        acquire(project, work)
    assert not list(work.rglob("acquisition.json"))
    assert not list(work.rglob(".acquire-*"))


def test_wrong_snapshot_pin_with_correct_manifest_pin(tmp_path, monkeypatch):
    project, _ = fixture(tmp_path, monkeypatch)
    spec = project.model_dump(mode="json")
    spec["sources"][0]["acquisition"]["retained_subset"]["snapshot_sha256"] = "0" * 64
    refused(
        Project.model_validate(spec), tmp_path / "work", "snapshot identity mismatch"
    )


def test_manifest_input_cap_precedes_hashing(tmp_path, monkeypatch):
    project, parent = fixture(tmp_path, monkeypatch)
    with (parent / "manifest.json").open("r+b") as stream:
        stream.truncate(
            256 * 1024**2 + 1
        )  # Sparse owned fixture; never allocate this in RAM.
    monkeypatch.setattr(
        retained_subset,
        "sha256_file",
        lambda *_: pytest.fail("oversized manifest was hashed"),
    )
    refused(project, tmp_path / "work", "metadata input-byte cap exceeded")


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("kind", "input-byte cap or source shape"),
        ("files", "input-byte cap or source shape"),
        ("size", "input-byte cap or source shape"),
        ("shards", "requires one parent shard"),
        ("extension", "requires parent JSONL"),
        ("output_path", "requires parent JSONL"),
    ],
)
def test_parent_shape_refusals(tmp_path, monkeypatch, fault, reason):
    project, parent = fixture(tmp_path, monkeypatch)

    def mutate(manifest, body):
        if fault == "kind":
            manifest["declaration"].update(
                kind="git", canonical_uri="https://github.com/example/fixture"
            )
            manifest["declaration"]["acquisition"] = {
                "include": ["*.jsonl"],
                "max_bytes": 1024**2,
            }
            manifest["adapter"]["id"] = "git"
        elif fault == "files":
            manifest["files"].append(
                {
                    "path": "second.jsonl",
                    "size": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                }
            )
        elif fault == "size":
            manifest["declaration"]["acquisition"]["max_bytes"] = len(body) - 1
        elif fault == "shards":
            manifest["retrieval"]["shards"].append(
                copy.deepcopy(manifest["retrieval"]["shards"][0])
            )
        elif fault == "extension":
            manifest["files"][0]["path"] = "fixture.bin"
            manifest["retrieval"]["shards"][0]["output_path"] = "fixture.bin"
        else:
            manifest["retrieval"]["shards"][0]["output_path"] = "other.jsonl"
        return manifest, body

    project, _ = reseal_parent(project, parent, mutate)
    refused(project, tmp_path / "work", reason)


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("drop", "row inventory mismatch"),
        ("add", "row inventory mismatch"),
        ("envelope", "row ancestry mismatch"),
        ("index", "row ancestry mismatch"),
        ("row_sha", "row ancestry mismatch"),
        ("receipt_sha", "row ancestry mismatch"),
        ("shard_path", "row ancestry mismatch"),
        ("shard_sha", "row ancestry mismatch"),
        ("revision", "row ancestry mismatch"),
        ("row_cap", "parent row cap exceeded"),
        ("lf", "must end with LF"),
    ],
)
def test_resealed_parent_row_refusals(tmp_path, monkeypatch, fault, reason):
    project, parent = fixture(tmp_path, monkeypatch)

    def mutate(manifest, body):
        rows = [json.loads(line) for line in body.splitlines()]
        envelope = rows[0]["_sparselab_source"]
        if fault == "drop":
            rows.pop()
        elif fault == "add":
            rows.append(rows[-1])
        elif fault == "envelope":
            rows[0]["_sparselab_source"] = None
        elif fault == "index":
            envelope["source_row_index"] += 1
        elif fault == "row_sha":
            envelope["source_row_sha256"] = "0" * 64
        elif fault == "receipt_sha":
            manifest["retrieval"]["shards"][0]["selected_rows"][0][
                "source_row_sha256"
            ] = "0" * 64
        elif fault == "shard_path":
            envelope["source_shard_path"] = "wrong.jsonl"
        elif fault == "shard_sha":
            envelope["source_shard_sha256"] = "0" * 64
        elif fault == "revision":
            envelope["dataset_revision"] = "e" * 40
        elif fault == "row_cap":
            manifest["declaration"]["acquisition"]["max_rows"] = 1
        body = b"".join(canonical_json(row) + b"\n" for row in rows)
        return manifest, body.rstrip(b"\n") if fault == "lf" else body

    project, target = reseal_parent(project, parent, mutate)
    if fault == "lf":
        # Find a fixture-only profile that selects the unterminated last row.
        spec = project.model_dump(mode="json")
        retained = spec["sources"][0]["acquisition"]["retained_subset"]
        receipt = json.loads((target / "manifest.json").read_text())["retrieval"][
            "shards"
        ][0]["selected_rows"][-1]
        for number in range(2000):
            label = f"unterminated-fixture-{number}"
            rank = hashlib.sha256(
                canonical_json(
                    [
                        label,
                        retained["seed"],
                        "one",
                        retained["snapshot_sha256"],
                        receipt["source_row_index"],
                        receipt["source_row_sha256"],
                    ]
                )
            ).hexdigest()
            if int(rank, 16) % 100 == 0:
                retained["profile_id"] = label
                break
        else:
            pytest.fail("no selecting fixture label")
        project = Project.model_validate(spec)
    refused(project, tmp_path / "work", reason)


@pytest.mark.parametrize("fault", ["file", "manifest"])
def test_parent_changes_after_last_yield(tmp_path, monkeypatch, fault):
    project, parent = fixture(tmp_path, monkeypatch)
    original = retained_subset.records_from_path

    def changed(path, **kwargs):
        yield from original(path, **kwargs)
        target = path if fault == "file" else parent / "manifest.json"
        with target.open("ab") as stream:
            stream.write(b" ")

    monkeypatch.setattr(retained_subset, "records_from_path", changed)
    refused(project, tmp_path / "work", "parent changed during selection")


@pytest.mark.parametrize("fault", ["files", "retrieval"])
def test_independent_cold_comparison_halves(tmp_path, monkeypatch, fault):
    project, _ = fixture(tmp_path, monkeypatch)
    child = Path(acquire(project, tmp_path / "work")["sources"]["one"]["snapshot_path"])
    manifest = json.loads((child / "manifest.json").read_text())
    if fault == "files":
        path = child / "files" / manifest["files"][0]["path"]
        body = path.read_bytes().replace(b"Retained", b"RetAined", 1)
        path.write_bytes(body)
        manifest["files"][0]["sha256"] = sha256_file(path)
    else:
        manifest["retrieval"]["retained_subset"]["excluded_long_rows"] += 1
    manifest["snapshot_sha256"] = _digest(
        {
            k: manifest[k]
            for k in ("declaration_sha256", "adapter", "files", "retrieval")
        }
    )
    (child / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    with pytest.raises(ValueError, match="cold parent replay"):
        verify_snapshot(child, _staged=True)


def test_exact_partial_long_row_exclusions(tmp_path, monkeypatch):
    project, parent = fixture(tmp_path, monkeypatch)

    def varied(manifest, body):
        rows = [json.loads(line) for line in body.splitlines()]
        for index, row in enumerate(rows):
            row["text"] += "x" * (512 * (index % 2))
            digest = hashlib.sha256(
                canonical_json(
                    {k: v for k, v in row.items() if k != "_sparselab_source"}
                )
            ).hexdigest()
            row["_sparselab_source"]["source_row_sha256"] = digest
            manifest["retrieval"]["shards"][0]["selected_rows"][index][
                "source_row_sha256"
            ] = digest
        return manifest, b"".join(canonical_json(row) + b"\n" for row in rows)

    project, parent = reseal_parent(project, parent, varied)
    full = Path(acquire(project, tmp_path / "full")["sources"]["one"]["snapshot_path"])
    manifest = verify_snapshot(full)
    lines = (
        (full / "files" / manifest["files"][0]["path"])
        .read_bytes()
        .splitlines(keepends=True)
    )
    limit = min(map(len, lines))
    kept = [line for line in lines if len(line) <= limit]
    assert 0 < len(kept) < len(lines)
    spec = project.model_dump(mode="json")
    spec["sources"][0]["acquisition"]["retained_subset"]["max_record_bytes"] = limit
    selected = Path(
        acquire(Project.model_validate(spec), tmp_path / "limited")["sources"]["one"][
            "snapshot_path"
        ]
    )
    result = verify_snapshot(selected)
    assert result["retrieval"]["retained_subset"]["excluded_long_rows"] == len(
        lines
    ) - len(kept)
    assert (selected / "files" / result["files"][0]["path"]).read_bytes() == b"".join(
        kept
    )


def test_plain_acquire_and_cold_verify_scratch_stays_in_work_root(
    tmp_path, monkeypatch
):
    from sparselab.corpus import bounded_metadata

    project, _ = fixture(tmp_path, monkeypatch)
    work = tmp_path / "work"
    original = bounded_metadata.MetadataStore.__init__
    directories = []

    def checked(self, directory, **kwargs):
        directories.append(Path(directory))
        assert Path(directory).is_relative_to(work)
        original(self, directory, **kwargs)

    monkeypatch.setattr(bounded_metadata.MetadataStore, "__init__", checked)
    child = Path(acquire(project, work)["sources"]["one"]["snapshot_path"])
    verify_snapshot(child)
    assert directories
