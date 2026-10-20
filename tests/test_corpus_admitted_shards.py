"""Tiny verified v2 admission through the existing streaming shard preparer."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_corpus_row_qualification import code_policy, code_row, declared_fixture

from sparselab.corpus.bounded_metadata import metadata_operation, read_metadata
from sparselab.corpus.large_build import _prepare_file, _verify_prepared
from sparselab.corpus.pipeline import _records_for_file
from sparselab.corpus.progress import BuildProgress
from sparselab.corpus.project import SourceDeclaration
from sparselab.corpus.rights import verify_record_admission
from sparselab.corpus.row_qualification import RowUsePolicy, use_evidence
from sparselab.training.manifest import canonical_json


def fixture(root, profile):
    options = {}
    if profile == "web":
        options = {
            "policy": {"profile": "web_text", "family_fields": ["metadata.url"]},
            "rows": [
                {
                    "text": "Ordinary explanatory fixture prose. " * 5,
                    "metadata": {"url": "https://example.invalid/a"},
                },
                {"text": "Missing provenance remains quarantined. " * 5},
            ],
        }
    if profile == "code":
        options = {
            "policy": code_policy().model_dump(mode="json"),
            "rows": [
                {**code_row(), "license": None},
                {**code_row(), "code": "print(43)"},
            ],
        }
    manifest, source, snapshot, raw = declared_fixture(root, **options)
    manifest["schema_version"] = 2
    if profile == "code":
        source["acquisition"]["text_field"] = "code"
        policy = RowUsePolicy(
            mode="record_attribution",
            restricted_rows={
                hashlib.sha256(canonical_json(options["rows"][1])).hexdigest(): {
                    "basis": "Explicit fixture training restriction",
                    "evidence": "Retained fixture notice",
                }
            },
        )
        entry = manifest["sources"][0]
        entry["row_use_policy"] = policy.model_dump(mode="json")
        for choice, row in zip(entry["records"], options["rows"], strict=True):
            choice["row_evidence"] = use_evidence(row, code_policy(), "code", policy)
        entry["records"][1].update(
            decision="exclude", reason="explicit training-use restriction on source row"
        )
    # Physical blank lines must not change the parser's nonblank ordinal.
    raw = b"\n" + b"\n".join(raw.splitlines(keepends=True))
    file = snapshot["files"][0]
    file.update(sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))
    manifest["sources"][0]["sample_sha256"] = file["sha256"]
    path = root / source["id"] / snapshot["snapshot_sha256"] / "files" / file["path"]
    path.write_bytes(raw)
    return manifest, source, snapshot, raw, path


def kwargs(root, source, snapshot, path, admitted, digest):
    return {
        "build_id": "e" * 64,
        "prepared_root": root / "prepared",
        "source": SourceDeclaration.model_validate(source),
        "file": snapshot["files"][0],
        "snapshot_sha": snapshot["snapshot_sha256"],
        "source_path": path,
        "decision": admitted["rights"],
        "record_admission": admitted,
        "admission_sha256": digest,
        "progress": BuildProgress(root / "progress.jsonl", "e" * 64),
    }


@pytest.mark.parametrize("profile", ["synthetic", "web", "code"])
def test_streamed_admitted_shard_matches_native_v2_bytes(
    tmp_path, monkeypatch, profile
):
    manifest, source, snapshot, raw, path = fixture(tmp_path, profile)
    declaration = tmp_path / "admission.json"
    declaration.write_bytes(canonical_json(manifest))
    digest = hashlib.sha256(declaration.read_bytes()).hexdigest()
    with metadata_operation(tmp_path / "scratch"):
        admitted = verify_record_admission(
            read_metadata(declaration),
            {source["id"]: source},
            {source["id"]: snapshot},
            tmp_path,
        )[source["id"], snapshot["files"][0]["path"]]
        args = kwargs(tmp_path, source, snapshot, path, admitted, digest)
        rejected = []
        pairs = _records_for_file(
            raw,
            args["file"]["path"],
            args["source"],
            args["snapshot_sha"],
            file_rights=admitted["rights"],
            record_admission=admitted,
            rejected_records=rejected,
        )
        real_read = Path.read_bytes

        def forbid_shard_read(self):
            if self == path:
                raise AssertionError("whole shard read")
            return real_read(self)

        monkeypatch.setattr(Path, "read_bytes", forbid_shard_read)
        output = _prepare_file(**args)
        for name, rows in [
            ("docs.jsonl", [x[0] for x in pairs]),
            ("spans.jsonl", [x[1] for x in pairs]),
            ("rejected.jsonl", rejected),
        ]:
            assert (output / name).read_bytes() == b"".join(
                canonical_json(row) + b"\n" for row in rows
            )
        assert len(pairs) == len(rejected) == 1
        if profile == "code":
            assert "exclude" in rejected[0]["reason"]
            assert (
                json.loads(pairs[0][0]["metadata"]["row_attribution_json"])[
                    "license_label"
                ]
                is None
            )
        else:
            assert "quarantine" in rejected[0]["reason"]
        receipt = json.loads((output / "receipt.json").read_text())
        assert receipt["identity"]["admission_sha256"] == digest
        assert _prepare_file(**args) == output
        # A different complete reviewed artifact cannot reuse this receipt.
        other = _prepare_file(**{**args, "admission_sha256": "f" * 64})
        assert other != output
        (output / "docs.jsonl").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="changed"):
            _verify_prepared(output, receipt["identity"])
        with pytest.raises(ValueError, match="changed"):
            _prepare_file(**args)
        assert (output / "docs.jsonl").read_bytes() == b"tampered"


@pytest.mark.parametrize(
    "fault",
    [
        "missing_decision",
        "missing_digest",
        "changed_bytes",
        "oversized_record",
        "row_digest",
    ],
)
def test_admitted_shard_refusals_leave_no_published_shard(tmp_path, monkeypatch, fault):
    from sparselab.corpus import large_build

    manifest, source, snapshot, raw, path = fixture(tmp_path, "synthetic")
    if fault == "row_digest":
        manifest["sources"][0]["records"][0]["source_row_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="provenance"):
            verify_record_admission(
                manifest, {source["id"]: source}, {source["id"]: snapshot}, tmp_path
            )
        assert not (tmp_path / "prepared").exists()
        return
    admitted = verify_record_admission(
        manifest, {source["id"]: source}, {source["id"]: snapshot}, tmp_path
    )[source["id"], snapshot["files"][0]["path"]]
    args = kwargs(
        tmp_path,
        source,
        snapshot,
        path,
        admitted,
        hashlib.sha256(canonical_json(manifest)).hexdigest(),
    )
    if fault == "missing_decision":
        admitted["decisions"] = deepcopy(admitted["decisions"])
        del admitted["decisions"][0]
    elif fault == "missing_digest":
        args["admission_sha256"] = None
    elif fault == "changed_bytes":
        path.write_bytes(raw.replace(b"Generated", b"Corrupted"))
    else:
        monkeypatch.setattr(large_build, "_ADMITTED_RECORD_MAX_BYTES", 32)
    with pytest.raises(ValueError):
        _prepare_file(**args)
    assert not list((tmp_path / "prepared").rglob("receipt.json"))
    assert not list((tmp_path / "prepared").rglob(".incomplete-*"))
