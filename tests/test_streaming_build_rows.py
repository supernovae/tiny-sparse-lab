"""Tiny comparisons against the unchanged native v2 deduplication policy."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.corpus.bounded_metadata import metadata_operation
from sparselab.corpus.large_build import _deduplicate, _index_prepared
from sparselab.corpus.progress import BuildProgress
from sparselab.corpus.split_runs import SplitStages
from sparselab.corpus.streaming_build_rows import StreamBuildRows
from sparselab.training.manifest import canonical_json


def fixture(tmp_path, monkeypatch, *, fault=None, external=False):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    ids = [f"{i:064x}" for i in range(4)]
    choices = {identifier: "train" for identifier in ids}
    if fault == "split":
        choices[ids[2]] = "validation"
    policy = {"schema_version": 1, "unit": "document", "assignments": choices}
    if external:
        encoded = b"".join(
            canonical_json({"document_id": key, "split": value}) + b"\n"
            for key, value in sorted(choices.items())
        )
        digest = hashlib.sha256(encoded).hexdigest()
        (tmp_path / ("assignments-" + digest + ".jsonl")).write_bytes(encoded)
        policy.update(assignments={}, assignment_file={"sha256": digest, "rows": 4})
    project = SimpleNamespace(
        sources=[SimpleNamespace(id="fixture", kind="huggingface_dataset")],
        root=tmp_path,
        config=SimpleNamespace(splits="splits.yaml"),
        splits=SimpleNamespace(model_dump=lambda **_: policy),
        release=SimpleNamespace(schema_version=2),
    )
    shard = tmp_path / "source-shard"
    shard.mkdir()
    documents = []
    # Raw links 0 -> 2, normalized links 2 -> 3: preserve transitive min-ID choice.
    for i in [3, 1, 2, 0]:
        documents.append(
            {
                "document_id": ids[i],
                "source_id": "fixture",
                "source_family": "family",
                "document_kind": "webpage",
                "metadata": {},
                "raw_content_sha256": "a" * 64 if i in {0, 2} else f"{100 + i:064x}",
                "content_sha256": "b" * 64 if i in {2, 3} else f"{200 + i:064x}",
                "text": f"Fixture text {i}",
                "license": "BSD" if fault == "rights" and i == 3 else "MIT",
                "redistribution": "metadata_reconstruction_only",
                "rights": {
                    "training_eligibility": "eligible",
                    "license_references": [],
                    "notices": [],
                    "training_restriction": None,
                },
                "structure": {"decision": "exclude_lm_metadata_only"}
                if i == 1
                else None,
            }
        )
    (shard / "docs.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in documents)
    )
    (shard / "spans.jsonl").write_bytes(
        b"".join(
            canonical_json({"record_id": row["document_id"]}) + b"\n"
            for row in documents
        )
    )
    progress = BuildProgress(tmp_path / "progress.jsonl", "d" * 64)
    stages = SplitStages(tmp_path / "stages", {"producer": "fixture", "case": fault})
    return project, [shard], progress, stages


@pytest.mark.parametrize("external", [False, True])
def test_compact_references_match_native_v2_order_representatives_and_audit(
    tmp_path, monkeypatch, external
):
    project, prepared, progress, stages = fixture(
        tmp_path, monkeypatch, external=external
    )
    with metadata_operation(tmp_path / "metadata"):
        connection = _index_prepared(
            tmp_path / "oracle.sqlite", prepared, project, progress
        )
        try:
            groups = tmp_path / "oracle-groups.jsonl"
            expected = _deduplicate(
                connection,
                groups,
                diagnostics=tmp_path / "diagnostic.json",
                progress=progress,
                schema_version=2,
            )
            native_rows = list(
                connection.execute(
                    "SELECT payload,split,representative_id,drop_reason FROM documents ORDER BY document_id"
                )
            )
            native_spans = [
                row[0]
                for row in connection.execute(
                    "SELECT span FROM documents ORDER BY rowid"
                )
            ]
            streamed = StreamBuildRows.create(stages, prepared, project)
            assert list(streamed.documents()) == native_rows
            assert list(streamed.spans()) == native_spans
            assert (
                Path(streamed.deduplicated["groups"]).read_bytes()
                == groups.read_bytes()
            )
            assert (
                streamed.deduplicated["group_count"],
                streamed.deduplicated["dropped"],
            ) == expected
            assert Path(streamed.deduplicated["parents"]).stat().st_size == 8 * 4
            assert not list(stages.root.rglob("*.sqlite"))
        finally:
            connection.close()


@pytest.mark.parametrize(
    "fault,message",
    [
        ("split", "cross-split exact text overlap"),
        ("rights", "duplicate source text has incompatible rights evidence"),
    ],
)
def test_compact_dedup_refuses_native_v2_conflicts(
    tmp_path, monkeypatch, fault, message
):
    project, prepared, progress, stages = fixture(tmp_path, monkeypatch, fault=fault)
    with metadata_operation(tmp_path / "metadata"):
        connection = _index_prepared(
            tmp_path / "oracle.sqlite", prepared, project, progress
        )
        try:
            with pytest.raises(ValueError, match=message):
                _deduplicate(
                    connection,
                    tmp_path / "groups.jsonl",
                    diagnostics=tmp_path / "error.json",
                    progress=progress,
                    schema_version=2,
                )
            with pytest.raises(ValueError, match=message):
                StreamBuildRows.create(stages, prepared, project)
        finally:
            connection.close()


def test_interruption_reuses_committed_reference_and_join_phases(tmp_path, monkeypatch):
    project, prepared, _progress, stages = fixture(tmp_path, monkeypatch)
    original = SplitStages.phase

    def interrupt(self, name, produce):
        if name == "build-deduplicate":
            raise RuntimeError("fixture interruption")
        return original(self, name, produce)

    monkeypatch.setattr(SplitStages, "phase", interrupt)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        StreamBuildRows.create(stages, prepared, project)
    monkeypatch.setattr(SplitStages, "phase", original)
    from sparselab.corpus import streaming_build_rows as module

    monkeypatch.setattr(
        module, "_references", lambda *_: pytest.fail("reference scan repeated")
    )
    resumed = SplitStages(tmp_path / "stages", {"producer": "fixture", "case": None})
    result = StreamBuildRows.create(resumed, prepared, project)
    assert "build-document-references" in resumed.reused
    assert "build-join-splits" in resumed.reused
    assert len(list(result.documents())) == 4
