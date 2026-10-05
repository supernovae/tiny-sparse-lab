"""Behavioral checks for the resumable, bounded-memory LM-only builder."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from test_corpus_rights_integration import _prospective

from sparselab.corpus import large_build
from sparselab.corpus.acquisition import acquire
from sparselab.corpus.large_build import (
    _deduplicate,
    _index_prepared,
    _prepare_file,
    _prepare_shard,
)
from sparselab.corpus.pipeline import _records_for_file, build
from sparselab.corpus.progress import BuildProgress
from sparselab.corpus.project import Project, load_project
from sparselab.corpus.release import freeze, verify_release
from sparselab.corpus.rights import resolve_file_rights
from sparselab.training.manifest import sha256_file


def test_v3_lm_builder_freezes_and_reuses_verified_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _prospective(tmp_path).model_dump(mode="json")
    for source in spec["sources"]:
        source["schema_version"] = 3
        source["explicit_training_restriction"] = "none_found"
    spec["transforms"] = [
        item for item in spec["transforms"] if item["kind"] == "lm_text"
    ]
    spec["release"].update(
        schema_version=3,
        training_use_policy="allowed_unless_explicitly_prohibited",
        chat={"selected": False, "training_splits": []},
    )
    project = Project.model_validate(spec)
    acquire(project, tmp_path)
    with monkeypatch.context() as interruption:

        def fail_after_preparation(*args: object) -> None:
            raise RuntimeError("interrupted after verified source preparation")

        interruption.setattr(large_build, "_index_prepared", fail_after_preparation)
        with pytest.raises(RuntimeError, match="interrupted after verified"):
            build(project, tmp_path, offline=True)
    root = build(project, tmp_path, offline=True)
    release = freeze(root, tmp_path)
    verify_release(release)
    report = json.loads((root / "report.json").read_text())
    assert report["view_counts"]["lm"]["train"] > 0
    assert report["rights"]["unresolved_rights_files"] >= 1
    assert build(project, tmp_path, offline=True) == root
    progress = root.parent.parent / "progress" / f"{root.name}.jsonl"
    events = [json.loads(line) for line in progress.read_text().splitlines()]
    assert events[-1]["event"] == "complete"
    assert events[-1]["peak_rss_bytes"] > 0
    assert any(event["event"] == "reused_verified_shard" for event in events)


def test_streamed_and_legacy_lm_outputs_have_same_hashes(tmp_path: Path) -> None:
    spec = _prospective(tmp_path).model_dump(mode="json")
    for source in spec["sources"]:
        source["schema_version"] = 3
        source["explicit_training_restriction"] = "none_found"
        if source["id"] == "sample_docs":
            empty = tmp_path / "empty-record.md"
            empty.write_bytes(b"")
            source["acquisition"]["files"].append(
                {"path": str(empty), "name": "empty-record.md"}
            )
    lm = next(item for item in spec["transforms"] if item["kind"] == "lm_text")
    lexical = next(
        item for item in spec["transforms"] if item["kind"] == "lexical_candidates"
    )
    spec["transforms"] = [lm]
    spec["release"].update(
        schema_version=3,
        training_use_policy="allowed_unless_explicitly_prohibited",
        chat={"selected": False, "training_splits": []},
    )
    streamed = Project.model_validate(spec)
    acquire(streamed, tmp_path / "streamed")
    new = build(streamed, tmp_path / "streamed", offline=True)

    # An additional transform selects the existing legacy build path without
    # changing source normalization or the LM transform's selected records.
    spec["transforms"] = [lm, lexical]
    legacy = Project.model_validate(spec)
    acquire(legacy, tmp_path / "legacy")
    old = build(legacy, tmp_path / "legacy", offline=True)
    for name in (
        "documents.jsonl",
        "spans.jsonl",
        "rejected.jsonl",
        "audit.json",
        "splits.json",
        "sources.json",
        "license-report.json",
        "lm/train.jsonl",
        "lm/validation.jsonl",
        "lm/test.jsonl",
        "lm/train.lineage.jsonl",
        "lm/validation.lineage.jsonl",
        "lm/test.lineage.jsonl",
        f"stages/{lm['id']}.jsonl",
    ):
        assert sha256_file(new / name) == sha256_file(old / name), name
    assert any(
        row["path"] == "empty-record.md" and row["reason"] == "empty input"
        for row in (
            json.loads(line)
            for line in (new / "rejected.jsonl").read_text().splitlines()
        )
    )
    legacy_document_lineage = [
        line
        for line in (old / "lineage.jsonl").read_bytes().splitlines(keepends=True)
        if json.loads(line)["record_kind"] == "document"
    ]
    assert (new / "lineage.jsonl").read_bytes() == b"".join(legacy_document_lineage)
    old_report = json.loads((old / "report.json").read_text())
    new_report = json.loads((new / "report.json").read_text())
    for key in ("split_counts", "source_counts", "source_scale", "train_source_scale"):
        assert new_report[key] == old_report[key]


def test_hf_rows_are_streamed_with_original_ids_and_verified_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = load_project("corpora/devmind-v2/corpus.yaml")
    source = next(s for s in project.sources if s.id == "fineweb_edu_sample_10bt")
    source = source.model_copy(
        update={"acquisition": source.acquisition.model_copy(update={"max_rows": 3})}
    )
    path = tmp_path / "sample.jsonl"
    with path.open("wb") as output:
        for index in range(5):
            output.write(
                json.dumps(
                    {
                        "text": f"Educational article {index} "
                        + "reproducible text " * 20,
                        "url": f"https://example.org/articles/{index}",
                    }
                ).encode()
                + b"\n"
            )
    file = {"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
    decision = resolve_file_rights(
        source.rights, path.name, b"", prospective_private_research=True
    )
    expected = _records_for_file(
        path.read_bytes(), path.name, source, "0" * 64, file_rights=decision
    )
    original_read = Path.read_bytes

    def no_source_read(self: Path) -> bytes:
        if self == path:
            raise AssertionError("materialized HF snapshot")
        return original_read(self)

    monkeypatch.setattr(Path, "read_bytes", no_source_read)
    progress = BuildProgress(tmp_path / "progress.jsonl", "shard-check")
    kwargs = {
        "build_id": "shard-check",
        "prepared_root": tmp_path / "prepared",
        "source": source,
        "file": file,
        "snapshot_sha": "0" * 64,
        "source_path": path,
        "decision": decision,
        "progress": progress,
    }
    prepared = _prepare_file(**kwargs)
    docs = [
        json.loads(line) for line in (prepared / "docs.jsonl").read_text().splitlines()
    ]
    spans = [
        json.loads(line) for line in (prepared / "spans.jsonl").read_text().splitlines()
    ]
    assert list(zip(docs, spans, strict=True)) == expected
    assert len(docs) == 3
    assert _prepare_file(**kwargs) == prepared
    assert any(
        json.loads(line)["event"] == "reused_verified_shard"
        for line in (tmp_path / "progress.jsonl").read_text().splitlines()
    )
    parallel_kwargs = {key: value for key, value in kwargs.items() if key != "progress"}
    parallel_kwargs["prepared_root"] = tmp_path / "parallel-prepared"
    with ProcessPoolExecutor(max_workers=2) as executor:
        result = executor.submit(_prepare_shard, **parallel_kwargs).result()
    child, documents, input_bytes, output_records, reused = result
    assert (documents, input_bytes, output_records, reused) == (
        3,
        file["size"],
        3,
        False,
    )
    for name in ("docs.jsonl", "spans.jsonl", "rejected.jsonl"):
        assert sha256_file(child / name) == sha256_file(prepared / name)
    with (prepared / "docs.jsonl").open("ab") as output:
        output.write(b"{}\n")
    with pytest.raises(ValueError, match="prepared source shard changed"):
        _prepare_file(**kwargs)


def test_shard_process_capacity_requires_memory_and_independent_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psutil

    class Memory:
        total = 16 * 1024**3
        available = 2 * 512 * 1024**2

    monkeypatch.setattr(psutil, "virtual_memory", Memory)
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 8 if logical else 4)

    class Affinity:
        def cpu_affinity(self):
            return list(range(4))

    monkeypatch.setattr(psutil, "Process", Affinity)
    assert large_build._process_shard_workers(2) == 1
    Memory.available = 8 * 1024**3
    assert large_build._process_shard_workers(1) == 1
    assert large_build._process_shard_workers(2) == 2


def test_sqlite_dedup_keeps_heldout_and_rejects_validation_test_overlap(
    tmp_path: Path,
) -> None:
    project = load_project("corpora/devmind-v2/corpus.yaml")
    shard = tmp_path / "prepared"
    shard.mkdir()
    records = [
        (
            "train-page",
            "fineweb_edu_sample_10bt",
            "fineweb_edu_sample_10bt_train",
            "Draft text one",
            "https://example.org/a?utm_source=crawl",
        ),
        (
            "validation-page",
            "v2_git_technical",
            "v2_git_technical_validation",
            "Edited text on a page",
            "https://example.org/a",
        ),
        (
            "independent",
            "fineweb_edu_sample_10bt",
            "fineweb_edu_sample_10bt_train",
            "Independent item",
            "https://example.org/b",
        ),
        (
            "test-exact",
            "v2_prometheus_promql",
            "v2_prometheus_promql_test",
            "Identical secret example",
            "https://example.org/c",
        ),
        (
            "train-exact",
            "fineweb_edu_sample_10bt",
            "fineweb_edu_sample_10bt_train",
            "Identical secret example",
            "https://example.org/d",
        ),
    ]

    def write_rows(values: list[tuple[str, str, str, str, str]]) -> None:
        with (
            (shard / "docs.jsonl").open("w") as docs,
            (shard / "spans.jsonl").open("w") as spans,
        ):
            for identifier, source, family, text, url in values:
                content_hash = hashlib.sha256(text.encode()).hexdigest()
                docs.write(
                    json.dumps(
                        {
                            "document_id": identifier,
                            "source_id": source,
                            "source_family": family,
                            "raw_content_sha256": content_hash,
                            "content_sha256": content_hash,
                            "document_kind": "prose",
                            "metadata": {"url": url},
                            "text": text,
                        }
                    )
                    + "\n"
                )
                spans.write(json.dumps({"record_id": identifier}) + "\n")

    write_rows(records)
    progress = BuildProgress(tmp_path / "progress.jsonl", "dedup-fixture")
    connection = _index_prepared(tmp_path / "rows.sqlite", [shard], project, progress)
    groups, dropped = _deduplicate(
        connection,
        tmp_path / "groups.jsonl",
        diagnostics=tmp_path / "diagnostic.json",
        progress=progress,
    )
    assert groups >= 2
    assert dropped == 2
    assert dict(
        connection.execute("SELECT document_id,drop_reason FROM documents")
    ) == {
        "train-page": "contaminated_heldout",
        "validation-page": None,
        "independent": None,
        "test-exact": None,
        "train-exact": "contaminated_heldout",
    }
    connection.close()

    write_rows(
        records
        + [
            (
                "test-page",
                "v2_prometheus_promql",
                "v2_prometheus_promql_test",
                "Another copy of page a",
                "https://example.org/a",
            )
        ]
    )
    connection = _index_prepared(
        tmp_path / "conflict.sqlite", [shard], project, progress
    )
    with pytest.raises(ValueError, match="cross-split exact text overlap"):
        _deduplicate(
            connection,
            tmp_path / "conflict-groups.jsonl",
            diagnostics=tmp_path / "conflict.json",
            progress=progress,
        )
    assert json.loads((tmp_path / "conflict.json").read_text())["error"] == (
        "cross-split exact text overlap"
    )
    connection.close()


@pytest.mark.parametrize("malformed", [False, True])
def test_process_shard_reduction_preserves_full_build_bytes_and_failure_counters(
    tmp_path: Path, monkeypatch, malformed: bool
) -> None:
    from sparselab.corpus import acquisition

    project = load_project("corpora/devmind-v2/corpus.yaml")
    source = next(s for s in project.sources if s.id == "fineweb_edu_sample_10bt")
    project = project.model_copy(
        update={
            "sources": (source,),
            "transforms": tuple(t for t in project.transforms if t.kind == "lm_text"),
        }
    )
    snapshot = tmp_path / "snapshot"
    (snapshot / "files").mkdir(parents=True)
    row = {
        "text": "Educational reproducible document " + "ordinary source text " * 20,
        "url": "https://example.org/source",
    }
    for name, count in (("a.jsonl", 100), ("b.jsonl", 3)):
        content = b"".join(
            json.dumps({**row, "text": row["text"] + str(i)}).encode() + b"\n"
            for i in range(count)
        )
        if malformed and name == "a.jsonl":
            content = content.splitlines(keepends=True)[0] + b"{malformed\n"
        (snapshot / "files" / name).write_bytes(content)
    files = [
        {"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
        for path in sorted((snapshot / "files").iterdir())
    ]
    manifest = {"files": files}
    monkeypatch.setattr(acquisition, "verify_snapshot", lambda *a, **kw: manifest)
    monkeypatch.setattr(large_build, "_PROCESS_SHARD_MIN_BYTES", 1)
    lock = {
        "sources": {
            source.id: {
                "snapshot_path": str(snapshot),
                "snapshot_sha256": "0" * 64,
            }
        }
    }
    identity = {
        "project_id": project.config.id,
        "release": project.release.model_dump(mode="json"),
        "implementation_sha256": "1" * 64,
    }
    results, counters = [], []
    for workers in (1, 2):
        monkeypatch.setattr(
            large_build,
            "_process_shard_workers",
            lambda count, workers=workers: min(workers, max(1, count)),
        )
        workspace = tmp_path / f"workers-{workers}"
        progress = BuildProgress(workspace / "progress.jsonl", "0" * 64)
        built = large_build.build_large(
            project,
            workspace,
            lock,
            identity,
            "0" * 64,
            workspace / "result",
            progress,
        )
        results.append(
            {
                path.relative_to(built).as_posix(): sha256_file(path)
                for path in sorted(built.rglob("*"))
                if path.is_file()
            }
        )
        counters.append(
            (progress.documents, progress.input_bytes, progress.output_records)
        )
    assert results[0] == results[1]
    assert counters[0] == counters[1]
