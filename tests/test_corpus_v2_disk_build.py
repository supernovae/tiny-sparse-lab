"""Public v2 disk dispatch must preserve native tiny fixture output bytes."""

from test_kml_50m_preparation_sequence import _prior

from sparselab.corpus import large_build, pipeline
from sparselab.corpus.project import load_project
from sparselab.corpus.release import freeze, verify_release


def test_public_v2_disk_build_matches_native_in_memory_outputs(tmp_path, monkeypatch):
    root = tmp_path / "work"
    root.mkdir()
    calls = []
    native = large_build.build_large

    def tracked(*args, **kwargs):
        assert kwargs["record_admission"]
        assert kwargs["admission_path"].is_file()
        output = native(*args, **kwargs)
        calls.append(output)
        return output

    monkeypatch.setattr(large_build, "build_large", tracked)
    release, _, _ = _prior(root)
    assert len(calls) == 1
    disk = calls[0]
    project = load_project(root / "fixture-prior" / "build.yaml")
    assert pipeline.build(project, root, offline=True) == disk
    assert len(calls) == 1  # Immutable verified reuse, not a second normalization.
    monkeypatch.setattr(pipeline, "_uses_disk_lm_builder", lambda *_: False)
    memory = pipeline.build(project, root, offline=True)
    assert memory != disk
    for name in [
        "record-admission.json",
        "documents.jsonl",
        "spans.jsonl",
        "lineage.jsonl",
        "sources.json",
        "license-report.json",
        "rejected.jsonl",
        "audit.json",
        "report.json",
        "report.md",
        "splits.json",
        *[
            str(path.relative_to(disk))
            for directory in ("lm", "chat", "stages")
            for path in (disk / directory).glob("*.jsonl")
        ],
    ]:
        assert (disk / name).read_bytes() == (memory / name).read_bytes(), name
    assert verify_release(release)["release_id"] == release.name
    assert verify_release(freeze(memory, root))["release_id"]
    assert not list(root.rglob(".metadata-*"))
