from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparselab.config.models import DatasetConfig
from sparselab.data import local_stories
from sparselab.data.snapshot_import import FORMAT, import_legacy_snapshot, verify_import
from sparselab.training.manifest import canonical_json


@pytest.fixture
def historical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    rows = {"train": ["a", "a", "b"], "validation": ["a", "c"]}
    monkeypatch.setattr(
        local_stories,
        "_hub_stream",
        lambda split, _: ({"text": text} for text in rows[split]),
    )
    path = local_stories.snapshot(tmp_path / "old", train_count=2, validation_count=1)

    def no_fetch(*args, **kwargs):
        pytest.fail("snapshot import must not fetch upstream source data")

    monkeypatch.setattr(local_stories, "_hub_stream", no_fetch)
    monkeypatch.setattr("datasets.load_dataset", no_fetch)
    monkeypatch.setattr("sparselab.data.sources._hub_identity", no_fetch)
    return path


def test_import_has_distinct_authenticated_identity_and_portable_bytes(
    historical: Path, tmp_path: Path
) -> None:
    original = {p.name: p.read_bytes() for p in historical.parent.iterdir()}
    manifest_path = import_legacy_snapshot(historical, tmp_path / "new")
    manifest = verify_import(manifest_path)
    assert manifest["format"] == FORMAT
    assert manifest["lock"]["source"]["kind"] == "legacy_snapshot"
    assert (
        manifest["lock"]["provenance"]["manifest_sha256"]
        == hashlib.sha256(original["manifest.json"]).hexdigest()
    )
    assert (
        manifest["manifest_sha256"] != manifest["lock"]["provenance"]["manifest_sha256"]
    )
    for name, data in original.items():
        assert (historical.parent / name).read_bytes() == data
        assert (manifest_path.parent / "legacy" / name).read_bytes() == data
    historical.parent.rename(tmp_path / "historical-moved")
    manifest_path.parent.rename(tmp_path / "import-moved")
    assert verify_import(tmp_path / "import-moved/manifest.json") == manifest


def test_import_coverage_is_bounded_even_if_all_retained_documents_prepared(
    historical: Path, tmp_path: Path
) -> None:
    from sparselab.data.coverage import _source_coverage

    manifest = verify_import(import_legacy_snapshot(historical, tmp_path / "new"))
    row = _source_coverage(
        manifest,
        "train",
        {
            "acquired_documents": 2,
            "retained_documents": 2,
            "skipped_documents": 0,
            "truncated_documents": 0,
        },
    )
    assert row["full"] is False
    assert row["source_exhausted"] is False
    assert row["all_retained_records_prepared"] is True
    assert row["excluded_records"] == 1
    assert row["stop_reason"] == "legacy_bounded_import"


@pytest.mark.parametrize(
    "tamper", ["historical", "export", "coverage", "provenance", "extra", "symlink"]
)
def test_import_rejects_tampering_even_after_manifest_reseal(
    historical: Path, tmp_path: Path, tamper: str
) -> None:
    path = import_legacy_snapshot(historical, tmp_path / "new")
    if tamper == "historical":
        (path.parent / "legacy/train.jsonl").write_bytes(b"changed\n")
    elif tamper == "export":
        (path.parent / "train.jsonl").write_bytes(b"changed\n")
    elif tamper in {"coverage", "provenance"}:
        raw = json.loads(path.read_text())
        if tamper == "coverage":
            raw["source_exhausted"] = True
            raw["splits"]["train"]["source_exhausted"] = True
        else:
            raw["lock"]["provenance"]["manifest_sha256"] = "0" * 64
        raw.pop("manifest_sha256")
        raw["manifest_sha256"] = hashlib.sha256(canonical_json(raw)).hexdigest()
        path.write_bytes(canonical_json(raw))
    elif tamper == "extra":
        (path.parent / "untracked").mkdir()
    else:
        (path.parent / "extra-link").symlink_to(historical)
    with pytest.raises(ValueError):
        verify_import(path)


def test_import_rejects_runtime_binding_mismatch(
    historical: Path, tmp_path: Path
) -> None:
    path = import_legacy_snapshot(historical, tmp_path / "new")
    config = DatasetConfig(
        source="snapshot",
        revision=local_stories.REVISION,
        license=local_stories.LICENSE,
        train_path=path.parent / "train.jsonl",
        validation_path=path.parent / "validation.jsonl",
        source_manifest_path=path,
        cache_dir=tmp_path / "cache",
        train_max_documents=2,
        validation_max_documents=1,
        train_max_tokens=100,
        validation_max_tokens=100,
    )
    verify_import(config)
    with pytest.raises(ValueError, match="config path mismatch"):
        verify_import(
            config.model_copy(update={"train_path": historical.parent / "train.jsonl"})
        )
    with pytest.raises(ValueError, match="config provenance mismatch"):
        verify_import(config.model_copy(update={"revision": "0" * 40}))


def test_import_never_overwrites_and_failure_does_not_publish(
    historical: Path, tmp_path: Path
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        import_legacy_snapshot(historical, output)
    raw = json.loads(historical.read_text())
    raw["splits"]["train"]["path"] = "../escape.jsonl"
    historical.write_bytes(canonical_json(raw))
    with pytest.raises(ValueError, match="unsafe"):
        import_legacy_snapshot(historical, tmp_path / "new")
    assert not (tmp_path / "new").exists()
    assert not list(tmp_path.glob(".snapshot-import-*"))
