"""Pinned source acquisition, bounded journals, and offline snapshot contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparselab.config.models import DatasetConfig
from sparselab.data import sources


def test_published_source_schema_is_current():
    path = Path(__file__).resolve().parents[1] / "schemas/dataset-source-v1.schema.json"
    assert json.loads(path.read_text()) == sources.DatasetSource.model_json_schema()


def declaration(**updates):
    return {
        "repo_id": "fixture/data",
        "revision": "a" * 40,
        "config": "default",
        "splits": {"train": "training", "validation": "heldout"},
        "text_field": "text",
        "attribution": "fixture",
        "license": "MIT",
        "selection": {"mode": "exhaustion"},
        "resources": {
            "max_source_records": 20,
            "max_text_bytes": 100000,
            "max_record_bytes": 10000,
            "max_work_bytes": 10000000,
            "min_free_bytes": 0,
        },
        **updates,
    }


def inventory(source, **kwargs):
    return {
        "repo_id": source.repo_id,
        "revision": source.revision,
        "loader": "parquet",
        "files": {
            split: [{"path": f"data/{split}.parquet", "size": 42, "blob_id": "b" * 40}]
            for split in source.splits
        },
    }


@pytest.fixture
def offline(monkeypatch):
    rows = {
        "train": [{"text": "train one"}, {"text": "train two"}],
        "validation": [{"text": "validation one"}],
    }
    monkeypatch.setattr(sources, "_hub_identity", inventory)
    monkeypatch.setattr(
        sources, "_stream", lambda source, split, cache, **kw: iter(rows[split])
    )
    return rows


def acquire(root, payload=None, *, resume=False):
    root.mkdir(parents=True, exist_ok=True)
    config = root / "source.yaml"
    lock = root / "lock.json"
    if not resume:
        config.write_text(yaml.safe_dump(payload or declaration()))
        sources.lock_source(config, lock, root / "cache")
    return sources.snapshot_source(
        lock, root / "snapshot", root / "cache", resume=resume
    )


def runtime(manifest):
    return DatasetConfig(
        source="snapshot",
        revision="a" * 40,
        license="MIT",
        dataset_config="default",
        source_manifest_path=manifest,
        train_path=manifest.parent / "train.jsonl",
        validation_path=manifest.parent / "validation.jsonl",
        cache_dir=manifest.parent.parent / "prepared",
        train_max_documents=20,
        validation_max_documents=20,
        train_max_tokens=1000,
        validation_max_tokens=1000,
    )


def test_exhaustion_and_relocation(tmp_path, offline):
    manifest = acquire(tmp_path)
    verified = sources.verify_snapshot(runtime(manifest))
    assert verified["source_exhausted"] is True
    assert list(sources.iter_snapshot(runtime(manifest), "train")) == [
        "train one",
        "train two",
    ]
    shutil.copytree(manifest.parent, tmp_path / "moved")
    moved = tmp_path / "moved/manifest.json"
    assert sources.verify_snapshot(runtime(moved)) == verified


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "c" * 40),
        ("license", "other"),
        ("dataset_config", "other"),
        ("train_path", "other.jsonl"),
    ],
)
def test_config_binding_rejects_changes(tmp_path, offline, field, value):
    manifest = acquire(tmp_path)
    if field.endswith("path"):
        path = tmp_path / value
        path.write_bytes((manifest.parent / "train.jsonl").read_bytes())
        value = path
    bad = runtime(manifest).model_copy(update={field: value})
    with pytest.raises(ValueError, match="mismatch"):
        sources.verify_snapshot(bad)


@pytest.mark.parametrize(
    "member", ["train.jsonl", "validation.jsonl", "events.jsonl", "excluded.jsonl"]
)
def test_changed_inventory_fails(tmp_path, offline, member):
    manifest = acquire(tmp_path)
    target = manifest.parent / member
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        sources.verify_snapshot(manifest)


def test_extra_split_is_verified(tmp_path, offline):
    offline["test"] = [{"text": "third split"}]
    payload = declaration(
        splits={"train": "training", "validation": "heldout", "test": "test"},
        dedup={"priority": ["validation", "train", "test"]},
    )
    manifest = acquire(tmp_path, payload)
    assert sources.verify_snapshot(runtime(manifest))["splits"]["test"]["count"] == 1
    (manifest.parent / "test.jsonl").unlink()
    with pytest.raises(ValueError, match="inventory"):
        sources.verify_snapshot(manifest)


@pytest.mark.parametrize("split", ["events", "excluded", "manifest"])
def test_reserved_split_names_rejected(split):
    with pytest.raises(ValueError, match="split mapping"):
        sources.DatasetSource.model_validate(
            declaration(splits={split: "upstream"}, dedup={"priority": [split]})
        )


@pytest.mark.parametrize(
    "suffix",
    [
        "license: other\n",
        "selection:\n  mode: bounded\n  mode: exhaustion\n",
        "true: value\n",
    ],
)
def test_strict_yaml_rejects_ambiguous_keys(tmp_path, suffix):
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(declaration()) + suffix)
    with pytest.raises((ValueError, TypeError)):
        sources.load_source(path)


@pytest.mark.parametrize("artifact", ["lock", "snapshot"])
@pytest.mark.parametrize("extra", ["duplicate", "nonfinite"])
def test_locked_documents_use_strict_json(tmp_path, offline, artifact, extra):
    manifest = acquire(tmp_path)
    path = tmp_path / "lock.json" if artifact == "lock" else manifest
    raw = path.read_text().rstrip()
    if extra == "duplicate":
        field = ',"format":' + json.dumps(json.loads(raw)["format"])
    else:
        field = ',"bad":NaN'
    path.write_text(raw[:-1] + field + "}")
    with pytest.raises(ValueError, match="duplicate JSON|non-finite"):
        (sources.verify_lock if artifact == "lock" else sources.verify_snapshot)(path)


@pytest.mark.parametrize("policy", ["exclude", "error"])
def test_overlap_keep_does_not_bypass_within_split(tmp_path, offline, policy):
    offline["validation"] = [{"text": "shared"}]
    offline["train"] = [{"text": "shared"}, {"text": "shared"}]
    payload = declaration(dedup={"overlap": "keep", "within_split": policy})
    if policy == "error":
        with pytest.raises(ValueError, match="forbidden duplicate"):
            acquire(tmp_path, payload)
        assert not (tmp_path / "snapshot").exists()
    else:
        manifest = acquire(tmp_path, payload)
        result = sources.verify_snapshot(manifest)
        assert result["splits"]["train"]["count"] == 1
        assert result["splits"]["train"]["duplicate"] == 1


@pytest.mark.parametrize(
    "limit,value",
    [
        ("max_source_records", 1),
        ("max_text_bytes", 5),
        ("max_record_bytes", 5),
        ("max_work_bytes", 1024),
    ],
)
def test_resource_caps_never_publish(tmp_path, offline, limit, value):
    payload = declaration()
    payload["resources"][limit] = value
    with pytest.raises(ValueError, match="resource bound"):
        acquire(tmp_path, payload)
    assert not (tmp_path / "snapshot").exists()
    assert (tmp_path / ".snapshot.work").is_dir()
    with pytest.raises(ValueError, match="resource bound"):
        acquire(tmp_path, resume=True)
    assert not (tmp_path / "snapshot").exists()


def test_exact_record_cap_can_establish_eof(tmp_path, offline):
    payload = declaration()
    payload["resources"]["max_source_records"] = 2
    assert (
        sources.verify_snapshot(acquire(tmp_path, payload))["source_exhausted"] is True
    )


def test_document_target_does_not_claim_exhaustion(tmp_path, offline):
    payload = declaration(
        selection={"mode": "bounded", "documents": {"train": 2, "validation": 1}}
    )
    result = sources.verify_snapshot(acquire(tmp_path, payload))
    assert result["source_exhausted"] is False
    assert result["splits"]["train"]["stop_reason"] == "document_target"


@pytest.mark.parametrize("drift", [False, True])
def test_interruption_replays_committed_prefix(tmp_path, offline, monkeypatch, drift):
    def interrupted(source, split, cache, **kw):
        yield offline[split][0]
        raise InterruptedError("fixture interruption")

    monkeypatch.setattr(sources, "_stream", interrupted)
    with pytest.raises(InterruptedError):
        acquire(tmp_path)
    assert not (tmp_path / "snapshot").exists()
    if drift:
        offline["validation"][0] = {"text": "changed upstream"}
    monkeypatch.setattr(
        sources, "_stream", lambda source, split, cache, **kw: iter(offline[split])
    )
    if drift:
        with pytest.raises(ValueError, match="prefix mismatch"):
            acquire(tmp_path, resume=True)
    else:
        manifest = acquire(tmp_path, resume=True)
        assert sources.verify_snapshot(manifest)["splits"]["train"]["count"] == 2


def test_interrupted_export_replays_without_duplicate_output(
    tmp_path, offline, monkeypatch
):
    original = sources._rename_noreplace

    def fail_publication(source, destination):
        if destination.name == "snapshot":
            raise InterruptedError("export interrupted")
        return original(source, destination)

    monkeypatch.setattr(sources, "_rename_noreplace", fail_publication)
    with pytest.raises(InterruptedError):
        acquire(tmp_path)
    monkeypatch.setattr(sources, "_rename_noreplace", original)
    manifest = acquire(tmp_path, resume=True)
    assert sources.verify_snapshot(manifest)["splits"]["train"]["count"] == 2


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/data.parquet",
        "hf://datasets/fixture/data@main/a.parquet",
        "hf://datasets/other/data@" + "a" * 40 + "/a.parquet",
        "https://huggingface.co/datasets/fixture/data/resolve/"
        + "a" * 40
        + "/../a.parquet",
    ],
)
def test_inventory_rejects_external_or_unpinned_files(url):
    with pytest.raises(ValueError):
        sources._hub_file(sources.DatasetSource.model_validate(declaration()), url)


def test_hub_lock_resolves_file_inventory_and_stream_uses_only_it(
    tmp_path, monkeypatch
):
    source = sources.DatasetSource.model_validate(declaration())
    files = {
        name: [f"hf://datasets/{source.repo_id}@{source.revision}/data/{name}.parquet"]
        for name in source.splits.values()
    }
    monkeypatch.setattr(
        "huggingface_hub.HfApi.dataset_info",
        lambda *a, **kw: SimpleNamespace(
            sha=source.revision,
            siblings=[
                SimpleNamespace(
                    rfilename=f"data/{name}.parquet",
                    size=42,
                    blob_id="b" * 40,
                    lfs=SimpleNamespace(sha256="c" * 64),
                )
                for name in source.splits.values()
            ],
        ),
    )
    monkeypatch.setattr(
        "datasets.load_dataset_builder",
        lambda *a, **kw: SimpleNamespace(
            info=SimpleNamespace(builder_name="parquet"),
            config=SimpleNamespace(data_files=files),
        ),
    )
    identity = sources._hub_identity(source)
    assert identity["files"]["train"][0]["path"] == "data/training.parquet"
    assert identity["files"]["train"][0]["lfs_sha256"] == "c" * 64

    def load(files, options, selected_source, cache, auth):
        assert selected_source == source
        assert cache == tmp_path
        assert files == [
            f"https://huggingface.co/datasets/{source.repo_id}/resolve/{source.revision}/data/training.parquet"
        ]
        return iter([{"text": "locked data"}])

    monkeypatch.setattr(sources, "_stream_parquet", load)
    monkeypatch.setattr(
        "datasets.load_dataset", lambda *a, **kw: pytest.fail("threaded parquet loader")
    )
    assert list(sources._stream(source, "train", tmp_path, inventory=identity)) == [
        {"text": "locked data"}
    ]


@pytest.mark.parametrize("bounded", [False, True])
def test_parquet_reads_synchronously_and_closes_on_bounded_stop(
    tmp_path, monkeypatch, bounded
):
    import io
    import threading

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "fixture.parquet"
    pq.write_table(
        pa.table({"text": [f"row {i}" for i in range(12)]}),
        parquet_path,
        row_group_size=3,
    )
    owner = threading.get_ident()
    handles = []

    class CheckedFile(io.BytesIO):
        def read(self, *args):
            assert threading.get_ident() == owner, "asynchronous Arrow read"
            return super().read(*args)

        def readinto(self, *args):
            assert threading.get_ident() == owner, "asynchronous Arrow readinto"
            return super().readinto(*args)

        def seek(self, *args):
            assert threading.get_ident() == owner, "asynchronous Arrow seek"
            return super().seek(*args)

    def open_file(path, mode, **kwargs):
        handle = CheckedFile(parquet_path.read_bytes())
        handles.append(handle)
        return handle

    monkeypatch.setattr("datasets.utils.file_utils.xopen", open_file)
    source = sources.DatasetSource.model_validate(declaration())
    stream = sources._stream_parquet(
        ["first", "second"], {"batch_size": 2}, source, tmp_path, {}
    )
    if bounded:
        assert next(stream) == {"text": "row 0"}
        stream.close()
        assert len(handles) == 1
    else:
        assert list(stream) == [{"text": f"row {i}"} for i in range(12)] * 2
        assert len(handles) == 2
    assert all(handle.closed for handle in handles)


def test_parquet_preserves_filter_projection_and_feature_cast(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Features, Value

    path = tmp_path / "fixture.parquet"
    pq.write_table(
        pa.table({"text": [1, 2, 3, 4], "selected": [False, True, True, False]}),
        path,
        row_group_size=2,
    )
    source = sources.DatasetSource.model_validate(declaration())
    options = {
        "batch_size": 1,
        "columns": ["text"],
        "filters": [["selected", "=", True]],
        "features": Features({"text": Value("string")}).to_dict(),
    }
    assert list(
        sources._stream_parquet([str(path)], options, source, tmp_path, {})
    ) == [{"text": "2"}, {"text": "3"}]


@pytest.mark.parametrize("declared_batch_size", [None, 1000000, 7])
def test_parquet_caps_decoder_batches(tmp_path, monkeypatch, declared_batch_size):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "fixture.parquet"
    pq.write_table(pa.table({"text": ["row"] * 3000}), path, row_group_size=3000)
    original = pq.ParquetFile.iter_batches
    observed = []

    def batches(self, **kwargs):
        observed.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", batches)
    stream = sources._stream_parquet(
        [str(path)],
        {"batch_size": declared_batch_size},
        sources.DatasetSource.model_validate(declaration()),
        tmp_path,
        {},
    )
    assert next(stream) == {"text": "row"}
    stream.close()
    assert observed == [
        {
            "batch_size": min(1024, declared_batch_size or 1024),
            "columns": None,
            "use_threads": False,
        }
    ]


def test_bounded_parquet_subprocess_exits(tmp_path):
    import subprocess
    import sys

    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "fixture.parquet"
    pq.write_table(pa.table({"text": ["row"] * 5000}), path, row_group_size=5000)
    program = """
import io, json, sys
from pathlib import Path
from unittest.mock import patch
from sparselab.data.sources import DatasetSource, _stream_parquet
path = Path(sys.argv[1])
source = DatasetSource.model_validate(json.loads(sys.argv[2]))
with patch('datasets.utils.file_utils.xopen',
           side_effect=lambda *a, **k: io.BytesIO(path.read_bytes())):
    stream = _stream_parquet([str(path)], {}, source, path.parent, {})
    assert next(stream) == {'text': 'row'}
    stream.close()
print('complete', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path), json.dumps(declaration())],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stdout.strip() == "complete"


def test_forge_locations_do_not_change_scientific_lock(tmp_path, offline, monkeypatch):
    monkeypatch.setattr(
        sources,
        "_forge_bindings",
        lambda source: {split: "c" * 64 for split in source.splits},
    )
    payload = declaration(
        kind="forge_files",
        forge_files={
            split: {"snapshot_path": f"first/{split}", "files": ["data.jsonl"]}
            for split in ("train", "validation")
        },
    )
    first = acquire(tmp_path / "first", payload)
    for binding in payload["forge_files"].values():
        binding["snapshot_path"] = "relocated"
    second = acquire(tmp_path / "second", payload)
    assert first.read_bytes() == second.read_bytes()
    assert (tmp_path / "first/lock.json").read_bytes() == (
        tmp_path / "second/lock.json"
    ).read_bytes()
    assert b"snapshot_path" not in first.read_bytes()


def test_import_dispatch_preserves_distinct_verifier(tmp_path, monkeypatch):
    from sparselab.data import snapshot_import

    manifest = tmp_path / "manifest.json"
    payload = sources._seal(
        {"format": "sparselab-dataset-import-v1"}, "manifest_sha256"
    )
    manifest.write_text(json.dumps(payload))
    monkeypatch.setattr(
        snapshot_import, "verify_import", lambda config: {"verified": str(config)}
    )
    assert sources.verify_snapshot(manifest) == {"verified": str(manifest)}
