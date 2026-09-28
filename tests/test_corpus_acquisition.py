"""Acquisition identity, strict recipes, and offline byte verification."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from sparselab.corpus.acquisition import acquire, verify_acquisition, verify_snapshot
from sparselab.corpus.project import load_project


def _yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    kind: str = "local",
    acquisition: dict | None = None,
    revision: str = "v1",
    uri: str = "fixture://source",
    license: str = "MIT",
) -> Path:
    root = tmp_path / "recipe"
    root.mkdir(exist_ok=True)
    _yaml(
        root / "corpus.yaml",
        {
            "schema_version": 1,
            "id": "example",
            "sources": ["sources/one.yaml"],
            "transforms": [],
            "splits": "splits.yaml",
            "release": "release.yaml",
        },
    )
    _yaml(
        root / "splits.yaml",
        {"schema_version": 1, "unit": "document", "assignments": {"document": "train"}},
    )
    _yaml(
        root / "release.yaml",
        {
            "schema_version": 1,
            "mixture": {"technical_docs": 1.0},
            "lm": {"selected": True, "training_splits": ["train", "validation"]},
            "chat": {"selected": False, "training_splits": []},
        },
    )
    _yaml(
        root / "sources/one.yaml",
        {
            "schema_version": 1,
            "id": "one",
            "kind": kind,
            "canonical_uri": uri,
            "revision": revision,
            "license": license,
            "redistribution": "redistributable",
            "domains": ["technical_docs"],
            "document_kinds": ["markdown"],
            "source_family": "docs",
            "acquisition": acquisition
            or {
                "files": [{"path": "sources/data.md", "name": "data.md"}],
                "max_bytes": 1024,
            },
        },
    )
    return root / "corpus.yaml"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_local_offline_change_and_tamper(tmp_path: Path) -> None:
    recipe = _fixture(tmp_path)
    file = recipe.parent / "sources/data.md"
    file.write_text("# Before\n", encoding="utf-8")
    project = load_project(recipe)
    first = acquire(project, tmp_path / "work")
    original = first["sources"]["one"]
    assert acquire(project, tmp_path / "work", offline=True) == first
    file.write_text("# After\n", encoding="utf-8")
    assert acquire(project, tmp_path / "work", offline=True) == first
    second = acquire(project, tmp_path / "work")
    assert second["sources"]["one"]["snapshot_sha256"] != original["snapshot_sha256"]
    frozen = Path(original["snapshot_path"])
    assert (
        verify_snapshot(frozen)["files"][0]["sha256"]
        == hashlib.sha256(b"# Before\n").hexdigest()
    )
    active = Path(second["sources"]["one"]["snapshot_path"])
    (active / "files/data.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_acquisition(project, tmp_path / "work")
    with pytest.raises(ValueError, match="mismatch"):
        acquire(project, tmp_path / "work", offline=True)


def test_git_pinned_revision_and_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test")
    (repo / "docs").mkdir()
    document = repo / "docs/guide.md"
    document.write_text("first\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "first")
    first_revision = _git(repo, "rev-parse", "HEAD")
    recipe = _fixture(
        tmp_path,
        kind="git",
        acquisition={"include": ["docs/*.md"], "max_bytes": 1024},
        revision=first_revision,
        uri=str(repo),
    )
    project = load_project(recipe)
    first = acquire(project, tmp_path / "work")
    assert first["sources"]["one"]["snapshot_sha256"]
    document.write_text("second\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "second")
    latest = _git(repo, "rev-parse", "HEAD")
    source = yaml.safe_load((recipe.parent / "sources/one.yaml").read_text())
    source["revision"] = latest
    _yaml(recipe.parent / "sources/one.yaml", source)
    second = acquire(load_project(recipe), tmp_path / "work")
    assert (
        second["sources"]["one"]["snapshot_sha256"]
        != first["sources"]["one"]["snapshot_sha256"]
    )
    (repo / "docs/link.md").symlink_to("guide.md")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "symlink")
    source["revision"] = _git(repo, "rev-parse", "HEAD")
    _yaml(recipe.parent / "sources/one.yaml", source)
    with pytest.raises(ValueError, match="symlink"):
        acquire(load_project(recipe), tmp_path / "work")
    assert (
        json.loads((tmp_path / "work/corpora/example/acquisition.json").read_text())[
            "sources"
        ]["one"]
        == second["sources"]["one"]
    )


def test_http_hash_bound_and_offline(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        body = b"# bounded document\n"
        calls = 0

        def do_GET(self) -> None:
            Handler.calls += 1
            self.send_response(200)
            self.send_header("ETag", "test")
            self.end_headers()
            self.wfile.write(Handler.body)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        uri = f"http://127.0.0.1:{server.server_port}/manual.md"
        recipe = _fixture(
            tmp_path,
            kind="http_document",
            acquisition={
                "expected_sha256": hashlib.sha256(Handler.body).hexdigest(),
                "max_bytes": 100,
            },
            uri=uri,
        )
        project = load_project(recipe)
        first = acquire(project, tmp_path / "work")
        before = Handler.calls
        assert acquire(project, tmp_path / "work", offline=True) == first
        assert Handler.calls == before
        Handler.body = b"# changed document\n"
        with pytest.raises(ValueError, match="SHA-256"):
            acquire(project, tmp_path / "work")
        assert verify_acquisition(project, tmp_path / "work") == first
        source = yaml.safe_load((recipe.parent / "sources/one.yaml").read_text())
        source["acquisition"]["expected_sha256"] = hashlib.sha256(
            Handler.body
        ).hexdigest()
        _yaml(recipe.parent / "sources/one.yaml", source)
        second = acquire(load_project(recipe), tmp_path / "work")
        assert (
            second["sources"]["one"]["snapshot_sha256"]
            != first["sources"]["one"]["snapshot_sha256"]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_hf_pinned_selection_and_unsupported_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    revision = "a" * 40
    location = tmp_path / revision
    (location / "default/train").mkdir(parents=True)
    selected = location / "default/train/data.jsonl"
    selected.write_text('{"text":"hello"}\n', encoding="utf-8")
    calls = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(location)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    recipe = _fixture(
        tmp_path,
        kind="huggingface_dataset",
        acquisition={
            "config": "default",
            "split": "train",
            "include": ["default/train/*"],
            "text_field": "text",
            "max_rows": 10,
            "max_bytes": 1024,
        },
        revision=revision,
        uri="org/dataset",
    )
    project = load_project(recipe)
    first = acquire(project, tmp_path / "work")
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == revision
    assert acquire(project, tmp_path / "work", offline=True) == first
    assert len(calls) == 1
    (location / "default/train/unsupported.bin").write_bytes(b"binary")
    source = yaml.safe_load((recipe.parent / "sources/one.yaml").read_text())
    source["revision"] = "b" * 40
    _yaml(recipe.parent / "sources/one.yaml", source)
    location.rename(tmp_path / ("b" * 40))
    location = tmp_path / ("b" * 40)
    with pytest.raises(ValueError, match="unsupported HF file format"):
        acquire(load_project(recipe), tmp_path / "work")
    assert verify_acquisition(project, tmp_path / "work") == first


def test_hf_parquet_rows_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    revision = "c" * 40
    snapshot = tmp_path / revision
    folder = snapshot / "default/train"
    folder.mkdir(parents=True)
    pq.write_table(pa.table({"text": ["one", "two"]}), folder / "data.parquet")
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **_kwargs: str(snapshot)
    )
    recipe = _fixture(
        tmp_path,
        kind="huggingface_dataset",
        acquisition={
            "config": "default",
            "split": "train",
            "include": ["default/train/*.parquet"],
            "text_field": "text",
            "max_rows": 1,
            "max_bytes": 4096,
        },
        revision=revision,
        uri="org/parquet",
    )
    with pytest.raises(ValueError, match="max_rows"):
        acquire(load_project(recipe), tmp_path / "work")
    assert not (tmp_path / "work/corpora/example/acquisition.json").exists()
    source = yaml.safe_load((recipe.parent / "sources/one.yaml").read_text())
    source["acquisition"]["max_rows"] = 2
    _yaml(recipe.parent / "sources/one.yaml", source)
    lock = acquire(load_project(recipe), tmp_path / "work")
    assert (
        verify_snapshot(lock["sources"]["one"]["snapshot_path"])["retrieval"]["rows"]
        == 2
    )


def test_invalid_declarations_and_symlink(tmp_path: Path) -> None:
    recipe = _fixture(tmp_path)
    source_file = recipe.parent / "sources/one.yaml"
    original = yaml.safe_load(source_file.read_text())
    for key, value in (("license", " "), ("revision", "")):
        invalid = {**original, key: value}
        _yaml(source_file, invalid)
        with pytest.raises(ValueError):
            load_project(recipe)
    _yaml(source_file, original)
    actual = recipe.parent / "sources/data.md"
    actual.write_text("valid", encoding="utf-8")
    link = recipe.parent / "sources/linked.md"
    link.symlink_to(actual)
    original["acquisition"]["files"][0]["path"] = "sources/linked.md"
    _yaml(source_file, original)
    with pytest.raises(ValueError, match="symlink"):
        acquire(load_project(recipe), tmp_path / "work")
    assert not (tmp_path / "work/corpora/example/acquisition.json").exists()


def test_generator_receipt_and_rejected_source(tmp_path: Path) -> None:
    recipe = _fixture(tmp_path)
    (recipe.parent / "sources/data.md").write_text("hello", encoding="utf-8")
    config = yaml.safe_load(recipe.read_text())
    config["sources"] = [
        "sources/generator.yaml",
        "sources/one.yaml",
        "sources/rejected.yaml",
    ]
    _yaml(recipe, config)
    template = yaml.safe_load((recipe.parent / "sources/one.yaml").read_text())
    generator = {
        **template,
        "id": "generator",
        "kind": "deterministic_generator",
        "canonical_uri": "generator://pathlib_path_suffix_v1",
        "acquisition": {
            "generator": "pathlib_path_suffix_v1",
            "generator_version": "1",
        },
    }
    _yaml(recipe.parent / "sources/generator.yaml", generator)
    _yaml(
        recipe.parent / "sources/rejected.yaml",
        {
            **template,
            "id": "rejected",
            "redistribution": "rejected",
            "rejection_reason": "No redistribution permission",
            "acquisition": {
                "files": [{"path": "sources/unavailable.md", "name": "unavailable.md"}],
                "max_bytes": 100,
            },
        },
    )
    project = load_project(recipe)
    lock = acquire(project, tmp_path / "work")
    assert lock["sources"]["rejected"]["snapshot_path"] is None
    assert (
        lock["sources"]["rejected"]["receipt"]["reason"]
        == "No redistribution permission"
    )
    generator_receipt = lock["sources"]["generator"]
    assert generator_receipt["receipt"]["status"] == "generator"
    assert verify_snapshot(generator_receipt["snapshot_path"])["files"] == []
    assert acquire(project, tmp_path / "work", offline=True) == lock
