"""Acquisition identity, strict recipes, and offline byte verification."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import os
import signal
import sqlite3
import subprocess
import threading
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError

import pytest
import yaml

from sparselab.corpus.acquisition import (
    _copy_hf_body,
    _read_metadata_body,
    _read_with_deadline,
    _set_response_deadline,
    acquire,
    declaration_sha256,
    verify_acquisition,
    verify_snapshot,
)
from sparselab.corpus.project import (
    load_project,
    release_declaration_payload,
    source_declaration_payload,
)
from sparselab.corpus.transport_budget import TransportBudget


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


def test_verified_pinned_git_snapshot_survives_adapter_module_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.corpus import acquisition

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test")
    (repo / "guide.md").write_text("A pinned source document.\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "source")
    recipe = _fixture(
        tmp_path,
        kind="git",
        acquisition={"include": ["guide.md"], "max_bytes": 1024},
        revision=_git(repo, "rev-parse", "HEAD"),
        uri=str(repo),
    )
    project = load_project(recipe)
    work = tmp_path / "work"
    original = acquire(project, work)
    original_adapter = acquisition._adapter
    monkeypatch.setattr(
        acquisition,
        "_adapter",
        lambda source: {**original_adapter(source), "module_sha256": "0" * 64},
    )
    monkeypatch.setattr(
        acquisition,
        "_acquire_git",
        lambda *_args, **_kwargs: pytest.fail("pinned snapshot was refetched"),
    )
    assert acquire(project, work) == original
    (work / "corpora" / project.config.id / "acquisition.json").unlink()
    assert acquire(project, work) == original


def test_git_v2_acquires_pinned_nested_license_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test")
    (repo / "docs").mkdir()
    (repo / "docs/guide.md").write_text("# SPDX-License-Identifier: MIT\nSource text\n")
    metadata = {"files": [{"path": "docs/guide.md", "license": "MIT"}]}
    (repo / "license-metadata.json").write_text(json.dumps(metadata))
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "pinned rights")
    recipe = _fixture(
        tmp_path,
        kind="git",
        acquisition={"include": ["docs/*.md"], "max_bytes": 4096},
        revision=_git(repo, "rev-parse", "HEAD"),
        uri=str(repo),
    )
    source_path = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_path.read_text())
    source["schema_version"] = 2
    source.pop("redistribution")
    source["license_url"] = "https://example.org/upstream/LICENSE"
    source["rights"] = {
        "training_eligibility": "eligible",
        "redistribution_mode": "metadata_reconstruction_only",
        "spdx_expression": "MIT",
        "license_references": ["https://example.org/upstream/LICENSE"],
        "nested_metadata_path": "license-metadata.json",
    }
    _yaml(source_path, source)
    release_path = recipe.parent / "release.yaml"
    release = yaml.safe_load(release_path.read_text())
    release.update(schema_version=2, publication_mode="metadata_reconstruction_only")
    _yaml(release_path, release)
    lock = acquire(load_project(recipe), tmp_path / "work")
    source = load_project(recipe).sources[0]
    legacy_payload = source.model_dump(mode="json")
    legacy_payload.pop("explicit_training_restriction")
    legacy_payload["acquisition"].pop("tree_oid")
    legacy_payload["acquisition"].pop("bounded_blobs")
    assert source_declaration_payload(source) == legacy_payload
    snapshot = verify_snapshot(Path(lock["sources"]["one"]["snapshot_path"]))
    assert {item["path"] for item in snapshot["files"]} == {
        "docs/guide.md",
        "license-metadata.json",
    }
    assert (
        Path(lock["sources"]["one"]["snapshot_path"]) / "files/license-metadata.json"
    ).read_text() == json.dumps(metadata)
    assert acquire(load_project(recipe), tmp_path / "work", offline=True) == lock


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


def _bounded_hf_fixture(
    tmp_path: Path, data: bytes, path: str, *, rows: int = 3
) -> Path:
    return _fixture(
        tmp_path,
        kind="huggingface_dataset",
        acquisition={
            "config": "default",
            "split": "train",
            "bounded_shards": [
                {
                    "path": path,
                    "expected_sha256": hashlib.sha256(data).hexdigest(),
                    "max_shard_bytes": len(data),
                    "max_scanned_rows": rows,
                    "hash_modulus": 3,
                    "hash_remainders": [0, 2],
                }
            ],
            "text_field": "text",
            "max_rows": 5,
            "max_bytes": 4096,
        },
        revision="d" * 40,
        uri="org/dataset",
    )


def _mock_hf_stream(monkeypatch: pytest.MonkeyPatch, content: bytes) -> list[str]:
    import huggingface_hub

    calls: list[str] = []

    def no_snapshot(**_kwargs: object) -> None:
        raise AssertionError("bounded HF mode must not download a snapshot")

    class Response(io.BytesIO):
        url = "https://cdn.example.test/pinned-shard"

        def __init__(self, body: bytes):
            super().__init__(body)
            self.headers = {"Content-Length": str(len(body))}

    class Opener:
        def open(self, request: urllib.request.Request, timeout: int) -> Response:
            calls.append(request.full_url)
            return Response(content)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_snapshot)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: Opener())
    return calls


def test_hf_legacy_declaration_hash_and_receipt_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b'{"text":"one"}\n'
    recipe = _bounded_hf_fixture(tmp_path, content, "default/train/data.jsonl", rows=1)
    project = load_project(recipe)
    source = project.sources[0]
    expected = source.model_dump(mode="json")
    expected.pop("rights")
    expected.pop("explicit_training_restriction")
    expected["acquisition"].pop("include")
    expected["acquisition"].pop("max_decompressed_bytes")
    for shard in expected["acquisition"]["bounded_shards"]:
        shard.pop("declared_config")
        shard.pop("declared_split")
    assert source_declaration_payload(source) == expected
    assert (
        declaration_sha256(source)
        == hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert (
        declaration_sha256(source)
        == "feae644fd3f3070beab148748c16f77ccae4654af7e322abedb00c39e8926b25"
    )
    calls = _mock_hf_stream(monkeypatch, content)
    first = acquire(project, tmp_path / "work")
    manifest = verify_snapshot(first["sources"]["one"]["snapshot_path"])
    assert manifest["declaration_sha256"] == declaration_sha256(source)
    assert "transport_budget" not in manifest["retrieval"]
    assert acquire(project, tmp_path / "work") == first
    assert len(calls) == 1


def _bounded_git_fixture(tmp_path: Path, content: bytes) -> tuple[Path, str]:
    oid = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    recipe = _fixture(
        tmp_path,
        kind="git",
        uri="https://github.com/org/repo",
        revision="a" * 40,
        acquisition={
            "max_bytes": len(content),
            "tree_oid": "b" * 40,
            "bounded_blobs": [
                {"path": "docs/one.md", "git_blob_oid": oid, "max_bytes": len(content)}
            ],
        },
    )
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    config["transport_budget"] = {
        "attempt_id": "git_fixture",
        "max_source_body_bytes": len(content) * 2,
        "max_metadata_body_bytes": 10000,
        "max_transfers": 2,
        "max_retries_per_shard": 1,
        "max_wall_seconds": 120,
        "max_disk_bytes": 1000000,
    }
    _yaml(recipe, config)
    return recipe, oid


def _mock_bounded_git_http(
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
    oid: str,
    *,
    wrong_tree: bool = False,
    corrupt: bool = False,
    interrupt: bool = False,
    extra_body: bool = False,
) -> None:
    class Response(io.BytesIO):
        def __init__(self, body: bytes, url: str):
            super().__init__(body)
            self.url = url
            self.status = 200
            self.headers = {"Content-Length": str(len(body))}

        def read(self, size: int = -1) -> bytes:
            if interrupt and "raw.githubusercontent.com" in self.url:
                if self.tell():
                    raise OSError("interrupted bounded Git transfer")
                return super().read(min(2, size))
            return super().read(size)

    class Opener:
        def open(self, request: urllib.request.Request, timeout: float) -> Response:
            url = request.full_url
            if "/git/commits/" in url:
                body = {"sha": "a" * 40, "tree": {"sha": "b" * 40}}
            elif "/git/trees/" in url:
                body = {
                    "sha": "b" * 40,
                    "truncated": False,
                    "tree": [
                        {
                            "path": "docs/wrong.md" if wrong_tree else "docs/one.md",
                            "type": "blob",
                            "mode": "100644",
                            "sha": oid,
                            "size": len(content),
                        }
                    ],
                }
            else:
                assert (
                    url
                    == "https://raw.githubusercontent.com/org/repo/"
                    + "a" * 40
                    + "/docs/one.md"
                )
                return Response(
                    content + b"x"
                    if extra_body
                    else bytes([content[0] ^ 1]) + content[1:]
                    if corrupt
                    else content,
                    url,
                )
            return Response(json.dumps(body).encode(), url)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: Opener())


def test_bounded_git_blob_metadata_checksum_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"# Pinned text\n"
    recipe, oid = _bounded_git_fixture(tmp_path, content)
    project = load_project(recipe)
    root = tmp_path / "work"
    ledger_path = root / "corpora/example/transport-budget.sqlite"
    with pytest.raises(ValueError, match="ledger missing"):
        acquire(project, root)
    TransportBudget.initialize(ledger_path, project)
    _mock_bounded_git_http(monkeypatch, content, oid, wrong_tree=True)
    with pytest.raises(ValueError, match="missing or ambiguous"):
        acquire(project, root)
    assert TransportBudget(ledger_path, project).receipt()["source_charged"] == 0
    _mock_bounded_git_http(monkeypatch, content, oid, interrupt=True)
    with pytest.raises(OSError, match="interrupted"):
        acquire(project, root)
    _mock_bounded_git_http(monkeypatch, content, oid)
    lock = acquire(project, root)
    assert (
        verify_snapshot(lock["sources"]["one"]["snapshot_path"])["files"][0][
            "git_blob_id"
        ]
        == oid
    )
    state = TransportBudget(ledger_path, project).receipt()
    assert state["source_charged"] == 2 * len(content)
    assert [item["status"] for item in state["transfers"]] == [
        "interrupted",
        "complete",
    ]
    assert acquire(project, root) == lock

    # A later file in the same bounded Git source can interrupt after this
    # one completed; a second invocation must re-fetch verified earlier blobs
    # within the same shared byte and per-blob retry allowance.
    replay_root = tmp_path / "replay-work"
    replay_ledger_path = replay_root / "corpora/example/transport-budget.sqlite"
    replay = TransportBudget.initialize(replay_ledger_path, project)
    first = replay.reserve_transfer("one", "docs/one.md", len(content))
    replay.charge_transfer_actual(first, len(content))
    replay.finish_transfer(first, success=True)
    second = replay.reserve_transfer(
        "one", "docs/one.md", len(content), allow_verified_retry=True
    )
    replay.charge_transfer_actual(second, len(content))
    replay.finish_transfer(second, success=True)
    with pytest.raises(ValueError, match="retry allowance exhausted"):
        replay.reserve_transfer(
            "one", "docs/one.md", len(content), allow_verified_retry=True
        )

    bad_root = tmp_path / "bad-work"
    TransportBudget.initialize(
        bad_root / "corpora/example/transport-budget.sqlite", project
    )
    _mock_bounded_git_http(monkeypatch, content, oid, corrupt=True)
    with pytest.raises(ValueError, match="SHA-1 mismatch"):
        acquire(project, bad_root)

    oversized_root = tmp_path / "oversized-work"
    TransportBudget.initialize(
        oversized_root / "corpora/example/transport-budget.sqlite", project
    )
    _mock_bounded_git_http(monkeypatch, content, oid, extra_body=True)
    with pytest.raises(ValueError, match="Content-Length mismatch"):
        acquire(project, oversized_root)

    exhausted_root = tmp_path / "exhausted-work"
    exhausted_ledger = exhausted_root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(exhausted_ledger, project)
    _mock_bounded_git_http(monkeypatch, content, oid, interrupt=True)
    for _ in range(2):
        with pytest.raises(OSError, match="interrupted"):
            acquire(project, exhausted_root)
    with pytest.raises(ValueError, match="retry allowance exhausted"):
        acquire(project, exhausted_root)
    assert TransportBudget(exhausted_ledger, project).receipt()["source_charged"] == (
        2 * len(content)
    )


def test_bounded_git_rejects_ambiguous_declaration(tmp_path: Path) -> None:
    recipe, _ = _bounded_git_fixture(tmp_path, b"# one\n")
    source_file = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_file.read_text())
    source["acquisition"]["include"] = ["docs/*.md"]
    _yaml(source_file, source)
    with pytest.raises(ValueError, match="bounded Git requires exact blobs"):
        load_project(recipe)


def _budgeted_hf_fixture(tmp_path: Path, content: bytes) -> Path:
    recipe = _bounded_hf_fixture(tmp_path, content, "complete-0001.json.gz", rows=1)
    source_file = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_file.read_text(encoding="utf-8"))
    shard = source["acquisition"]["bounded_shards"][0]
    shard["declared_config"] = "default"
    shard["declared_split"] = "train"
    source["acquisition"]["max_decompressed_bytes"] = 4096
    _yaml(source_file, source)
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    config["transport_budget"] = {
        "attempt_id": "fixture_attempt",
        "max_source_body_bytes": len(content) * 2,
        "max_metadata_body_bytes": 1_000_000,
        "max_transfers": 2,
        "max_retries_per_shard": 1,
        "max_wall_seconds": 120,
        "max_disk_bytes": 1_000_000,
    }
    _yaml(recipe, config)
    return recipe


def _mock_budgeted_hf_http(
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
    *,
    interrupt_first: bool = False,
    checksum: str | None = None,
    ambiguous: bool = False,
    payload: bytes | None = None,
    fail_http: bool = False,
    tree_size: int | None = None,
    xet: bool = False,
    lfs_oid: bool = False,
) -> tuple[list[str], list[int]]:
    calls: list[str] = []
    reads: list[int] = []
    transfers = 0
    digest = checksum or hashlib.sha256(content).hexdigest()
    tree = json.dumps(
        [
            {
                "path": "complete-0001.json.gz",
                "type": "file",
                "size": len(content) if tree_size is None else tree_size,
                "lfs": None
                if xet
                else {"oid": digest, "size": len(content)}
                if lfs_oid
                else {"sha256": digest},
                **({"xetHash": "a" * 64, "oid": "b" * 40} if xet else {}),
            }
        ]
    ).encode()
    split_rows = [{"dataset": "org/dataset", "config": "default", "split": "train"}]
    if ambiguous:
        split_rows.append(
            {"dataset": "org/dataset", "config": "other", "split": "train"}
        )
    splits = json.dumps({"splits": split_rows}).encode()

    class Response(io.BytesIO):
        def __init__(
            self,
            body: bytes,
            url: str,
            *,
            status: int = 200,
            location: str | None = None,
            fail: bool = False,
        ):
            super().__init__(body)
            self.url = url
            self.status = status
            self.headers = {"Content-Length": str(len(body))}
            if location is not None:
                self.headers["Location"] = location
            self.fail = fail

        def read(self, size: int = -1) -> bytes:
            reads.append(size)
            if (
                self.url == "https://cdn.example.test/shard"
                and size > len(self.getbuffer()) - self.tell()
            ):
                raise AssertionError("source response was read past its byte cap")
            if self.fail and self.tell() > 0:
                raise OSError("simulated interrupted transfer")
            if self.fail:
                return super().read(min(size, 4))
            return super().read(size)

    class Opener:
        def open(self, request: urllib.request.Request, timeout: float) -> Response:
            nonlocal transfers
            url = request.full_url
            calls.append(url)
            if "/api/datasets/" in url:
                assert request.data is not None
                assert b"paths=complete-0001.json.gz" in request.data
                return Response(tree, url)
            if "/splits?" in url:
                assert request.get_header("Authorization") is None
                return Response(splits, url)
            if "huggingface.co/datasets/" in url:
                return Response(
                    b"redirect-body",
                    url,
                    status=302,
                    location="https://cdn.example.test/shard",
                )
            assert url == "https://cdn.example.test/shard"
            assert request.get_header("Authorization") is None
            transfers += 1
            if fail_http:
                raise HTTPError(
                    url,
                    503,
                    "unavailable",
                    {"Content-Length": "9"},
                    io.BytesIO(b"error-503"),
                )
            return Response(
                payload if payload is not None else content,
                url,
                fail=interrupt_first and transfers == 1,
            )

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: Opener())
    return calls, reads


def test_hf_explicit_metadata_redirect_and_budget_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = gzip.compress(b'{"text":"one"}\n', mtime=0)
    recipe = _budgeted_hf_fixture(tmp_path, content)
    project = load_project(recipe)
    root = tmp_path / "work"
    with pytest.raises(ValueError, match="ledger missing"):
        acquire(project, root)
    ledger_path = root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(ledger_path, project)
    calls, reads = _mock_budgeted_hf_http(monkeypatch, content)
    lock = acquire(project, root)
    assert len(calls) == 4
    state = TransportBudget(ledger_path, project).receipt()
    assert state["source_charged"] == len(content)
    assert state["source_actual"] == len(content)
    assert state["metadata_actual"] == len(b"redirect-body") + len(
        json.dumps(
            [
                {
                    "path": "complete-0001.json.gz",
                    "type": "file",
                    "size": len(content),
                    "lfs": {"sha256": hashlib.sha256(content).hexdigest()},
                }
            ]
        ).encode()
    ) + len(
        json.dumps(
            {
                "splits": [
                    {"dataset": "org/dataset", "config": "default", "split": "train"}
                ]
            }
        ).encode()
    )
    assert state["metadata_charged"] >= state["metadata_actual"]
    assert all(size <= len(content) or size <= 65536 for size in reads)
    receipt = verify_snapshot(lock["sources"]["one"]["snapshot_path"])["retrieval"]
    assert receipt["transport_budget"]["source_actual"] == len(content)
    assert acquire(project, root) == lock
    assert len(calls) == 4


def test_hf_deadline_reaches_nested_urllib_socket(tmp_path: Path) -> None:
    recipe = _budgeted_hf_fixture(tmp_path, gzip.compress(b"{}\n", mtime=0))
    project = load_project(recipe)
    ledger = TransportBudget.initialize(
        tmp_path / "work/corpora/example/transport-budget.sqlite", project
    )
    timeouts: list[float] = []
    socket = types.SimpleNamespace(settimeout=timeouts.append)
    response = types.SimpleNamespace(
        fp=types.SimpleNamespace(
            fp=types.SimpleNamespace(raw=types.SimpleNamespace(_sock=socket))
        )
    )
    _set_response_deadline(response, ledger)
    assert len(timeouts) == 1
    assert 0 < timeouts[0] <= 30
    with pytest.raises(ValueError, match="cannot enforce transport deadline"):
        _set_response_deadline(types.SimpleNamespace(fp=object()), ledger)


def test_hf_exact_length_body_does_not_recheck_closed_socket(tmp_path: Path) -> None:
    recipe = _budgeted_hf_fixture(tmp_path, gzip.compress(b"{}\n", mtime=0))
    project = load_project(recipe)
    ledger = TransportBudget.initialize(
        tmp_path / "work/corpora/example/transport-budget.sqlite", project
    )
    socket = types.SimpleNamespace(settimeout=lambda _: None)

    class ClosingResponse:
        def __init__(self, body: bytes) -> None:
            self.body = body
            self.headers = {"Content-Length": str(len(body))}
            self.fp = types.SimpleNamespace(raw=types.SimpleNamespace(_sock=socket))

        def read(self, size: int) -> bytes:
            chunk, self.body = self.body[:size], self.body[size:]
            if not self.body:
                self.fp = None
            return chunk

    assert _read_metadata_body(ClosingResponse(b"{}"), ledger) == b"{}"
    transfer = ledger.reserve_transfer("one", "shard", 3)
    digest, size = _copy_hf_body(
        ClosingResponse(b"abc"), tmp_path / "shard", 3, ledger, transfer
    )
    assert size == 3
    assert digest == hashlib.sha256(b"abc").hexdigest()


def test_accounted_reads_bound_hidden_urllib_socket(tmp_path: Path) -> None:
    content = gzip.compress(b"{}\n", mtime=0)
    recipe = _budgeted_hf_fixture(tmp_path, content)
    project = load_project(recipe)
    ledger = TransportBudget.initialize(
        tmp_path / "work/corpora/example/transport-budget.sqlite", project
    )

    class HiddenSocketResponse:
        def __init__(self, body: bytes) -> None:
            self.body = body
            self.headers = {"Content-Length": str(len(body))}
            self.fp = object()
            self.reads: list[int] = []

        def read(self, size: int) -> bytes:
            self.reads.append(size)
            chunk, self.body = self.body[:size], self.body[size:]
            return chunk

    metadata = HiddenSocketResponse(b"x" * 70_000)
    assert _read_metadata_body(metadata, ledger) == b"x" * 70_000
    assert metadata.reads == [65_536, 4_464]
    transfer = ledger.reserve_transfer("one", "complete-0001.json.gz", len(content))
    source = HiddenSocketResponse(content)
    target = tmp_path / "source.gz"
    assert _copy_hf_body(source, target, len(content), ledger, transfer) == (
        hashlib.sha256(content).hexdigest(),
        len(content),
    )
    assert target.read_bytes() == content


def test_hidden_socket_blocked_read_times_out_and_restores_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _budgeted_hf_fixture(tmp_path, gzip.compress(b"{}\n", mtime=0))
    project = load_project(recipe)
    ledger = TransportBudget.initialize(
        tmp_path / "work/corpora/example/transport-budget.sqlite", project
    )
    monkeypatch.setattr(ledger, "remaining_seconds", lambda: 0.2)
    read_fd, write_fd = os.pipe()

    class BlockedResponse:
        def __init__(self) -> None:
            self.headers = {"Content-Length": "1"}
            self.fp = object()

        def read(self, _size: int) -> bytes:
            return os.read(read_fd, 1)

    previous_handler = signal.getsignal(signal.SIGALRM)

    def sentinel_handler(_number: int, _frame: object) -> None:
        raise AssertionError("previous signal handler ran during bounded read")

    signal.signal(signal.SIGALRM, sentinel_handler)
    try:
        with pytest.raises(TimeoutError, match="transport deadline"):
            _read_metadata_body(BlockedResponse(), ledger)
        assert signal.getsignal(signal.SIGALRM) is sentinel_handler
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
        assert ledger.receipt()["metadata_actual"] == 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        os.close(read_fd)
        os.close(write_fd)


def test_hidden_socket_success_restores_handler(tmp_path: Path) -> None:
    recipe = _budgeted_hf_fixture(tmp_path, gzip.compress(b"{}\n", mtime=0))
    project = load_project(recipe)
    ledger = TransportBudget.initialize(
        tmp_path / "work/corpora/example/transport-budget.sqlite", project
    )

    class Response:
        fp = object()

        def read(self, _size: int) -> bytes:
            return b"ok"

    previous_handler = signal.getsignal(signal.SIGALRM)

    def sentinel_handler(_number: int, _frame: object) -> None:
        raise AssertionError("previous handler must not run")

    signal.signal(signal.SIGALRM, sentinel_handler)
    try:
        assert _read_with_deadline(Response(), 2, ledger) == b"ok"
        assert signal.getsignal(signal.SIGALRM) is sentinel_handler
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def test_hidden_socket_refuses_occupied_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _budgeted_hf_fixture(tmp_path, gzip.compress(b"{}\n", mtime=0))
    project = load_project(recipe)
    ledger = TransportBudget.initialize(
        tmp_path / "work/corpora/example/transport-budget.sqlite", project
    )
    monkeypatch.setattr(signal, "getitimer", lambda _kind: (1.0, 0.0))

    class UnreadResponse:
        fp = object()

        def read(self, _size: int) -> bytes:
            raise AssertionError("occupied timer must refuse before reading")

    with pytest.raises(ValueError, match="existing HTTP deadline timer"):
        _read_with_deadline(UnreadResponse(), 1, ledger)


def test_hidden_socket_refuses_non_main_thread(tmp_path: Path) -> None:
    recipe = _budgeted_hf_fixture(tmp_path, gzip.compress(b"{}\n", mtime=0))
    project = load_project(recipe)
    ledger = TransportBudget.initialize(
        tmp_path / "work/corpora/example/transport-budget.sqlite", project
    )

    class UnreadResponse:
        fp = object()

        def read(self, _size: int) -> bytes:
            raise AssertionError("worker thread must refuse before reading")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_read_with_deadline, UnreadResponse(), 1, ledger)
        with pytest.raises(ValueError, match="cannot enforce transport deadline"):
            future.result()


def test_hf_xet_metadata_requires_downloaded_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = gzip.compress(b'{"text":"one"}\n', mtime=0)
    recipe = _budgeted_hf_fixture(tmp_path, content)
    project = load_project(recipe)
    root = tmp_path / "work"
    ledger_path = root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(ledger_path, project)
    _mock_budgeted_hf_http(monkeypatch, content, xet=True)
    lock = acquire(project, root)
    assert TransportBudget(ledger_path, project).receipt()["source_actual"] == len(
        content
    )
    assert verify_snapshot(lock["sources"]["one"]["snapshot_path"])["retrieval"]

    bad_root = tmp_path / "bad-work"
    bad_ledger_path = bad_root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(bad_ledger_path, project)
    corrupted = bytes([content[0] ^ 1]) + content[1:]
    _mock_budgeted_hf_http(monkeypatch, content, xet=True, payload=corrupted)
    with pytest.raises(ValueError, match="shard SHA-256 mismatch"):
        acquire(project, bad_root)


def test_hf_lfs_oid_is_pinned_content_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = gzip.compress(b'{"text":"one"}\n', mtime=0)
    recipe = _budgeted_hf_fixture(tmp_path, content)
    project = load_project(recipe)
    root = tmp_path / "work"
    TransportBudget.initialize(
        root / "corpora/example/transport-budget.sqlite", project
    )
    _mock_budgeted_hf_http(monkeypatch, content, lfs_oid=True)
    lock = acquire(project, root)
    assert verify_snapshot(lock["sources"]["one"]["snapshot_path"])["retrieval"]


def test_hf_transport_interruption_resume_and_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = gzip.compress(b'{"text":"one"}\n', mtime=0)
    recipe = _budgeted_hf_fixture(tmp_path, content)
    project = load_project(recipe)
    root = tmp_path / "work"
    ledger_path = root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(ledger_path, project)
    calls, reads = _mock_budgeted_hf_http(monkeypatch, content, interrupt_first=True)
    with pytest.raises(OSError, match="interrupted"):
        acquire(project, root)
    state = TransportBudget(ledger_path, project).receipt()
    assert state["source_charged"] == len(content)
    assert state["source_actual"] == 4
    assert state["transfers"][0]["status"] == "interrupted"
    assert all(size <= len(content) or size <= 65536 for size in reads)
    acquire(load_project(recipe), root)
    state = TransportBudget(ledger_path, project).receipt()
    assert state["source_charged"] == 2 * len(content)
    assert state["source_actual"] == len(content) + 4
    assert [item["status"] for item in state["transfers"]] == [
        "interrupted",
        "complete",
    ]
    assert len(calls) == 8
    with pytest.raises(ValueError, match="already exists"):
        TransportBudget.initialize(ledger_path, project)


def test_hf_transport_rejects_ambiguous_metadata_checksum_and_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = gzip.compress(b'{"text":"one"}\n', mtime=0)
    recipe = _budgeted_hf_fixture(tmp_path, content)
    source_file = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_file.read_text(encoding="utf-8"))
    source["acquisition"]["bounded_shards"][0]["declared_split"] = "test"
    _yaml(source_file, source)
    with pytest.raises(ValueError, match="contradicts"):
        load_project(recipe)
    source["acquisition"]["bounded_shards"][0]["declared_split"] = "train"
    _yaml(source_file, source)
    project = load_project(recipe)
    root = tmp_path / "work"
    ledger_path = root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(ledger_path, project)
    _mock_budgeted_hf_http(monkeypatch, content, ambiguous=True)
    with pytest.raises(ValueError, match="ambiguous"):
        acquire(project, root)
    assert TransportBudget(ledger_path, project).receipt()["source_charged"] == 0
    _mock_budgeted_hf_http(monkeypatch, content, tree_size=len(content) + 1)
    with pytest.raises(ValueError, match="size/type mismatch"):
        acquire(project, root)
    assert TransportBudget(ledger_path, project).receipt()["source_charged"] == 0
    source["acquisition"]["bounded_shards"][0]["expected_sha256"] = "0" * 64
    _yaml(source_file, source)
    with pytest.raises(ValueError, match="binding changed"):
        TransportBudget(ledger_path, load_project(recipe))
    # Restore the declaration, then make the received source differ from its pinned digest.
    source["acquisition"]["bounded_shards"][0]["expected_sha256"] = hashlib.sha256(
        content
    ).hexdigest()
    _yaml(source_file, source)
    calls, _ = _mock_budgeted_hf_http(monkeypatch, content, payload=content[:-1] + b"x")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        acquire(load_project(recipe), root)
    state = TransportBudget(ledger_path, project).receipt()
    assert state["source_charged"] == len(content)
    assert state["source_actual"] == len(content)
    assert state["transfers"][0]["status"] == "failed"
    with pytest.raises(ValueError, match="terminal"):
        acquire(project, root)
    assert len(calls) == 4  # terminal shard is refused before another request


def test_hf_transport_caps_stop_before_source_or_redirect_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = gzip.compress(b'{"text":"one"}\n', mtime=0)
    recipe = _budgeted_hf_fixture(tmp_path, content)
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    config["transport_budget"]["max_source_body_bytes"] = len(content) - 1
    _yaml(recipe, config)
    project = load_project(recipe)
    root = tmp_path / "work"
    ledger_path = root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(ledger_path, project)
    calls, _ = _mock_budgeted_hf_http(monkeypatch, content)
    with pytest.raises(ValueError, match="source body allowance exhausted"):
        acquire(project, root)
    assert calls == []
    assert TransportBudget(ledger_path, project).receipt()["source_actual"] == 0

    other_root = tmp_path / "other"
    config["transport_budget"]["max_source_body_bytes"] = len(content)
    config["transport_budget"]["max_metadata_body_bytes"] = 241
    _yaml(recipe, config)
    project = load_project(recipe)
    other_ledger = other_root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(other_ledger, project)
    calls, _ = _mock_budgeted_hf_http(monkeypatch, content)
    with pytest.raises(ValueError, match="metadata body allowance exhausted"):
        acquire(project, other_root)
    state = TransportBudget(other_ledger, project).receipt()
    assert state["metadata_actual"] <= 241
    assert state["source_actual"] == 0
    assert not any("cdn.example.test" in url for url in calls)


def test_hf_http_error_body_is_charged_and_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = gzip.compress(b'{"text":"one"}\n', mtime=0)
    recipe = _budgeted_hf_fixture(tmp_path, content)
    project = load_project(recipe)
    root = tmp_path / "work"
    path = root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(path, project)
    calls, _ = _mock_budgeted_hf_http(monkeypatch, content, fail_http=True)
    with pytest.raises(HTTPError):
        acquire(project, root)
    state = TransportBudget(path, project).receipt()
    assert state["metadata_actual"] >= len(b"redirect-body") + len(b"error-503")
    assert state["source_charged"] == len(content)
    assert state["source_actual"] == 0
    assert state["transfers"][0]["status"] == "failed"
    assert len(calls) == 4


def test_hf_transport_disk_decompression_and_deadline_are_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = gzip.compress(b'{"text":"one"}\n', mtime=0)
    recipe = _budgeted_hf_fixture(tmp_path, content)
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    config["transport_budget"]["max_disk_bytes"] = 100
    _yaml(recipe, config)
    project = load_project(recipe)
    root = tmp_path / "disk"
    ledger_path = root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(ledger_path, project)
    calls, _ = _mock_budgeted_hf_http(monkeypatch, content)
    with pytest.raises(ValueError, match="disk allowance"):
        acquire(project, root)
    assert calls == []

    assert TransportBudget(ledger_path, project).receipt()["source_charged"] == 0

    config["transport_budget"]["max_disk_bytes"] = 1_000_000
    _yaml(recipe, config)
    source_file = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_file.read_text(encoding="utf-8"))
    source["acquisition"]["max_decompressed_bytes"] = 5
    _yaml(source_file, source)
    project = load_project(recipe)
    root = tmp_path / "decompress"
    TransportBudget.initialize(
        root / "corpora/example/transport-budget.sqlite", project
    )
    _mock_budgeted_hf_http(monkeypatch, content)
    with pytest.raises(ValueError, match="max_decompressed_bytes"):
        acquire(project, root)
    state = TransportBudget(
        root / "corpora/example/transport-budget.sqlite", project
    ).receipt()
    assert state["source_actual"] == len(content)
    assert state["failures"][-1]["error_type"] == "ValueError"

    root = tmp_path / "deadline"
    ledger_path = root / "corpora/example/transport-budget.sqlite"
    TransportBudget.initialize(ledger_path, project)
    created = TransportBudget(ledger_path, project).receipt()["created_at"]
    import sparselab.corpus.transport_budget as budget_module

    monkeypatch.setattr(
        budget_module, "time", types.SimpleNamespace(time=lambda: created + 121)
    )
    calls, _ = _mock_budgeted_hf_http(monkeypatch, content)
    with pytest.raises(ValueError, match="deadline"):
        acquire(project, root)
    assert calls == []


def test_hf_transport_ledger_reopens_and_rejects_exhaustion_or_corruption(
    tmp_path: Path,
) -> None:
    content = gzip.compress(b'{"text":"one"}\n', mtime=0)
    recipe = _budgeted_hf_fixture(tmp_path, content)
    project = load_project(recipe)
    path = tmp_path / "work/corpora/example/transport-budget.sqlite"
    ledger = TransportBudget.initialize(path, project)
    first = ledger.reserve_transfer("one", "complete-0001.json.gz", len(content))
    ledger.charge_transfer_actual(first, 3)
    ledger.finish_transfer(first, success=False, interrupted=True)
    reopened = TransportBudget(path, project)
    second = reopened.reserve_transfer("one", "complete-0001.json.gz", len(content))
    reopened.finish_transfer(second, success=False, interrupted=True)
    with pytest.raises(ValueError, match="retry allowance exhausted"):
        TransportBudget(path, project).reserve_transfer(
            "one", "complete-0001.json.gz", len(content)
        )
    state = reopened.receipt()
    state["source_charged"] = 0
    with sqlite3.connect(path) as db:
        db.execute("UPDATE state SET body=? WHERE id=1", (json.dumps(state),))
    with pytest.raises(ValueError, match="counters are invalid"):
        TransportBudget(path, project)


@pytest.mark.parametrize("format", ["jsonl", "json.gz", "parquet"])
def test_hf_bounded_replay_hash_and_row_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format: str
) -> None:
    rows = [
        {
            "text": f"passage {index}",
            "id": f"source-{index}",
            "url": f"https://example.org/{index}",
        }
        for index in range(6)
    ]
    if format == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        buffer = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows), buffer)
        content = buffer.getvalue()
    else:
        raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
        content = gzip.compress(raw, mtime=0) if format.endswith(".gz") else raw
    shard_path = f"default/train/data.{format}"
    recipe = _bounded_hf_fixture(tmp_path, content, shard_path, rows=5)
    calls = _mock_hf_stream(monkeypatch, content)
    project = load_project(recipe)
    first = acquire(project, tmp_path / "first")
    second = acquire(project, tmp_path / "second")
    assert len(calls) == 2
    assert all("datasets/org/dataset/resolve/" in url for url in calls)
    assert (
        first["sources"]["one"]["snapshot_sha256"]
        == second["sources"]["one"]["snapshot_sha256"]
    )
    snapshot = verify_snapshot(first["sources"]["one"]["snapshot_path"])
    retrieval = snapshot["retrieval"]
    selected_indices = [
        index
        for index in range(5)
        if int(
            hashlib.sha256(
                json.dumps(
                    ["d" * 40, shard_path, index], separators=(",", ":"), sort_keys=True
                ).encode()
            ).hexdigest(),
            16,
        )
        % 3
        in {0, 2}
    ]
    selection = retrieval["shards"][0]
    assert selection["scanned_rows"] == 5
    assert [
        row["source_row_index"] for row in selection["selected_rows"]
    ] == selected_indices
    assert selection["source_shard_sha256"] == hashlib.sha256(content).hexdigest()
    emitted = (
        (
            Path(first["sources"]["one"]["snapshot_path"])
            / "files"
            / selection["output_path"]
        )
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert [json.loads(line)["text"] for line in emitted] == [
        rows[index]["text"] for index in selected_indices
    ]
    assert all(
        json.loads(line)["_sparselab_source"]["source_row_index"] == index
        and json.loads(line)["_sparselab_source"]["id"] == rows[index]["id"]
        and json.loads(line)["_sparselab_source"]["source_shard_sha256"]
        == hashlib.sha256(content).hexdigest()
        and json.loads(line)["_sparselab_source"]["source_row_sha256"]
        == selection["selected_rows"][position]["source_row_sha256"]
        for position, (line, index) in enumerate(zip(emitted, selected_indices))
    )


def test_hf_bounded_rejects_caps_bad_shards_and_bad_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b'{"text":"first"}\n{"text":123}\n'
    path = "default/train/data.jsonl"
    recipe = _bounded_hf_fixture(tmp_path, content, path, rows=2)
    _mock_hf_stream(monkeypatch, content)
    with pytest.raises(TypeError, match="non-string"):
        acquire(load_project(recipe), tmp_path / "work")
    source_path = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_path.read_text())
    source["acquisition"]["bounded_shards"][0]["path"] = "../train/data.jsonl"
    _yaml(source_path, source)
    with pytest.raises(ValueError, match="unsafe logical path"):
        load_project(recipe)
    source["acquisition"]["bounded_shards"][0]["path"] = path
    source["acquisition"]["bounded_shards"][0]["max_shard_bytes"] = len(content) - 1
    _yaml(source_path, source)
    with pytest.raises(ValueError, match="max_bytes"):
        acquire(load_project(recipe), tmp_path / "work")
    source["acquisition"]["bounded_shards"][0]["max_shard_bytes"] = len(content)
    source["acquisition"]["bounded_shards"][0]["expected_sha256"] = "0" * 64
    _yaml(source_path, source)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        acquire(load_project(recipe), tmp_path / "work")
    assert not (tmp_path / "work/corpora/example/acquisition.json").exists()


def test_hf_bounded_output_caps_and_config_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b'{"text":"one"}\n{"text":"two"}\n'
    recipe = _bounded_hf_fixture(tmp_path, data, "default/train/data.jsonl", rows=2)
    _mock_hf_stream(monkeypatch, data)
    source_file = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_file.read_text(encoding="utf-8"))
    shard = source["acquisition"]["bounded_shards"][0]
    shard["hash_modulus"] = 1
    shard["hash_remainders"] = [0]
    source["acquisition"]["max_rows"] = 1
    _yaml(source_file, source)
    with pytest.raises(ValueError, match="max_rows"):
        acquire(load_project(recipe), tmp_path / "work")
    source["acquisition"]["max_rows"] = 2
    source["acquisition"]["max_bytes"] = 10
    _yaml(source_file, source)
    with pytest.raises(ValueError, match="max_bytes"):
        acquire(load_project(recipe), tmp_path / "work")
    source["acquisition"]["max_bytes"] = 4096
    shard["path"] = "default/test/data.jsonl"
    _yaml(source_file, source)
    with pytest.raises(ValueError, match="declared split"):
        load_project(recipe)
    shard["path"] = "other/train/data.jsonl"
    _yaml(source_file, source)
    with pytest.raises(ValueError, match="declared config"):
        load_project(recipe)
    shard["path"] = "default/train/data.zst"
    _yaml(source_file, source)
    with pytest.raises(ValueError, match="JSONL stream"):
        load_project(recipe)


def test_v3_training_policy_requires_explicit_source_review(
    tmp_path: Path,
) -> None:
    recipe = _fixture(tmp_path)
    source_file = recipe.parent / "sources/one.yaml"
    release_file = recipe.parent / "release.yaml"
    source = yaml.safe_load(source_file.read_text(encoding="utf-8"))
    release = yaml.safe_load(release_file.read_text(encoding="utf-8"))
    source["schema_version"] = 3
    source.pop("redistribution")
    source["explicit_training_restriction"] = "none_found"
    source["rights"] = {
        "training_eligibility": "eligible",
        "redistribution_mode": "redistributable_under_source_terms",
        "spdx_expression": "MIT",
    }
    release["schema_version"] = 3
    release["publication_mode"] = "metadata_reconstruction_only"
    release["training_use_policy"] = "allowed_unless_explicitly_prohibited"
    _yaml(source_file, source)
    _yaml(release_file, release)
    project = load_project(recipe)
    assert (
        source_declaration_payload(project.sources[0])["explicit_training_restriction"]
        == "none_found"
    )
    assert (
        release_declaration_payload(project.release)["training_use_policy"]
        == "allowed_unless_explicitly_prohibited"
    )
    source["explicit_training_restriction"] = "incompatible"
    _yaml(source_file, source)
    with pytest.raises(ValueError, match="prohibited basis"):
        load_project(recipe)
    source["rights"]["training_eligibility"] = "ineligible"
    source["rights"]["training_restriction"] = {
        "kind": "prohibited",
        "basis": "Explicit no-training condition",
    }
    _yaml(source_file, source)
    assert (
        load_project(recipe).sources[0].explicit_training_restriction == "incompatible"
    )
    release.pop("training_use_policy")
    _yaml(release_file, release)
    with pytest.raises(ValueError, match="training_use_policy"):
        load_project(recipe)


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


def _wikimedia_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, xml: bytes
) -> tuple[Path, bytes]:
    name = "enwikibooks-20260901-pages-articles-multistream.xml.bz2"
    prefix = "https://dumps.wikimedia.org/enwikibooks/20260901/"
    content = bz2.compress(xml)
    sha1 = hashlib.sha1(content).hexdigest()
    recipe = _fixture(
        tmp_path,
        kind="wikimedia_dump",
        revision="20260901",
        uri=prefix + name,
        acquisition={
            "expected_sha1": sha1,
            "expected_sha256": hashlib.sha256(content).hexdigest(),
            "checksum_uri": prefix + "enwikibooks-20260901-sha1sums.txt",
            "max_compressed_bytes": len(content),
            "max_decompressed_bytes": len(xml) + 1024,
            "max_scanned_pages": 20,
            "max_selected_pages": 20,
            "max_emitted_bytes": 10000,
        },
    )

    class Response(io.BytesIO):
        def __init__(self, url: str, body: bytes) -> None:
            super().__init__(body)
            self.url = url

    class Opener:
        def open(self, request: urllib.request.Request, *, timeout: int) -> Response:
            if request.full_url.endswith("sha1sums.txt"):
                return Response(request.full_url, f"{sha1}  {name}\n".encode())
            assert request.full_url == prefix + name
            return Response(request.full_url, content)

    monkeypatch.setattr(urllib.request, "build_opener", lambda: Opener())
    return recipe, content


def test_wikimedia_bounded_xml_provenance_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = b"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" xml:lang="en">
<siteinfo><dbname>enwikibooks</dbname></siteinfo>
<page><title>Programming/Python</title><ns>0</ns><id>17</id>
<revision><id>93</id><timestamp>2026-08-31T00:00:00Z</timestamp>
<text>== Code ==&#10;Use &lt;code&gt;print(1)&lt;/code&gt; and &lt;math&gt;x^2&lt;/math&gt;.
[[File:example.png]] [[Python|Python language]] {{citation|unknown}}</text></revision></page>
<page><title>Private secrets</title><ns>0</ns><id>18</id>
<revision><id>94</id><timestamp>2026-08-31T00:00:00Z</timestamp><text>Secret key</text></revision></page>
<page><title>Redirect</title><ns>0</ns><id>19</id><redirect title="Elsewhere"/>
<revision><id>95</id><timestamp>2026-08-31T00:00:00Z</timestamp><text>Skip me</text></revision></page>
</mediawiki>"""
    recipe, content = _wikimedia_fixture(tmp_path, monkeypatch, xml)
    project = load_project(recipe)
    first = acquire(project, tmp_path / "first")
    second = acquire(project, tmp_path / "second")
    assert (
        first["sources"]["one"]["snapshot_sha256"]
        == second["sources"]["one"]["snapshot_sha256"]
    )
    snapshot_path = Path(first["sources"]["one"]["snapshot_path"])
    manifest = verify_snapshot(snapshot_path)
    receipt = manifest["retrieval"]
    assert receipt["source_sha256"] == hashlib.sha256(content).hexdigest()
    assert receipt["scanned_pages"] == 3
    assert [(r["page_id"], r["revision_id"]) for r in receipt["selected_pages"]] == [
        ("17", "93")
    ]
    row = json.loads((snapshot_path / "files" / receipt["output_path"]).read_text())
    assert row["_sparselab_source"]["page_uri"] == "https://en.wikibooks.org/?curid=17"
    assert row["_sparselab_source"]["revision_timestamp"] == "2026-08-31T00:00:00Z"
    assert "print(1)" in row["text"] and "x^2" in row["text"]
    assert "example.png" not in row["text"] and "citation" not in row["text"]
    assert acquire(project, tmp_path / "first", offline=True) == first
    manifest["retrieval"]["selected_pages"][0]["page_id"] = "999"
    (snapshot_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="snapshot identity mismatch"):
        verify_snapshot(snapshot_path)


def test_interrupted_wikimedia_acquisition_reuses_verified_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = b"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" xml:lang="en">
<siteinfo><dbname>enwikibooks</dbname></siteinfo>
<page><title>Programming/Python</title><ns>0</ns><id>17</id>
<revision><id>93</id><timestamp>2026-08-31T00:00:00Z</timestamp>
<text>Use Python to write useful programs with careful examples.</text></revision></page>
</mediawiki>"""
    recipe, _ = _wikimedia_fixture(tmp_path, monkeypatch, xml)
    project = load_project(recipe)
    work = tmp_path / "work"
    first = acquire(project, work)
    (work / "corpora" / project.config.id / "acquisition.json").unlink()

    def no_network() -> None:
        raise AssertionError("verified interrupted snapshot must not be fetched again")

    monkeypatch.setattr(urllib.request, "build_opener", no_network)
    restored = acquire(project, work)
    assert restored["sources"]["one"] == first["sources"]["one"]


def test_verified_snapshot_alias_reused_across_project_ids_without_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = b"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" xml:lang="en">
<siteinfo><dbname>enwikibooks</dbname></siteinfo>
<page><title>Programming/Python</title><ns>0</ns><id>17</id>
<revision><id>93</id><timestamp>2026-08-31T00:00:00Z</timestamp>
<text>Useful short programming explanation.</text></revision></page>
</mediawiki>"""
    recipe, _ = _wikimedia_fixture(tmp_path, monkeypatch, xml)
    work = tmp_path / "work"
    old = load_project(recipe)
    old_lock = acquire(old, work)
    old_snapshot = Path(old_lock["sources"]["one"]["snapshot_path"])
    declaration = yaml.safe_load(recipe.read_text())
    declaration["id"] = "new-project"
    _yaml(recipe, declaration)
    new = load_project(recipe)
    alias = work / "corpora" / new.config.id / "snapshots" / "one" / old_snapshot.name
    alias.parent.mkdir(parents=True)
    alias.symlink_to(old_snapshot, target_is_directory=True)

    def no_network() -> None:
        raise AssertionError("verified snapshot alias must prevent a transfer")

    monkeypatch.setattr(urllib.request, "build_opener", no_network)
    new_lock = acquire(new, work)
    assert new_lock["sources"]["one"]["snapshot_sha256"] == old_snapshot.name
    assert acquire(new, work, offline=True) == new_lock


def test_wikimedia_rejects_checksum_caps_and_doctype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = b"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" xml:lang="en">
<siteinfo><dbname>enwikibooks</dbname></siteinfo>
<page><title>English programming</title><ns>0</ns><id>3</id>
<revision><id>5</id><timestamp>2026-08-01T00:00:00Z</timestamp>
<text>Useful English programming explanation.</text></revision></page></mediawiki>"""
    recipe, content = _wikimedia_fixture(tmp_path, monkeypatch, xml)
    source_path = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_path.read_text())
    source["acquisition"]["expected_sha256"] = "0" * 64
    _yaml(source_path, source)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        acquire(load_project(recipe), tmp_path / "wrong")
    source["acquisition"]["expected_sha256"] = hashlib.sha256(content).hexdigest()
    source["acquisition"]["max_compressed_bytes"] = len(content) - 1
    _yaml(source_path, source)
    with pytest.raises(ValueError, match="max_bytes"):
        acquire(load_project(recipe), tmp_path / "compressed-cap")
    source["acquisition"]["max_compressed_bytes"] = len(content)
    source["acquisition"]["max_emitted_bytes"] = 10
    _yaml(source_path, source)
    with pytest.raises(ValueError, match="emitted JSONL"):
        acquire(load_project(recipe), tmp_path / "output-cap")
    malicious = xml.replace(
        b"<mediawiki",
        b'<!DOCTYPE mediawiki [<!ENTITY bad SYSTEM "file:///etc/passwd">]><mediawiki',
        1,
    )
    recipe, _ = _wikimedia_fixture(tmp_path, monkeypatch, malicious)
    with pytest.raises(ValueError, match="DTD/entities"):
        acquire(load_project(recipe), tmp_path / "doctype")
