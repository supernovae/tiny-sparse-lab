"""Offline scale check for the pinned Scoutflo bounded-Git acquisition shape."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path
from urllib.parse import unquote

import yaml

from sparselab.corpus.acquisition import acquire, verify_snapshot
from sparselab.corpus.project import SourceDeclaration, load_project
from sparselab.corpus.transport_budget import TransportBudget


def test_scoutflo_pinned_declaration_is_complete_and_bounded() -> None:
    path = (
        Path(__file__).parents[1]
        / "experiments/research/kernel-memory-lab/corpus-scale/sources/scoutflo.yaml"
    )
    source = SourceDeclaration.model_validate(yaml.safe_load(path.read_text()))
    blobs = source.acquisition.bounded_blobs
    assert source.id == "kml_scale_scoutflo"
    assert source.revision == "acc55da0224fda5eda580939a15bc60990e8e69f"
    assert source.acquisition.tree_oid == "1e77048f12715248fd7a1cab6c4d0bdc4eb2ec86"
    assert len(blobs) == 433
    assert (
        sum(blob.max_bytes for blob in blobs)
        == source.acquisition.max_bytes
        == 2_087_503
    )
    assert {blob.path for blob in blobs if blob.path in {"LICENSE", "README.md"}} == {
        "LICENSE",
        "README.md",
    }
    assert all(
        blob.path in {"LICENSE", "README.md"}
        or blob.path.startswith(
            ("AWS Playbooks/", "K8s Playbooks/", "Sentry Playbooks/")
        )
        and blob.path.endswith(".md")
        for blob in blobs
    )


def test_bounded_git_433_blob_scale_offline(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "recipe"
    (root / "sources").mkdir(parents=True)
    revision = "a" * 40
    tree_oid = "b" * 40
    contents = {
        "LICENSE": b"MIT License\n",
        "README.md": b"# Synthetic repository\n",
        **{
            f"{('AWS Playbooks', 'K8s Playbooks', 'Sentry Playbooks')[index % 3]}/playbook-{index:04d}.md": f"# Playbook {index}\n".encode()
            for index in range(431)
        },
    }
    blobs = [
        {
            "path": path,
            "git_blob_oid": hashlib.sha1(
                f"blob {len(body)}\0".encode() + body
            ).hexdigest(),
            "max_bytes": len(body),
        }
        for path, body in sorted(contents.items())
    ]
    body_bytes = sum(len(body) for body in contents.values())
    (root / "corpus.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "scoutflo_scale_fixture",
                "sources": ["sources/scoutflo.yaml"],
                "transforms": [],
                "splits": "splits.yaml",
                "release": "release.yaml",
                "transport_budget": {
                    "attempt_id": "scoutflo_scale_fixture",
                    "max_source_body_bytes": 2 * body_bytes,
                    "max_metadata_body_bytes": 4_194_304,
                    "max_transfers": 866,
                    "max_retries_per_shard": 1,
                    "max_wall_seconds": 600,
                    "max_disk_bytes": 16 * 1024 * 1024,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "splits.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "unit": "document",
                "assignments": {"synthetic-placeholder": "train"},
            }
        ),
        encoding="utf-8",
    )
    (root / "release.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "publication_mode": "metadata_reconstruction_only",
                "mixture": {"incident_response_docs": 1.0},
                "lm": {"selected": False, "training_splits": []},
                "chat": {"selected": False, "training_splits": []},
            }
        ),
        encoding="utf-8",
    )
    (root / "sources/scoutflo.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "id": "scoutflo",
                "kind": "git",
                "canonical_uri": "https://github.com/org/repo",
                "revision": revision,
                "license": "MIT repository claim, per-file review required",
                "rights": {
                    "training_eligibility": "review_required",
                    "redistribution_mode": "review_required",
                },
                "domains": ["incident_response_docs"],
                "document_kinds": ["markdown"],
                "source_family": "scoutflo",
                "acquisition": {
                    "max_bytes": body_bytes,
                    "tree_oid": tree_oid,
                    "bounded_blobs": blobs,
                },
            }
        ),
        encoding="utf-8",
    )

    class Response(io.BytesIO):
        def __init__(self, body: bytes, url: str):
            super().__init__(body)
            self.url = url
            self.status = 200
            self.headers = {"Content-Length": str(len(body))}

    class Opener:
        def open(self, request: urllib.request.Request, timeout: float) -> Response:
            url = request.full_url
            if "/git/commits/" in url:
                body = json.dumps({"sha": revision, "tree": {"sha": tree_oid}}).encode()
            elif "/git/trees/" in url:
                body = json.dumps(
                    {
                        "sha": tree_oid,
                        "truncated": False,
                        "tree": [
                            {
                                "path": blob["path"],
                                "type": "blob",
                                "mode": "100644",
                                "sha": blob["git_blob_oid"],
                                "size": blob["max_bytes"],
                            }
                            for blob in blobs
                        ],
                    }
                ).encode()
            else:
                prefix = f"https://raw.githubusercontent.com/org/repo/{revision}/"
                assert url.startswith(prefix)
                body = contents[unquote(url.removeprefix(prefix))]
            return Response(body, url)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: Opener())
    project = load_project(root / "corpus.yaml")
    ledger = tmp_path / "work/corpora/scoutflo_scale_fixture/transport-budget.sqlite"
    TransportBudget.initialize(ledger, project)
    lock = acquire(project, tmp_path / "work")
    snapshot = verify_snapshot(lock["sources"]["scoutflo"]["snapshot_path"])
    assert len(snapshot["files"]) == 433
    assert sum(item["size"] for item in snapshot["files"]) == body_bytes
    receipt = TransportBudget(ledger, project).receipt()
    assert receipt["source_actual"] == body_bytes
    assert receipt["source_charged"] == body_bytes
    assert receipt["metadata_actual"] > 0
    assert len(receipt["transfers"]) == 433
    assert all(item["status"] == "complete" for item in receipt["transfers"])
