"""Tiny pinned Parquet and loopback HTTP; no Hub or model downloads."""

import hashlib
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from sparselab.data import sources
from sparselab.data.bounded_acquisition import AcquisitionInterrupted, DatasetTransport
from sparselab.experiments.acquisition_admission import AcquisitionAdmission


def pinned_files(tmp_path, monkeypatch, *, lfs=False):
    import pyarrow as pa
    import pyarrow.parquet as pq

    tmp_path.mkdir(parents=True, exist_ok=True)
    files = {}
    inventory = {}
    for split in ("train", "validation"):
        path = tmp_path / (split + ".parquet")
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {"text": f"{split} local row {i}: the fox explored the forest."}
                    for i in range(4)
                ]
            ),
            path,
        )
        data = path.read_bytes()
        files[path.name] = data
        row = {
            "path": path.name,
            "size": len(data),
            "blob_id": hashlib.sha1(
                b"blob " + str(len(data)).encode() + b"\0" + data
            ).hexdigest(),
        }
        if lfs:
            row.update(blob_id="a" * 40, lfs_sha256=hashlib.sha256(data).hexdigest())
        inventory[split] = [row]
    declaration = {
        "kind": "huggingface",
        "repo_id": "fixture/portable",
        "revision": "b" * 40,
        "config": "default",
        "splits": {"train": "train", "validation": "validation"},
        "text_field": "text",
        "license": "MIT",
        "attribution": "Locally authored fixture",
        "selection": {"mode": "exhaustion"},
        "resources": {
            "max_source_records": 20,
            "max_text_bytes": 100000,
            "max_record_bytes": 10000,
            "max_work_bytes": 10000000,
            "min_free_bytes": 0,
        },
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(declaration))
    with monkeypatch.context() as context:
        context.setattr(
            sources,
            "_hub_identity",
            lambda source, **kw: {
                "repo_id": source.repo_id,
                "revision": source.revision,
                "loader": "parquet",
                "loader_options": {},
                "files": inventory,
            },
        )
        lock_path = sources.lock_source(
            source, tmp_path / "source.lock.json", tmp_path / "unused"
        )
    return lock_path, sources.verify_lock(lock_path), files


def admission(lock, **changes):
    value = {
        "acquisition_version": 1,
        "provider": "huggingface",
        "repo_id": lock["source"]["repo_id"],
        "revision": lock["source"]["revision"],
        "lock_sha256": lock["lock_sha256"],
        "allow_network": True,
        "decoder_profile": "scalar-parquet-v1",
        "max_source_body_bytes": 1000000,
        "max_metadata_body_bytes": 10000,
        "max_transfers": 8,
        "max_requests": 20,
        "max_retries_per_shard": 2,
        "max_wall_seconds": 600,
        "max_cache_bytes": 1000000,
        "max_cache_files": 8,
        "max_scanned_rows": 1000,
        "max_verification_bytes": 10000000,
        "max_observed_owned_bytes": 20000000,
        "max_observed_owned_inodes": 1000,
    }
    return AcquisitionAdmission.model_validate({**value, **changes})


@pytest.fixture
def http_files(monkeypatch):
    state = {"files": {}, "requests": [], "mode": "ok"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            state["requests"].append(
                {"path": self.path, "authorization": self.headers.get("Authorization")}
            )
            if state.get("on_request"):
                state["on_request"]()
            name = self.path.split("?")[0].rsplit("/", 1)[-1]
            data = state["files"][name]
            if state["mode"] == "redirect" and "redirected" not in self.path:
                body = b"redirect body"
                self.send_response(302)
                self.send_header("Location", "https://cdn-lfs.hf.co/redirected/" + name)
            elif state["mode"] == "error":
                body = b"error body"
                self.send_response(503)
            else:
                body = data
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state["mode"] == "short":
                self.wfile.write(body[:3])
            else:
                self.wfile.write(body)
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    build = urllib.request.build_opener

    class RoutedOpener:
        def __init__(self, handlers):
            self.opener = build(*handlers)

        def open(self, request, timeout):
            parsed = urlsplit(request.full_url)
            # Only the connection endpoint is substituted; production redirects,
            # bounds, framing, hashing, accounting and decoding all remain real.
            routed = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}" + parsed.path,
                headers=dict(request.header_items()),
            )
            return self.opener.open(routed, timeout=timeout)

    monkeypatch.setattr(
        urllib.request, "build_opener", lambda *handlers: RoutedOpener(handlers)
    )
    monkeypatch.setattr("sparselab.hf_auth.hub_auth_kwargs", dict)
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def transport(tmp_path, lock, **limits):
    work = tmp_path / "work"
    return DatasetTransport(
        lock,
        admission(lock, **limits),
        work / "cache",
        work / "snapshot",
        work,
        {"fixture": 1},
    )


@pytest.mark.parametrize("lfs", [False, True])
def test_typed_file_identity_and_verified_cache(tmp_path, monkeypatch, http_files, lfs):
    _, lock, files = pinned_files(tmp_path / "input", monkeypatch, lfs=lfs)
    http_files["files"] = files
    current = transport(tmp_path, lock)
    current.initialize()
    row = lock["provenance"]["files"]["train"][0]
    path = current.file(row)
    assert path.read_bytes() == files[row["path"]]
    assert current.file(row) == path
    assert len(http_files["requests"]) == 1
    receipt = current.receipt()
    assert receipt["source_actual"] == receipt["source_charged"] == row["size"]
    assert receipt["verification_bytes"] == 2 * row["size"]
    assert receipt["cache_hits"] == 1
    path.write_bytes(b"x" * row["size"])
    with pytest.raises(ValueError, match="typed content hash"):
        current.file(row)
    assert len(http_files["requests"]) == 1


def test_redirect_error_bodies_and_credentials(tmp_path, monkeypatch, http_files):
    _, lock, files = pinned_files(tmp_path / "input", monkeypatch)
    http_files.update(files=files, mode="redirect")
    monkeypatch.setattr(
        "sparselab.hf_auth.hub_auth_kwargs", lambda: {"token": "fixture-token"}
    )
    current = transport(tmp_path, lock)
    current.initialize()
    current.file(lock["provenance"]["files"]["train"][0])
    assert current.receipt()["metadata_actual"] == len(b"redirect body")
    assert [r["authorization"] for r in http_files["requests"]] == [
        "Bearer fixture-token",
        None,
    ]
    http_files["mode"] = "error"
    with pytest.raises(AcquisitionInterrupted):
        current.file(lock["provenance"]["files"]["validation"][0])
    assert current.receipt()["metadata_actual"] == len(b"redirect bodyerror body")


def test_crash_reservation_and_retry_are_never_refunded(
    tmp_path, monkeypatch, http_files
):
    _, lock, files = pinned_files(tmp_path / "input", monkeypatch)
    http_files["files"] = files
    current = transport(tmp_path, lock)
    current.initialize()
    row = lock["provenance"]["files"]["train"][0]
    # Simulate death after durable reservation, before actual-byte recording.
    current.reserve_transfer(current.spec.repo_id, row["path"], row["size"])
    resumed = transport(tmp_path, lock)
    resumed.initialize()
    resumed.file(row)
    receipt = resumed.receipt()
    assert receipt["source_charged"] == 2 * row["size"]
    assert receipt["source_actual"] == row["size"]
    assert receipt["transfers"][0]["status"] == "interrupted"
    assert receipt["observed_body_bytes_may_be_incomplete_after_crash"]


@pytest.mark.parametrize(
    "guarantee",
    [
        "require_hard_disk_bytes",
        "require_hard_decoder_memory_bytes",
        "require_expanded_bytes",
    ],
)
def test_unavailable_guarantees_refuse_before_network(
    tmp_path, monkeypatch, http_files, guarantee
):
    _, lock, _ = pinned_files(tmp_path / "input", monkeypatch)
    with pytest.raises(ValueError, match="guarantee is unavailable"):
        transport(tmp_path, lock, **{guarantee: 100000})
    assert not http_files["requests"]
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("limit", ["max_source_body_bytes", "max_cache_bytes"])
def test_full_file_reservation_before_request(tmp_path, monkeypatch, http_files, limit):
    _, lock, files = pinned_files(tmp_path / "input", monkeypatch)
    http_files["files"] = files
    current = transport(tmp_path, lock, **{limit: 1})
    current.initialize()
    with pytest.raises(ValueError, match="budget|reservation"):
        current.file(lock["provenance"]["files"]["train"][0])
    assert not http_files["requests"]
    assert current.receipt()["source_actual"] == 0


def test_metadata_request_and_retry_limits(tmp_path, monkeypatch, http_files):
    _, lock, files = pinned_files(tmp_path / "input", monkeypatch)
    http_files.update(files=files, mode="redirect")
    current = transport(tmp_path, lock, max_requests=1)
    current.initialize()
    row = lock["provenance"]["files"]["train"][0]
    with pytest.raises(ValueError, match="requests"):
        current.file(row)
    assert len(http_files["requests"]) == 1
    assert current.receipt()["metadata_actual"] == len(b"redirect body")
    other = transport(tmp_path / "metadata", lock, max_metadata_body_bytes=1)
    other.initialize()
    with pytest.raises(ValueError, match="metadata"):
        other.file(row)
    assert other.receipt()["metadata_actual"] == 0
    http_files["mode"] = "short"
    retry = transport(tmp_path / "retry", lock, max_retries_per_shard=0)
    retry.initialize()
    with pytest.raises(AcquisitionInterrupted):
        retry.file(row)
    count = len(http_files["requests"])
    with pytest.raises(ValueError, match="retry"):
        retry.file(row)
    assert len(http_files["requests"]) == count


def test_cancellation_and_deadline_survive_reopen(tmp_path, monkeypatch, http_files):
    _, lock, files = pinned_files(tmp_path / "input", monkeypatch)
    http_files["files"] = files
    current = transport(tmp_path, lock)
    current.initialize()
    current.cancel.touch()
    with pytest.raises(AcquisitionInterrupted, match="cancellation"):
        transport(tmp_path, lock).initialize()
    current.cancel.unlink()
    deadline = current.receipt()["deadline"]
    monkeypatch.setattr("time.time", lambda: deadline + 1)
    with pytest.raises(ValueError, match="deadline|wall"):
        transport(tmp_path, lock).initialize()
    assert not http_files["requests"]


@pytest.mark.parametrize(
    "damage",
    ["missing_ledger", "foreign", "guard_symlink", "journal_symlink", "cache_symlink"],
)
def test_cache_and_ledger_fail_closed(tmp_path, monkeypatch, http_files, damage):
    _, lock, files = pinned_files(tmp_path / "input", monkeypatch)
    http_files["files"] = files
    current = transport(tmp_path, lock)
    current.initialize()
    row = lock["provenance"]["files"]["train"][0]
    current.file(row)
    count = len(http_files["requests"])
    if damage == "missing_ledger":
        current.path.unlink()
    elif damage == "foreign":
        (current.cache / ".owner").write_text("other")
    elif damage in {"guard_symlink", "journal_symlink"}:
        member = "guard" if damage == "guard_symlink" else "transport.sqlite-journal"
        (current.root / member).symlink_to(tmp_path / "outside")
    else:
        (current.cache / "unexpected").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="ledger missing|another acquisition|symlink"):
        transport(tmp_path, lock).initialize()
    assert len(http_files["requests"]) == count


@pytest.mark.parametrize(
    "option",
    [
        {"filters": []},
        {"columns": ["text"]},
        {"features": {"text": {"_type": "Image"}}},
        {"on_bad_files": "skip"},
    ],
)
def test_unsupported_decode_profile_before_network(
    tmp_path, monkeypatch, http_files, option
):
    _, lock, _ = pinned_files(tmp_path / "input", monkeypatch)
    lock["provenance"]["loader_options"] = option
    with pytest.raises(ValueError, match="unsupported|unavailable|cannot skip"):
        transport(tmp_path, lock)
    assert not http_files["requests"]


def test_scan_charge_precedes_decode_and_verification_is_cumulative(
    tmp_path, monkeypatch, http_files
):
    _, lock, files = pinned_files(tmp_path / "input", monkeypatch)
    http_files["files"] = files
    current = transport(tmp_path, lock, max_scanned_rows=1)
    current.initialize()
    source = sources.DatasetSource.model_validate(lock["source"])
    stream = current.stream(source, "train", lock["provenance"])
    assert next(stream)["text"]
    with pytest.raises(ValueError, match="scanned_rows"):
        next(stream)
    assert current.receipt()["scanned_rows"] == 1
    row = lock["provenance"]["files"]["train"][0]
    other = transport(tmp_path / "verify", lock, max_verification_bytes=row["size"])
    other.initialize()
    other.file(row)
    with pytest.raises(ValueError, match="verification_bytes"):
        other.file(row)
    assert other.receipt()["verification_bytes"] == row["size"]
