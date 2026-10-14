"""Guard-level behavioral fences. HTTP is an in-memory opener with fake credentials."""

import copy
import hashlib
import io
from email.message import Message
from types import SimpleNamespace

import pytest
from test_bounded_acquisition import admission, pinned_files, transport

from sparselab.data import bounded_acquisition as bounded
from sparselab.data.sources import DatasetSource


@pytest.fixture
def current(tmp_path, monkeypatch):
    _, lock, _ = pinned_files(tmp_path / "input", monkeypatch)
    result = transport(tmp_path, lock)
    result.initialize()
    return result


class Response(io.BytesIO):
    def __init__(self, body=b"", status=200, **headers):
        super().__init__(body)
        self.status = status
        self.headers = Message()
        for key, value in {"Content-Length": str(len(body)), **headers}.items():
            if value is not None:
                self.headers[key] = value


@pytest.fixture
def opener(monkeypatch):
    requests = []
    responses = []

    def open_request(request, timeout):
        requests.append(request)
        assert timeout > 0
        assert responses, "unexpected additional request"
        return responses.pop(0)

    monkeypatch.setattr(
        bounded.urllib.request,
        "build_opener",
        lambda *a: SimpleNamespace(open=open_request),
    )
    monkeypatch.setattr(
        "sparselab.hf_auth.hub_auth_kwargs", lambda: {"token": "fake-test-token"}
    )
    # The stub has no socket; forbid all real socket operations as a second fence.
    monkeypatch.setattr(
        "socket.socket.connect", lambda *a: pytest.fail("external connection")
    )
    return requests, responses


@pytest.mark.parametrize(
    "destination,match",
    [
        ("http://huggingface.co/file", "ordinary HTTPS"),
        ("https://user:pw@huggingface.co/file", "ordinary HTTPS"),
        ("https://huggingface.co:8443/file", "ordinary HTTPS"),
        ("https://evil.example/file", "admitted Hub delivery host"),
        ("https://huggingface.co.evil.example/file", "admitted Hub delivery host"),
    ],
)
def test_redirect_authority_before_follow(current, opener, destination, match):
    requests, responses = opener
    responses.extend([Response(status=302, Location=destination), Response()])
    with pytest.raises(ValueError, match=match):
        current._open("https://huggingface.co/file")
    assert len(requests) == 1
    assert current.receipt()["requests"] == 1


@pytest.mark.parametrize(
    "initial,destination,expected",
    [
        ("https://cdn-lfs.hf.co/file", None, [None]),
        # Same hostname, different netloc: isolates redirect stripping from host stripping.
        (
            "https://huggingface.co/file",
            "https://huggingface.co:443/file",
            ["Bearer fake-test-token", None],
        ),
        (
            "https://huggingface.co/file",
            "https://cdn-lfs.hf.co/file",
            ["Bearer fake-test-token", None],
        ),
    ],
)
def test_each_credential_boundary(current, opener, initial, destination, expected):
    requests, responses = opener
    if destination:
        responses.append(Response(status=302, Location=destination))
    responses.append(Response())
    with current._open(initial):
        pass
    assert [r.get_header("Authorization") for r in requests] == expected


@pytest.mark.parametrize(
    "status,headers,match",
    [
        (200, {"Content-Encoding": "gzip"}, "content encoding"),
        (500, {}, "unexpected acquisition HTTP status"),
        (206, {}, "unexpected acquisition HTTP status"),
        (302, {}, "lacks Location"),
    ],
)
def test_response_protocol_refusals(current, opener, status, headers, match):
    requests, responses = opener
    response = Response(b"body", status=status, **headers)
    responses.append(response)
    with pytest.raises(ValueError, match=match):
        current._open("https://huggingface.co/file")
    assert response.closed and len(requests) == 1


def test_redirect_chain_cap(current, opener):
    requests, responses = opener
    responses.extend(Response(status=302, Location=f"/file/{i}") for i in range(7))
    with pytest.raises(ValueError, match="too many acquisition redirects"):
        current._open("https://huggingface.co/file")
    assert len(requests) == 6
    assert current.receipt()["requests"] == 6


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Length": None},
        {"Content-Length": "1"},
        {"Transfer-Encoding": "chunked"},
    ],
)
def test_file_framing_precedes_read(current, opener, headers):
    requests, responses = opener
    row = current.lock["provenance"]["files"]["train"][0]
    body = (current.work_dir.parent / "input" / row["path"]).read_bytes()
    responses.append(Response(body, **headers))
    with pytest.raises(ValueError, match="exact Content-Length"):
        current.file(row)
    assert len(requests) == 1
    assert current.receipt()["source_actual"] == 0
    assert not list(current.cache.glob("*.partial"))


def test_cached_size_before_hashing(current):
    path = current.cache / ("a" * 64 + ".bin")
    path.write_bytes(b"abc")
    # Valid content hash deliberately decouples size from the hash guard.
    row = {"size": 4, "lfs_sha256": hashlib.sha256(b"abc").hexdigest()}
    with pytest.raises(ValueError, match="cached pinned file size mismatch"):
        current._authenticate(path, row)
    assert current.receipt()["verification_bytes"] == 0


@pytest.mark.parametrize("field", ["root", "cache", "output", "journal", "ancestor"])
def test_storage_location_symlinks(current, tmp_path, field):
    outside = tmp_path / "outside"
    outside.mkdir()
    if field == "ancestor":
        link = tmp_path / "linked-parent"
        link.symlink_to(outside, target_is_directory=True)
        current.output = link / "child"
    else:
        path = (
            current.publication.parent
            if field == "journal"
            else getattr(current, field)
        )
        if path.exists():
            path.rename(path.with_name(path.name + "-retained"))
        path.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink in acquisition storage location"):
        current.validate_locations()
    assert not list(outside.iterdir())


@pytest.mark.parametrize(
    "name",
    [
        "guard",
        "transport.sqlite",
        "transport.sqlite-journal",
        "transport.sqlite-wal",
        "transport.sqlite-shm",
        "receipt.json",
        "verification",
    ],
)
def test_operational_symlinks(current, tmp_path, name):
    path = current.root / name
    if path.exists():
        path.rename(path.with_name(name + "-retained"))
    path.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink in acquisition operational storage"):
        current.validate_locations()


def test_unowned_cache_before_owner_creation(current):
    (current.cache / ".owner").unlink()
    (current.cache / ("a" * 64 + ".bin")).write_bytes(b"x")
    with pytest.raises(ValueError, match="cache lacks ownership"):
        current.initialize()
    assert not (current.cache / ".owner").exists()


@pytest.mark.parametrize("name", ["a.bin", "z" * 64 + ".bin", "a" * 64 + ".txt"])
def test_unexpected_cache_member(current, name):
    (current.cache / name).write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected acquisition cache member"):
        current._cache_inventory()


def test_cache_owner_rechecked(current):
    (current.cache / ".owner").write_text("other")
    with pytest.raises(ValueError, match="cache ownership changed"):
        current._cache_inventory()


@pytest.mark.parametrize("limit", ["max_cache_bytes", "max_cache_files"])
def test_existing_cache_quota(current, limit):
    current.spec = current.spec.model_copy(update={limit: 1})
    for name in ("a", "b"):
        (current.cache / (name * 64 + ".bin")).write_bytes(b"xx")
    with pytest.raises(ValueError, match="cache allowance exceeded"):
        current._cache_inventory()


@pytest.mark.parametrize(
    "limit", ["max_observed_owned_bytes", "max_observed_owned_inodes"]
)
def test_observed_owned_stop(current, limit):
    current.spec = current.spec.model_copy(update={limit: 1})
    with pytest.raises(ValueError, match="observed owned-storage stop exceeded"):
        current.observe()


def test_network_permission_is_explicit(current):
    with pytest.raises(ValueError, match="explicit network authority"):
        admission(current.lock, allow_network=False)


@pytest.mark.parametrize(
    "damage,match",
    [
        ("loader", "scalar Parquet only"),
        ("extension", "nonempty Parquet"),
        ("size", "nonempty Parquet"),
        ("dtype", "non-scalar features"),
    ],
)
def test_lock_decoder_profile(current, damage, match):
    lock = copy.deepcopy(current.lock)
    inventory = lock["provenance"]
    if damage == "loader":
        inventory["loader"] = "json"
    elif damage == "dtype":
        inventory["loader_options"] = {
            "features": {"text": {"_type": "Value", "dtype": "binary"}}
        }
    else:
        inventory["files"]["train"][0]["path" if damage == "extension" else "size"] = (
            "train.json" if damage == "extension" else 0
        )
    with pytest.raises(ValueError, match=match):
        bounded.validate_profile(lock, current.spec)


def test_changed_stream_inventory(current):
    inventory = copy.deepcopy(current.lock["provenance"])
    inventory["files"]["train"] = []
    with pytest.raises(ValueError, match="transport inventory changed"):
        list(
            current.stream(
                DatasetSource.model_validate(current.lock["source"]), "train", inventory
            )
        )


@pytest.mark.parametrize(
    "damage,match",
    [
        ("columns", "scalar/text columns"),
        ("text", "text field must be a string"),
        ("missing_text", "text field must be a string"),
        ("cast", "feature casts"),
    ],
)
def test_parquet_schema_before_decode(current, monkeypatch, tmp_path, damage, match):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"text": ["hello"], "extra": [1]})
    if damage == "columns":
        table = pa.table({"text": ["hello"], "extra": [[1, 2]]})
        # Independently exercise the Arrow column guard, below feature validation.
        monkeypatch.setattr(bounded, "_validate_features", lambda features: None)
    elif damage == "text":
        table = pa.table({"text": [1]})
    elif damage == "missing_text":
        table = pa.table({"other": ["hello"]})
    elif damage == "cast":
        current.lock["provenance"]["loader_options"] = {
            "features": {
                "text": {"_type": "Value", "dtype": "large_string"},
                "extra": {"_type": "Value", "dtype": "int64"},
            }
        }
    path = tmp_path / "schema.parquet"
    pq.write_table(table, path)
    monkeypatch.setattr(current, "file", lambda row: path)
    with pytest.raises(ValueError, match=match):
        next(
            current.stream(
                DatasetSource.model_validate(current.lock["source"]),
                "train",
                current.lock["provenance"],
            )
        )
    assert current.receipt()["scanned_rows"] == 0


def test_decoder_batch_cannot_exceed_reserved_rows(current, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    batch = pa.record_batch({"text": ["a", "b"]})

    class Parquet:
        schema_arrow = batch.schema
        metadata = SimpleNamespace(num_rows=1)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_batches(self, **kw):
            yield batch

    monkeypatch.setattr(pq, "ParquetFile", lambda *a, **kw: Parquet())
    monkeypatch.setattr(current, "file", lambda row: None)
    # Observe is unrelated to decode; avoid its ledger consistency check masking
    # a missing batch guard after the invalid observed count has been recorded.
    monkeypatch.setattr(current, "observe", lambda: None)
    with pytest.raises(ValueError, match="row batch exceeds reservation"):
        next(
            current.stream(
                DatasetSource.model_validate(current.lock["source"]),
                "train",
                current.lock["provenance"],
            )
        )
    assert current.receipt()["scanned_rows_observed"] == 0


def test_cumulative_snapshot_bytes_and_repeated_verification(current):
    peer = current.work_dir / "peer"
    for root in (current.output, peer):
        root.mkdir()
        (root / "manifest.json").write_bytes(b"123")
    current.snapshot_roots.append((peer,))
    current.snapshot_limit = 5
    with pytest.raises(ValueError, match="cumulative snapshot byte admission"):
        current.before_verification(current.output / "manifest.json")
    assert current.receipt()["verification_bytes"] == 0
    current.snapshot_limit = 6
    current.before_verification(current.output / "manifest.json")
    current.before_verification(current.output / "manifest.json")
    assert current.receipt()["verification_bytes"] == 6
