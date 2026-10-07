"""Pinned source acquisition and independently verifiable immutable byte snapshots."""

from __future__ import annotations

import bz2
import fnmatch
import gzip
import hashlib
import html
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin, urlparse

from sparselab.corpus.project import (
    GitAcquisition,
    HttpAcquisition,
    HuggingFaceAcquisition,
    LocalAcquisition,
    Project,
    SourceDeclaration,
    WikimediaDumpAcquisition,
    project_path,
    safe_name,
    source_declaration_payload,
)
from sparselab.corpus.transport_budget import TransportBudget
from sparselab.engram.packs import _rename_noreplace
from sparselab.hf_auth import HUB_ACCESS_ERRORS, hub_auth_kwargs, raise_for_hub_auth
from sparselab.training.manifest import canonical_json, sha256_file

if TYPE_CHECKING:
    from sparselab.verification_proofs import ProofStore, VerificationMode

ADAPTER_VERSION = "corpus-acquisition-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _identity_declaration(source: SourceDeclaration) -> dict[str, Any]:
    declaration = source_declaration_payload(source)
    if source.kind == "local":
        declaration["acquisition"]["files"] = [
            {"name": entry.name} for entry in source.acquisition.files
        ]
    return declaration


def declaration_sha256(source: SourceDeclaration) -> str:
    return _digest(_identity_declaration(source))


def _adapter(source: SourceDeclaration) -> dict[str, str]:
    return {
        "id": source.kind,
        "version": ADAPTER_VERSION,
        "module_sha256": sha256_file(Path(__file__)),
    }


def _reusable_immutable_adapter(
    source: SourceDeclaration, manifest: dict[str, Any]
) -> bool:
    """A pinned, verified snapshot survives incidental adapter-module edits.

    The version still binds adapter semantics; the module digest stays recorded
    in the snapshot for provenance rather than forcing a new network retrieval.
    Mutable local and HTTP sources must still be reacquired.
    """
    actual = manifest["adapter"]
    expected = _adapter(source)
    return actual == expected or (
        source.kind in {"git", "huggingface_dataset", "wikimedia_dump"}
        and actual.get("id") == expected["id"]
        and actual.get("version") == expected["version"]
    )


def _write_json(path: Path, value: object) -> None:
    with path.open("wb") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_stream(source: Any, target: Path, remaining: int) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with target.open("xb") as output:
        while chunk := source.read(min(1024 * 1024, remaining - size + 1)):
            size += len(chunk)
            if size > remaining:
                raise ValueError("source exceeds declared max_bytes")
            digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest(), size


def _regular(path: Path) -> None:
    if path.is_symlink() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise ValueError(f"source is not a regular, nonsymlink file: {path}")


def _acquire_local(
    source: SourceDeclaration, root: Path, staging: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = source.acquisition
    assert isinstance(spec, LocalAcquisition)
    total = 0
    inventory = []
    for item in sorted(spec.files, key=lambda f: f.name):
        location = Path(item.path)
        location = location if location.is_absolute() else project_path(root, item.path)
        _regular(location)
        with location.open("rb") as stream:
            digest, size = _copy_stream(
                stream, staging / "files" / item.name, spec.max_bytes - total
            )
        total += size
        inventory.append({"path": item.name, "sha256": digest, "size": size})
    return inventory, {"acquisition_paths": [item.model_dump() for item in spec.files]}


def _git(args: list[str], *, cwd: Path | None = None) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True
    ).stdout


def _acquire_git(
    source: SourceDeclaration, staging: Path, cache_root: Path, offline: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if offline:
        raise ValueError("offline acquisition cannot fetch Git")
    spec = source.acquisition
    assert isinstance(spec, GitAcquisition)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / _digest(
        {"uri": source.canonical_uri, "revision": source.revision}
    )
    if not cache.exists():
        cache.mkdir()
        _git(["init", "--bare", str(cache)])
    _git(
        [
            "--git-dir",
            str(cache),
            "fetch",
            "--depth=1",
            "--no-tags",
            source.canonical_uri,
            source.revision,
        ]
    )
    commit = (
        _git(["--git-dir", str(cache), "rev-parse", "FETCH_HEAD^{commit}"])
        .decode()
        .strip()
    )
    if commit.lower() != source.revision.lower():
        raise ValueError("Git fetched commit differs from pinned revision")
    tree = _git(["--git-dir", str(cache), "ls-tree", "-rz", "--full-tree", commit])
    inventory = []
    total = 0
    selected = set()
    metadata_path = source.rights.nested_metadata_path if source.rights else None
    metadata_seen = False
    for record in tree.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8", errors="strict")
        is_metadata = path == metadata_path
        if not is_metadata and not any(
            fnmatch.fnmatchcase(path, pattern) for pattern in spec.include
        ):
            continue
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in spec.exclude):
            if is_metadata:
                raise ValueError("rights metadata excluded from pinned Git selection")
            continue
        safe_name(path)
        if mode not in ("100644", "100755") or kind != "blob":
            raise ValueError(f"Git selection contains unsafe symlink/submodule: {path}")
        if is_metadata:
            metadata_seen = True
        else:
            selected.add(path)
        size = int(_git(["--git-dir", str(cache), "cat-file", "-s", blob]))
        if size > spec.max_bytes - total:
            raise ValueError("Git selection exceeds max_bytes")
        target = staging / "files" / path
        with subprocess.Popen(
            ["git", "--git-dir", str(cache), "cat-file", "blob", blob],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            assert process.stdout is not None
            digest, copied = _copy_stream(
                process.stdout, target, spec.max_bytes - total
            )
            if process.wait() != 0 or copied != size:
                raise ValueError(f"Git blob read failed: {path}")
        total += copied
        inventory.append(
            {"path": path, "sha256": digest, "size": copied, "git_blob_id": blob}
        )
    if not selected:
        raise ValueError("Git include patterns selected no files")
    if metadata_path and not metadata_seen:
        raise ValueError("pinned Git rights metadata file is missing")
    return sorted(inventory, key=lambda item: item["path"]), {"commit": commit}


class _PrivateHubRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward a Hub bearer credential to another origin."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None:
            before = urlparse(request.full_url)
            after = urlparse(newurl)
            if (before.scheme, before.netloc) != (after.scheme, after.netloc):
                redirected.remove_header("Authorization")
        return redirected


class _ManualRedirect(urllib.request.HTTPRedirectHandler):
    """Return redirects to the caller so their bodies can be charged first."""

    def http_error_302(
        self, request: Any, fp: Any, code: int, msg: str, headers: Any
    ) -> Any:
        return fp

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _response_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    raw = headers.get("Content-Length") if headers is not None else None
    if raw is None:
        return None
    if not str(raw).isdigit():
        raise ValueError("invalid HTTP Content-Length")
    return int(raw)


def _set_response_deadline(response: Any, budget: TransportBudget) -> None:
    remaining = budget.remaining_seconds()
    stream = response
    for _ in range(5):
        if isinstance(stream, io.BytesIO):
            return  # deterministic local fixture, including HTTPError bodies
        sock = getattr(stream, "_sock", None)
        if sock is not None:
            sock.settimeout(min(30, remaining))
            return
        next_stream = getattr(stream, "fp", None)
        if next_stream is None:
            next_stream = getattr(stream, "raw", None)
        if next_stream is None or next_stream is stream:
            break
        stream = next_stream
    raise ValueError("HTTP response socket cannot enforce transport deadline")


def _read_metadata_body(response: Any, budget: TransportBudget) -> bytes:
    length = _response_length(response)
    if length is not None:
        budget.reserve_metadata(length)
    chunks: list[bytes] = []
    received = 0
    while True:
        _set_response_deadline(response, budget)
        if length is not None:
            allowance = length - received
            if allowance == 0:
                break
            request_size = min(65536, allowance)
        else:
            remaining = (
                budget.spec.max_metadata_body_bytes
                - budget.receipt()["metadata_charged"]
            )
            if remaining <= 0:
                raise ValueError("metadata response length is unverified at body cap")
            request_size = min(65536, remaining)
            budget.reserve_metadata(request_size)
        chunk = response.read(request_size)
        if not chunk:
            if length is not None:
                raise ValueError("short HTTP metadata body")
            break
        if len(chunk) > request_size:
            raise ValueError("HTTP metadata body exceeded requested read")
        received += len(chunk)
        budget.charge_metadata_actual(len(chunk))
        chunks.append(chunk)
    return b"".join(chunks)


def _open_accounted(
    url: str,
    headers: dict[str, str],
    budget: TransportBudget,
    *,
    data: bytes | None = None,
) -> tuple[Any, str]:
    opener = urllib.request.build_opener(_ManualRedirect())
    current = url
    if urlparse(current).hostname != "huggingface.co":
        headers = {
            key: value
            for key, value in headers.items()
            if key.lower() != "authorization"
        }
    for _ in range(6):
        budget.remaining_seconds()
        request = urllib.request.Request(current, headers=headers, data=data)
        try:
            response = opener.open(request, timeout=min(30, budget.remaining_seconds()))
        except HTTPError as error:
            # Even an error response consumes the attempt's metadata allowance.
            with error:
                _read_metadata_body(error, budget)
            raise
        status = getattr(response, "status", None) or getattr(response, "code", 200)
        if status not in {301, 302, 303, 307, 308}:
            if urlparse(response.url).scheme != "https":
                response.close()
                raise ValueError("HF response must use HTTPS")
            return response, current
        try:
            _read_metadata_body(response, budget)
            location = response.headers.get("Location")
        finally:
            response.close()
        if not location:
            raise ValueError("HTTP redirect lacks Location")
        next_url = urljoin(current, location)
        before, after = urlparse(current), urlparse(next_url)
        if after.scheme != "https":
            raise ValueError("HF redirect must use HTTPS")
        if (before.scheme, before.netloc) != (after.scheme, after.netloc):
            headers = {
                key: value
                for key, value in headers.items()
                if key.lower() != "authorization"
            }
            if data is not None:
                raise ValueError("pinned HF metadata POST redirected across origins")
        current = next_url
    raise ValueError("too many HF redirects")


def _hf_pinned_metadata(
    source: SourceDeclaration,
    spec: HuggingFaceAcquisition,
    shard: Any,
    headers: dict[str, str],
    budget: TransportBudget,
) -> dict[str, Any]:
    repo = source.canonical_uri.removeprefix("https://huggingface.co/datasets/")
    tree_url = (
        f"https://huggingface.co/api/datasets/{repo}/paths-info/{source.revision}"
    )
    splits_url = "https://datasets-server.huggingface.co/splits?" + urlencode(
        {"dataset": repo, "revision": source.revision}
    )
    records = []
    for url in (tree_url, splits_url):
        data = (
            urlencode({"paths": shard.path, "expand": "false"}).encode()
            if url == tree_url
            else None
        )
        response, _ = _open_accounted(url, headers.copy(), budget, data=data)
        with response:
            try:
                records.append(json.loads(_read_metadata_body(response, budget)))
            except json.JSONDecodeError as error:
                raise ValueError("invalid pinned HF metadata") from error
    tree, splits = records
    if not isinstance(tree, list) or not isinstance(splits, dict):
        raise TypeError("invalid pinned HF metadata shape")
    matches = [
        item
        for item in tree
        if isinstance(item, dict) and item.get("path") == shard.path
    ]
    if len(matches) != 1:
        raise ValueError("pinned HF shard path missing or ambiguous")
    item = matches[0]
    if item.get("type") != "file" or item.get("size") != shard.max_shard_bytes:
        raise ValueError("pinned HF shard size/type mismatch")
    lfs = item.get("lfs")
    if (
        not isinstance(lfs, dict)
        or lfs.get("sha256", "").lower() != shard.expected_sha256.lower()
    ):
        raise ValueError("pinned HF shard metadata checksum mismatch")
    entries = splits.get("splits")
    if not isinstance(entries, list):
        raise TypeError("pinned HF config/split metadata missing")
    if not entries or any(
        not isinstance(entry, dict)
        or entry.get("dataset") != repo
        or not isinstance(entry.get("config"), str)
        or not isinstance(entry.get("split"), str)
        for entry in entries
    ):
        raise ValueError("pinned HF config/split metadata missing or ambiguous")
    pairs = {(entry.get("config"), entry.get("split")) for entry in entries}
    if pairs != {(spec.config, spec.split)}:
        raise ValueError("pinned HF config/split metadata missing or ambiguous")
    return {"tree_url": tree_url, "splits_url": splits_url, "size": item["size"]}


def _copy_hf_body(
    response: Any,
    target: Path,
    limit: int,
    budget: TransportBudget | None,
    transfer_index: int | None,
) -> tuple[str, int]:
    length = _response_length(response)
    if length is None:
        raise ValueError("HF source requires Content-Length for strict body cap")
    if length is not None and length > limit:
        raise ValueError("HF source exceeds declared max_bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with target.open("xb") as output:
        while True:
            if budget is not None:
                _set_response_deadline(response, budget)
            remaining = limit - size
            if remaining == 0:
                if length is None or length == size:
                    break
                raise ValueError("HF source exceeds declared max_bytes")
            request_size = min(1024 * 1024, remaining)
            if length is not None:
                if length == size:
                    break
                request_size = min(request_size, length - size)
            chunk = response.read(request_size)
            if not chunk:
                if length is not None and size != length:
                    raise ValueError("short HF source body")
                break
            if len(chunk) > request_size:
                raise ValueError("HF source body exceeded requested read")
            if budget is not None:
                assert transfer_index is not None
                budget.charge_transfer_actual(transfer_index, len(chunk))
            digest.update(chunk)
            output.write(chunk)
            size += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest(), size


def _acquire_http(
    source: SourceDeclaration, staging: Path, offline: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if offline:
        raise ValueError("offline acquisition cannot fetch HTTP")
    spec = source.acquisition
    assert isinstance(spec, HttpAcquisition)
    if urlparse(source.canonical_uri).scheme not in ("http", "https"):
        raise ValueError("HTTP source requires an HTTP(S) canonical URI")
    suffix = Path(urlparse(source.canonical_uri).path).suffix.lower()
    if suffix not in (".txt", ".md", ".markdown"):
        raise ValueError("HTTP source must explicitly name a text/Markdown document")
    name = "document" + suffix
    headers = {"User-Agent": "SparseLab-Corpus-Forge/1"}
    origin = urlparse(source.canonical_uri)
    if origin.scheme == "https" and origin.hostname == "huggingface.co":
        token = hub_auth_kwargs().get("token")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(source.canonical_uri, headers=headers)
    try:
        with urllib.request.build_opener(_PrivateHubRedirect()).open(
            request, timeout=30
        ) as response:
            if urlparse(response.url).scheme not in ("http", "https"):
                raise ValueError("HTTP redirect has unsafe scheme")
            digest, size = _copy_stream(
                response, staging / "files" / name, spec.max_bytes
            )
            retrieval = {
                "final_url": response.url,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
            }
    except HTTPError as error:
        if origin.scheme == "https" and origin.hostname == "huggingface.co":
            raise_for_hub_auth(error, credential_supplied="Authorization" in headers)
        raise
    if digest != spec.expected_sha256:
        raise ValueError("HTTP expected SHA-256 mismatch")
    return [{"path": name, "sha256": digest, "size": size}], retrieval


def _validate_hf_rows(path: Path, spec: HuggingFaceAcquisition, available: int) -> int:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        if spec.text_field not in parquet.schema_arrow.names:
            raise ValueError(f"HF text field missing: {spec.text_field}")
        count = parquet.metadata.num_rows
        if count > available:
            raise ValueError("HF selection exceeds max_rows")
        return count
    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            raise ValueError("HF JSON requires object or array of objects")
        if len(records) > available:
            raise ValueError("HF selection exceeds max_rows")
    else:
        with path.open(encoding="utf-8") as handle:
            records = []
            for line in handle:
                if len(records) >= available:
                    raise ValueError("HF selection exceeds max_rows")
                records.append(json.loads(line))
    if any(
        not isinstance(record, dict) or not isinstance(record.get(spec.text_field), str)
        for record in records
    ):
        raise ValueError(f"HF text field missing/non-string: {spec.text_field}")
    return len(records)


def _bounded_hf_rows(
    path: Path,
    *,
    limit: int,
    max_line_bytes: int,
    max_decompressed_bytes: int | None = None,
    budget: TransportBudget | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.name.endswith(".parquet"):
        import pyarrow.parquet as pq

        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=min(limit, 256)):
            for row in batch.to_pylist():
                if index >= limit:
                    return
                yield index, row
                index += 1
        return
    stream = gzip.open if path.name.endswith(".gz") else open
    decompressed = 0
    with stream(path, "rb") as handle:
        for index in range(limit):
            if budget is not None:
                budget.remaining_seconds()
            line = handle.readline(max_line_bytes + 1)
            if not line:
                break
            decompressed += len(line)
            if (
                max_decompressed_bytes is not None
                and decompressed > max_decompressed_bytes
            ):
                raise ValueError(
                    "HF decompressed stream exceeds max_decompressed_bytes"
                )
            if len(line) > max_line_bytes:
                raise ValueError("HF decompressed JSONL row exceeds max_bytes")
            try:
                yield index, json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"malformed HF JSONL row {index}") from error


def _bounded_hf_acquire(
    source: SourceDeclaration,
    spec: HuggingFaceAcquisition,
    staging: Path,
    budget: TransportBudget | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import hf_hub_url

    assert spec.bounded_shards is not None
    repo_id = source.canonical_uri.removeprefix("https://huggingface.co/datasets/")
    auth = hub_auth_kwargs()
    inventory: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    total_bytes = 0
    total_rows = 0
    for shard in spec.bounded_shards:
        if shard.declared_config is not None and budget is None:
            raise ValueError("explicit HF shard metadata requires a transport budget")
        url = hf_hub_url(
            repo_id=repo_id,
            filename=shard.path,
            repo_type="dataset",
            revision=source.revision,
        )
        if (
            urlparse(url).scheme != "https"
            or urlparse(url).hostname != "huggingface.co"
        ):
            raise ValueError("HF shard URL must point to the Hugging Face Hub")
        headers = {"User-Agent": "SparseLab-Corpus-Forge/1"}
        if auth.get("token"):
            headers["Authorization"] = f"Bearer {auth['token']}"
        request = urllib.request.Request(url, headers=headers)
        input_path = staging / "hf-input" / shard.path
        transfer_index = None
        try:
            if budget is not None:
                budget.assert_transfer_available(
                    source.id, shard.path, shard.max_shard_bytes
                )
                projected = budget.projected_disk_bytes()
                if projected > budget.spec.max_disk_bytes:
                    raise ValueError("transport budget disk allowance exhausted")
                if shutil.disk_usage(staging).free < projected:
                    raise ValueError("insufficient disk space for bounded HF shard")
                _hf_pinned_metadata(source, spec, shard, headers, budget)
                transfer_index = budget.reserve_transfer(
                    source.id, shard.path, shard.max_shard_bytes
                )
                response, _ = _open_accounted(url, headers.copy(), budget)
            else:
                response = urllib.request.build_opener(_PrivateHubRedirect()).open(
                    request, timeout=30
                )
            with response:
                if urlparse(response.url).scheme != "https":
                    raise ValueError("HF shard redirect must use HTTPS")
                digest, input_bytes = _copy_hf_body(
                    response, input_path, shard.max_shard_bytes, budget, transfer_index
                )
        except (*HUB_ACCESS_ERRORS, HTTPError) as error:
            if budget is not None and transfer_index is not None:
                budget.finish_transfer(transfer_index, success=False, interrupted=False)
            raise_for_hub_auth(error, credential_supplied=bool(auth.get("token")))
        except OSError, TimeoutError, EOFError:
            if budget is not None and transfer_index is not None:
                budget.finish_transfer(transfer_index, success=False, interrupted=True)
            raise
        except Exception:
            if budget is not None and transfer_index is not None:
                budget.finish_transfer(transfer_index, success=False, interrupted=False)
            raise
        if digest != shard.expected_sha256.lower():
            if budget is not None and transfer_index is not None:
                budget.finish_transfer(transfer_index, success=False)
            raise ValueError(f"HF shard SHA-256 mismatch: {shard.path}")
        if budget is not None and transfer_index is not None:
            budget.finish_transfer(transfer_index, success=True)

        output_name = shard.path + ".sample.jsonl"
        target = staging / "files" / output_name
        target.parent.mkdir(parents=True, exist_ok=True)
        selected: list[dict[str, Any]] = []
        output_digest = hashlib.sha256()
        output_bytes = 0
        scanned = 0
        with target.open("wb") as output:
            for index, row in _bounded_hf_rows(
                input_path,
                limit=shard.max_scanned_rows,
                max_line_bytes=spec.max_bytes,
                max_decompressed_bytes=spec.max_decompressed_bytes,
                budget=budget,
            ):
                scanned += 1
                if not isinstance(row, dict) or spec.text_field not in row:
                    raise ValueError(
                        f"HF text field missing at {shard.path} row {index}"
                    )
                if not isinstance(row[spec.text_field], str):
                    raise TypeError(
                        f"HF text field non-string at {shard.path} row {index}"
                    )
                if "_sparselab_source" in row:
                    raise ValueError(
                        "HF source row collides with reserved provenance field"
                    )
                try:
                    row_digest = hashlib.sha256(canonical_json(row)).hexdigest()
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"HF row is not canonical JSON: {shard.path} row {index}"
                    ) from error
                row_hash = hashlib.sha256(
                    canonical_json([source.revision, shard.path, index])
                ).hexdigest()
                if int(row_hash, 16) % shard.hash_modulus not in shard.hash_remainders:
                    continue
                if total_rows >= spec.max_rows:
                    raise ValueError("HF bounded selection exceeds max_rows")
                envelope: dict[str, Any] = {
                    "source_shard_path": shard.path,
                    "source_shard_sha256": digest,
                    "dataset_revision": source.revision,
                    "source_row_index": index,
                    "source_row_sha256": row_digest,
                }
                for key in (
                    "url",
                    "id",
                    "dump",
                    "score",
                    "language",
                    "language_score",
                    "token_count",
                    "blob_id",
                    "repo_name",
                    "path",
                    "detected_licenses",
                ):
                    value = row.get(key)
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        if key in row:
                            envelope[key] = value
                    elif key == "detected_licenses":
                        envelope[key] = canonical_json(value).decode("utf-8")
                line = canonical_json({**row, "_sparselab_source": envelope}) + b"\n"
                if len(line) > spec.max_bytes - total_bytes - output_bytes:
                    raise ValueError("HF bounded selection exceeds max_bytes")
                output.write(line)
                output_digest.update(line)
                output_bytes += len(line)
                total_rows += 1
                selected.append(
                    {
                        "source_row_index": index,
                        "source_row_sha256": row_digest,
                        "selection_hash": row_hash,
                    }
                )
            output.flush()
            os.fsync(output.fileno())
        input_path.unlink()
        if not selected:
            target.unlink()
        else:
            total_bytes += output_bytes
            inventory.append(
                {
                    "path": output_name,
                    "sha256": output_digest.hexdigest(),
                    "size": output_bytes,
                }
            )
        receipts.append(
            {
                "source_shard_path": shard.path,
                "source_shard_sha256": digest,
                "source_shard_bytes": input_bytes,
                "scanned_rows": scanned,
                "selected_rows": selected,
                "output_path": output_name if selected else None,
            }
        )
    if not inventory:
        raise ValueError("HF bounded selection produced no rows")
    retrieval = {
        "config": spec.config,
        "split": spec.split,
        "text_field": spec.text_field,
        "max_rows": spec.max_rows,
        "rows": total_rows,
        "dataset_revision": source.revision,
        "sampling": "sha256-revision-path-zero-index-modulus-v1",
        "shards": receipts,
    }
    if budget is not None:
        retrieval["transport_budget"] = budget.receipt()
    return inventory, retrieval


def _acquire_hf(
    source: SourceDeclaration,
    staging: Path,
    cache_root: Path,
    offline: bool,
    budget: TransportBudget | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if offline:
        raise ValueError("offline acquisition cannot fetch Hugging Face")
    from huggingface_hub import snapshot_download

    spec = source.acquisition
    if spec.bounded_shards is not None:
        return _bounded_hf_acquire(source, spec, staging, budget)
    assert isinstance(spec, HuggingFaceAcquisition)
    auth = hub_auth_kwargs()
    try:
        location = Path(
            snapshot_download(
                repo_id=source.canonical_uri.removeprefix(
                    "https://huggingface.co/datasets/"
                ),
                repo_type="dataset",
                revision=source.revision,
                allow_patterns=list(spec.include),
                cache_dir=str(cache_root),
                **auth,
            )
        )
    except HUB_ACCESS_ERRORS as error:
        raise_for_hub_auth(error, credential_supplied=bool(auth))
    if location.name.lower() != source.revision.lower():
        raise ValueError("HF snapshot revision does not match pinned commit")
    inventory = []
    total = rows = 0
    for path in sorted(location.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(location).as_posix()
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in spec.include):
            continue
        safe_name(relative)
        if path.is_symlink():
            # hub cache often provides symlinks to immutable blobs; resolve within the cache.
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(cache_root.resolve()):
                raise ValueError("HF snapshot symlink escapes cache")
            path = resolved
        _regular(path)
        parts = relative.split("/")
        if spec.config not in parts and not Path(relative).name.startswith(
            spec.config + "-"
        ):
            raise ValueError(
                f"HF file is not unambiguously in config {spec.config}: {relative}"
            )
        if spec.split not in parts and not any(
            part.startswith((spec.split + "-", spec.split + ".")) for part in parts
        ):
            raise ValueError(
                f"HF file is not unambiguously in split {spec.split}: {relative}"
            )
        if path.suffix.lower() not in (".jsonl", ".json", ".parquet"):
            raise ValueError(f"unsupported HF file format: {relative}")
        if path.stat().st_size > spec.max_bytes - total:
            raise ValueError("HF selection exceeds max_bytes")
        rows += _validate_hf_rows(path, spec, spec.max_rows - rows)
        with path.open("rb") as stream:
            digest, size = _copy_stream(
                stream, staging / "files" / relative, spec.max_bytes - total
            )
        total += size
        inventory.append({"path": relative, "sha256": digest, "size": size})
    if not inventory:
        raise ValueError("HF include patterns selected no files")
    return inventory, {
        "config": spec.config,
        "split": spec.split,
        "text_field": spec.text_field,
        "max_rows": spec.max_rows,
        "rows": rows,
    }


_WIKI_NS = "{http://www.mediawiki.org/xml/export-0.11/}"
_WIKI_BLOCKED = re.compile(
    r"(?i)\b(?:confidential|copyright violation|copyvio|non-free|"
    r"fair use|do not train|noai|imported from|transwiki)\b"
)
_WIKI_PRIVATE_TITLE = re.compile(r"(?i)\b(?:private secrets?|confidential data)\b")


class _BoundedXML:
    """Bound decompressed XML and reject document types before the XML parser sees them."""

    def __init__(self, stream: Any, limit: int):
        self.stream = stream
        self.limit = limit
        self.size = 0
        self.previous = b""

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(min(65536, size if size >= 0 else 65536))
        self.size += len(chunk)
        if self.size > self.limit:
            raise ValueError("Wikimedia decompressed XML exceeds limit")
        combined = (self.previous + chunk).upper()
        if b"<!DOCTYPE" in combined or b"<!ENTITY" in combined:
            raise ValueError("Wikimedia XML DTD/entities are forbidden")
        self.previous = combined[-8:]
        return chunk


def _wiki_text(wikitext: str) -> str:
    # Remove entire templates, references, tables, files and categories; never
    # expand templates or fetch links. Keep literal code and math element bodies.
    text = re.sub(r"(?is)<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/>", "", wikitext)
    text = re.sub(r"(?s)\{\|.*?\|\}", "", text)
    while "{{" in text:
        stripped = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if stripped == text:
            return ""
        text = stripped
    text = re.sub(r"(?i)\[\[(?:file|image|category):[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[(?:[^]|]*\|)?([^]|]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+(?:\s+([^\]]+))?\]", lambda m: m[1] or "", text)
    text = re.sub(r"(?is)<!--.*?-->", "", text)
    text = re.sub(r"(?i)</?(?!math\b|code\b|pre\b)[a-z][^>]*>", "", text)
    text = re.sub(r"(?i)</?(?:math|code|pre)\b[^>]*>", "\n", text)
    text = re.sub(r"(?m)^={2,}\s*(.*?)\s*={2,}\s*$", r"\1", text)
    text = html.unescape(text.replace("'''", "").replace("''", ""))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def _wiki_row(page: ET.Element, source: SourceDeclaration) -> dict[str, Any] | None:
    title = page.findtext(_WIKI_NS + "title")
    page_id = page.findtext(_WIKI_NS + "id")
    if (
        page.findtext(_WIKI_NS + "ns") != "0"
        or page.find(_WIKI_NS + "redirect") is not None
        or not title
        or not page_id
        or _WIKI_BLOCKED.search(title)
        or _WIKI_PRIVATE_TITLE.search(title)
    ):
        return None
    revisions = page.findall(_WIKI_NS + "revision")
    if not revisions:
        return None
    revision = max(
        revisions,
        key=lambda item: (
            item.findtext(_WIKI_NS + "timestamp") or "",
            int(item.findtext(_WIKI_NS + "id") or 0),
        ),
    )
    content = revision.find(_WIKI_NS + "text")
    if content is None or content.get("deleted") is not None or not content.text:
        return None
    raw = content.text
    if _WIKI_BLOCKED.search(raw) or re.search(
        r"(?i)\[\[(?:category|file|image):[^\]]*(?:import|non-free|copyright)", raw
    ):
        return None
    text = _wiki_text(raw)
    if not text or not re.search(r"[a-zA-Z]", text):
        return None
    # Skip predominantly non-Latin prose; retain mathematics/code with English context.
    letters = [char for char in text if char.isalpha()]
    if letters and sum(char.isascii() for char in letters) * 2 < len(letters):
        return None
    revision_id = revision.findtext(_WIKI_NS + "id")
    timestamp = revision.findtext(_WIKI_NS + "timestamp")
    if not revision_id or not timestamp:
        return None
    project = urlparse(source.canonical_uri).path.split("/")[1]
    if project not in {"enwikibooks", "enwiki"}:
        raise ValueError("unsupported Wikimedia project for page attribution")
    site = "wikipedia" if project == "enwiki" else "wikibooks"
    return {
        "text": text,
        "_sparselab_source": {
            "source_uri": source.canonical_uri,
            "dump_revision": source.revision,
            "page_title": title,
            "page_id": page_id,
            "revision_id": revision_id,
            "revision_timestamp": timestamp,
            "page_uri": f"https://en.{site}.org/?curid={page_id}",
        },
    }


def _acquire_wikimedia(
    source: SourceDeclaration, staging: Path, offline: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if offline:
        raise ValueError("offline acquisition cannot fetch Wikimedia dump")
    spec = source.acquisition
    assert isinstance(spec, WikimediaDumpAcquisition)
    opener = urllib.request.build_opener()
    headers = {
        "User-Agent": (
            "SparseLab-Corpus-Forge/1 (+https://github.com/supernovae/tiny-sparse-lab)"
        )
    }
    with opener.open(
        urllib.request.Request(spec.checksum_uri, headers=headers), timeout=30
    ) as response:
        if response.url != spec.checksum_uri:
            raise ValueError("Wikimedia checksum redirect differs from pinned URI")
        checksum = response.read(4_194_305)
    if len(checksum) > 4_194_304:
        raise ValueError("Wikimedia checksum manifest exceeds limit")
    name = Path(urlparse(source.canonical_uri).path).name
    matches = [
        line.split()
        for line in checksum.decode("ascii").splitlines()
        if line.split()[1:] == [name]
    ]
    if len(matches) != 1 or matches[0][0].lower() != spec.expected_sha1.lower():
        raise ValueError("Wikimedia official SHA-1 manifest mismatch")
    compressed = staging / "wikimedia-input" / name
    with opener.open(
        urllib.request.Request(source.canonical_uri, headers=headers), timeout=60
    ) as response:
        if response.url != source.canonical_uri:
            raise ValueError("Wikimedia dump redirect differs from pinned URI")
        digest, compressed_bytes = _copy_stream(
            response, compressed, spec.max_compressed_bytes
        )
    if spec.expected_sha256 and digest != spec.expected_sha256.lower():
        raise ValueError("Wikimedia compressed SHA-256 mismatch")
    source_sha1 = hashlib.sha1()
    with compressed.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            source_sha1.update(chunk)
    if source_sha1.hexdigest() != spec.expected_sha1.lower():
        raise ValueError("Wikimedia compressed SHA-1 mismatch")
    output_name = name + ".sample.jsonl"
    output_path = staging / "files" / output_name
    emitted = scanned = 0
    rows: list[dict[str, str]] = []
    with bz2.open(compressed, "rb") as decompressed, output_path.open("wb") as output:
        guarded = _BoundedXML(decompressed, spec.max_decompressed_bytes)
        context = ET.iterparse(guarded, events=("start", "end"))
        root: ET.Element | None = None
        english_site = False
        for event, element in context:
            if root is None:
                root = element
                if root.tag != _WIKI_NS + "mediawiki":
                    raise ValueError("unsupported Wikimedia XML namespace")
            if event == "end" and element.tag == _WIKI_NS + "siteinfo":
                language = element.findtext(_WIKI_NS + "lang") or root.get(
                    "{http://www.w3.org/XML/1998/namespace}lang"
                )
                expected_db = urlparse(source.canonical_uri).path.split("/")[1]
                if (
                    language != "en"
                    or element.findtext(_WIKI_NS + "dbname") != expected_db
                ):
                    raise ValueError(
                        "Wikimedia dump site identity or language mismatch"
                    )
                english_site = True
            if event != "end" or element.tag != _WIKI_NS + "page":
                continue
            if not english_site:
                raise ValueError("Wikimedia dump lacks English siteinfo")
            if (
                scanned >= spec.max_scanned_pages
                or len(rows) >= spec.max_selected_pages
            ):
                break
            scanned += 1
            row = _wiki_row(element, source)
            if row is not None:
                line = canonical_json(row) + b"\n"
                if len(line) > spec.max_emitted_bytes - emitted:
                    raise ValueError("Wikimedia emitted JSONL exceeds limit")
                output.write(line)
                emitted += len(line)
                rows.append(
                    {
                        "page_id": row["_sparselab_source"]["page_id"],
                        "revision_id": row["_sparselab_source"]["revision_id"],
                        "row_sha256": hashlib.sha256(line).hexdigest(),
                    }
                )
            root.clear()
        output.flush()
        os.fsync(output.fileno())
    compressed.unlink()
    if not rows:
        raise ValueError("Wikimedia bounded selection produced no pages")
    return [
        {
            "path": output_name,
            "sha256": sha256_file(output_path),
            "size": emitted,
        }
    ], {
        "source_uri": source.canonical_uri,
        "checksum_uri": spec.checksum_uri,
        "source_sha1": spec.expected_sha1.lower(),
        "source_sha256": digest,
        "compressed_bytes": compressed_bytes,
        "scanned_pages": scanned,
        "selected_pages": rows,
        "output_path": output_name,
    }


def _verify_wikimedia_receipt(path: Path, manifest: dict[str, Any]) -> None:
    receipt = manifest["retrieval"]
    source = SourceDeclaration.model_validate(manifest["declaration"])
    spec = source.acquisition
    assert isinstance(spec, WikimediaDumpAcquisition)
    if (
        receipt["source_uri"] != source.canonical_uri
        or receipt["checksum_uri"] != spec.checksum_uri
        or receipt["source_sha1"] != spec.expected_sha1.lower()
        or (
            spec.expected_sha256
            and receipt["source_sha256"] != spec.expected_sha256.lower()
        )
        or receipt["compressed_bytes"] > spec.max_compressed_bytes
        or receipt["scanned_pages"] > spec.max_scanned_pages
        or len(receipt["selected_pages"]) > spec.max_selected_pages
        or len(manifest["files"]) != 1
        or receipt["output_path"] != manifest["files"][0]["path"]
        or manifest["files"][0]["size"] > spec.max_emitted_bytes
    ):
        raise ValueError("Wikimedia receipt mismatch")
    rows = []
    with (path / "files" / receipt["output_path"]).open("rb") as handle:
        for line in handle:
            row = json.loads(line)
            metadata = row["_sparselab_source"]
            if (
                not isinstance(row["text"], str)
                or metadata["source_uri"] != source.canonical_uri
                or metadata["dump_revision"] != source.revision
            ):
                raise ValueError("Wikimedia row provenance mismatch")
            rows.append(
                {
                    "page_id": metadata["page_id"],
                    "revision_id": metadata["revision_id"],
                    "row_sha256": hashlib.sha256(line).hexdigest(),
                }
            )
    if rows != receipt["selected_pages"]:
        raise ValueError("Wikimedia page receipt mismatch")


def verify_snapshot(
    path: Path | str,
    *,
    _staged: bool = False,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    _domain_cold: bool = False,
) -> dict[str, Any]:
    """Authenticate a snapshot; trusted receipts are optional for published roots."""
    path = Path(path)
    if (
        not _staged
        and not _domain_cold
        and proof_store is not None
        and verification_mode == "verified_reuse"
    ):
        from sparselab.experiments.artifacts import verify_artifact
        from sparselab.experiments.plan import Artifact

        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        verify_artifact(
            Artifact(
                kind="source_snapshot",
                version=manifest["schema_version"],
                producer="sparselab",
                identifier=manifest["source_id"],
                sha256=manifest["snapshot_sha256"],
                path=str(path),
            ),
            path / "manifest.json",
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        return manifest
    return _verify_snapshot_cold(path, _staged=_staged)


def _verify_snapshot_cold(path: Path, *, _staged: bool = False) -> dict[str, Any]:
    path = Path(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["schema_version"] != 1
            or manifest["source_id"] != manifest["declaration"]["id"]
        ):
            raise ValueError("snapshot metadata mismatch")
        source = SourceDeclaration.model_validate(manifest["declaration"])
        if (
            declaration_sha256(source) != manifest["declaration_sha256"]
            or manifest["adapter"]["id"] != source.kind
        ):
            raise ValueError("snapshot declaration/adapter identity mismatch")
        identity_fields = {
            "declaration_sha256": manifest["declaration_sha256"],
            "adapter": manifest["adapter"],
            "files": manifest["files"],
        }
        if source.kind == "wikimedia_dump":
            identity_fields["retrieval"] = manifest["retrieval"]
        expected = _digest(identity_fields)
        if expected != manifest["snapshot_sha256"] or (
            not _staged and path.name != expected
        ):
            raise ValueError("snapshot identity mismatch")
        names = set()
        for entry in manifest["files"]:
            name = safe_name(entry["path"])
            if name in names:
                raise ValueError("duplicate snapshot path")
            names.add(name)
            file = path / "files" / name
            _regular(file)
            if (
                not file.resolve().is_relative_to((path / "files").resolve())
                or file.stat().st_size != entry["size"]
                or sha256_file(file) != entry["sha256"]
            ):
                raise ValueError(f"snapshot byte mismatch: {name}")
        actual = {
            p.relative_to(path / "files").as_posix()
            for p in (path / "files").rglob("*")
            if p.is_file() or p.is_symlink()
        }
        if actual != names:
            raise ValueError("snapshot file inventory mismatch")
        if source.kind == "wikimedia_dump":
            _verify_wikimedia_receipt(path, manifest)
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid/incomplete snapshot: {path}") from error


def _project_sha(project: Project) -> str:
    """Only acquisition declarations bind the lock; release variants share snapshots."""
    return _digest(
        {
            "project_id": project.config.id,
            "sources": [
                source_declaration_payload(s)
                for s in sorted(project.sources, key=lambda s: s.id)
            ],
        }
    )


def verify_acquisition(
    project: Project,
    work_root: Path | str,
    *,
    lock_path: Path | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    base = Path(work_root) / "corpora" / project.config.id
    selected = Path(lock_path) if lock_path is not None else base / "acquisition.json"
    if selected.is_symlink() or any(parent.is_symlink() for parent in selected.parents):
        raise ValueError("symlinked acquisition lock")
    try:
        lock = json.loads(selected.read_text(encoding="utf-8"))
        if (
            lock["schema_version"] != 1
            or lock["project_id"] != project.config.id
            or lock["project_sha256"] != _project_sha(project)
            or set(lock["sources"]) != {s.id for s in project.sources}
        ):
            raise ValueError("acquisition lock does not match project")
        for source in project.sources:
            entry = lock["sources"][source.id]
            if entry["declaration_sha256"] != declaration_sha256(source):
                raise ValueError(f"declaration changed: {source.id}")
            if source.redistribution == "rejected":
                if (
                    entry["receipt"]["status"] != "rejected"
                    or entry["receipt"]["reason"] != source.rejection_reason
                    or entry["snapshot_path"] is not None
                    or entry["snapshot_sha256"] is not None
                ):
                    raise ValueError("invalid rejected receipt")
                continue
            snapshot = base / "snapshots" / source.id / entry["snapshot_sha256"]
            if entry["snapshot_path"] != str(snapshot.resolve()):
                raise ValueError("snapshot path mismatch")
            manifest = verify_snapshot(
                snapshot, proof_store=proof_store, verification_mode=verification_mode
            )
            if (
                manifest["snapshot_sha256"] != entry["snapshot_sha256"]
                or manifest["declaration_sha256"] != entry["declaration_sha256"]
                or manifest["declaration"] != source_declaration_payload(source)
            ):
                raise ValueError("snapshot provenance mismatch")
            status = "generator" if source.kind.endswith("generator") else "acquired"
            if entry["receipt"] != {
                "status": status,
                "retrieval": manifest["retrieval"],
            }:
                raise ValueError("snapshot receipt mismatch")
        return lock
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("missing or invalid acquisition lock") from error


def acquire(
    project: Project,
    work_root: Path | str,
    offline: bool = False,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Publish verified snapshots independently; replace the active lock only on success."""
    base = Path(work_root) / "corpora" / project.config.id
    if offline:
        return verify_acquisition(
            project,
            work_root,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
    snapshot_root = base / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    budget = (
        TransportBudget(base / "transport-budget.sqlite", project)
        if project.config.transport_budget is not None
        else None
    )
    entries = {}
    for source in project.sources:
        declared_digest = declaration_sha256(source)
        if source.redistribution == "rejected":
            entries[source.id] = {
                "declaration_sha256": declared_digest,
                "snapshot_sha256": None,
                "snapshot_path": None,
                "receipt": {"status": "rejected", "reason": source.rejection_reason},
            }
            continue
        # A pre-existing lock is advisory for reuse, but never trusted without verification.
        try:
            old = json.loads((base / "acquisition.json").read_text(encoding="utf-8"))[
                "sources"
            ][source.id]
            old_path = snapshot_root / source.id / old["snapshot_sha256"]
            if old["declaration_sha256"] == declared_digest:
                manifest = verify_snapshot(
                    old_path,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
                if (
                    source.kind not in ("local", "http_document")
                    and manifest["declaration"] == source_declaration_payload(source)
                    and _reusable_immutable_adapter(source, manifest)
                ):
                    entries[source.id] = old
                    continue
        except OSError, ValueError, TypeError, KeyError, json.JSONDecodeError:
            pass
        # An interrupted campaign can publish verified source snapshots before
        # writing its final lock. Only immutable upstream revisions are reusable.
        parent = snapshot_root / source.id
        if (
            source.kind in {"git", "huggingface_dataset", "wikimedia_dump"}
            and parent.exists()
        ):
            for candidate in sorted(parent.iterdir()):
                if not candidate.is_dir() or candidate.name.startswith("."):
                    continue
                try:
                    manifest = verify_snapshot(
                        candidate,
                        proof_store=proof_store,
                        verification_mode=verification_mode,
                    )
                except OSError, ValueError, KeyError, TypeError, json.JSONDecodeError:
                    continue
                if (
                    manifest["declaration_sha256"] != declared_digest
                    or manifest["declaration"] != source_declaration_payload(source)
                    or not _reusable_immutable_adapter(source, manifest)
                ):
                    continue
                entries[source.id] = {
                    "declaration_sha256": declared_digest,
                    "snapshot_sha256": manifest["snapshot_sha256"],
                    "snapshot_path": str(candidate.resolve()),
                    "receipt": {
                        "status": "acquired",
                        "retrieval": manifest["retrieval"],
                    },
                }
                break
            if source.id in entries:
                continue
        parent = snapshot_root / source.id
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".acquire-", dir=parent))
        try:
            (staging / "files").mkdir()
            retrieval: dict[str, Any] = {}
            if source.kind == "local":
                files, retrieval = _acquire_local(source, project.root, staging)
            elif source.kind == "git":
                files, retrieval = _acquire_git(
                    source, staging, base / "git-cache", False
                )
            elif source.kind == "http_document":
                files, retrieval = _acquire_http(source, staging, False)
            elif source.kind == "huggingface_dataset":
                files, retrieval = _acquire_hf(
                    source, staging, base / "hf-cache", False, budget
                )
            elif source.kind == "wikimedia_dump":
                files, retrieval = _acquire_wikimedia(source, staging, False)
            else:
                files = []
                retrieval = {
                    "generator_config_sha256": _digest(
                        source.acquisition.model_dump(mode="json")
                    )
                }
            adapter = _adapter(source)
            identity_fields = {
                "declaration_sha256": declared_digest,
                "adapter": adapter,
                "files": files,
            }
            if source.kind == "wikimedia_dump":
                identity_fields["retrieval"] = retrieval
            identity = _digest(identity_fields)
            manifest = {
                "schema_version": 1,
                "source_id": source.id,
                "declaration": source_declaration_payload(source),
                "declaration_sha256": declared_digest,
                "adapter": adapter,
                "files": files,
                "retrieval": retrieval,
                "snapshot_sha256": identity,
            }
            _write_json(staging / "manifest.json", manifest)
            if verify_snapshot(staging, _staged=True)["snapshot_sha256"] != identity:
                raise ValueError("staged snapshot identity mismatch")
            destination = parent / identity
            if destination.exists():
                verify_snapshot(
                    destination,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
            else:
                for directory in sorted((staging / "files").rglob("*"), reverse=True):
                    if directory.is_dir():
                        _sync_dir(directory)
                _sync_dir(staging / "files")
                _sync_dir(staging)
                _rename_noreplace(staging, destination)
                _sync_dir(parent)
                verify_snapshot(
                    destination,
                    proof_store=proof_store,
                    verification_mode=verification_mode,
                )
            entries[source.id] = {
                "declaration_sha256": declared_digest,
                "snapshot_sha256": identity,
                "snapshot_path": str(destination.resolve()),
                "receipt": {
                    "status": "generator"
                    if not files and source.kind.endswith("generator")
                    else "acquired",
                    "retrieval": retrieval,
                },
            }
        except Exception as error:
            if budget is not None:
                budget.record_failure(source.id, error)
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    lock = {
        "schema_version": 1,
        "project_id": project.config.id,
        "project_sha256": _project_sha(project),
        "sources": entries,
    }
    handle, name = tempfile.mkstemp(prefix=".acquisition-", dir=base)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(canonical_json(lock) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, base / "acquisition.json")
        _sync_dir(base)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return verify_acquisition(
        project, work_root, proof_store=proof_store, verification_mode=verification_mode
    )
