"""Freeze an explicitly retained native payload without copying or relinking it.

Historical completion remains historical: authenticate its original signature and
unchanged producer, inputs and outputs. Never reseal it as current production.
"""

import hashlib
import json
import os
from pathlib import Path

from sparselab.corpus.admission_inspection_cache import capture, unchanged
from sparselab.corpus.completed_preparation import _producer
from sparselab.corpus.preparation_inputs import corpus_work_root, local_scope
from sparselab.corpus.retained_render_adoption import _sha, _small
from sparselab.corpus.split_runs import SplitStages, _stamp, write_rows
from sparselab.result_publication import sync_directory
from sparselab.training.manifest import canonical_json, sha256_file

_NOTE = "release-publication.json"


def in_place(path, manifest):
    return (
        local_scope(manifest) is not None
        and path.parent.name == "builds"
        and path.name == manifest.get("build_id")
    )


def retained_payload(path):
    """Read authenticated executed completion; do not replay a cache miss."""
    path = Path(path).resolve(strict=True)
    root = corpus_work_root(path)
    build = json.loads(_small(path / "build.json", root))
    scope = local_scope(build)
    if (
        scope is None
        or _sha(build["identity"]) != path.name
        or build["build_id"] != path.name
        or scope["request"]["producer"] != _producer()
    ):
        raise ValueError("retained native build identity or producer changed")
    unchanged(scope["source_bindings"])
    request_id = _sha(scope["request"])
    directory = path.parent.parent / "build-stages"
    if not (directory / request_id).is_dir():
        raise ValueError("retained native build completion unavailable")
    stages = SplitStages(directory, scope["request"], work_root=root)
    matches = list(stages.root.glob("completion-*.json"))
    candidates = []
    for completion in matches:
        event = json.loads(_small(completion, root))
        if event["record"]["phase"] == "native-build-publication":
            candidates.append(event)
    if len(candidates) != 1:
        raise ValueError("retained native build publication missing or ambiguous")
    event = candidates[0]
    body, payload = event["record"], event["record"]["result"]
    if (
        _sha(body) != event["sha256"]
        or body["request"] != request_id
        or payload.get("build_path") != str(path)
        or payload.get("build_id") != path.name
        or stages.completed.get("native-build-publication") != payload
    ):
        raise ValueError("retained native build execution binding differs")
    request = {"request": request_id, "completion": event["sha256"]}
    cache = stages.cache
    if cache.proofs is None:
        raise ValueError("retained native build proof store unavailable")
    cache_path = cache._path("split-phase", request)
    raw = _small(cache_path, root)
    value = json.loads(raw)
    if set(value) != {"dependencies", "payload"} or value["payload"] != payload:
        raise ValueError("retained native build cache differs")
    paths = value["dependencies"]
    if not isinstance(paths, list) or any(
        not isinstance(name, str)
        or not Path(name).is_absolute()
        or ".." in Path(name).parts
        or not Path(name).is_relative_to(root)
        for name in paths
    ):
        raise ValueError("unsafe retained native build dependencies")
    dependencies = capture(paths)
    binding = cache._binding(cache_path, raw, request, dependencies)
    receipt = cache.proofs._receipt(binding)
    historical = _small(receipt, root, private=True)
    saved = json.loads(historical)["proof"]["binding"]
    if not cache.proofs.lookup(saved):
        raise ValueError("retained native build historical proof invalid or changed")
    before, after = saved["upstream_identities"], binding["upstream_identities"]
    if before["authority"]["streaming_producer"] != _producer():
        raise ValueError("retained native build signed producer changed")
    # Only verifier authority may differ. Mutable declaration fingerprints and
    # every other signed binding field must still match exactly.
    normalized = {
        **binding,
        "upstream_identities": {**after, "authority": before["authority"]},
    }
    if "mutable_dependencies" not in before:
        normalized["upstream_identities"].pop("mutable_dependencies", None)
    normalized["manifest_inventory_sha256"] = _sha(normalized["upstream_identities"])
    if normalized != saved:
        raise ValueError("retained native build semantic binding differs")
    inventory = {item["path"]: item for item in payload["files"]}
    expected = {str(path / name) for name in build["files"]} | {
        str(path / "build.json")
    }
    if set(inventory) != expected:
        raise ValueError("retained native build output inventory differs")
    for name, item in inventory.items():
        output = Path(name)
        if (
            not output.is_relative_to(path)
            or ".." in output.parts
            or _stamp(output) != item["stamp"]
        ):
            raise ValueError("retained native build output changed")
    for name, item in build["files"].items():
        recorded = inventory[str(path / name)]
        if recorded["sha256"] != item["sha256"] or recorded["bytes"] != item["size"]:
            raise ValueError("retained native build manifest differs from execution")
    if inventory[str(path / "build.json")]["sha256"] != sha256_file(
        path / "build.json"
    ):
        raise ValueError("retained native build manifest changed")
    unchanged(dependencies)
    evidence = {
        "kind": "adopted_execution",
        "method": "in_place_retained_native_build",
        "build_id": build["build_id"],
        "completion_sha256": event["sha256"],
        "historical_proof_sha256": hashlib.sha256(historical).hexdigest(),
        "historical_authority_sha256": _sha(before["authority"]),
        "build_manifest_sha256": sha256_file(path / "build.json"),
    }
    return build, evidence


def _publish_metadata(path, value, root):
    encoded = canonical_json(value) + b"\n"
    if path.exists() or path.is_symlink():
        if _small(path, root) != encoded:
            raise ValueError("conflicting completed release metadata")
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        sync_directory(path.parent)
        return
    try:
        write_rows(path, [value])
    except FileExistsError:
        if _small(path, root) != encoded:
            raise ValueError("conflicting completed release metadata") from None
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        sync_directory(path.parent)


def freeze_retained(path, work_root):
    from sparselab.corpus.release import _manifest_payload

    path = Path(path).resolve(strict=True)
    root = corpus_work_root(path)
    if root != Path(work_root).resolve(strict=True):
        raise ValueError("retained release must stay in its owned corpus work root")
    build, evidence = retained_payload(path)
    payload = _manifest_payload(build)
    release_id = _sha(payload)
    evidence = {**evidence, "release_id": release_id}
    # No payload inode, filename, timestamp or historical proof is modified.
    # manifest.json is the final commit; an interrupted metadata-only freeze is
    # repeatable without copying or rebuilding any completed data.
    _publish_metadata(path / _NOTE, evidence, root)
    retained_payload(path)
    _publish_metadata(
        path / "manifest.json", {**payload, "release_id": release_id}, root
    )
    verify_retained_release(path)
    return path


def verify_retained_release(path, manifest=None, *, expected_id=None):
    from sparselab.corpus.release import _manifest_payload

    path = Path(path).resolve(strict=True)
    root = corpus_work_root(path)
    manifest = manifest or json.loads(_small(path / "manifest.json", root))
    build, evidence = retained_payload(path)
    payload = _manifest_payload(build)
    release_id = _sha(payload)
    if (
        not in_place(path, manifest)
        or manifest != {**payload, "release_id": release_id}
        or (expected_id is not None and expected_id != release_id)
    ):
        raise ValueError("invalid retained local release identity")
    if json.loads(_small(path / _NOTE, root)) != {**evidence, "release_id": release_id}:
        raise ValueError("retained local release publication evidence differs")
    return manifest
