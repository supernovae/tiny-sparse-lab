"""Bounded subprocess entry point; import producers only from the materialized Git tree."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("worker requires request and response paths")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if (
        type(request) is not dict
        or set(request)
        != {
            "version",
            "source_root",
            "project",
            "work_root",
            "allow_network",
            "expected_build_sha256",
            "expected_release_sha256",
            "components",
        }
        or type(request["version"]) is not int
        or request["version"] != 1
    ):
        raise ValueError("invalid historical worker request")
    if (
        type(request["allow_network"]) is not bool
        or type(request["components"]) is not dict
    ):
        raise ValueError("invalid historical worker request types")
    for key in ("expected_release_sha256", "expected_build_sha256"):
        value = request[key]
        if value is not None and (
            type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
        ):
            raise ValueError(f"invalid historical worker {key}")
    for key in ("source_root", "project", "work_root"):
        if type(request[key]) is not str or not Path(request[key]).is_absolute():
            raise ValueError(f"invalid historical worker {key}")
    source = Path(request["source_root"]).resolve(strict=True)
    project_path = Path(request["project"])
    work_path = Path(request["work_root"])
    if (
        project_path.is_symlink()
        or not project_path.is_file()
        or not project_path.resolve(strict=True).is_relative_to(source)
    ):
        raise ValueError("invalid historical worker project outside source tree")
    if work_path.resolve(strict=True).is_relative_to(source) or source.is_relative_to(
        work_path.resolve(strict=True)
    ):
        raise ValueError("invalid historical worker root overlap")
    if set(request["components"]) != {
        "acquisition",
        "pipeline",
        "project",
        "provenance",
        "rights",
        "large_build",
        "release",
        "export",
        "training_manifest",
        "project_config",
        "pyproject",
        "uv_lock",
    }:
        raise ValueError("invalid historical worker component set")
    sys.path.insert(0, str(source / "src"))
    phase = "producer_preflight"
    artifacts = {}
    try:
        modules = {}
        for name, entry in request["components"].items():
            if (
                type(entry) is not dict
                or not {
                    "role",
                    "affected_stage",
                    "source_commit",
                    "path",
                    "pinned_sha256",
                    "current_sha256",
                    "direct_identity_hash",
                    "status",
                }.issubset(entry)
                or entry["role"] != name
                or entry["source_commit"] != source.name
                or type(entry["affected_stage"]) is not str
                or type(entry["path"]) is not str
                or type(entry["pinned_sha256"]) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", entry["pinned_sha256"])
            ):
                raise ValueError(f"invalid historical worker component: {name}")
            if (
                not (source / entry["path"]).absolute().is_relative_to(source)
                or ".." in Path(entry["path"]).parts
            ):
                raise ValueError(f"invalid historical worker component path: {name}")
            expected = source / entry["path"]
            if not expected.is_file() or expected.is_symlink():
                raise ValueError(f"MISSING_IMPLEMENTATION: missing historical {name}")
            if (
                hashlib.sha256(expected.read_bytes()).hexdigest()
                != entry["pinned_sha256"]
            ):
                raise ValueError(
                    f"SOURCE_TAMPERED: historical {name} differs from pinned blob"
                )
            if name in {"pyproject", "uv_lock"}:
                continue
            module_name = (
                "sparselab.training.manifest"
                if name == "training_manifest"
                else "sparselab.config.models"
                if name == "project_config"
                else f"sparselab.corpus.{name}"
            )
            module = importlib.import_module(module_name)
            if Path(module.__file__).resolve(strict=True) != expected:
                raise ValueError(
                    f"MISSING_IMPLEMENTATION: historical {name} imported from {module.__file__}"
                )
            modules[name] = module
        acquisition = modules["acquisition"]
        pipeline = modules["pipeline"]
        project = modules["project"]
        release = modules["release"]
        phase = "project"
        corpus = project.load_project(Path(request["project"]))
        work = Path(request["work_root"])
        remote = any(
            item.kind not in {"local", "deterministic_generator"}
            for item in corpus.sources
            if item.redistribution != "rejected"
        )
        if (
            remote
            and not request["allow_network"]
            and not (work / "corpora" / corpus.config.id / "acquisition.json").is_file()
        ):
            raise ValueError(
                "NETWORK_PERMISSION_REQUIRED: historical acquisition requires network"
            )
        phase = "acquisition"
        lock = acquisition.acquire(
            corpus, work, offline=remote and not request["allow_network"]
        )
        acquisition.verify_acquisition(corpus, work)
        artifacts["snapshots"] = [
            {"source_id": key, "sha256": value["snapshot_sha256"]}
            for key, value in lock["sources"].items()
            if value["snapshot_sha256"]
        ]
        phase = "build"
        built = pipeline.build(corpus, work, offline=True)
        build = release.verify_build(built)
        artifacts["build_sha256"] = build["build_id"]
        artifacts["build_path"] = str(built)
        if (
            request["expected_build_sha256"] is not None
            and build["build_id"] != request["expected_build_sha256"]
        ):
            raise ValueError(
                f"EXPECTED_BUILD_MISMATCH: actual={build['build_id']} expected={request['expected_build_sha256']}"
            )
        phase = "freeze"
        frozen = release.freeze(built, work)
        result = release.verify_release(frozen)
        artifacts["release_sha256"] = result["release_id"]
        artifacts["path"] = str(frozen)
        phase = "release_verification"
        if (
            request["expected_release_sha256"] is not None
            and result["release_id"] != request["expected_release_sha256"]
        ):
            raise ValueError(
                f"EXPECTED_DIGEST_MISMATCH: actual={result['release_id']} expected={request['expected_release_sha256']}"
            )
        payload = {
            "version": 1,
            "status": "MATCH",
            "path": str(frozen),
            "build_path": str(built),
            "build_sha256": build["build_id"],
            "release_sha256": result["release_id"],
            "snapshots": [
                {"source_id": key, "sha256": value["snapshot_sha256"]}
                for key, value in lock["sources"].items()
                if value["snapshot_sha256"]
            ],
            "modules": {
                name: str(Path(module.__file__).resolve())
                for name, module in modules.items()
            },
        }
    except Exception as error:
        logger.exception("Historical Corpus Forge replay failed in %s", phase)
        payload = {
            "version": 1,
            "status": "FAILED",
            "phase": phase,
            "artifacts": artifacts,
            "error": str(error),
            "error_type": type(error).__name__,
        }
    Path(sys.argv[2]).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return 0 if payload["status"] == "MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
