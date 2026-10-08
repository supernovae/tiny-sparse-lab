"""Offline checks of tracked research declarations, never their external payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from sparselab.experiments.plan import read_document
from sparselab.recovery.evidence import KINDS
from sparselab.recovery.provenance import declaration_paths, declaration_reference

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_RESEARCH = Path("experiments/research")
_FORBIDDEN_DIRS = {
    "checkpoints",
    "checkpoint",
    "cache",
    "caches",
    ".cache",
    "__pycache__",
    "datasets",
    "dataset",
    "logs",
    "runs",
    "run",
    "outputs",
    "output",
    "scratch",
    "tmp",
    "prepared",
    "downloads",
    "wandb",
    "tensorboard",
}
_FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".npy",
    ".npz",
    ".arrow",
    ".parquet",
    ".csv",
    ".tsv",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".log",
    ".pyc",
    ".pkl",
    ".pickle",
    ".zip",
    ".gz",
    ".tar",
    ".zst",
}
_RECORD_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".jsonl"}
_MUTABLE_NAMES = {"metrics.jsonl", "events.jsonl", "stdout", "stderr", "run.db"}
# Historical KML records are retained byte-for-byte. These bindings bypass only
# the unsupported-suffix/JSONL-panel classification below, never path safety,
# tracked-file, mutable-output, size, or declaration validation for other files.
_LEGACY_RECORD_SHA256 = {
    "experiments/research/kernel-memory-lab/corpus-scale/run-offline-continuation.sh": "7dd86e3701c76a1de1b5a7a10c635100b044fbfae21dd647b59897980f1f1260",
    "experiments/research/kernel-memory-lab/results/2026-10-07-card03-review-index.jsonl": "eaeca027b321913ea59a938d01e1b878c80d7a54ee9a3201e2173360c81dbb5d",
    "experiments/research/kernel-memory-lab/results/evidence/card05-v2-language-final-scores.jsonl": "5a5b76046ddc7a670bf4f386e94ad17c9260185cf3290fcb23538d66fce41f43",
}


def _sha(value: Any) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("expected a lowercase SHA-256 digest")
    return value


def _safe_file(root: Path, path: Path, tracked: set[str]) -> Path:
    """Reject links before reading bytes, including links in an ancestor."""
    if not path.is_relative_to(root):
        raise ValueError(f"reference outside repository: {path}")
    relative = path.relative_to(root)
    for component in (path, *path.parents):
        if component == root:
            break
        if component.is_symlink():
            raise ValueError(f"symlinked research reference: {relative}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"reference outside repository: {relative}")
    if relative.as_posix() not in tracked or not path.is_file():
        raise ValueError(f"missing tracked reference: {relative}")
    return path


def _reference(root: Path, source: Path, name: Any, tracked: set[str]) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or "\\" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError(f"unsafe repository reference: {name!r}")
    return _safe_file(root, declaration_reference(source, name), tracked)


def _binding(root: Path, source: Path, name: Any, sha: Any, tracked: set[str]) -> None:
    expected = _sha(sha)
    target = _reference(root, source, name, tracked)
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise ValueError(f"reference digest mismatch: {name}")


def _evidence(root: Path, source: Path, raw: dict, tracked: set[str]) -> None:
    # The export envelope describes a past verification. Do not open external_location.
    if raw.get("kind") not in KINDS:
        raise ValueError("unknown scientific evidence kind")
    _sha(raw.get("sha256"))
    commit = raw.get("source_commit")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError("evidence source_commit must be a full Git object ID")
    for key in ("identifier", "external_location", "verified_at"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"evidence requires nonempty {key}")
    expected_scope = (
        "observation" if raw["kind"] == "runtime_probe" else "verified_artifact"
    )
    if raw.get("verification_scope") != expected_scope:
        raise ValueError("evidence verification_scope disagrees with kind")
    declarations = raw.get("declaration_hashes")
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("evidence requires declaration_hashes")
    seen: set[str] = set()
    for item in declarations:
        if not isinstance(item, dict):
            raise TypeError("declaration_hashes entries must be objects")
        name = item.get("path")
        _reference(root, source, name, tracked)
        if name in seen:
            raise ValueError(f"duplicate declaration binding: {name}")
        seen.add(name)
        _binding(root, source, name, item.get("sha256"), tracked)
        if (
            item.get("head_sha256") is not None
            and item["head_sha256"] != item["sha256"]
        ):
            raise ValueError(f"declaration committed digest mismatch: {name}")


def _document(
    root: Path, source: Path, raw: dict, tracked: set[str], visited: set[Path]
) -> None:
    if source in visited:
        return
    visited.add(source)
    for version, kind in (
        ("plan_version", "experiment"),
        ("campaign_version", "campaign"),
        ("recovery_version", "recovery"),
    ):
        if version in raw:
            for target in declaration_paths(source, kind, include_local_inputs=False):
                _safe_file(root, target, tracked)
                if target != source and target.suffix.lower() in {
                    ".yaml",
                    ".yml",
                    ".json",
                }:
                    _document(root, target, read_document(target), tracked, visited)
            if kind == "experiment":
                from sparselab.experiments.plan import base_run_config, load_plan

                base_run_config(load_plan(source), source)
    if "producer_record_version" in raw:
        if (
            type(raw["producer_record_version"]) is not int
            or raw["producer_record_version"] != 1
        ):
            raise ValueError("unsupported producer record version")
        _reference(root, source, raw.get("protocol"), tracked)
        project = _reference(root, source, raw.get("corpus_project"), tracked)
        for target in declaration_paths(project, "corpus", include_local_inputs=False):
            _safe_file(root, target, tracked)
            _document(root, target, read_document(target), tracked, visited)
    if "recovery_version" in raw:
        from sparselab.config.loading import load_config, load_tokenizer_config
        from sparselab.recovery.manifest import load_manifest

        for step in load_manifest(source).steps:
            for field in ("config", "base_run"):
                name = getattr(step, field, None)
                if name:
                    target = _reference(root, source, name, tracked)
                    loader = (
                        load_tokenizer_config
                        if step.kind == "tokenizer_train"
                        else load_config
                    )
                    loader(target)
    if "readiness_version" in raw:
        from sparselab.evaluation.readiness import load_policy

        load_policy(source)
    if {"model", "dataset", "training"}.issubset(raw):
        from sparselab.config.models import RunConfig

        RunConfig.model_validate(raw)
    if "family_version" in raw:
        from sparselab.family.manifest import load_family

        family = load_family(source)
        for node in family.nodes:
            for name in (*node.lifecycle_receipts, node.recovery_manifest):
                if name:
                    _reference(root, source, name, tracked)
    if "evaluation_suite_version" in raw:
        from sparselab.evaluation.suite import load_suite

        suite = load_suite(source)
        for item in suite.evaluations:
            # surface_review sources are external run outputs, not declarations.
            if item.source and item.kind != "surface_review":
                _reference(root, source, item.source, tracked)
    for field in ("frozen_sha256", "inputs_sha256"):
        if field not in raw:
            continue
        frozen = raw[field]
        if not isinstance(frozen, dict) or not frozen:
            raise ValueError(f"{field} must be a nonempty path/digest mapping")
        for name, sha in frozen.items():
            _binding(root, source, name, sha, tracked)
    if (
        str(raw.get("format", "")).startswith("scientific-evidence-reference-")
        and raw["format"] != "scientific-evidence-reference-v1"
    ):
        raise ValueError("unsupported scientific evidence reference version")
    if (
        "declaration_hashes" in raw
        and raw.get("format") != "scientific-evidence-reference-v1"
    ):
        raise ValueError(
            "declaration_hashes requires a scientific evidence reference envelope"
        )
    if raw.get("format") == "scientific-evidence-reference-v1":
        _evidence(root, source, raw, tracked)
    # Explicit scientific bindings are current immutable declarations; unrelated
    # nested implementation/archive inventories remain historical metadata.
    if "bindings" in raw:
        if not isinstance(raw["bindings"], dict):
            raise ValueError("bindings must be a mapping")
        for binding in raw["bindings"].values():
            if not isinstance(binding, dict):
                raise TypeError("bindings entries must be path/digest objects")
            _binding(root, source, binding.get("path"), binding.get("sha256"), tracked)
    for key in ("protocol", "config", "base_run"):
        binding = raw.get(key)
        if isinstance(binding, dict) and "path" in binding:
            _binding(root, source, binding["path"], binding.get("sha256"), tracked)
    for split in ("development", "test"):
        field = f"source_{split}_files"
        if field in raw:
            files = raw[field]
            if not isinstance(files, dict) or not files:
                raise ValueError(f"{field} must be a nonempty path/digest mapping")
            for name, sha in files.items():
                _binding(root, source, f"{split}/{name}", sha, tracked)
    if "reference_checkpoint_receipt" in raw:
        _reference(root, source, raw["reference_checkpoint_receipt"], tracked)


def _review(root: Path, source: Path, rows: list, tracked: set[str]) -> None:
    """Recognize the existing content-addressed blind comparison panel."""
    fields = {
        "pair_id",
        "prompt",
        "category",
        "a",
        "b",
        "dimensions",
        "allowed_votes",
        "judgments",
    }
    if (
        source.name != "review-blind.json"
        or source.parent.name != "evidence"
        or not rows
    ):
        raise ValueError("JSON arrays are only allowed for frozen blind review panels")
    identifiers: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("invalid blind review panel row")
        if any(
            not isinstance(row[key], str) or not row[key]
            for key in ("pair_id", "prompt", "category", "a", "b")
        ):
            raise ValueError("blind review row requires nonempty text fields")
        if row["pair_id"] in identifiers:
            raise ValueError("duplicate blind review pair ID")
        identifiers.add(row["pair_id"])
        if (
            not isinstance(row["dimensions"], list)
            or not row["dimensions"]
            or any(not isinstance(item, str) or not item for item in row["dimensions"])
            or row["allowed_votes"] != ["A", "B", "tie", "uncertain"]
            or not isinstance(row["judgments"], list)
        ):
            raise ValueError("invalid blind review dimensions, votes, or judgments")
    summary = _safe_file(root, source.parent / "summary.json", tracked)
    blind = read_document(summary).get("blind_review")
    if not isinstance(blind, dict) or blind.get("pair_count") != len(rows):
        raise ValueError("blind review summary pair count mismatch")
    _binding(root, source, source.name, blind.get("blind_sha256"), tracked)


def _panel(root: Path, source: Path, tracked: set[str]) -> None:
    """Allow the existing frozen decoding evidence, not arbitrary JSONL datasets."""
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
    ]
    if not rows or not isinstance(rows[0], dict):
        raise ValueError("evidence panel requires a header")
    header = rows[0]
    split = header.get("split")
    if (
        header.get("format") != "dense_lm_decoding_cells_v1"
        or header.get("type") != "header"
        or split not in {"development", "test"}
        or source.parent.name != split
        or source.parent.parent.name != "evidence"
    ):
        raise ValueError("JSONL payload is not a frozen decoding evidence panel")
    if len(rows) < 2 or any(
        not isinstance(row, dict) or row.get("type") != "cell" for row in rows[1:]
    ):
        raise ValueError("evidence panel requires cell records")
    summary = _safe_file(root, source.parent.parent / "summary.json", tracked)
    raw = read_document(summary)
    files = raw.get(f"source_{split}_files")
    if not isinstance(files, dict):
        raise TypeError("evidence summary requires a source-file digest mapping")
    expected = files.get(source.name)
    _binding(root, source, source.name, expected, tracked)
    campaign = source.parent.parent.parent
    for key, name in (
        ("preregistration_sha256", "preregistration.json"),
        ("prompt_sha256", f"{split}.json"),
    ):
        _binding(
            root,
            source,
            (campaign / name).relative_to(root).as_posix(),
            header.get(key),
            tracked,
        )


def lint_research(root: Path) -> dict[str, Any]:
    """Lint tracked working-tree bytes (including staged additions), without writes."""
    root = Path(root).resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    tracked = set(result.stdout.decode().rstrip("\0").split("\0"))
    paths = sorted(name for name in tracked if Path(name).is_relative_to(_RESEARCH))
    errors: list[dict[str, str]] = []
    visited: set[Path] = set()
    for name in paths:
        source = root / name
        relative = Path(name).relative_to(_RESEARCH)
        try:
            _safe_file(root, source, tracked)
            if (
                {part.lower() for part in relative.parts[:-1]} & _FORBIDDEN_DIRS
                or source.suffix.lower() in _FORBIDDEN_SUFFIXES
                or source.name.lower() in _MUTABLE_NAMES
                or source.name.startswith("events.out.tfevents.")
            ):
                raise ValueError(
                    "forbidden mutable output or payload in research records"
                )
            if source.stat().st_size > 1024 * 1024:
                raise ValueError(
                    "research record exceeds the 1 MiB durable-record limit"
                )
            expected_legacy = _LEGACY_RECORD_SHA256.get(name)
            if expected_legacy is not None:
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected_legacy:
                    raise ValueError("pinned legacy research record digest mismatch")
                continue
            if source.suffix.lower() not in _RECORD_SUFFIXES:
                raise ValueError(
                    "unsupported research record format; payloads belong outside the checkout"
                )
            if source.suffix.lower() == ".json":
                raw = json.loads(source.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    _document(root, source, read_document(source), tracked, visited)
                elif isinstance(raw, list):
                    _review(root, source, raw, tracked)
                else:
                    raise ValueError("research JSON must be an object or review array")
            elif source.suffix.lower() in {".yaml", ".yml"}:
                _document(root, source, read_document(source), tracked, visited)
            elif source.suffix.lower() == ".jsonl":
                if "evidence" not in relative.parts[:-1]:
                    raise ValueError(
                        "JSONL payloads belong outside research; only frozen evidence panels are allowed"
                    )
                _panel(root, source, tracked)
        except (ValueError, TypeError, KeyError, OSError) as error:
            errors.append({"path": name, "message": str(error)})
    if not paths:
        errors.append(
            {"path": _RESEARCH.as_posix(), "message": "no tracked research records"}
        )
    return {
        "valid": not errors,
        "checked_files": len(paths),
        "errors": errors,
        "scope": "tracked declarations and references; external payloads not verified",
    }


def _handle(args: argparse.Namespace) -> None:
    report = lint_research(Path(args.repository))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"Research records: {report['checked_files']} checked, {len(report['errors'])} errors"
        )
        for error in report["errors"]:
            print(f"{error['path']}: {error['message']}")
        print(report["scope"])
    if not report["valid"]:
        raise SystemExit(1)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "lint", help="Validate tracked research declarations offline."
    )
    parser.add_argument("--repository", default=".")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=_handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--json", action="store_true")
    _handle(parser.parse_args())
