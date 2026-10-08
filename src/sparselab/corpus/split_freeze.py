"""Freeze reviewed, admission-aware document-family splits from native inventory."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from sparselab.corpus.acquisition import verify_acquisition
from sparselab.corpus.project import Project, project_path, source_declaration_payload
from sparselab.corpus.release import verify_release
from sparselab.corpus.rights import verify_record_admission
from sparselab.corpus.split_inventory import write_split_inventory
from sparselab.training.manifest import canonical_json, sha256_file
from sparselab.verification_proofs import verification_options

_SPLITS = ("train", "validation", "test")
_WEIGHTS = (8, 1, 1)


def _counts(total: int) -> tuple[int, int, int]:
    floors = [total * weight // 10 for weight in _WEIGHTS]
    remainders = [total * weight % 10 for weight in _WEIGHTS]
    for index in sorted(range(3), key=lambda i: (-remainders[i], i))[
        : total - sum(floors)
    ]:
        floors[index] += 1
    return floors[0], floors[1], floors[2]


def _publish_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=".freeze-splits-", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _prior_family_splits(
    binding: object, root: Path
) -> tuple[dict[str, str], dict[str, str], dict[str, tuple[str, str]]]:
    """Authenticate a previous release's kept family assignments before extending it."""
    if not isinstance(binding, dict) or set(binding) != {
        "path",
        "sha256",
        "release_path",
    }:
        raise ValueError("invalid prior family-inventory binding")
    prior_path = Path(binding["path"]).resolve()
    release = Path(binding["release_path"]).resolve()
    if (
        not prior_path.is_relative_to(root)
        or not release.is_relative_to(root)
        or prior_path.is_symlink()
        or release.is_symlink()
        or sha256_file(prior_path) != binding["sha256"]
    ):
        raise ValueError("prior family inventory path or digest mismatch")
    manifest = verify_release(release, **verification_options(root))
    if manifest["release_id"] != release.name:
        raise ValueError("prior release identity mismatch")
    kept = {}
    with (release / "documents.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                doc = json.loads(line)
                if doc["drop_reason"] is None:
                    kept[doc["document_id"]] = doc
    families: dict[str, str] = {}
    strata: dict[str, str] = {}
    content: dict[str, tuple[str, str]] = {}
    seen: set[str] = set()
    with prior_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or set(row)
                != {"document_id", "family_id", "split", "stratum", "content_sha256"}
                or row["document_id"] in seen
                or row["split"] not in _SPLITS
                or not isinstance(row["family_id"], str)
                or not row["family_id"]
            ):
                raise ValueError("invalid prior family inventory row")
            seen.add(row["document_id"])
            doc = kept.get(row["document_id"])
            if (
                doc is None
                or doc["split"] != row["split"]
                or doc["content_sha256"] != row["content_sha256"]
                or row["stratum"] not in doc["domains"]
            ):
                raise ValueError("prior family inventory differs from release")
            family = row["family_id"]
            if family in families and (
                families[family] != row["split"] or strata[family] != row["stratum"]
            ):
                raise ValueError("prior family spans split or stratum")
            families[family] = row["split"]
            strata[family] = row["stratum"]
            previous = content.setdefault(row["content_sha256"], (family, row["split"]))
            if previous != (family, row["split"]):
                raise ValueError("prior content spans families or splits")
    if seen != set(kept):
        raise ValueError("prior family inventory does not cover kept release")
    return families, strata, content


def freeze_splits(
    project: Project,
    work_root: Path,
    inventory_path: Path,
    clusters_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Assign eligible document families; excluded rows get held-out placeholders."""
    root = work_root.resolve()
    paths = [inventory_path.resolve(), clusters_path.resolve(), output.resolve()]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("split inputs and output must reside in the work root")
    if output.exists() or output.is_symlink():
        raise ValueError("frozen split output already exists")
    raw_clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw_clusters, dict)
        or not {
            "schema_version",
            "inventory_sha256",
            "seed",
            "source_strata",
            "merges",
            "reviewer",
            "reviewed_on",
        }.issubset(raw_clusters)
        or set(raw_clusters)
        - {
            "schema_version",
            "inventory_sha256",
            "seed",
            "source_strata",
            "merges",
            "reviewer",
            "reviewed_on",
            "prior_family_inventory",
        }
        or raw_clusters["schema_version"] != 1
        or not isinstance(raw_clusters["seed"], str)
        or not raw_clusters["seed"].strip()
        or not isinstance(raw_clusters["reviewer"], str)
        or not raw_clusters["reviewer"].strip()
        or not isinstance(raw_clusters["reviewed_on"], str)
        or not raw_clusters["reviewed_on"].strip()
        or raw_clusters["inventory_sha256"] != sha256_file(inventory_path)
        or not isinstance(raw_clusters["source_strata"], dict)
        or set(raw_clusters["source_strata"])
        != {source.id for source in project.sources}
        or any(
            stratum
            not in {"general_prose", "explanatory_prose", "incident_response_docs"}
            for stratum in raw_clusters["source_strata"].values()
        )
        or not isinstance(raw_clusters["merges"], list)
    ):
        raise ValueError("invalid reviewed family-cluster declaration")
    prior_families, prior_strata, prior_content = (
        _prior_family_splits(raw_clusters["prior_family_inventory"], root)
        if "prior_family_inventory" in raw_clusters
        else ({}, {}, {})
    )
    # Recompute through the same verified native parser, not caller-supplied IDs.
    check_dir = root / "corpora" / project.config.id / "split-inventory-check"
    check_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verify-", dir=check_dir) as folder:
        regenerated = Path(folder) / "inventory.jsonl"
        report = write_split_inventory(project, root, regenerated)
        if report["sha256"] != raw_clusters["inventory_sha256"]:
            raise ValueError("split inventory differs from verified snapshots")
    lock = verify_acquisition(project, root)
    reference = project.release.record_admission
    if reference is None:
        raise ValueError("frozen splits require a reviewed record admission")
    admission_path = project_path(project.root, reference.path)
    admission_bytes = admission_path.read_bytes()
    if hashlib.sha256(admission_bytes).hexdigest() != reference.sha256:
        raise ValueError("record admission SHA-256 mismatch")
    manifest = json.loads(admission_bytes)
    if manifest.get("schema_version") != 2:
        raise ValueError("frozen splits require complete v2 admission")
    sources = {
        source.id: source_declaration_payload(source) for source in project.sources
    }
    snapshots = {
        source_id: json.loads(
            (Path(entry["snapshot_path"]) / "manifest.json").read_text(encoding="utf-8")
        )
        for source_id, entry in lock["sources"].items()
    }
    admitted = verify_record_admission(
        manifest,
        sources,
        snapshots,
        root / "corpora" / project.config.id / "snapshots",
    )
    inventory = [
        json.loads(line)
        for line in inventory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row["document_id"] for row in inventory]
    if len(set(ids)) != len(ids):
        raise ValueError("split inventory has duplicate document IDs")
    hints = {row["family_hint"] for row in inventory}
    merged: dict[str, str] = {}
    for item in raw_clusters["merges"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"family_id", "member_hints"}
            or not isinstance(item["family_id"], str)
            or not item["family_id"].strip()
            or not isinstance(item["member_hints"], list)
            or len(item["member_hints"]) < 2
            or len(set(item["member_hints"])) != len(item["member_hints"])
            or any(hint not in hints or hint in merged for hint in item["member_hints"])
        ):
            raise ValueError("invalid reviewed family merge")
        for hint in item["member_hints"]:
            merged[hint] = item["family_id"]
    families: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    assignments: dict[str, str] = {}
    qualified_family: dict[str, tuple[str, str]] = {}
    eligible = 0
    for row in inventory:
        source_id = row["source_id"]
        path = row["source_location"].split("#", 1)[0]
        decision = admitted.get((source_id, path))
        if decision is None:
            raise ValueError("split inventory lacks file admission decision")
        if "decisions" in decision:
            index = row["source_row_index"]
            choice = decision["decisions"].get(index)
            if choice is None:
                raise ValueError("split inventory lacks HF row admission decision")
            state = choice["decision"]
        else:
            state = decision["decision"]
        if state != "qualify":
            assignments[row["document_id"]] = "test"
            continue
        hint = row["family_hint"]
        if not isinstance(hint, str) or hint.startswith("unresolved-family:"):
            raise ValueError("eligible document has unresolved family hint")
        stratum = raw_clusters["source_strata"][source_id]
        family = merged.get(hint, hint)
        previous = prior_content.get(row["content_sha256"])
        if previous is not None and previous[0] != family:
            raise ValueError("prior content changed reviewed family")
        families[stratum, family].append(row)
        qualified_family[row["document_id"]] = (stratum, family)
        eligible += 1
    # A family may not straddle source strata. It must be merged/reviewed first.
    family_strata: dict[str, set[str]] = defaultdict(set)
    for stratum, family in families:
        family_strata[family].add(stratum)
    if any(len(strata) > 1 for strata in family_strata.values()):
        raise ValueError("reviewed family spans multiple token strata")
    realized: dict[str, dict[str, Any]] = {}
    for stratum in ("general_prose", "explanatory_prose", "incident_response_docs"):
        names = sorted(
            (
                family
                for family_stratum, family in families
                if family_stratum == stratum
            ),
            key=lambda family: hashlib.sha256(
                canonical_json([raw_clusters["seed"], stratum, family])
            ).hexdigest(),
        )
        if len(names) < 10:
            raise ValueError(f"too few eligible families for {stratum} split")
        train, validation, test = _counts(len(names))
        realized[stratum] = {
            "families": len(names),
            "train": train,
            "validation": validation,
            "test": test,
            "fractions": {
                "train": f"{train}/{len(names)}",
                "validation": f"{validation}/{len(names)}",
                "test": f"{test}/{len(names)}",
            },
        }
        planned = dict(zip(_SPLITS, (train, validation, test), strict=True))
        pinned = {
            family: prior_families[family]
            for family in names
            if family in prior_families
        }
        if any(prior_strata[family] != stratum for family in pinned):
            raise ValueError("prior family changed token stratum")
        remaining = {
            split: planned[split] - sum(value == split for value in pinned.values())
            for split in _SPLITS
        }
        if any(count < 0 for count in remaining.values()):
            raise ValueError("prior family assignments exceed new split quota")
        new_names = [family for family in names if family not in pinned]
        if sum(remaining.values()) != len(new_names):
            raise ValueError("prior family split accounting mismatch")
        assigned = dict(pinned)
        for index, family in enumerate(new_names):
            assigned[family] = (
                "train"
                if index < remaining["train"]
                else "validation"
                if index < remaining["train"] + remaining["validation"]
                else "test"
            )
        for family in names:
            split = assigned[family]
            for row in families[stratum, family]:
                assignments[row["document_id"]] = split
    if set(assignments) != set(ids):
        raise ValueError("split assignments do not cover inventory")
    by_content: dict[str, set[str]] = defaultdict(set)
    for row in inventory:
        if row["document_id"] not in assignments:
            raise ValueError("missing document assignment")
        if row["document_id"] in qualified_family:
            by_content[row["content_sha256"]].add(assignments[row["document_id"]])
            previous = prior_content.get(row["content_sha256"])
            if previous is not None and assignments[row["document_id"]] != previous[1]:
                raise ValueError("prior held-out content changed split")
    if any(len(splits) > 1 for splits in by_content.values()):
        raise ValueError("cross-split exact content duplicate")
    declaration = {
        "schema_version": 1,
        "unit": "document",
        "family_key": None,
        "assignments": dict(sorted(assignments.items())),
    }
    payload = yaml.safe_dump(declaration, sort_keys=False).encode("utf-8")
    family_rows = [
        {
            "document_id": row["document_id"],
            "family_id": qualified_family[row["document_id"]][1],
            "split": assignments[row["document_id"]],
            "stratum": qualified_family[row["document_id"]][0],
            "content_sha256": row["content_sha256"],
        }
        for row in sorted(inventory, key=lambda row: row["document_id"])
        if row["document_id"] in qualified_family
    ]
    candidate_path = output.with_name(output.stem + ".family-candidates.jsonl")
    receipt_path = output.with_name(output.stem + ".receipt.json")
    if any(
        path.exists() or path.is_symlink() for path in (candidate_path, receipt_path)
    ):
        raise ValueError("frozen split companion output already exists")
    family_payload = b"".join(canonical_json(row) + b"\n" for row in family_rows)
    receipt = {
        "schema_version": 1,
        "split_sha256": hashlib.sha256(payload).hexdigest(),
        "family_candidate_sha256": hashlib.sha256(family_payload).hexdigest(),
        "inventory_sha256": report["sha256"],
        "admission_sha256": reference.sha256,
        "cluster_sha256": sha256_file(clusters_path),
        "family_counts": realized,
        **(
            {
                "prior_family_inventory_sha256": raw_clusters["prior_family_inventory"][
                    "sha256"
                ],
                "prior_release_id": Path(
                    raw_clusters["prior_family_inventory"]["release_path"]
                ).name,
            }
            if prior_families
            else {}
        ),
    }
    _publish_new(candidate_path, family_payload)
    _publish_new(output, payload)
    _publish_new(receipt_path, canonical_json(receipt) + b"\n")
    return {
        "schema_version": 1,
        "output": str(output),
        "sha256": sha256_file(output),
        "inventory_sha256": report["sha256"],
        "admission_sha256": reference.sha256,
        "cluster_sha256": sha256_file(clusters_path),
        "family_candidates": str(candidate_path),
        "family_candidate_sha256": receipt["family_candidate_sha256"],
        "receipt": str(receipt_path),
        "eligible_documents": eligible,
        "placeholder_documents": len(ids) - eligible,
        "family_counts": realized,
    }


def finalize_family_inventory(
    release: Path, frozen_splits: Path, output: Path
) -> dict[str, Any]:
    """Filter the authenticated family candidates to exactly kept release docs."""
    manifest = verify_release(release)
    receipt_path = frozen_splits.with_name(frozen_splits.stem + ".receipt.json")
    candidate_path = frozen_splits.with_name(
        frozen_splits.stem + ".family-candidates.jsonl"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("split_sha256") != sha256_file(frozen_splits)
        or receipt.get("family_candidate_sha256") != sha256_file(candidate_path)
        or manifest["build_identity"]["split"]
        != yaml.safe_load(frozen_splits.read_text(encoding="utf-8"))
    ):
        raise ValueError("family candidates differ from frozen release split")
    candidates = {
        row["document_id"]: row
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    docs = {
        row["document_id"]: row
        for line in (release / "documents.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    if len(candidates) != sum(
        1 for line in candidate_path.read_text().splitlines() if line.strip()
    ):
        raise ValueError("duplicate family candidate document")
    if set(candidates) != set(docs):
        raise ValueError("family candidates differ from release documents")
    kept = []
    for document_id, doc in sorted(docs.items()):
        row = candidates[document_id]
        if (
            set(row)
            != {"document_id", "family_id", "split", "stratum", "content_sha256"}
            or row["split"] != doc["split"]
            or row["content_sha256"] != doc["content_sha256"]
            or row["stratum"] not in doc["domains"]
        ):
            raise ValueError("family candidate disagrees with verified release")
        if doc["drop_reason"] is None:
            kept.append(row)
    if output.exists() or output.is_symlink():
        raise ValueError("final family inventory already exists")
    _publish_new(output, b"".join(canonical_json(row) + b"\n" for row in kept))
    return {
        "schema_version": 1,
        "release_id": manifest["release_id"],
        "output": str(output),
        "sha256": sha256_file(output),
        "kept_documents": len(kept),
        "duplicate_documents_removed": len(docs) - len(kept),
        "split_sha256": receipt["split_sha256"],
        "family_candidate_sha256": receipt["family_candidate_sha256"],
    }
