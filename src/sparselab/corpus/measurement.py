"""Descriptive, read-only measurements of verified corpus release views."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sparselab.training.manifest import canonical_json, sha256_file

SPLITS = ("train", "validation", "test")
DIMENSIONS = ("origin", "verification_status", "shape")
_WORD = re.compile(r"\w+", re.UNICODE)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _distribution(
    rows: list[tuple[dict[str, Any], int | None]], *, counted_tokens: bool
) -> dict[str, Any]:
    """Keep row and token denominators distinct; multi-valued domains overlap."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    tokens: dict[str, Counter[str]] = defaultdict(Counter)
    token_total = sum(count or 0 for _, count in rows) if counted_tokens else None
    for row, token_count in rows:
        shape = row.get("shape") or {}
        attributes = shape.get("attributes") or {}
        generator = row.get("generator") or {}
        verification = row.get("verification") or {}
        fields = {
            "origin": row.get("origin", "legacy_unclassified"),
            "verification_status": verification.get(
                "status", row.get("validation_status", "unverified")
            ),
            "shape": shape.get("id", "legacy_unclassified"),
            "generator_identity": row.get("generator_identity") or "none",
            "generator_provider": generator.get("provider") or "none",
            "generator_model": generator.get("model") or "none",
            "generator_adapter_id": generator.get("adapter_id") or "none",
            "generator_model_revision": generator.get("model_revision") or "none",
            "generator_version": generator.get("generator_version")
            or generator.get("adapter_version")
            or "none",
            "template_family": row.get("template_family_id") or "none",
            "template_id": generator.get("template_id")
            or row.get("template_family_id")
            or "none",
        }
        fields.update(
            {
                f"shape.{key}": value
                for key, value in attributes.items()
                if isinstance(value, str)
            }
        )
        for key, value in fields.items():
            counts[key][str(value)] += 1
            if token_count is not None:
                tokens[key][str(value)] += token_count
        for domain in attributes.get("source_domains", row.get("domains", [])):
            counts["source_domain"][domain] += 1
            if token_count is not None:
                tokens["source_domain"][domain] += token_count
        for modality in row.get("modalities", []):
            counts["source_modality"][modality] += 1
            if token_count is not None:
                tokens["source_modality"][modality] += token_count
    return {
        "records": len(rows),
        "actual_tokens": token_total,
        "token_count_reason": None if counted_tokens else "tokenizer_not_declared",
        "dimensions": {
            key: {
                value: {
                    "records": count,
                    "record_fraction": count / len(rows),
                    "actual_tokens": tokens[key][value] if counted_tokens else None,
                    "token_fraction": (
                        tokens[key][value] / token_total if token_total else None
                    ),
                }
                for value, count in sorted(values.items())
            }
            for key, values in sorted(counts.items())
        },
    }


def _concentration(
    texts: list[str], chat: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Cheap descriptive overlap statistics, never a quality or correctness score."""
    normalized = [
        unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
        for text in texts
    ]
    duplicate = sum(n - 1 for n in Counter(texts).values() if n > 1)
    normalized_duplicate = sum(n - 1 for n in Counter(normalized).values() if n > 1)
    prefixes = Counter(text[:32] for text in texts if len(text) >= 32)
    prefix_rows = sum(n - 1 for n in prefixes.values() if n > 1)
    ngrams: Counter[tuple[str, ...]] = Counter()
    for text in normalized:
        words = _WORD.findall(text.casefold())
        ngrams.update(zip(words, words[1:], words[2:], strict=False))
    ngram_total = sum(ngrams.values())
    advisories = []
    if normalized_duplicate:
        advisories.append(
            "Repeated exact or NFC/newline-normalized text; inspect origins."
        )
    if prefix_rows:
        advisories.append(
            "Shared rendered prefixes; inspect templates and boilerplate."
        )
    if ngram_total >= 20 and len(ngrams) * 4 < ngram_total:
        advisories.append(
            "Repeated 3-word sequences; inspect source and template concentration."
        )
    result = {
        "duplicate_text_rows": duplicate,
        "normalized_duplicate_rows": normalized_duplicate,
        "prefix_32_rows": prefix_rows,
        "ngram_3_distinct": len(ngrams) if ngram_total >= 20 else None,
        "ngram_3_total": ngram_total if ngram_total >= 20 else None,
        "advisories": advisories,
    }
    if chat is not None:
        prompt = [
            next(
                (
                    message["content"]
                    for message in row["messages"]
                    if message["role"] == "user"
                ),
                "",
            )
            for row in chat
        ]
        answers = [
            next(
                (
                    message["content"]
                    for message in reversed(row["messages"])
                    if message["role"] == "assistant"
                ),
                "",
            )
            for row in chat
        ]
        result["prompt_prefix_32_rows"] = sum(
            n - 1
            for n in Counter(text[:32] for text in prompt if len(text) >= 32).values()
            if n > 1
        )
        result["answer_prefix_32_rows"] = sum(
            n - 1
            for n in Counter(text[:32] for text in answers if len(text) >= 32).values()
            if n > 1
        )
    return result


def measure_views(
    root: Path,
    tokenizer: Path | None = None,
    *,
    release_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Count exact rendered tokenizer inputs and classify records independently."""
    root = Path(root)
    model = None
    if tokenizer is not None:
        from tokenizers import Tokenizer

        model = Tokenizer.from_file(str(tokenizer))
    lineage = {row["record_id"]: row for row in _rows(root / "lineage.jsonl")}
    generations = {row["record_id"]: row for row in _rows(root / "generations.jsonl")}
    scenarios = {row["scenario_id"]: row for row in _rows(root / "scenarios.jsonl")}
    views = {}
    training_rows: list[tuple[dict[str, Any], int | None]] = []
    for view in ("lm", "chat"):
        views[view] = {}
        for split in SPLITS:
            payload = root / view / f"{split}.jsonl"
            links = _rows(root / view / f"{split}.lineage.jsonl")
            if view == "lm":
                chat_rows = None
                texts = [row["text"] for row in _rows(payload)]
            else:
                from sparselab.data.conversations import iter_rendered_conversations

                chat_rows = _rows(payload)
                texts = [row.text for row in iter_rendered_conversations(payload)]
            if len(texts) != len(links):
                raise ValueError("view and lineage lengths disagree")
            classified = []
            for link, text in zip(links, texts, strict=True):
                if link["split"] != split or link["record_id"] not in lineage:
                    raise ValueError("view lineage reference mismatch")
                record = lineage[link["record_id"]]
                generation = generations.get(record.get("generation_id"))
                scenario = scenarios.get(record.get("scenario_id"))
                if generation:
                    record = {**record, "generator": generation["generator"]}
                elif scenario:
                    record = {
                        **record,
                        "generator": {
                            "adapter_id": scenario["generator_id"],
                            "generator_version": scenario["generator_version"],
                        },
                    }
                elif record.get("semantic_id"):
                    record = {
                        **record,
                        "generator": {"generator_version": "semantic_chat_v1"},
                    }
                classified.append(
                    (record, len(model.encode(text).ids) if model is not None else None)
                )
            if (
                release_spec is not None
                and split == "train"
                and release_spec[view]["selected"]
                and split in release_spec[view]["training_splits"]
            ):
                training_rows.extend(classified)
            summary = _distribution(classified, counted_tokens=model is not None)
            summary["concentration"] = _concentration(texts, chat_rows)
            source_ids = Counter(
                parent
                for record, _ in classified
                for parent in record.get("parent_document_ids", [])
            )
            summary["concentration"]["parent_document_counts"] = dict(
                sorted(source_ids.items())
            )
            families = Counter(
                family
                for record, _ in classified
                for family in record.get("source_family_ids", [])
            )
            summary["concentration"]["source_family_counts"] = dict(
                sorted(families.items())
            )
            for dimension in ("generator_identity", "template_family"):
                groups = summary["dimensions"].get(dimension, {})
                if (
                    (
                        item_count := max(
                            (item["records"] for item in groups.values()), default=0
                        )
                    )
                    > 1
                    and item_count == summary["records"]
                    and len(groups) == 1
                ):
                    summary["concentration"]["advisories"].append(
                        f"All {view}/{split} records share {dimension}; inspect lineage concentration."
                    )
            if source_ids and max(source_ids.values()) > 1:
                summary["concentration"]["advisories"].append(
                    f"Repeated parent document lineage in {view}/{split}; inspect ancestry."
                )
            if (
                families
                and len(classified) > 1
                and max(families.values()) == len(classified)
            ):
                summary["concentration"]["advisories"].append(
                    f"All {view}/{split} records inherit one source family; inspect source concentration."
                )
            views[view][split] = summary
    result = {
        "views": views,
        "denominators": {
            "records": "exported rows",
            "actual_tokens": "tokens in rendered tokenizer inputs; domain counts may overlap",
        },
    }
    if release_spec is not None:
        result["training_mixture"] = _distribution(
            training_rows, counted_tokens=model is not None
        )
    if tokenizer is not None:
        result["tokenizer_sha256"] = sha256_file(tokenizer)
    return result


def summarize_release(
    root: Path,
    *,
    release_spec: dict[str, Any] | None = None,
    tokenizer: Path | None = None,
) -> dict[str, Any]:
    """Frozen descriptive census, with measured tokens only for a pinned tokenizer."""
    summary = measure_views(root, tokenizer, release_spec=release_spec)
    views = summary["views"]
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for splits in views.values():
        for item in splits.values():
            for dimension, values in item["dimensions"].items():
                for value, count in values.items():
                    totals[dimension][value] += count["records"]
    ledger = {row["record_id"]: row for row in _rows(Path(root) / "lineage.jsonl")}
    training_ids = {
        link["record_id"]
        for view in ("lm", "chat")
        if release_spec is not None
        and release_spec[view]["selected"]
        and "train" in release_spec[view]["training_splits"]
        for link in _rows(Path(root) / view / "train.lineage.jsonl")
    }
    generated = [
        ledger[record_id]
        for record_id in training_ids
        if ledger[record_id]["origin"] not in {"primary_source", "human_authored"}
    ]
    unverified_generated = sum(
        row["verification"]["status"] in {"unverified", "schema_validated"}
        for row in generated
    )
    test_parents = {
        parent
        for view in ("lm", "chat")
        for link in _rows(Path(root) / view / "test.lineage.jsonl")
        for parent in ledger[link["record_id"]]["parent_document_ids"]
    }
    generated_test_overlap = sorted(
        {parent for row in generated for parent in row["parent_document_ids"]}
        & test_parents
    )
    test_ids = {
        link["record_id"]
        for view in ("lm", "chat")
        for link in _rows(Path(root) / view / "test.lineage.jsonl")
    }
    heldout_keys = (
        "parent_document_ids",
        "source_family_ids",
        "generator_world_id",
        "scenario_family_id",
        "template_family_id",
    )
    heldout = {
        key: {
            value
            for record_id in test_ids
            for value in (
                ledger[record_id].get(key, [])
                if key.endswith("_ids")
                else [ledger[record_id].get(key)]
            )
            if value is not None
        }
        for key in heldout_keys
    }
    shared_by_shape = {
        shape_id: {
            key: sorted(
                {
                    value
                    for row in generated
                    if row["shape"]["id"] == shape_id
                    for value in (
                        row.get(key, []) if key.endswith("_ids") else [row.get(key)]
                    )
                    if value is not None
                }
                & heldout[key]
            )
            for key in heldout_keys
        }
        for shape_id in sorted({row["shape"]["id"] for row in generated})
    }
    lexical = _rows(Path(root) / "lexical/candidates.jsonl")
    summary["inventory"] = {
        "lexical_candidate_count": len(lexical),
        "lexical_distinct_terms": len({row["term"] for row in lexical}),
        "view_record_distribution": {
            key: dict(sorted(values.items())) for key, values in sorted(totals.items())
        },
    }
    summary["inventory"]["generated_training_records"] = len(generated)
    summary["inventory"]["unverified_generated_training_records"] = unverified_generated
    summary["inventory"]["generated_test_shared_parent_ids"] = generated_test_overlap
    summary["inventory"]["generated_heldout_lineage_by_shape"] = shared_by_shape
    summary["advisories"] = sorted(
        {
            warning
            for splits in views.values()
            for item in splits.values()
            for warning in item["concentration"]["advisories"]
        }
    )
    if unverified_generated:
        summary["advisories"].append(
            f"{unverified_generated} generated training records have no correctness verification beyond schema validation."
        )
    if generated_test_overlap:
        summary["advisories"].append(
            "Generated training and test examples share source document lineage."
        )
    summary["advisories"].sort()
    return summary


def _verified_capability_results(
    run: Path, evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    """Read only built-in-card results bound to a verified run and checkpoint."""
    from sparselab.evaluation.capabilities import (
        _passes,
        capability_card,
        list_capability_cards,
    )
    from sparselab.training.manifest import read_manifest

    folder = run / "evaluations"
    if not folder.exists():
        return []
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("unsafe capability evidence directory")
    manifest = read_manifest(run / "manifest.json")
    artifacts = {
        item["relative_path"]: item["sha256"] for item in manifest["artifacts"]
    }
    checkpoints = {
        item["digest"]: item for item in evidence["checkpoints"] if item["verified"]
    }
    registered = {item["name"] for item in list_capability_cards()}
    results = []
    for path in sorted(folder.glob("capability-*.json")):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4_000_000:
            raise ValueError("unsafe capability result")
        record = json.loads(path.read_text(encoding="utf-8"))
        digest = record.pop("result_digest", None)
        identity = record.get("identity")
        if (
            not isinstance(digest, str)
            or hashlib.sha256(canonical_json(record)).hexdigest() != digest
            or not isinstance(identity, dict)
            or path.name
            != f"capability-{hashlib.sha256(canonical_json(str(identity.get('checkpoint_sha256') or 'unbound'))).hexdigest()[:12]}-{digest}.json"
            or identity.get("run_id") != evidence["run_id"]
            or identity.get("source_identity_sha256")
            != evidence["source_identity_sha256"]
            or identity.get("tokenizer_sha256") != artifacts.get("tokenizer.json")
            or not isinstance(identity.get("data_sha256"), dict)
            or identity["data_sha256"].get("validation")
            != artifacts.get("data/validation.npy")
            or identity.get("checkpoint_sha256") not in checkpoints
            or identity.get("tokens_seen")
            != checkpoints[identity["checkpoint_sha256"]]["tokens_seen"]
            or record.get("format") != "capability_result_v2"
            or record.get("valid") is not True
        ):
            raise ValueError(f"unverified capability result: {path.name}")
        if record.get("card") not in registered:
            raise ValueError("capability result must use a registered card")
        card = capability_card(record["card"])
        cases = record.get("results")
        if (
            record.get("card_digest") != card.digest
            or record.get("scorer") != card.scorer
            or not isinstance(cases, list)
            or len(cases) != len(card.cases)
            or record.get("case_count") != len(card.cases)
        ):
            raise ValueError("capability result protocol mismatch")
        for actual, expected in zip(cases, card.cases, strict=True):
            if (
                actual.get("id") != expected.identifier
                or actual.get("kind") != expected.kind
                or actual.get("prompt") != expected.prompt
                or actual.get("expected") != expected.expected
                or not isinstance(actual.get("response"), str)
                or actual.get("passed")
                is not _passes(card, actual["response"], expected.expected)
            ):
                raise ValueError("capability case or scorer mismatch")
        passed = sum(case["passed"] for case in cases)
        if record.get("passed") != passed or record.get("score") != passed / len(cases):
            raise ValueError("capability result count mismatch")
        results.append(
            {
                "card": card.name,
                "card_sha256": card.digest,
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "tokens_seen": identity["tokens_seen"],
                "passed": passed,
                "cases": len(cases),
                "score": passed / len(cases),
                "result_sha256": digest,
            }
        )
    return results


def capability_matrix(release: Path, run: Path) -> dict[str, Any] | None:
    """Bind shape composition to observed cards; never attribute causality to a shape."""
    from sparselab.corpus.release import verify_release
    from sparselab.evaluation.evidence import experiment_evidence
    from sparselab.training.manifest import read_manifest

    manifest = verify_release(release)
    identity = manifest["release_id"]
    run = Path(run)
    if not (run / "manifest.json").is_file():
        return None
    evidence = experiment_evidence(run)
    if evidence.get("format") != "experiment_evidence_v2":
        raise ValueError("unsupported model evidence schema")
    if evidence["corpus_release_sha256"] != identity:
        raise ValueError("model evidence belongs to a different corpus release")
    if not evidence["verified_checkpoints"]:
        return None
    cards = _verified_capability_results(run, evidence)
    if not cards:
        return None
    run_manifest = read_manifest(run / "manifest.json")
    measured = measure_views(
        release, release_spec=manifest["build_identity"]["release"]
    )["training_mixture"]
    return {
        "release_id": identity,
        "run_id": evidence["run_id"],
        "model_architecture_sha256": run_manifest["architecture_sha256"],
        "source_snapshots": manifest["snapshots"],
        "training_shape": measured["dimensions"].get("shape", {}),
        "training_origin": measured["dimensions"].get("origin", {}),
        "capabilities": cards,
        "transform_usefulness": [
            {
                "transform_id": stage["id"],
                "state": "UNKNOWN",
                "release_id": identity,
                "run_id": evidence["run_id"],
                "model_checkpoints": sorted(
                    {item["checkpoint_sha256"] for item in cards}
                ),
                "token_budgets": sorted({item["tokens_seen"] for item in cards}),
                "evaluation_suites": sorted({item["card_sha256"] for item in cards}),
                "reason": "An isolated paired transformation comparison is required.",
            }
            for stage in manifest["stages"]
        ],
        "interpretation": "Shape is corpus metadata; card outcomes are checkpoint-bound observations, not a per-shape effect.",
    }


def compare_capability_matrix(
    pairs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    """Compare only releases with common sources, checkpoint budgets and card suites."""
    if len(pairs) < 2:
        raise ValueError("at least two release/run pairs are required")
    rows = [capability_matrix(release, run) for release, run in pairs]
    if any(row is None for row in rows):
        raise ValueError("every comparison run needs verified capability evidence")
    assert all(row is not None for row in rows)
    baseline = rows[0]
    source = baseline["source_snapshots"]
    architecture = baseline["model_architecture_sha256"]
    suites = {
        (card["card_sha256"], card["tokens_seen"]) for card in baseline["capabilities"]
    }
    if any(
        row["source_snapshots"] != source
        or row["model_architecture_sha256"] != architecture
        or {(card["card_sha256"], card["tokens_seen"]) for card in row["capabilities"]}
        != suites
        for row in rows[1:]
    ):
        raise ValueError(
            "comparison requires common source snapshots, model architecture, cards and token budgets"
        )
    return {
        "schema_version": 1,
        "rows": rows,
        "interpretation": "Paired representation/outcome observations only; compare matched model configurations, seeds and evaluation protocols before a causal usefulness judgment.",
    }
