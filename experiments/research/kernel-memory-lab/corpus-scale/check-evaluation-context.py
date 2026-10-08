"""Offline Card 03 item lineage, prompt-fit and literal-leakage screen.

This is a mechanical screen. A passing receipt does not review item semantics.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict, deque
from pathlib import Path

from sparselab.corpus.release import verify_release
from sparselab.data.tokenizer import load_tokenizer
from sparselab.evaluation.kml_card03_items import _chunks, _families, _rows
from sparselab.training.manifest import canonical_json, sha256_file


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def token_count(tokenizer, value: str) -> int:
    return len(tokenizer.encode(value, add_special_tokens=False).ids)


def train_overlap_flags(items: list[dict], train_text: str) -> dict[int, set[str]]:
    """Flag partial eight-word overlap with train text for human leakage review."""
    queries: dict[tuple[str, ...], set[tuple[int, str]]] = defaultdict(set)
    for index, item in enumerate(items):
        for label, phrase in (
            ("question", item["question"]),
            *(("required_claim", value) for value in item["required_claims"]),
            *(("acceptable_paraphrase", value) for value in item["acceptable_paraphrases"]),
        ):
            words = re.findall(r"\w+", normalized(phrase))
            for start in range(len(words) - 7):
                queries[tuple(words[start : start + 8])].add((index, label))
    hits: dict[int, set[str]] = defaultdict(set)
    if not queries:
        return hits
    window: deque[str] = deque(maxlen=8)
    for match in re.finditer(r"\w+", train_text):
        window.append(match.group())
        if len(window) == 8:
            for index, label in queries.get(tuple(window), ()):
                hits[index].add(f"train_8gram_overlap:{label}")
    return hits


def prompt(question: str, selected: list[str], chunks: dict, docs: dict) -> tuple[str, str]:
    if selected:
        evidence = "\n".join(
            f"[{index}] {docs[chunks[chunk_id]['document_id']]['text'][chunks[chunk_id]['start']:chunks[chunk_id]['end']]}"
            for index, chunk_id in enumerate(selected, 1)
        )
        return f"Question: {question}\nEvidence:\n{evidence}\nAnswer:", evidence
    return f"Question: {question}\nEvidence:\n(none)\nAnswer:", "(none)"


def measure(draft: dict, release: Path, family_inventory: Path, tokenizer_path: Path) -> dict:
    release_manifest = verify_release(release)
    if draft.get("release_id") != release_manifest["release_id"]:
        raise ValueError("draft release ID differs from cold-verified release")
    if draft.get("family_inventory_sha256") != sha256_file(family_inventory):
        raise ValueError("draft family inventory differs")
    if draft.get("evidence_token_budget") != 512 or draft.get("decoder", {}).get(
        "max_output_tokens"
    ) != 128:
        raise ValueError("draft differs from bound 512/128 context policy")
    docs = {
        row["document_id"]: row
        for row in _rows(release / "documents.jsonl")
        if row["drop_reason"] is None
    }
    families = _families(family_inventory, docs)
    chunks = _chunks(draft["chunks"], docs, families)
    tokenizer = load_tokenizer(tokenizer_path)
    train_text = normalized(
        " ".join(row["text"] for row in _rows(release / "lm/train.jsonl"))
    )
    overlap_flags = train_overlap_flags(draft["items"], train_text)
    findings: list[dict] = []
    for index, item in enumerate(draft["items"]):
        errors: list[str] = []
        flags: list[str] = sorted(overlap_flags.get(index, ()))
        question = item["question"]
        question_tokens = token_count(tokenizer, question)
        if question_tokens > 64:
            errors.append("question_exceeds_64_tokens")
        parents = item["parent_document_ids"]
        if not parents or any(
            doc_id not in families or families[doc_id]["split"] != "test"
            for doc_id in parents
        ):
            errors.append("parent_not_heldout_test")
        for label, phrase in (
            ("question", question),
            *(("required_claim", value) for value in item["required_claims"]),
            *(("acceptable_paraphrase", value) for value in item["acceptable_paraphrases"]),
        ):
            searchable = normalized(phrase)
            if len(searchable) >= 32 and len(searchable.split()) >= 6 and searchable in train_text:
                flags.append(f"literal_train_overlap:{label}")
        for phrase in (*item["required_claims"], *item["acceptable_paraphrases"]):
            if token_count(tokenizer, phrase) > 128:
                errors.append("gold_answer_exceeds_128_tokens")
                break
        conditions: dict[str, dict[str, int]] = {}
        if item["suite"] == "closed_book":
            rendered = f"Question: {question}\nAnswer:"
            conditions["closed_book"] = {
                "evidence_tokens": 0,
                "prompt_tokens": token_count(tokenizer, rendered) + 1,
            }
        else:
            controls = item["controls"]
            if (
                item["answerability"] == "answerable"
                and controls["plausible_wrong"] == controls["shuffled_absent"]
            ):
                flags.append("wrong_and_absent_controls_identical")
            selected_conditions = (
                {name: selected for name, selected in controls.items() if selected is not None}
                if item["answerability"] == "answerable"
                else {"unanswerable": item["support_chunk_ids"]}
            )
            for name, selected in selected_conditions.items():
                if any(chunk_id not in chunks for chunk_id in selected):
                    errors.append(f"unknown_chunk:{name}")
                    continue
                rendered, evidence = prompt(question, selected, chunks, docs)
                conditions[name] = {
                    "evidence_tokens": token_count(tokenizer, evidence),
                    "prompt_tokens": token_count(tokenizer, rendered) + 1,
                }
                if name in {"plausible_wrong", "shuffled_absent"}:
                    content = normalized(evidence)
                    if any(
                        len(normalized(claim)) >= 8 and normalized(claim) in content
                        for claim in item["required_claims"]
                    ):
                        flags.append(f"control_literal_answer_overlap:{name}")
        for name, counts in conditions.items():
            if counts["evidence_tokens"] > 512:
                errors.append(f"evidence_exceeds_512_tokens:{name}")
            if counts["prompt_tokens"] + 128 > 1024:
                errors.append(f"prompt_plus_generation_exceeds_1024_tokens:{name}")
        findings.append(
            {
                "id": item["id"],
                "item_sha256": item["content_sha256"],
                "question_tokens": question_tokens,
                "conditions": conditions,
                "hard_errors": sorted(set(errors)),
                "review_flags": sorted(set(flags)),
            }
        )
    return {
        "schema_version": 1,
        "status": "MECHANICAL_ONLY",
        "release_id": release_manifest["release_id"],
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "family_inventory_sha256": sha256_file(family_inventory),
        "item_count": len(findings),
        "hard_error_count": sum(bool(row["hard_errors"]) for row in findings),
        "review_flag_count": sum(bool(row["review_flags"]) for row in findings),
        "items": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    result = measure(draft, args.release, args.families, args.tokenizer)
    result["draft_sha256"] = sha256_file(args.draft)
    with args.output.open("xb") as stream:
        stream.write(canonical_json(result) + b"\n")
    print(
        json.dumps(
            {
                "item_count": result["item_count"],
                "hard_error_count": result["hard_error_count"],
                "review_flag_count": result["review_flag_count"],
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    if result["hard_error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
