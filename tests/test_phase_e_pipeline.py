from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparselab.cli.main import build_parser
from sparselab.data.conversations import iter_conversations
from sparselab.evaluation.capabilities import capability_card, phase_e_behavior_suite


def _dispatch(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


def test_behavior_suite_keeps_language_model_and_task_scores_separate() -> None:
    suite = phase_e_behavior_suite()
    categories = {item["id"]: item for item in suite["categories"]}

    assert {
        "language_model",
        "lexical_recall",
        "factual_recall",
        "paraphrase",
        "api_behavior",
        "math",
        "application",
        "composition",
        "long_context",
        "override",
        "instruction_following",
        "conversation",
        "stale_source",
        "conflicting_source",
        "missing_source",
    } <= set(categories)
    assert categories["language_model"]["cards"] == []
    assert categories["lexical_recall"]["cards"] == ["lexical-recall-v1"]
    assert categories["conversation"]["cards"] == ["conversation-followup-v1"]
    lexical = capability_card("lexical-recall-v1")
    paraphrase = capability_card("paraphrase-recall-v1")
    conversation = capability_card("conversation-followup-v1")
    assert len(lexical.cases) == len(paraphrase.cases) == 12
    assert len(conversation.cases) == 6
    assert all(
        "Assistant: Understood.\n\nUser:" not in case.prompt
        for case in paraphrase.cases
    )
    assert all(
        "Assistant: Understood.\n\nUser:" in case.prompt for case in conversation.cases
    )
    assert "not combined" in suite["claim_boundary"]
    instruction_case = capability_card("instruction-over-memory-v1").cases[0]
    assert instruction_case.expected == "UNKNOWN"
    assert "capital of France" in instruction_case.prompt
    assert "no entry for the capital" in instruction_case.prompt
    for category in suite["categories"]:
        for name in category.get("cards", []):
            assert name in suite["card_digests"]
            assert capability_card(name).digest == suite["card_digests"][name]


def test_task_builder_emits_train_validation_and_frozen_test_card(
    tmp_path: Path,
) -> None:
    output = tmp_path / "math-task"

    _dispatch(
        [
            "research",
            "tasks",
            "build",
            "math-identities",
            "--output",
            str(output),
            "--seed",
            "17",
        ]
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_paths"] == ["train.jsonl"]
    assert manifest["validation_paths"] == ["validation.jsonl"]
    assert manifest["test_answers_used_for_training"] is False
    assert manifest["answer_method"] == "integer arithmetic over generated operands"
    assert len(list(iter_conversations(output / "train.jsonl"))) == 8
    assert len(list(iter_conversations(output / "validation.jsonl"))) == 8
    card_path = output / "cards" / "math-identity-novel-operands-v1.json"
    card = capability_card(str(card_path))
    test_cases = json.loads((output / "test-cases.json").read_text(encoding="utf-8"))
    assert len(card.cases) == len(test_cases) == 8
    assert {case.kind for case in card.cases} == {"math"}
    assert card.digest == json.loads(card_path.read_text(encoding="utf-8"))["digest"]


def test_wikidata_task_cli_builds_pinned_cards_and_licenses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.data import phase_e_tasks

    def fake_fetch(qid: str, revision: int) -> dict[str, object]:
        year = {
            "Q42": 1901,
            "Q937": 1902,
            "Q7259": 1903,
            "Q7186": 1904,
        }[qid]
        return {
            "entities": {
                qid: {
                    "id": qid,
                    "lastrevid": revision,
                    "labels": {"en": {"language": "en", "value": f"Entity {qid}"}},
                    "claims": {
                        "P569": [
                            {
                                "mainsnak": {
                                    "snaktype": "value",
                                    "datavalue": {
                                        "type": "time",
                                        "value": {
                                            "time": f"+{year}-02-03T00:00:00Z",
                                            "precision": 11,
                                        },
                                    },
                                }
                            }
                        ],
                        "P570": [],
                    },
                }
            }
        }

    monkeypatch.setattr(phase_e_tasks, "_fetch_entity", fake_fetch)
    output = tmp_path / "wikidata"
    _dispatch(
        [
            "research",
            "tasks",
            "build",
            "wikidata-mini",
            "--output",
            str(output),
        ]
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["license"] == "MIT prompts/format; CC0-1.0 Wikidata facts"
    assert manifest["source_data_license"] == "CC0-1.0"
    assert manifest["generated_artifact_license"] == "MIT"
    assert manifest["test_answers_used_for_training"] is False
    assert "source/manifest.json" in {item["path"] for item in manifest["files"]}
    train_text = (output / "source" / "train.jsonl").read_text(encoding="utf-8")
    assert "1904-02-03" not in train_text
    factual = capability_card(
        str(output / "cards" / "wikidata-mini-factual-recall-v1.json")
    )
    assert [case.expected for case in factual.cases] == ["1904-02-03"]


def test_review_cli_keeps_reveal_map_and_judgments_separate(tmp_path: Path) -> None:
    records = [
        {
            "case_id": case_id,
            "prompt": prompt,
            "condition": condition,
            "response": response,
            "source_id": source_id,
        }
        for case_id, prompt, left, right in (
            (
                "case-alpha",
                "Which code belongs to the first location?",
                "amber",
                "cobalt",
            ),
            (
                "case-beta",
                "Which code belongs to the second location?",
                "ivory",
                "jade",
            ),
        )
        for condition, response, source_id in (
            ("baseline", left, "checkpoint-" + "a" * 64),
            ("variant", right, "checkpoint-" + "b" * 64),
        )
    ]
    records_path = tmp_path / "records.json"
    criteria_path = tmp_path / "criteria.json"
    bundle_path = tmp_path / "review-bundle.json"
    reveal_path = tmp_path / "private-reveal.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    criteria_path.write_text(
        json.dumps(
            [
                {
                    "id": "grounding",
                    "description": "Rate whether the response follows the evidence.",
                    "minimum": 1,
                    "maximum": 5,
                }
            ]
        ),
        encoding="utf-8",
    )

    _dispatch(
        [
            "review",
            "bundle",
            "--records",
            str(records_path),
            "--criteria",
            str(criteria_path),
            "--seed",
            "23",
            "--bundle",
            str(bundle_path),
            "--reveal-map",
            str(reveal_path),
        ]
    )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    reveal_map = json.loads(reveal_path.read_text(encoding="utf-8"))
    blind_text = json.dumps(bundle, sort_keys=True)
    assert "baseline" not in blind_text and "variant" not in blind_text
    assert "checkpoint-" not in blind_text and "case-alpha" not in blind_text
    assert reveal_map["reveal_map_digest"] == bundle["reveal_map_digest"]
    assert len(bundle["cases"]) == 2

    judgments_path = tmp_path / "human-judgments.json"
    judgments_path.write_text(
        json.dumps(
            {
                "format": "sparselab-human-review-judgments",
                "version": 1,
                "bundle_digest": bundle["bundle_digest"],
                "criteria_digest": bundle["criteria_digest"],
                "judgments": [
                    {
                        "blind_case_id": case["blind_case_id"],
                        "ratings": {"grounding": {"A": 4, "B": 4}},
                    }
                    for case in bundle["cases"]
                ],
            }
        ),
        encoding="utf-8",
    )
    normalized_path = tmp_path / "validated-judgments.json"
    _dispatch(
        [
            "review",
            "validate",
            "--bundle",
            str(bundle_path),
            "--judgments",
            str(judgments_path),
            "--output",
            str(normalized_path),
        ]
    )

    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    assert normalized["bundle_digest"] == bundle["bundle_digest"]
    assert len(normalized["judgments"]) == 2
    assert "checkpoint-" not in json.dumps(normalized)


def test_review_cli_pairs_capability_results_without_revealing_identity(
    tmp_path: Path,
) -> None:
    prompt = "User: Return the code stated by the source.\nAssistant:"

    def result(checkpoint: str, response: str) -> dict[str, object]:
        payload = {
            "format": "capability_result_v2",
            "card_digest": "a" * 64,
            "evaluation_source_sha256": "b" * 64,
            "valid": True,
            "identity": {"checkpoint_sha256": checkpoint * 64},
            "results": [
                {
                    "id": "facts-1",
                    "prompt": prompt,
                    "expected": "hidden-ground-truth",
                    "response": response,
                    "passed": False,
                }
            ],
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return {
            **payload,
            "result_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }

    base_path, variant_path = tmp_path / "base.json", tmp_path / "variant.json"
    base_path.write_text(json.dumps(result("c", "response-a")), encoding="utf-8")
    variant_path.write_text(json.dumps(result("d", "response-b")), encoding="utf-8")
    criteria_path = tmp_path / "criteria.json"
    criteria_path.write_text(
        json.dumps(
            [
                {
                    "id": "accuracy",
                    "description": "Rate factual accuracy.",
                    "minimum": 1,
                    "maximum": 5,
                }
            ]
        ),
        encoding="utf-8",
    )
    bundle_path, reveal_path = tmp_path / "blind.json", tmp_path / "reveal.json"

    _dispatch(
        [
            "review",
            "bundle",
            "--base-result",
            str(base_path),
            "--variant-result",
            str(variant_path),
            "--criteria",
            str(criteria_path),
            "--seed",
            "31",
            "--bundle",
            str(bundle_path),
            "--reveal-map",
            str(reveal_path),
        ]
    )

    bundle_text = bundle_path.read_text(encoding="utf-8")
    reveal_text = reveal_path.read_text(encoding="utf-8")
    assert "hidden-ground-truth" not in bundle_text
    assert "baseline" not in bundle_text and "variant" not in bundle_text
    assert "c" * 64 not in bundle_text and "d" * 64 not in bundle_text
    assert "baseline" in reveal_text and "variant" in reveal_text
