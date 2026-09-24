"""Regression coverage for safe Phase E task construction."""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path

import pytest

import sparselab.data.phase_e_tasks as tasks


def _operands(cases: tuple[tasks.TaskCase, ...]) -> set[int]:
    values: set[int] = set()
    for case in cases:
        match = re.search(r"\((\d+) \+ (\d+)\)\^2", case.prompt)
        assert match is not None
        values.update(map(int, match.groups()))
    return values


def test_math_splits_are_deterministic_and_operand_disjoint() -> None:
    train = tasks.math_identity_training_cases(17)
    validation = tasks.math_identity_validation_cases(17)
    test = tasks.math_identity_test_cases(17)

    assert train == tasks.math_identity_training_cases(17)
    assert all(case.kind == "math" for case in test)
    assert not (_operands(train) & _operands(validation))
    assert not (_operands(train) & _operands(test))
    assert not (_operands(validation) & _operands(test))
    train_documents = "\n".join(tasks.math_identity_training_documents(17))
    rendered_operands = {
        int(operand)
        for match in re.finditer(r"\((\d+) \+ (\d+)\)\^2", train_documents)
        for operand in match.groups()
    }
    assert rendered_operands == _operands(train)
    assert not (rendered_operands & _operands(test))
    assert tasks.math_identity_training_documents() == tuple(
        f"User: {case.prompt}\nAssistant: {case.expected}" for case in train
    )


def test_python_cases_are_structured_disjoint_and_use_direct_stdlib_results() -> None:
    train = tasks.python_stdlib_training_cases()
    validation = tasks.python_stdlib_validation_cases()
    test = tasks.python_stdlib_test_cases()
    rendered = {
        "train": "\n".join(tasks.python_stdlib_training_documents()),
        "validation": "\n".join(tasks.python_stdlib_validation_documents()),
    }

    assert train == tasks.python_stdlib_training_cases()
    assert all(case.kind == "api" for case in test)
    assert all(case.prompt not in rendered["validation"] for case in train)
    assert all(case.prompt not in rendered["train"] for case in test)
    train_prompts = {case.prompt for case in train}
    validation_prompts = {case.prompt for case in validation}
    test_prompts = {case.prompt for case in test}
    assert not (train_prompts & validation_prompts)
    assert not (train_prompts & test_prompts)
    assert not (validation_prompts & test_prompts)
    for case in (*train, *validation, *test):
        if "html.escape" in case.prompt:
            value = re.search(r"html\.escape\((.+), quote=True\)", case.prompt).group(1)
            assert case.expected == tasks.html.escape(
                ast.literal_eval(value), quote=True
            )
        elif "urllib.parse.quote" in case.prompt:
            value = re.search(
                r"urllib\.parse\.quote\((.+), safe=''\)", case.prompt
            ).group(1)
            assert case.expected == tasks.urllib.parse.quote(
                ast.literal_eval(value), safe=""
            )
        else:
            value = re.search(r"posixpath\.normpath\((.+)\)", case.prompt).group(1)
            assert case.expected == tasks.posixpath.normpath(ast.literal_eval(value))


def _claim(property_id: str, date: str) -> dict[str, object]:
    return {
        "mainsnak": {
            "snaktype": "value",
            "datavalue": {"type": "time", "value": {"time": date, "precision": 11}},
        },
        "property": property_id,
    }


def _pinned_payload(qid: str, revision: int) -> dict[str, object]:
    return {
        "entities": {
            qid: {
                "id": qid,
                "lastrevid": revision,
                "labels": {"en": {"language": "en", "value": f"Person {qid}"}},
                "claims": {
                    "P569": [_claim("P569", "+1901-02-03T00:00:00Z")],
                    "P570": [_claim("P570", "+1984-05-06T00:00:00Z")],
                },
            }
        }
    }


def test_wikidata_source_manifest_is_answer_free_and_pinned() -> None:
    spec = tasks.wikidata_mini_spec()
    assert spec["license"] == "CC0-1.0"
    assert [
        (item["id"], item["revision"], item["split"]) for item in spec["entities"]
    ] == [
        ("Q42", 2547848680, "train"),
        ("Q937", 2548339644, "train"),
        ("Q7259", 2548536803, "validation"),
        ("Q7186", 2548356481, "test"),
    ]
    raw = Path(tasks._resource_manifest_path()).read_text(encoding="utf-8")
    assert "labels" not in raw
    assert "claims" not in raw
    assert "birth" not in raw


def test_wikidata_build_uses_injected_pinned_json_and_loader_verifies_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_fetch(qid: str, revision: int) -> dict[str, object]:
        return _pinned_payload(qid, revision)

    monkeypatch.setattr(tasks, "_fetch_entity", fake_fetch)
    output = tmp_path / "mini"
    assert tasks.build_wikidata_mini(output) == output
    cases = tasks.load_wikidata_mini_cases(output)
    assert {case.kind for case in cases} == {"factual_recall", "paraphrase"}
    assert {case.expected for case in cases} == {"1901-02-03", "1984-05-06"}
    assert (output / "train.jsonl").read_text(encoding="utf-8").count("\n") == 2
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_data_license"] == "CC0-1.0"
    assert manifest["generated_artifact_license"] == "MIT"
    assert [source["revision"] for source in manifest["sources"]] == [
        2547848680,
        2548339644,
        2548536803,
        2548356481,
    ]

    with (output / "test.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        tasks.load_wikidata_mini_cases(output)


def test_wikidata_build_rejects_unexpected_revision_without_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_revision(qid: str, revision: int) -> dict[str, object]:
        return _pinned_payload(qid, revision + 1)

    monkeypatch.setattr(tasks, "_fetch_entity", unexpected_revision)
    output = tmp_path / "mini"
    with pytest.raises(ValueError, match="revision mismatch"):
        tasks.build_wikidata_mini(output)
    assert not output.exists()


def test_wikidata_parser_rejects_injected_wrong_entity() -> None:
    with pytest.raises(ValueError, match="exactly requested entity"):
        tasks._extract_entity(_pinned_payload("Q43", 1), "Q42", 1)


def test_task_cases_are_immutable_and_reject_unsafe_identifiers() -> None:
    case = tasks.TaskCase("safe-case", "Prompt", "Answer", "application")
    with pytest.raises(dataclasses.FrozenInstanceError):
        case.expected = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="unsafe task identifier"):
        tasks.TaskCase("../unsafe", "Prompt", "Answer", "math")


def test_wikidata_date_claim_requires_full_valid_positive_date() -> None:
    for raw, precision in (
        ("+1901-02-03T00:00:00Z", 10),
        ("-1901-02-03T00:00:00Z", 11),
        ("+1901-04-31T00:00:00Z", 11),
    ):
        claim = _claim("P569", raw)
        claim["mainsnak"]["datavalue"]["value"]["precision"] = precision
        entity = {"claims": {"P569": [claim]}}
        assert tasks._date_claim(entity, "P569", "Q42") is None


def test_wikidata_source_records_language_neutral_label_when_english_is_absent() -> (
    None
):
    payload = _pinned_payload("Q937", 2548339644)
    entity = payload["entities"]["Q937"]
    entity["labels"] = {"mul": {"language": "mul", "value": "Albert Einstein"}}

    fact = tasks._extract_entity(payload, "Q937", 2548339644)

    assert fact["label"] == "Albert Einstein"
    assert fact["label_language"] == "mul"
