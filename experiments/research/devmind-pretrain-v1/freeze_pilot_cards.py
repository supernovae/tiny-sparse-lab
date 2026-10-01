"""Freeze diagnostic PILOT cards only; never qualify these for the 200-case promotion gate.

Usage: python experiments/research/devmind-pretrain-v1/freeze_pilot_cards.py RELEASE OUTPUT
The release must contain v2 synthetic worlds with test seeds 900–999. The
independent ≥200-case promotion freezer is freeze_cards.py (unchanged).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import freeze_cards as strict

from sparselab.corpus.release import verify_release
from sparselab.evaluation.capabilities import (
    CapabilityCase,
    CapabilityCaseLineage,
    capability_card_payload,
    check_capability_family_exclusion,
    load_capability_card,
)
from sparselab.training.manifest import canonical_json

PILOT_VERSION = 1
V2_GENERATORS = frozenset(
    {
        "filesystem_judgment_v2",
        "platform_fault_v2",
        "deployment_change_v2",
        "code_test_workflow_v2",
    }
)


def causal_hash(item: dict) -> str:
    world = item["world_state"]
    if (
        not isinstance(world, dict)
        or not world
        or item["oracle_receipt"]["world_facts"] != world
    ):
        raise ValueError("world facts disagree with oracle receipt")
    if "fixture_id" in world or "world_seed" in world:
        raise ValueError("identity cannot count as causal world facts")
    return strict.digest(world)


def check_worlds(scenarios: list[dict]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for row in scenarios:
        if row["generator_id"] not in V2_GENERATORS or row["generator_version"] != "2":
            raise ValueError(
                "pilot release requires v2 scenario generators exclusively"
            )
        seed = row["world_seed"]
        expected = "train" if seed < 800 else "validation" if seed < 900 else "test"
        if seed >= 1000 or row["split"] != expected:
            raise ValueError("v2 world seed crosses its split partition")
        identity = (row["generator_id"], causal_hash(row))
        if identity in seen:
            raise ValueError(
                f"repeated causal world facts: {seen[identity]} and {row['generator_world_id']}"
            )
        seen[identity] = row["generator_world_id"]


def check_release_lineage(
    documents: list[dict], scenarios: list[dict]
) -> dict[str, list[dict]]:
    """Reject identical source parents, text, worlds or templates across splits."""
    check_worlds(scenarios)
    seen: dict[tuple[str, object], str] = {}
    ledger: dict[str, list[dict]] = {"train": [], "validation": []}
    for doc in documents:
        split = doc["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError("unknown source split")
        for key in (
            ("family", doc["source_family"]),
            (
                "parent",
                doc["source_id"],
                doc["source_revision"],
                doc["source_location"].split("#", 1)[0],
            ),
            ("content", doc["content_sha256"]),
        ):
            previous = seen.setdefault(key, split)
            if previous != split:
                raise ValueError("source parent or content crosses splits")
        if split != "test":
            ledger[split].append(
                {
                    "split": split,
                    "source_document_family": doc["source_family"],
                    "world_id": None,
                    "template_family": f"source_{split}_v1",
                    "parent_content_hashes": list(strict.parent_hashes(doc)),
                }
            )
    for row in scenarios:
        split = row["split"]
        for key in (
            ("world_id", row["generator_world_id"]),
            ("world", row["generator_id"], causal_hash(row)),
            ("template", row["template_family_id"]),
        ):
            previous = seen.setdefault(key, split)
            if previous != split:
                raise ValueError("scenario world or template crosses splits")
        if split != "test":
            ledger[split].append(
                {
                    "split": split,
                    "source_document_family": None,
                    "world_id": row["generator_world_id"],
                    "template_family": row["template_family_id"],
                    "parent_content_hashes": [causal_hash(row)],
                }
            )
    return ledger


def pilot_source_cases(
    name: str, generated: list[dict], documents: dict[str, dict]
) -> list[CapabilityCase]:
    if name == "code_config_understanding":
        allowed = {"code", "python", "go", "rust", "shell", "yaml", "json", "toml"}
    elif name == "technical_comprehension":
        allowed = {"prose"}
    else:
        allowed = None
    matching = [
        row
        for row in generated
        if row.get("validation_status") == "source_entailed"
        and (
            allowed is None
            or documents[row["evidence"]["document_id"]]["document_kind"] in allowed
        )
    ]
    return strict.source_cases(name, matching, documents)


def pilot_scenario_cases(name: str, scenarios: list[dict]) -> list[CapabilityCase]:
    cases = []
    generator_by_card = {
        "abstention_unknown": "filesystem_judgment_v2",
        "operational_risk": "filesystem_judgment_v2",
        "troubleshooting_cause": "platform_fault_v2",
        "next_diagnostic": "platform_fault_v2",
        "inert_tool_protocol": "platform_fault_v2",
        "verification": "deployment_change_v2",
        "state_tracking": "code_test_workflow_v2",
        "stop_after_success": "code_test_workflow_v2",
    }
    for row in sorted(scenarios, key=lambda item: item["generator_world_id"]):
        if row["split"] != "test" or row["generator_id"] != generator_by_card[name]:
            continue
        world, receipt = row["world_state"], row["oracle_receipt"]
        facts = json.dumps(world, sort_keys=True)
        if name in {"abstention_unknown", "operational_risk", "verification"}:
            prompt = f"Fictional {row['generator_id']} world: {facts}\nChoose proceed, inspect, ask, or stop. Output one word."
            answer = receipt["judgment"]
        elif name in {"troubleshooting_cause", "inert_tool_protocol"}:
            prompt = f"INERT platform observation: {receipt['evidence']}. Declared world: {facts}\nName only the cause; do not execute tools."
            answer = world["cause"]
        elif name == "next_diagnostic":
            prompt = f"Fictional platform observations: {facts}\nState only the next bounded diagnostic."
            answer = receipt["next_diagnostic"]
        elif name == "state_tracking":
            prompt = f"Disposable code/test world: {facts}\nState the current condition only."
            answer = world["condition"]
        else:
            prompt = f"Disposable code/test world: {facts}\nIs another intervention needed? Output the oracle judgment only."
            answer = receipt["judgment"]
        cases.append(
            CapabilityCase(
                f"{name}-{row['generator_world_id']}",
                prompt,
                answer,
                "custom",
                lineage=CapabilityCaseLineage(
                    "test",
                    None,
                    row["generator_world_id"],
                    row["template_family_id"],
                    (causal_hash(row),),
                ),
            )
        )
    return cases


def freeze(release: Path, output: Path) -> dict:
    verified = verify_release(release)
    if output.exists():
        raise ValueError("sealed pilot card output already exists")
    manifest_bytes = (release / "manifest.json").read_bytes()
    if (
        verified["release_id"] != release.name
        or json.loads(manifest_bytes)["release_id"] != release.name
    ):
        raise ValueError("release path does not match frozen manifest identity")
    documents = list(strict.rows(release / "documents.jsonl"))
    by_id = {doc["document_id"]: doc for doc in documents}
    if len(by_id) != len(documents):
        raise ValueError("duplicate document identities")
    scenarios = list(strict.rows(release / "scenarios.jsonl"))
    ledger = check_release_lineage(documents, scenarios)
    generated = list(strict.rows(release / "generations.jsonl"))
    cards = []
    report = {}
    case_parents: dict[str, set[str]] = {}
    for name in strict.SPECIFICATIONS:
        cases = (
            pilot_source_cases(name, generated, by_id)
            if name in strict.SOURCE_CARDS
            else pilot_scenario_cases(name, scenarios)
        )
        parents = [case.lineage.parent_content_hashes[-1] for case in cases]
        if len(set(parents)) != len(parents):
            raise ValueError(f"repeated independent parent/world facts in {name}")
        case_parents[name] = set(parents)
        count = len(cases)
        report[name] = {
            "effective_n": count,
            "underpowered_for_promotion": count < strict.MIN_CASES,
            "shared_world_with": {},
            "shared_source_parent_with": {},
        }
        if count:
            card = replace(
                strict.make_card(name, cases),
                limitations=(
                    f"PILOT diagnostic only: {count} distinct parents/world facts for this "
                    "card; correlated cards share worlds or source parents as recorded "
                    "in receipt.json. Not promotion-eligible: the independent ≥200-case "
                    "gate remains in freeze_cards.py. "
                    + (
                        "Literal-span proxy, not technical reasoning."
                        if name in strict.SOURCE_CARDS
                        else "Simulated causal facts, not an operational run."
                    )
                ),
            )
            check_capability_family_exclusion(card, ledger)
            cards.append(card)
    for name, info in report.items():
        for other, identities in case_parents.items():
            shared = len(case_parents[name] & identities)
            if not shared or name == other:
                continue
            field = (
                "shared_source_parent_with"
                if name in strict.SOURCE_CARDS
                else "shared_world_with"
            )
            info[field][other] = shared
    output.mkdir(parents=True)
    receipt = {
        "schema_version": 1,
        "pilot_freezer_version": PILOT_VERSION,
        "release_id": verified["release_id"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "test_worlds": {
            generator: {
                "distinct_causal_states": sum(
                    row["generator_id"] == generator and row["split"] == "test"
                    for row in scenarios
                ),
                "fewer_than_100": sum(
                    row["generator_id"] == generator and row["split"] == "test"
                    for row in scenarios
                )
                < 100,
            }
            for generator in sorted(V2_GENERATORS)
        },
        "promotion_eligible": False,
        "promotion_gate": f"not evaluated; promotion requires >= {strict.MIN_CASES} independently sourced cases for each card",
        "cards": report,
    }
    for card in cards:
        path = output / f"{card.name}.json"
        path.write_bytes(canonical_json(capability_card_payload(card)) + b"\n")
        check_capability_family_exclusion(load_capability_card(path), ledger)
        receipt["cards"][card.name]["sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (output / "receipt.json").write_bytes(canonical_json(receipt) + b"\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args.release, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
