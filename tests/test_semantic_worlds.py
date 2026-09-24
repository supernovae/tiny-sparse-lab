"""Contracts for deterministic structured toy-world semantic retrieval."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sparselab.data.toy_worlds import (
    MemoryCondition,
    QueryCase,
    StructuredKey,
    World,
    evaluate_case,
    knowledge_swap_ratio,
    materialize_toy_worlds,
    validate_toy_worlds,
)
from sparselab.engram.packs import load_pack
from sparselab.engram.semantic import SemanticRetriever


def _case(benchmark, suffix: str):
    return next(case for case in benchmark.cases() if case.id.endswith(suffix))


def test_world_partitions_assignments_and_producer_inputs_are_disjoint(
    tmp_path,
) -> None:
    benchmark = materialize_toy_worlds(tmp_path / "worlds", seed=19)

    train = [world for world in benchmark.worlds if world.partition == "train"]
    heldout = [world for world in benchmark.worlds if world.partition == "heldout"]
    assert {world.id for world in train}.isdisjoint(world.id for world in heldout)
    assert {
        case.template_name for world in train for case in world.query_cases
    }.isdisjoint(case.template_name for world in heldout for case in world.query_cases)
    assert len({world.assignment for world in benchmark.worlds}) == len(
        benchmark.worlds
    )
    assert benchmark.producer_ids.isdisjoint(benchmark.query_ids)
    for by_condition in benchmark.packs.values():
        for pack in by_condition.values():
            if pack.path is not None:
                loaded = load_pack(pack.path, expected_pack_id=pack.pack_id)
                assert loaded.semantic_metadata is not None
                assert (
                    loaded.semantic_metadata.key_encoder
                    == benchmark.key_encoder.identity
                )
    audit = benchmark.audit()
    heldout_producers = set(audit.heldout_producer_record_ids)
    assert audit.benchmark_id == benchmark.identity
    assert set(audit.train_producer_record_ids).isdisjoint(heldout_producers)
    assert set(audit.query_case_ids) == benchmark.query_ids
    assert set(audit.train_world_ids).isdisjoint(audit.heldout_world_ids)
    assert set(audit.train_templates).isdisjoint(audit.heldout_templates)
    for scope, packs in benchmark.packs.items():
        if scope.startswith("train-world-"):
            for pack in packs.values():
                assert set(pack.record_ids).isdisjoint(heldout_producers)


def test_materialization_is_deterministic_for_a_seed(tmp_path) -> None:
    first = materialize_toy_worlds(tmp_path / "first", seed=41)
    second = materialize_toy_worlds(tmp_path / "second", seed=41)

    assert first.identity == second.identity
    assert first.worlds == second.worlds
    assert {
        scope: {condition: pack.pack_id for condition, pack in by_condition.items()}
        for scope, by_condition in first.packs.items()
    } == {
        scope: {condition: pack.pack_id for condition, pack in by_condition.items()}
        for scope, by_condition in second.packs.items()
    }


def test_exact_one_two_and_three_hop_cases_have_no_shortcuts(tmp_path) -> None:
    benchmark = materialize_toy_worlds(tmp_path / "worlds", seed=23)
    world = benchmark.worlds[0]
    edges = {
        (fact.key.subject, fact.key.relation, fact.value)
        for fact in world.producer_facts
    }
    for suffix, hops in (("one-hop", 1), ("two-hop", 2), ("three-hop", 3)):
        case = _case(benchmark, suffix)
        result = evaluate_case(benchmark, case, MemoryCondition.CORRECT)
        assert result.status == "retrieved"
        assert result.value == case.expected_value
        assert len(result.trace_record_ids) == hops
        assert result.trace_record_ids == tuple(
            fact.id for fact in world.producer_facts[:hops]
        )
        if hops > 1:
            assert (
                case.keys[0].subject,
                case.keys[0].relation,
                case.expected_value,
            ) not in edges


def test_conflict_ties_are_ordered_and_temporal_and_unknown_statuses(tmp_path) -> None:
    benchmark = materialize_toy_worlds(tmp_path / "worlds", seed=29)
    conflict = _case(benchmark, "conflict")
    pack = benchmark.pack_for(conflict, MemoryCondition.CORRECT)
    retriever = SemanticRetriever.from_pack(
        pack.path,
        expected_pack_id=pack.pack_id,
    )
    outcome = retriever.retrieve(
        torch.from_numpy(benchmark.key_encoder.encode(conflict.keys[0])),
        key_encoder=benchmark.key_encoder.identity,
        top_k=2,
        min_score=3.5,
    )
    assert outcome.status == "conflict"
    assert tuple(hit.record_id for hit in outcome.hits) == tuple(
        sorted(hit.record_id for hit in outcome.hits)
    )
    assert (
        evaluate_case(
            benchmark, _case(benchmark, "temporal"), MemoryCondition.CORRECT
        ).value
        == "summer"
    )
    assert (
        evaluate_case(
            benchmark, _case(benchmark, "temporal-miss"), MemoryCondition.CORRECT
        ).status
        == "temporal_miss"
    )
    assert (
        evaluate_case(
            benchmark, _case(benchmark, "unknown"), MemoryCondition.CORRECT
        ).status
        == "unknown"
    )


def test_validator_rejects_direct_start_to_final_shortcuts(tmp_path) -> None:
    benchmark = materialize_toy_worlds(tmp_path / "worlds", seed=43)
    world = benchmark.worlds[0]
    case = _case(benchmark, "three-hop")
    shortcut = replace(
        world.producer_facts[0],
        id=world.producer_facts[0].id + ":shortcut",
        value=case.expected_value,
    )
    altered = World(
        world.id,
        world.partition,
        world.assignment,
        (*world.producer_facts, shortcut),
        world.query_cases,
    )
    with pytest.raises(ValueError, match="shortcut edge"):
        validate_toy_worlds((altered, *benchmark.worlds[1:]))


def test_each_deterministic_memory_control_has_distinct_behavior(tmp_path) -> None:
    benchmark = materialize_toy_worlds(tmp_path / "worlds", seed=31)
    one_hop = next(
        case for case in benchmark.cases("heldout") if case.id.endswith("one-hop")
    )
    three_hop = next(
        case for case in benchmark.cases("heldout") if case.id.endswith("three-hop")
    )

    assert evaluate_case(
        benchmark, one_hop, MemoryCondition.CORRECT
    ).followed_attached_pack
    assert (
        evaluate_case(benchmark, one_hop, MemoryCondition.DISABLED).status == "disabled"
    )
    assert not evaluate_case(
        benchmark, one_hop, MemoryCondition.RANDOM
    ).followed_attached_pack
    assert (
        evaluate_case(benchmark, one_hop, MemoryCondition.CONFLICTING).status
        == "conflict"
    )
    assert (
        evaluate_case(benchmark, three_hop, MemoryCondition.INCOMPLETE).status
        == "unknown"
    )


def test_knowledge_swap_ratio_is_per_task_and_safe_at_zero_denominator(
    tmp_path,
) -> None:
    benchmark = materialize_toy_worlds(tmp_path / "worlds", seed=37)
    changed = [case for case in benchmark.cases("heldout") if case.changed_assignment]
    heldout_worlds = [
        world for world in benchmark.worlds if world.partition == "heldout"
    ]
    by_kind = {
        kind: [case for case in changed if case.id.endswith(kind)]
        for kind in ("one-hop", "two-hop", "three-hop")
    }
    for cases in by_kind.values():
        evaluations = {}
        for case in cases:
            attached_world = next(
                world for world in heldout_worlds if world.id != case.world_id
            )
            attached_case = next(
                candidate
                for candidate in attached_world.query_cases
                if candidate.id.endswith(case.id.rsplit(":", 1)[-1])
            )
            evaluation = evaluate_case(
                benchmark,
                case,
                MemoryCondition.CORRECT,
                attachment_scope=attached_world.id,
            )
            evaluations[case.id] = evaluation
            assert evaluation.value == attached_case.expected_value
            assert evaluation.value != case.expected_value
            assert evaluation.followed_attached_pack
        ratio = knowledge_swap_ratio(cases, evaluations)
        assert ratio.value == 1.0
        assert ratio.denominator == 2
        assert ratio.reason is None
    with pytest.raises(ValueError, match="one task kind"):
        knowledge_swap_ratio(
            changed,
            {
                case.id: evaluate_case(benchmark, case, MemoryCondition.CORRECT)
                for case in changed
            },
        )
    one_hop = by_kind["one-hop"]
    with pytest.raises(ValueError, match="one attachment condition"):
        knowledge_swap_ratio(
            one_hop,
            {
                one_hop[0].id: evaluate_case(
                    benchmark, one_hop[0], MemoryCondition.CORRECT
                ),
                one_hop[1].id: evaluate_case(
                    benchmark, one_hop[1], MemoryCondition.RANDOM
                ),
            },
        )

    unchanged = QueryCase(
        "query:zero",
        "heldout-unused",
        "heldout",
        (StructuredKey("zero", "label"),),
        None,
        "unknown",
    )
    zero = knowledge_swap_ratio((unchanged,), {})
    assert zero.value is None
    assert zero.denominator == 0
    assert zero.reason == "no changed-assignment query cases"
