"""Exact retained execution can continue without changing the default gate."""

import json
import os
from pathlib import Path

import pytest

from sparselab.corpus import admission_inspection_cache as caches
from sparselab.corpus.admission_inspection_cache import capture
from sparselab.corpus.retained_render_adoption import retained_renders, write_adoption
from sparselab.corpus.split_runs import SplitStages, write_rows


@pytest.fixture
def retained(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    previous = os.umask(0o077)
    root = tmp_path / "work"
    root.mkdir()
    source = root / "source"
    source.write_text("retained input\n")
    producer = {"renderer.py": "unchanged"}
    request = {"producer": producer, "inputs": capture([source])}
    directory = root / "source-stages"
    first = SplitStages(directory / "source-a", request, work_root=root)
    result = first.phase(
        "native-shard-render", lambda d: {"files": [write_rows(d / "rows", [[1], [2]])]}
    )
    assert first.cache.proofs.recorded == 1
    proof_files = {
        p: p.read_bytes() for p in first.cache.proofs.directory.glob("*.json")
    }
    original = caches.verifier_authority
    monkeypatch.setattr(
        caches,
        "verifier_authority",
        lambda *args: {**original(*args), "assessed_cache_bookkeeping": True},
    )
    project = {"id": "fixed-project"}
    decision = root / "retained-renders.json"
    written = write_adoption(
        root,
        directory,
        ["source-a"],
        project,
        producer,
        decision,
        from_source="1" * 40,
        to_source="2" * 40,
        rationale="Reviewed verifier bookkeeping only; retain exact old execution.",
    )
    try:
        yield (
            root,
            source,
            directory,
            request,
            result,
            decision,
            written["sha256"],
            project,
            proof_files,
        )
    finally:
        os.umask(previous)


def test_explicit_transition_reuses_and_records_old_execution_without_resealing(
    retained,
):
    root, _, directory, request, expected, decision, digest, project, proofs = retained
    assert (
        SplitStages(directory / "source-a", request, work_root=root).completed_phase(
            "native-shard-render"
        )
        is None
    )
    with retained_renders(root, decision, digest, project):
        resumed = SplitStages(directory / "source-a", request, work_root=root)
        assert (
            resumed.phase(
                "native-shard-render", lambda _: pytest.fail("render replayed")
            )
            == expected
        )
        assert resumed.reused == ["native-shard-render"]
        assert (
            resumed.cache.proofs.diagnostics()["reasons"][
                "explicit_retained_render_adoption"
            ]
            == 1
        )
    assert proofs == {
        p: p.read_bytes() for p in resumed.cache.proofs.directory.glob("*.json")
    }
    events = [
        json.loads(line)
        for path in root.glob("retained-renders.uses-*.jsonl")
        for line in path.read_bytes().splitlines()
    ]
    assert len(events) == 1 and events[0]["declaration_sha256"] == digest
    # Leaving the one invocation never changes future default authority policy.
    assert (
        SplitStages(directory / "source-a", request, work_root=root).completed_phase(
            "native-shard-render"
        )
        is None
    )


@pytest.mark.parametrize(
    "damage",
    [
        "output",
        "source",
        "proof",
        "cache",
        "authority",
        "request",
        "completion",
        "missing_completion",
    ],
)
def test_explicit_transition_refuses_changed_state_without_replay(
    retained, monkeypatch, damage
):
    root, source, directory, request, result, decision, digest, project, proofs = (
        retained
    )
    if damage == "output":
        Path(result["files"][0]["path"]).write_text("different\n")
    elif damage == "source":
        source.write_text("modified input\n")
    elif damage == "proof":
        next(iter(proofs)).write_text("{}\n")
    elif damage == "cache":
        for path in directory.glob("*/*/sealed/*.json"):
            path.write_text("{}\n")
    elif damage == "authority":
        authority = caches.verifier_authority
        monkeypatch.setattr(
            caches,
            "verifier_authority",
            lambda *a: {**authority(*a), "unassessed_change": True},
        )
    elif damage in {"completion", "missing_completion"}:
        completion = next(directory.glob("*/*/completion-*.json"))
        if damage == "missing_completion":
            completion.unlink()
        else:
            value = json.loads(completion.read_bytes())
            value["record"]["elapsed_seconds"] = -1
            completion.write_text(json.dumps(value))
    else:
        request = {**request, "normalizer": "changed"}
    with (
        retained_renders(root, decision, digest, project),
        pytest.raises(RuntimeError, match="retained render|retained-render"),
    ):
        resumed = SplitStages(directory / "source-a", request, work_root=root)
        resumed.phase("native-shard-render", lambda _: pytest.fail("render replayed"))


def test_explicit_transition_binds_project_and_declaration_bytes(retained):
    root, _, _, _, _, decision, digest, project, _ = retained
    with (
        pytest.raises(ValueError, match="invocation differs"),
        retained_renders(root, decision, digest, {**project, "changed": True}),
    ):
        pytest.fail("changed project accepted")
    decision.write_bytes(decision.read_bytes() + b"\n")
    with (
        pytest.raises(ValueError, match="digest differs"),
        retained_renders(root, decision, digest, project),
    ):
        pytest.fail("changed declaration accepted")


def test_transition_requires_authentic_historical_execution(retained):
    root, _, directory, request, _, _, _, project, proofs = retained
    next(iter(proofs)).write_text("{}\n")
    with pytest.raises((ValueError, KeyError)):
        write_adoption(
            root,
            directory,
            ["source-a"],
            project,
            request["producer"],
            root / "second.json",
            from_source="1" * 40,
            to_source="2" * 40,
            rationale="Explicit decision cannot invent execution.",
        )
    assert not (root / "second.json").exists()
