"""Real Campaign execution binds descriptive panels to collected evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from test_campaign import by_id, make_full_campaign

from sparselab.campaign.engine import CampaignEngine
from sparselab.evaluation.panel import verify_panel_result
from sparselab.evaluation.suite import verify_evaluation_index


@pytest.fixture(scope="module", params=["real", "negative"])
def completed_panel(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"campaign-panel-{request.param}")
    with pytest.MonkeyPatch.context() as patch:
        source = make_full_campaign(root, patch)
        declaration = {
            "generation_panel_version": 1,
            "id": "campaign-descriptive",
            "role": "descriptive_not_quality_gate",
            "prompts": ["def ", "SELECT ", "#!/bin/sh\n"],
            "decoder": {
                "temperature": 0,
                "top_k": 0,
                "max_new_tokens": 3,
                "seed": 42,
            },
            "checkpoint_selection": "same immutable generation as heldout evaluation",
        }
        panel_path = root / "panel.json"
        panel_path.write_text(json.dumps(declaration))
        value = yaml.safe_load(source.read_text())
        value["stages"].append(
            {
                "id": "panel",
                "kind": "generation_panel",
                "scope": "evaluation",
                "requires": ["collect", "evaluation", "runtime"],
                "collect": "collect",
                "evaluation": "evaluation",
                "runtime": "runtime",
                "panel": "panel.json",
            }
        )
        source.write_text(yaml.safe_dump(value))
        calls = []
        if request.param == "negative":
            from sparselab.evaluation import panel

            def negative_generator(model, tokenizer, prompt, *args, **kwargs):
                calls.append(prompt)
                if prompt == declaration["prompts"][0]:
                    return prompt, []
                if prompt == declaration["prompts"][1]:
                    return prompt + "aaa", [4, 4, 4]
                raise RuntimeError("preserved descriptive generation failure")

            patch.setattr(panel, "generate_with_token_ids", negative_generator)
        readiness_before_panel = {}

        def after_commit(stage_id, state):
            if stage_id == "model":
                rows = by_id(engine.inspect("status"))
                path = Path(rows["model"]["availability"]["path"])
                readiness_before_panel.update(path=path, content=path.read_bytes())

        engine = CampaignEngine(source, root / "work", after_commit=after_commit)
        engine.apply(allow_uncommitted_declaration=True)
        engine.approve("gate")
        rows = by_id(
            engine.apply(
                execute_runs=True,
                max_wait_seconds=600,
                allow_uncommitted_declaration=True,
            )
        )
        assert rows["panel"]["state"] == "COMPLETE", rows["panel"]
        yield {
            "engine": engine,
            "rows": rows,
            "panel": panel_path,
            "readiness": readiness_before_panel,
            "mode": request.param,
            "calls": calls,
        }


def test_panel_binds_the_collected_and_evaluated_immutable_generation(completed_panel):
    rows = completed_panel["rows"]
    panel = verify_panel_result(Path(rows["panel"]["availability"]["path"]))
    index = verify_evaluation_index(Path(rows["evaluation"]["availability"]["path"]))
    collected = rows["collect"]["measurements"]
    assert panel["checkpoint"] == f"checkpoints/{collected['generation']}"
    assert panel["checkpoint"] == index["checkpoint"]
    assert panel["checkpoint_sha256"] == collected["sha256"]
    assert panel["checkpoint_sha256"] == index["checkpoint_sha256"]
    assert panel["evaluation_index_sha256"] == index["index_sha256"]
    assert panel["run_id"] == rows["run"]["outputs"][0]["identifier"]
    assert (
        panel["runtime_binding"]["runtime_acceptance_sha256"]
        == rows["runtime"]["outputs"][0]["sha256"]
    )
    assert [row["prompt"] for row in panel["rows"]] == panel["declaration"]["prompts"]
    assert all(row["attempts"] == 1 for row in panel["rows"])
    if completed_panel["mode"] == "real":
        assert all(row["status"] == "COMPLETED" for row in panel["rows"])


def test_descriptive_panel_preserves_negative_observations_and_readiness(
    completed_panel,
):
    rows = completed_panel["rows"]
    panel = verify_panel_result(Path(rows["panel"]["availability"]["path"]))
    readiness = completed_panel["readiness"]
    assert readiness["path"].read_bytes() == readiness["content"]
    assert rows["model"]["state"] == "COMPLETE"
    assert panel["role"] == "descriptive_not_quality_gate"
    assert rows["panel"]["outcome"] == "DESCRIPTIVE_EVIDENCE"
    assert "no readiness assessment" in rows["panel"]["reason"]
    if completed_panel["mode"] == "negative":
        empty, repetitive, failed = panel["rows"]
        assert empty["status"] == "COMPLETED"
        assert empty["completion"] == ""
        assert empty["token_ids"] == []
        assert repetitive["completion"] == "aaa"
        assert repetitive["token_ids"] == [4, 4, 4]
        assert failed["status"] == "FAILED"
        assert failed["completion"] is None
        assert failed["error"]["message"] == "preserved descriptive generation failure"
        assert completed_panel["calls"] == panel["declaration"]["prompts"]


def test_reapply_reopens_panel_without_reroll_or_new_training(
    completed_panel, monkeypatch
):
    from sparselab.evaluation import panel
    from sparselab.workers.controller import Controller

    def no_reroll(*args, **kwargs):
        pytest.fail("a completed panel must not generate a second trajectory")

    monkeypatch.setattr(panel, "generate_with_token_ids", no_reroll)
    rows = completed_panel["rows"]
    path = Path(rows["panel"]["availability"]["path"])
    before = path.read_bytes()
    controller = Controller(
        Path(rows["run"]["availability"]["workspace"]) / "controller", read_only=True
    )
    attempts = controller.list_experiments()
    replay = by_id(completed_panel["engine"].apply(allow_uncommitted_declaration=True))
    assert replay["panel"]["outputs"] == rows["panel"]["outputs"]
    assert path.read_bytes() == before
    assert controller.list_experiments() == attempts


@pytest.mark.parametrize("change", ["prompt", "decoder"])
def test_changed_panel_declaration_cannot_reuse_campaign_receipt(
    completed_panel, change
):
    source = completed_panel["panel"]
    original = source.read_bytes()
    declaration = json.loads(original)
    if change == "prompt":
        declaration["prompts"][0] += "changed"
    else:
        declaration["decoder"]["seed"] += 1
    source.write_text(json.dumps(declaration))
    try:
        with pytest.raises(ValueError, match="DECLARATION_IDENTITY_CHANGED"):
            completed_panel["engine"].apply(allow_uncommitted_declaration=True)
        with pytest.raises(ValueError, match="declaration changed"):
            verify_panel_result(
                Path(completed_panel["rows"]["panel"]["availability"]["path"])
            )
    finally:
        source.write_bytes(original)


@pytest.mark.parametrize(
    "change",
    [
        "missing_dependency",
        "different_collect",
        "different_runtime_plan",
        "readiness",
        "unsafe_path",
    ],
)
def test_generation_panel_rejects_invalid_typed_plan(completed_panel, change):
    import copy

    from sparselab.campaign.plan import CampaignPlan

    plan = yaml.safe_load(completed_panel["engine"].source.read_text())
    stages = {stage["id"]: stage for stage in plan["stages"]}
    panel = stages["panel"]
    if change == "missing_dependency":
        panel["requires"].remove("evaluation")
    elif change == "different_collect":
        other = copy.deepcopy(stages["collect"])
        other["id"] = "other-collect"
        plan["stages"].append(other)
        panel["collect"] = "other-collect"
        panel["requires"].append("other-collect")
    elif change == "different_runtime_plan":
        other = copy.deepcopy(stages["plan"])
        other["id"] = "other-plan"
        plan["stages"].append(other)
        other_runtime = copy.deepcopy(stages["runtime"])
        other_runtime.update(
            id="other-runtime", plan="other-plan", requires=["other-plan"]
        )
        plan["stages"].append(other_runtime)
        panel["runtime"] = "other-runtime"
        panel["requires"].append("other-runtime")
    elif change == "readiness":
        stages["model"].update(evaluation="panel", requires=["panel"])
    else:
        panel["panel"] = "../outside.json"
    with pytest.raises(ValueError):
        CampaignPlan.model_validate(plan)


@pytest.mark.parametrize(
    "change", ["checkpoint", "generation", "index", "runtime", "run"]
)
def test_panel_receipt_rejects_mismatched_upstream_identity(completed_panel, change):
    import copy

    rows = copy.deepcopy(completed_panel["rows"])
    if change == "checkpoint":
        rows["collect"]["measurements"]["sha256"] = "a" * 64
    elif change == "generation":
        rows["collect"]["measurements"]["generation"] = "different-generation"
    elif change == "index":
        rows["evaluation"]["outputs"][0]["sha256"] = "a" * 64
    elif change == "runtime":
        rows["runtime"]["outputs"][0]["sha256"] = "a" * 64
    else:
        rows["run"]["outputs"][0]["identifier"] = "different-run"
    engine = completed_panel["engine"]
    rows["panel"]["stage_input_sha256"] = engine.store.input_sha(
        engine.stages["panel"], rows
    )
    with pytest.raises(ValueError, match="changed"):
        engine._verify(engine.stages["panel"], rows["panel"], rows)
