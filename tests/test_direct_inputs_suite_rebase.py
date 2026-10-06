"""Campaign relocation explicitly preserves a template's authored suite location."""

from __future__ import annotations

import pytest
import yaml
from test_experiment_direct_inputs import inputs

from sparselab.experiments.direct_inputs import bind_direct_inputs
from sparselab.experiments.lock import open_lock, publish_lock, resolve_plan
from sparselab.experiments.plan import load_plan
from sparselab.training.manifest import sha256_file


@pytest.mark.parametrize("rebase", [False, True])
def test_direct_suite_relocation_is_explicit(tmp_path, monkeypatch, rebase):
    config, root, run, template = inputs(tmp_path, monkeypatch)
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "evaluation_suite_version: 1\nid: fixture\nevaluations:\n"
        "  - id: loss\n    kind: heldout_lm\n    role: descriptive\n"
    )
    raw = yaml.safe_load(template.read_text())
    raw["evaluation_suite"] = "suite.yaml"
    template.write_text(yaml.safe_dump(raw))
    destination = tmp_path / "campaign" / "native"
    destination.mkdir(parents=True)
    output = destination / "plan.yaml"
    if not rebase:
        with pytest.raises(ValueError, match="suite changes location"):
            bind_direct_inputs(run, template, root, output)
        assert not output.exists()
        return
    bound = bind_direct_inputs(run, template, root, output, rebase_suite=True)
    assert bound.evaluation_suite == "suite-inputs/suite.yaml"
    copied = output.parent / bound.evaluation_suite
    assert copied.read_bytes() == suite.read_bytes()
    assert bound.base_run == config
    assert load_plan(output).evaluation_suite == bound.evaluation_suite
    assert yaml.safe_load(template.read_text())["evaluation_suite"] == "suite.yaml"
    locked = resolve_plan(bound, output)
    assert locked.evaluation_suite["sha256"] == sha256_file(suite)
    assert (
        open_lock(publish_lock(locked, tmp_path / "experiment")).evaluation_suite
        == locked.evaluation_suite
    )


def test_suite_rebase_preserves_unsafe_reference_rejection(tmp_path, monkeypatch):
    _, root, run, template = inputs(tmp_path, monkeypatch)
    raw = yaml.safe_load(template.read_text())
    raw["evaluation_suite"] = "../outside.yaml"
    template.write_text(yaml.safe_dump(raw))
    output = tmp_path / "rejected.yaml"
    with pytest.raises(ValueError):
        bind_direct_inputs(run, template, root, output, rebase_suite=True)
    assert not output.exists()


@pytest.mark.parametrize("collision", ["none", "same", "different", "symlink"])
def test_suite_materialization_copies_nested_dependencies_write_new(
    tmp_path, monkeypatch, collision
):
    _, root, run, template = inputs(tmp_path, monkeypatch)
    suite_dir = tmp_path / "evaluations"
    suite_dir.mkdir()
    suite = suite_dir / "suite.yaml"
    suite.write_text(
        "evaluation_suite_version: 1\nid: fixture\nevaluations:\n"
        "  - id: evidence\n    kind: evidence_reference\n"
        "    role: descriptive\n    source: evidence.json\n"
    )
    evidence = suite_dir / "evidence.json"
    evidence.write_text('{"observation": "retained"}\n')
    raw = yaml.safe_load(template.read_text())
    raw["evaluation_suite"] = "evaluations/suite.yaml"
    template.write_text(yaml.safe_dump(raw))
    destination = tmp_path / "campaign"
    destination.mkdir()
    target = destination / "suite-inputs/evaluations/evidence.json"
    if collision != "none":
        target.parent.mkdir(parents=True)
        if collision == "symlink":
            target.symlink_to(evidence)
        else:
            target.write_bytes(
                evidence.read_bytes() if collision == "same" else b"different"
            )
    output = destination / "plan.yaml"
    if collision in {"different", "symlink"}:
        with pytest.raises(ValueError, match="collision|symlink"):
            bind_direct_inputs(run, template, root, output, rebase_suite=True)
        assert not output.exists()
        assert not target.with_name("suite.yaml").exists()
        return
    bound = bind_direct_inputs(run, template, root, output, rebase_suite=True)
    assert target.read_bytes() == evidence.read_bytes()
    copied = destination / bound.evaluation_suite
    assert copied.read_bytes() == suite.read_bytes()
    assert resolve_plan(bound, output).evaluation_suite["sha256"] == sha256_file(suite)
