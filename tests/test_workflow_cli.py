"""Public CLI boundaries; native preparation/queueing only, no worker tick."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_workflow_execution import admitted, execution_fixture
from test_workflow_recipe import native_workflow  # noqa: F401

from sparselab.cli.entry import main
from sparselab.experiments.workflow_recipe import load_workflow


def invoke(monkeypatch, capsys, *args):
    monkeypatch.setattr(
        sys, "argv", ["sparselab", "experiment", "recipe", "workflow", *map(str, args)]
    )
    main()
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def authored(native_workflow, tmp_path):  # noqa: F811
    old_source, old_bindings = native_workflow
    old = load_workflow(old_source, old_bindings)
    source, bindings = execution_fixture(
        tmp_path / "committed",
        old.context.read_document(old.context.reference(old.source, old.plan.base_run)),
        {
            name: item.model_dump(mode="json")
            for name, item in old.descriptor.artifacts.items()
        },
    )
    bindings.write_bytes(old_bindings.read_bytes())
    context = admitted(source, bindings, tmp_path / "work")
    admission = tmp_path / "admission.json"
    admission.write_text(context.admission.model_dump_json())
    return context, [
        source,
        "--bindings",
        bindings,
        "--admission",
        admission,
        "--work-dir",
        context.work_dir,
        "--json",
    ]


def test_public_approval_queue_replay_and_report(
    authored, monkeypatch, capsys, tmp_path
):
    from sparselab.workers.controller import Controller

    monkeypatch.setattr(
        Controller,
        "tick",
        lambda *a, **kw: pytest.fail("CLI test must not launch a worker"),
    )
    _, args = authored
    first = invoke(monkeypatch, capsys, "apply", *args)
    rows = {row["id"]: row for row in first["stages"]}
    assert rows["gate"]["state"] == "AWAITING_APPROVAL"
    workspace = Path(rows["plan"]["availability"]["workspace"])
    assert not (workspace / "controller" / "experiments.sqlite3").exists()
    pending = invoke(
        monkeypatch, capsys, "apply", *args, "--execute-runs", "--max-wait-seconds", "0"
    )
    assert (
        next(row for row in pending["stages"] if row["id"] == "gate")["state"]
        == "AWAITING_APPROVAL"
    )
    assert not (workspace / "controller" / "experiments.sqlite3").exists()
    approved = invoke(
        monkeypatch, capsys, "approve", *args, "gate", "--note", "fixture only"
    )
    assert (
        next(row for row in approved["stages"] if row["id"] == "gate")["state"]
        == "COMPLETE"
    )
    invoke(
        monkeypatch, capsys, "apply", *args, "--execute-runs", "--max-wait-seconds", "0"
    )
    controller = Controller(workspace / "controller", read_only=True)
    attempts = controller.list_experiments()
    assert len(attempts) == 1
    invoke(monkeypatch, capsys, "resume", *args, "--max-wait-seconds", "0")
    assert controller.list_experiments() == attempts
    # Read-only planned report needs no execution admission or runtime selection.
    output = tmp_path / "planned.json"
    lock = rows["plan"]["availability"]["path"]
    result = invoke(monkeypatch, capsys, "report", lock, "--output", output)
    assert result["executable"] is False
    report = json.loads(output.read_text())
    assert report["readiness"]["execution_coverage"] != "complete"
    before = output.read_bytes()
    invoke(monkeypatch, capsys, "report", lock, "--output", output)
    assert output.read_bytes() == before
    assert controller.list_experiments() == attempts
    from sparselab.experiments.evidence import collect_evidence
    from sparselab.experiments.lock import open_lock

    evidence = collect_evidence(open_lock(Path(lock)), workspace)
    index = Path(evidence["index_path"])
    evidence_output = tmp_path / "evidence.md"
    invoke(
        monkeypatch,
        capsys,
        "report",
        lock,
        "--workspace",
        workspace,
        "--index",
        index,
        "--format",
        "markdown",
        "--output",
        evidence_output,
    )
    assert evidence_output.read_text().startswith("#")
    # Existing output cannot be replaced, and corrupt evidence cannot be rendered.
    with pytest.raises(SystemExit) as stopped:
        invoke(monkeypatch, capsys, "report", lock, "--output", evidence_output)
    assert stopped.value.code == 1
    assert "different bytes" in json.loads(capsys.readouterr().out)["error"]["reason"]
    index.write_text("{}")
    bad_output = tmp_path / "invalid.json"
    with pytest.raises(SystemExit) as stopped:
        invoke(
            monkeypatch,
            capsys,
            "report",
            lock,
            "--workspace",
            workspace,
            "--index",
            index,
            "--output",
            bad_output,
        )
    assert stopped.value.code == 1
    assert json.loads(capsys.readouterr().out)["state"] == "FAILED"
    assert not bad_output.exists()


@pytest.mark.parametrize("damage", ["bindings", "admission", "inventory"])
def test_public_changed_identity_refuses_before_store(
    authored, monkeypatch, capsys, damage
):
    context, args = authored
    if damage == "bindings":
        context.bindings.write_text(context.bindings.read_text() + "\n")
    elif damage == "admission":
        path = Path(args[4])
        raw = json.loads(path.read_text())
        raw["reviewed_commit"] = "0" * 40
        path.write_text(json.dumps(raw))
    else:
        path = context.source.parent / "configs/run.json"
        path.write_text(path.read_text() + "\n")
    with pytest.raises(SystemExit) as stopped:
        invoke(monkeypatch, capsys, "apply", *args, "--execute-runs")
    assert stopped.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "FAILED"
    assert not context.work_dir.exists()


@pytest.mark.parametrize("command", ["apply", "resume", "approve", "report"])
def test_public_help_is_lightweight(command):
    code = f'''
import sys
from sparselab.cli.entry import main
sys.argv = ["sparselab", "experiment", "recipe", "workflow", "{command}", "--help"]
try:
    main()
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("help must exit")
assert not any(name in sys.modules for name in ("torch", "numpy", "tokenizers", "sparselab.campaign.engine", "sparselab.experiments.workflow_execution", "sparselab.experiments.publication_report"))
'''
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True)


@pytest.mark.parametrize("wait", ["nan", "inf", "-1"])
def test_bad_wait_refuses_at_parse(monkeypatch, capsys, wait):
    with pytest.raises(SystemExit) as stopped:
        invoke(
            monkeypatch,
            capsys,
            "apply",
            "absent",
            "--bindings",
            "absent",
            "--admission",
            "absent",
            "--work-dir",
            "/absent",
            "--max-wait-seconds",
            wait,
        )
    assert stopped.value.code == 2
    assert "finite and nonnegative" in capsys.readouterr().err


def test_report_requires_paired_evidence_without_output(monkeypatch, capsys, tmp_path):
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as stopped:
        invoke(
            monkeypatch,
            capsys,
            "report",
            "absent",
            "--workspace",
            tmp_path,
            "--output",
            output,
        )
    assert stopped.value.code == 1
    assert "supplied together" in json.loads(capsys.readouterr().out)["error"]["reason"]
    assert not output.exists()


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("recipe_sha256", "recipe or bindings differ"),
        ("binding_sha256", "recipe or bindings differ"),
        ("inside_git", "outside Git"),
        ("dirty", "unchanged clean declarations"),
        ("cpu_python", "CPU environment differs"),
    ],
)
def test_public_admission_guards_before_engine(
    tmp_path, monkeypatch, capsys, damage, reason
):
    from test_workflow_recipe import repin, update

    from sparselab.campaign.engine import CampaignEngine

    source, bindings = execution_fixture(tmp_path / "recipe")
    context = admitted(source, bindings, tmp_path / "work")
    raw = context.admission.model_dump(mode="json")
    work = context.work_dir
    if damage in {"recipe_sha256", "binding_sha256"}:
        raw[damage] = "0" * 64
    elif damage == "cpu_python":
        raw[damage] = str(tmp_path / "different-python")
    elif damage == "inside_git":
        work = source.parent / "inside-work"
    else:
        update(source.parent / "suite.json", id="changed-consistently")
        repin(source)
        loaded = load_workflow(source, bindings)
        raw.update(
            recipe_sha256=loaded.recipe_sha256, binding_sha256=loaded.binding_sha256
        )
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps(raw))
    monkeypatch.setattr(
        CampaignEngine,
        "__init__",
        lambda *a, **kw: pytest.fail("invalid admission reached engine"),
    )
    with pytest.raises(SystemExit) as stopped:
        invoke(
            monkeypatch,
            capsys,
            "apply",
            source,
            "--bindings",
            bindings,
            "--admission",
            admission,
            "--work-dir",
            work,
            "--execute-runs",
        )
    assert stopped.value.code == 1
    assert reason in json.loads(capsys.readouterr().out)["error"]["reason"]
    assert not work.exists()


def test_old_approval_refuses_rebound_admission(authored, monkeypatch, capsys):
    from sparselab.workers.controller import Controller

    context, args = authored
    invoke(monkeypatch, capsys, "apply", *args)
    invoke(monkeypatch, capsys, "approve", *args, "gate")
    context.bindings.write_text(context.bindings.read_text() + "\n")
    loaded = load_workflow(context.source, context.bindings)
    admission = Path(args[4])
    raw = json.loads(admission.read_text())
    raw["binding_sha256"] = loaded.binding_sha256
    admission.write_text(json.dumps(raw))
    monkeypatch.setattr(
        Controller, "tick", lambda *a, **kw: pytest.fail("old approval dispatched work")
    )
    with pytest.raises(SystemExit) as stopped:
        invoke(
            monkeypatch,
            capsys,
            "apply",
            *args,
            "--execute-runs",
            "--max-wait-seconds",
            "0",
        )
    assert stopped.value.code == 1
    assert (
        "WORKFLOW_CONTEXT_CHANGED: receipt context differs"
        in json.loads(capsys.readouterr().out)["error"]["reason"]
    )


def test_failed_stage_exit_and_progress_channel(authored, monkeypatch, capsys):
    from sparselab.campaign.engine import CampaignEngine

    _, args = authored

    def failed(self, **kwargs):
        print("adapter progress")
        return {"stages": [{"id": "run", "state": "FAILED"}]}

    monkeypatch.setattr(CampaignEngine, "apply", failed)
    with pytest.raises(SystemExit) as stopped:
        invoke(monkeypatch, capsys, "apply", *args)
    assert stopped.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["stages"][0]["state"] == "FAILED"
    assert "adapter progress" in captured.err


@pytest.mark.parametrize("selector", [1, 2, 3, 4, True, "3"])
def test_obsolete_admission_refuses_before_loading_or_runtime(
    tmp_path, monkeypatch, capsys, selector
):
    from sparselab.experiments import workflow_execution

    def forbidden(*args, **kwargs):
        pytest.fail("obsolete admission reached workflow loading or runtime setup")

    monkeypatch.setattr(workflow_execution, "load_workflow", forbidden)
    monkeypatch.setattr(workflow_execution, "_environment", forbidden)
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"workflow_admission_version": selector}))
    with pytest.raises(SystemExit) as error:
        invoke(
            monkeypatch,
            capsys,
            "apply",
            tmp_path / "missing-recipe.json",
            "--bindings",
            tmp_path / "missing-bindings.json",
            "--admission",
            admission,
            "--work-dir",
            tmp_path / "work",
            "--execute-runs",
        )
    assert error.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "FAILED"
    assert "obsolete workflow admission; regenerate" in result["error"]["reason"]
    assert not (tmp_path / "work").exists()
