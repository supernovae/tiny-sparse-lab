"""Local synthetic fixtures: shared context through native Campaign lifecycle."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_preparation_recipe import recipe
from test_workflow_recipe import repin

from sparselab.experiments.preparation_execution import (
    PreparationAdmission,
    PreparationExecution,
)
from sparselab.experiments.preparation_recipe import load_preparation_recipe


def git(root, *args):
    return (
        subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)
        .decode()
        .strip()
    )


def setup(tmp_path, reuse=False):
    source, original_bindings = recipe(tmp_path / "recipe", reuse)
    for name in ("run.yaml", "tokenizer.yaml"):
        path = source.parent / name
        config = json.loads(path.read_text())
        if name == "run.yaml":
            config["model"]["vocab_size"] = 264
        else:
            config["vocab_size"] = 264
        path.write_text(json.dumps(config))
    repin(source)
    bindings = tmp_path / "bindings.json"
    original_bindings.rename(bindings)
    work = tmp_path / "work"
    raw = json.loads(bindings.read_text())
    for fields in raw["config_paths"].values():
        for field, value in fields.items():
            fields[field] = str(work / value)
    if reuse:
        from sparselab.config.models import TokenizerTrainConfig
        from sparselab.data.tokenizer import train_tokenizer
        from sparselab.training.manifest import sha256_file

        cfg = json.loads((source.parent / "tokenizer.yaml").read_text())
        cfg["output_dir"] = str(tmp_path / "external" / "tok")
        cfg["dataset"]["cache_dir"] = str(tmp_path / "external" / "cache")
        token = train_tokenizer(TokenizerTrainConfig.model_validate(cfg))
        raw["artifacts"]["reuse"] = str(token)
        raw["config_paths"]["run.yaml"]["tokenizer.path"] = str(token)
        campaign = source.parent / "campaign.json"
        declaration = json.loads(campaign.read_text())
        declaration["stages"][1]["artifact"]["sha256"] = sha256_file(token)
        campaign.write_text(json.dumps(declaration))
        repin(source)
    bindings.write_text(json.dumps(raw))
    git(source.parent, "init", "-q")
    git(source.parent, "add", ".")
    git(
        source.parent,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "reviewed synthetic fixture",
    )
    loaded = load_preparation_recipe(source, bindings)
    admission = PreparationAdmission(
        preparation_admission_version=1,
        reviewed_commit=git(source.parent, "rev-parse", "HEAD"),
        recipe_sha256=loaded.recipe_sha256,
        binding_sha256=loaded.binding_sha256,
        cpu_python=Path(sys.executable),
        max_documents=32,
        max_tokens=1024,
        max_vocab_size=512,
        min_free_bytes=1024,
        min_free_inodes=16,
    )
    return source, bindings, work, admission


def rows(result):
    return {row["id"]: row for row in result["stages"]}


@pytest.mark.parametrize("reuse", [False, True])
def test_two_graphs_complete_and_replay_native_receipts(tmp_path, monkeypatch, reuse):
    args = setup(tmp_path, reuse)
    context = PreparationExecution(*args)
    engine = context.engine()
    result = engine.apply()
    assert all(row["state"] == "COMPLETE" for row in result["stages"]), result
    for row in result["stages"]:
        assert row["declaration_provenance"]["preparation_context"] == context.identity
    monkeypatch.setattr(
        engine, "dispatch", lambda *a, **kw: pytest.fail("replayed adapter")
    )
    assert rows(engine.apply())["pack"]["outputs"] == rows(result)["pack"]["outputs"]
    assert rows(context.engine().inspect("status"))["pack"]["state"] == "COMPLETE"


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("reviewed_commit", "a" * 40, "reviewed commit"),
        ("recipe_sha256", "a" * 64, "CONTEXT_CHANGED"),
        ("binding_sha256", "a" * 64, "CONTEXT_CHANGED"),
        ("cpu_python", Path("/unselected/python"), "CPU Python"),
        ("max_tokens", 1, "limits"),
        ("max_documents", 1, "limits"),
        ("max_vocab_size", 260, "limits"),
        ("min_free_bytes", 10**30, "storage/permission"),
    ],
)
def test_admission_refuses_before_any_output(tmp_path, field, value, match):
    source, bindings, work, admission = setup(tmp_path)
    admission = admission.model_copy(update={field: value})
    with pytest.raises(ValueError, match=match):
        PreparationExecution(source, bindings, work, admission)
    assert not work.exists()


def test_dirty_inventory_and_output_collision_refuse_without_store(tmp_path):
    args = setup(tmp_path)
    context = PreparationExecution(*args)
    engine = context.engine()
    output = context.loaded.configs["train"].output_dir
    output.mkdir(parents=True)
    (output / "unowned").write_text("retain")
    with pytest.raises(ValueError, match="already exists"):
        engine.apply()
    assert not engine.store.root.exists()
    assert (output / "unowned").read_text() == "retain"


def test_context_change_before_apply_and_after_dispatch_is_refused(
    tmp_path, monkeypatch
):
    args = setup(tmp_path)
    context = PreparationExecution(*args)
    engine = context.engine()
    original = engine.dispatch

    def changed(*a, **kw):
        result = original(*a, **kw)
        args[1].write_text(args[1].read_text() + "\n")
        return result

    monkeypatch.setattr(engine, "dispatch", changed)
    with pytest.raises(ValueError, match="CONTEXT_CHANGED"):
        engine.apply()
    assert not list(engine.store.root.glob("receipts/**/*.json"))
    assert rows(engine.store.read())["train"]["state"] == "RUNNING"
    with pytest.raises(ValueError, match="CONTEXT_CHANGED"):
        engine.apply(resume=True)


def test_crash_after_receipt_reconciles_without_redispatch(tmp_path, monkeypatch):
    args = setup(tmp_path)
    engine = PreparationExecution(*args).engine()
    original = engine.store.commit

    def crash(stage, row):
        original(stage, row)
        raise KeyboardInterrupt("after immutable receipt before state save")

    monkeypatch.setattr(engine.store, "commit", crash)
    with pytest.raises(KeyboardInterrupt):
        engine.apply()
    recovered = PreparationExecution(*args).engine()
    dispatch = recovered.dispatch
    calls = []

    def track(stage, *a, **kw):
        calls.append(stage.id)
        return dispatch(stage, *a, **kw)

    monkeypatch.setattr(recovered, "dispatch", track)
    result = recovered.apply(resume=True)
    assert calls == ["pack"]
    assert rows(result)["pack"]["state"] == "COMPLETE"


def test_interrupted_attempt_is_preserved_and_never_rerun(tmp_path, monkeypatch):
    args = setup(tmp_path)
    engine = PreparationExecution(*args).engine()

    def interrupt(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "dispatch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        engine.apply()
    monkeypatch.setattr(
        engine, "dispatch", lambda *a, **kw: pytest.fail("unsafe rerun")
    )
    with pytest.raises(ValueError, match="uncommitted attempt"):
        engine.apply(resume=True)
    row = rows(engine.store.read())["train"]
    assert row["state"] == "INTERRUPTED"
    assert row["attempts"][-1]["state"] == "INTERRUPTED"


def test_new_binding_identity_cannot_reuse_old_receipts(tmp_path, monkeypatch):
    source, bindings, work, admission = setup(tmp_path)
    engine = PreparationExecution(source, bindings, work, admission).engine()
    engine.apply()
    bindings.write_text(bindings.read_text() + "\n")
    changed = admission.model_copy(
        update={
            "binding_sha256": load_preparation_recipe(source, bindings).binding_sha256,
        }
    )
    other = PreparationExecution(source, bindings, work, changed).engine()
    monkeypatch.setattr(
        other, "dispatch", lambda *a, **kw: pytest.fail("cross-binding dispatch")
    )
    with pytest.raises(ValueError, match="receipt context differs"):
        other.apply(resume=True)
    with pytest.raises(ValueError, match="receipt context differs"):
        other.inspect()


def test_legacy_route_cannot_reinterpret_portable_receipts(tmp_path):
    from sparselab.campaign.engine import CampaignEngine

    args = setup(tmp_path)
    context = PreparationExecution(*args)
    engine = context.engine()
    engine.apply()
    legacy = CampaignEngine(engine.source, engine.work_dir)
    with pytest.raises(ValueError, match="receipt context differs"):
        legacy.apply(allow_uncommitted_declaration=True)
    with pytest.raises(ValueError, match="require their execution context"):
        legacy.inspect("status")
    engine.store.index.unlink()  # Test crash recovery authority, not reexecution.
    with pytest.raises(ValueError, match="require their execution context"):
        legacy.inspect()


def test_output_drift_is_never_redispatched(tmp_path, monkeypatch):
    args = setup(tmp_path)
    engine = PreparationExecution(*args).engine()
    result = engine.apply()
    token = Path(rows(result)["train"]["availability"]["path"])
    token.write_text("corrupt fixture")
    monkeypatch.setattr(
        engine, "dispatch", lambda *a, **kw: pytest.fail("invalid output redispatched")
    )
    with pytest.raises(ValueError, match="provenance or digest mismatch"):
        engine.apply()
    assert token.read_text() == "corrupt fixture"


def test_changed_output_between_dispatch_and_commit_refuses_receipt(
    tmp_path, monkeypatch
):
    args = setup(tmp_path)
    engine = PreparationExecution(*args).engine()
    dispatch = engine.dispatch

    def corrupt(*a, **kw):
        result = dispatch(*a, **kw)
        Path(result["availability"]["path"]).write_text("corrupt fixture")
        return result

    monkeypatch.setattr(engine, "dispatch", corrupt)
    with pytest.raises(ValueError):
        engine.apply()
    assert not list(engine.store.root.glob("receipts/**/*.json"))


def test_changed_context_after_running_record_never_calls_adapter(
    tmp_path, monkeypatch
):
    args = setup(tmp_path)
    engine = PreparationExecution(*args).engine()
    record = engine._record

    def changed(state, row):
        record(state, row)
        if row["state"] == "RUNNING":
            args[1].write_text(args[1].read_text() + "\n")

    monkeypatch.setattr(engine, "_record", changed)
    monkeypatch.setattr(
        engine, "dispatch", lambda *a, **kw: pytest.fail("stale context dispatched")
    )
    with pytest.raises(ValueError, match="CONTEXT_CHANGED"):
        engine.apply()
    assert not list(engine.store.root.glob("receipts/**/*.json"))


def test_environment_drift_refuses_without_creating_store(tmp_path, monkeypatch):
    from sparselab.experiments import preparation_execution

    args = setup(tmp_path)
    engine = PreparationExecution(*args).engine()
    original = preparation_execution._environment
    monkeypatch.setattr(
        preparation_execution,
        "_environment",
        lambda: {**original(), "python_version": "changed"},
    )
    with pytest.raises(ValueError, match="environment differs"):
        engine.apply()
    assert not args[2].exists()


def test_dirty_index_is_not_reviewed_committed_content(tmp_path):
    args = setup(tmp_path)
    source = args[0]
    original = source.read_bytes()
    source.write_bytes(original + b"\n")
    git(source.parent, "add", "recipe.json")
    source.write_bytes(original)
    with pytest.raises(ValueError, match="clean declarations"):
        PreparationExecution(*args)
    assert not args[2].exists()


def test_unknown_and_legacy_execution_overrides_are_refused(tmp_path):
    args = setup(tmp_path)
    engine = PreparationExecution(*args).engine()
    for options in ({"execute_runs": True}, {"allow_uncommitted_declaration": True}):
        with pytest.raises(ValueError, match="does not allow overrides"):
            engine.apply(**options)
    assert not args[2].exists()


def test_dispatch_scopes_and_restores_process_locations(tmp_path, monkeypatch):
    import os
    import tempfile

    args = setup(tmp_path)
    context = PreparationExecution(*args)
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "other"))
    before = {
        name: os.environ.get(name)
        for name in ("SPARSELAB_WORK_DIR", "TMPDIR", "TEMP", "TMP")
    }
    temp = tempfile.tempdir
    context.engine().apply()
    assert {name: os.environ.get(name) for name in before} == before
    assert tempfile.tempdir == temp
    assert not (tmp_path / "other").exists()
    assert (args[2] / "scratch").is_dir()


def test_cli_apply_uses_selected_context_and_parser_help_stays_light(
    tmp_path, capsys, monkeypatch
):
    from sparselab.cli.entry import main

    source, bindings, work, admission = setup(tmp_path)
    document = tmp_path / "admission.json"
    document.write_text(admission.model_dump_json())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "--work-dir",
            str(tmp_path / "ignored-prefix"),
            "--hf-token-file",
            str(tmp_path / "must-not-read-token"),
            "experiment",
            "recipe",
            "preparation",
            "apply",
            str(source),
            "--bindings",
            str(bindings),
            "--admission",
            str(document),
            "--work-dir",
            str(work),
            "--json",
        ],
    )
    main()
    assert not (tmp_path / "ignored-prefix").exists()
    result = json.loads(capsys.readouterr().out)
    assert rows(result)["pack"]["state"] == "COMPLETE"
    for command in ("apply", "resume"):
        code = f'''import sys
from sparselab.cli.entry import main
sys.argv = ["sparselab", "experiment", "recipe", "preparation", "{command}", "--help"]
try:
    main()
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("help must exit")
assert not any(name in sys.modules for name in ("torch", "tokenizers", "numpy", "sparselab.campaign.engine", "sparselab.experiments.preparation_execution"))
'''
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_external_artifact_is_verified_before_store_creation(tmp_path):
    args = setup(tmp_path, reuse=True)
    context = PreparationExecution(*args)
    token = context.loaded.artifact_locations["reuse"]
    token.write_text("invalid external input")
    engine = context.engine()
    with pytest.raises(ValueError, match="provenance or digest mismatch"):
        engine.apply()
    assert not args[2].exists()


def test_new_contract_refuses_non_synthetic_but_legacy_config_still_loads(tmp_path):
    from sparselab.config.models import RunConfig

    source, bindings, work, admission = setup(tmp_path)
    config = source.parent / "run.yaml"
    raw = json.loads(config.read_text())
    raw["dataset"]["source"] = "tinystories"
    raw["dataset"]["revision"] = "fixture-pinned-revision"
    RunConfig.model_validate(raw)
    config.write_text(json.dumps(raw))
    repin(source)
    git(source.parent, "add", ".")
    git(
        source.parent,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "non-synthetic fixture",
    )
    loaded = load_preparation_recipe(source, bindings)
    updated = admission.model_copy(
        update={
            "reviewed_commit": git(source.parent, "rev-parse", "HEAD"),
            "recipe_sha256": loaded.recipe_sha256,
        }
    )
    with pytest.raises(ValueError, match="synthetic inputs only"):
        PreparationExecution(source, bindings, work, updated)
    assert not work.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("preparation_admission_version", True),
        ("preparation_admission_version", 2),
        ("max_documents", True),
        ("max_tokens", "1024"),
        ("max_tokens", float("inf")),
        ("min_free_bytes", 0),
    ],
)
def test_admission_uses_strict_new_contract(tmp_path, field, value):
    *_, admission = setup(tmp_path)
    raw = admission.model_dump(mode="python")
    raw[field] = value
    with pytest.raises(ValueError):
        PreparationAdmission.model_validate(raw)


def test_context_constructor_is_read_only_and_torch_free(tmp_path):
    source, bindings, work, admission = setup(tmp_path)
    document = tmp_path / "admission.json"
    document.write_text(admission.model_dump_json())
    code = """import json, sys
from pathlib import Path
from sparselab.experiments.preparation_execution import PreparationAdmission, PreparationExecution
source, bindings, work, document = map(Path, sys.argv[1:])
context = PreparationExecution(source, bindings, work, PreparationAdmission.model_validate_json(document.read_text()))
assert not work.exists()
assert not any(name in sys.modules for name in ("torch", "tokenizers", "numpy", "sparselab.campaign.engine"))
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(source),
            str(bindings),
            str(work),
            str(document),
        ],
        check=True,
        capture_output=True,
    )


def test_reused_tokenizer_vocabulary_must_fit_admission(tmp_path):
    source, bindings, work, admission = setup(tmp_path, reuse=True)
    smaller = admission.model_copy(update={"max_vocab_size": 260})
    with pytest.raises(ValueError, match="vocabulary limits"):
        PreparationExecution(source, bindings, work, smaller)
    assert not work.exists()


@pytest.mark.parametrize("package", ["torch", "datasets"])
def test_missing_transitive_cpu_dependency_refuses_before_work(
    tmp_path, monkeypatch, package
):
    from importlib.metadata import PackageNotFoundError

    from sparselab.experiments import preparation_execution

    args = setup(tmp_path)
    original = preparation_execution.version

    def missing(name):
        if name == package:
            raise PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(preparation_execution, "version", missing)
    with pytest.raises(ValueError, match="dependencies are unavailable"):
        PreparationExecution(*args)
    assert not args[2].exists()
