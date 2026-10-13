"""Bounded composition contracts: no new scheduler, admission or model work."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from sparselab.campaign.plan import CampaignPlan
from sparselab.campaign.preparation_plan import (
    BoundPreparationInput,
    PreparationArtifact,
    PreparationOperation,
    PreparationRequest,
    PreparationResult,
    compile_preparation,
)
from sparselab.config.models import RunConfig, TokenizerTrainConfig
from sparselab.training.manifest import canonical_json, config_sha256

FIXTURE = Path(__file__).parent / "fixtures/contracts/run_config_v2.json"


def declaration(reuse=False):
    producer = "reuse" if reuse else "train"
    upstream = (
        {
            "id": producer,
            "kind": "artifact_reference",
            "scope": "tokenizer",
            "artifact": {
                "kind": "tokenizer",
                "version": 1,
                "producer": "native",
                "identifier": "tok",
                "path": "tok/tokenizer.json",
                "sha256": "a" * 64,
            },
        }
        if reuse
        else {
            "id": producer,
            "kind": "tokenizer_train",
            "scope": "tokenizer",
            "config": "tokenizer.yaml",
        }
    )
    return {
        "campaign_version": 1,
        "id": "composition",
        "stages": [
            {
                "id": "pack",
                "kind": "data_prepare",
                "scope": "model",
                "config": "run.yaml",
                "tokenizer": producer,
                "requires": [producer],
            },
            upstream,
        ],
    }


def configs(root):
    raw = json.loads(FIXTURE.read_text())
    raw["model"]["vocab_size"] = 260
    raw["tokenizer"]["path"] = str(root / "tok/tokenizer.json")
    raw["dataset"]["cache_dir"] = str(root / "cache")
    raw["logging"]["root_dir"] = str(root / "runs")
    run = RunConfig.model_validate(raw)
    tokenizer = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        min_frequency=1,
        max_documents=8,
        output_dir=root / "tok",
        dataset=run.dataset,
    )
    return run, tokenizer


def request_for(root, kind="data_prepare"):
    operation = compile_preparation(
        CampaignPlan.model_validate(declaration())
    ).operation("pack" if kind == "data_prepare" else "train")
    run, tokenizer = configs(root)
    inputs = (
        ()
        if kind == "tokenizer_train"
        else (
            BoundPreparationInput(
                reference=operation.inputs[0],
                path=run.tokenizer.path,
                artifact=PreparationArtifact(
                    kind="tokenizer", identifier="tok", sha256="a" * 64
                ),
            ),
        )
    )
    return PreparationRequest(
        operation=operation,
        source=root / "campaign.yaml",
        config=run if kind == "data_prepare" else tokenizer,
        inputs=inputs,
    )


def test_two_graphs_share_ports_and_native_order_without_changing_declaration():
    projections = []
    for reuse in (False, True):
        plan = CampaignPlan.model_validate(declaration(reuse))
        before = canonical_json(plan.model_dump(mode="json"))
        compiled = compile_preparation(plan)
        assert compiled.ordered_stage_ids == tuple(
            stage.id for stage in plan.ordered_stages()
        )
        assert compiled.ordered_stage_ids == (
            ("reuse", "pack") if reuse else ("train", "pack")
        )
        assert canonical_json(plan.model_dump(mode="json")) == before
        pack = compiled.operation("pack")
        assert pack.inputs[0].producer == ("reuse" if reuse else "train")
        assert pack.inputs[0].kind == "tokenizer"
        assert pack.output.kind == "prepared_data"
        assert "sha256" not in pack.model_dump_json()  # Future content is not claimed.
        assert "path" not in pack.output.model_dump()
        projections.append(compiled)
    assert projections[0] != projections[1]
    with pytest.raises(ValidationError, match="frozen"):
        projections[0].operation("pack").inputs[0].producer = "other"


@pytest.mark.parametrize(
    "damage,match",
    [
        ("missing", "requires unknown"),
        ("edge", "must be in requires"),
        ("wrong-kind", "wrong producer kind"),
        ("cycle", "cycle"),
        ("duplicate", "duplicate campaign stage IDs"),
        ("version", "Input should be 1"),
    ],
)
def test_native_graph_refusals_precede_compilation(damage, match):
    raw = declaration(True)
    if damage == "missing":
        raw["stages"][0]["requires"] = ["absent"]
    elif damage == "edge":
        raw["stages"][0]["requires"] = []
    elif damage == "wrong-kind":
        raw["stages"][1]["scope"] = "model"
        raw["stages"][1]["artifact"]["kind"] = "prepared_data"
    elif damage == "cycle":
        raw["stages"][1]["requires"] = ["pack"]
    elif damage == "duplicate":
        raw["stages"].append(raw["stages"][1])
    else:
        raw["campaign_version"] = 2
    with pytest.raises(ValueError, match=match):
        compile_preparation(CampaignPlan.model_validate(raw))


@pytest.mark.parametrize(
    "damage,match",
    [
        ("kind", "requires tokenizer input"),
        ("duplicate", "duplicate preparation input port"),
        ("missing", "requires a tokenizer input"),
        ("edge", "declared dependency"),
        ("output", "requires prepared_data output"),
        ("version", "Input should be 1"),
    ],
)
def test_projection_contract_refusals(damage, match):
    raw = (
        compile_preparation(CampaignPlan.model_validate(declaration()))
        .operation("pack")
        .model_dump(mode="json")
    )
    if damage == "kind":
        raw["inputs"][0]["kind"] = "dataset_snapshot"
    elif damage == "duplicate":
        raw["inputs"].append(dict(raw["inputs"][0]))
    elif damage == "missing":
        raw["inputs"] = []
    elif damage == "edge":
        raw["dependencies"] = []
    elif damage == "output":
        raw["output"]["kind"] = "tokenizer"
    else:
        raw["operation_version"] = 2
    with pytest.raises(ValueError, match=match):
        PreparationOperation.model_validate(raw)


def test_bound_request_is_immutable_and_rejects_wrong_identity_or_path(tmp_path):
    request = request_for(tmp_path)
    assert request.config.tokenizer.path == request.inputs[0].path
    for damage, match in (
        ("path", "configured tokenizer differs"),
        ("kind", "wrong kind"),
        ("port", "bound inputs differ"),
    ):
        raw = request.model_dump(mode="python")
        if damage == "path":
            raw["inputs"][0]["path"] = tmp_path / "other"
        elif damage == "kind":
            raw["inputs"][0]["artifact"]["kind"] = "prepared_data"
        else:
            raw["inputs"] = ()
        with pytest.raises(ValueError, match=match):
            PreparationRequest.model_validate(raw)
    with pytest.raises(ValidationError, match="frozen"):
        request.inputs[0].artifact.sha256 = "b" * 64


@pytest.mark.parametrize("reuse", [False, True])
def test_engine_dispatch_uses_compiled_edges_and_preserves_result_shape(
    tmp_path, monkeypatch, reuse
):
    from sparselab.campaign import preparation
    from sparselab.campaign.engine import CampaignEngine

    raw = declaration(reuse)
    source = tmp_path / "campaign.yaml"
    source.write_text(yaml.safe_dump(raw))
    run, tokenizer = configs(tmp_path)
    for path, config in (("run.yaml", run), ("tokenizer.yaml", tokenizer)):
        (tmp_path / path).write_text(yaml.safe_dump(config.model_dump(mode="json")))
    engine = CampaignEngine(source, tmp_path / "work")
    producer = "reuse" if reuse else "train"
    artifact = {"kind": "tokenizer", "identifier": "tok", "sha256": "a" * 64}
    row = {"outputs": [artifact], "availability": {"path": str(run.tokenizer.path)}}
    seen = []

    def upstream(rows, name):
        seen.append(name)
        assert name == producer
        return row

    monkeypatch.setattr(engine, "_upstream", upstream)
    monkeypatch.setattr(engine, "_verification", dict)
    captured = []

    def dispatch(request, **options):
        captured.append(request)
        assert request.operation.inputs[0].producer == producer
        assert request.inputs[0].artifact.sha256 == "a" * 64
        assert config_sha256(request.config.model_dump(mode="json")) == config_sha256(
            run.model_dump(mode="json")
        )
        return PreparationResult(
            path=tmp_path / "cache/packed",
            artifact=PreparationArtifact(
                kind="prepared_data", identifier="packed", sha256="b" * 64
            ),
        )

    monkeypatch.setattr(preparation, "dispatch", dispatch)
    # Dispatch cannot rediscover a different input or config from a caller's
    # stage object; the compiled stage ID selects the immutable operation.
    decoy = engine.stages["pack"].model_copy(
        update={"tokenizer": "decoy", "config": "missing.yaml"}
    )
    result = CampaignEngine.dispatch.__wrapped__(
        engine, decoy, {"pack": {"stage_input_sha256": "c" * 64}}, 0
    )
    assert seen == [producer]
    assert result == {
        "state": "COMPLETE",
        "outcome": "READY_FOR_NEXT_STAGE",
        "reason": "verified",
        "deficits": [],
        "outputs": [
            {"kind": "prepared_data", "identifier": "packed", "sha256": "b" * 64}
        ],
        "availability": {"path": str(tmp_path / "cache/packed")},
    }
    row["outputs"][0]["sha256"] = "d" * 64
    assert captured[0].inputs[0].artifact.sha256 == "a" * 64
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("kind", ["tokenizer_train", "data_prepare"])
def test_adapter_invokes_only_native_computation_and_verification(
    tmp_path, monkeypatch, kind
):
    from sparselab.campaign import preparation
    from sparselab.data import packing, tokenizer

    request = request_for(tmp_path, kind)
    calls = []
    path = tmp_path / (
        "tok/tokenizer.json" if kind == "tokenizer_train" else "cache/packed"
    )
    path.parent.mkdir(parents=True)
    if kind == "tokenizer_train":
        path.write_text("tokenizer fixture")
    else:
        path.mkdir()
        (path / "manifest.json").write_text(
            json.dumps({"settings_sha256": "b" * 64, "manifest_sha256": "c" * 64})
        )
    token = object()
    monkeypatch.setattr(
        tokenizer,
        "load_tokenizer",
        lambda bound: (
            token
            if bound == request.config.tokenizer.path
            else pytest.fail("wrong tokenizer")
        ),
    )

    def train(config):
        calls.append("train")
        assert config == request.config
        return path

    def prepare(config, loaded, **kwargs):
        calls.append("pack")
        assert config == request.config and loaded is token
        assert kwargs == {
            "observer": "observer",
            "proof_store": None,
            "verification_mode": "cold",
        }
        return SimpleNamespace(root=path)

    monkeypatch.setattr(tokenizer, "train_tokenizer", train)
    monkeypatch.setattr(packing, "prepare_data", prepare)

    def verify(artifact, source, **kwargs):
        calls.append("verify")
        assert source == request.source and kwargs["dataset"] == request.config.dataset
        assert artifact.kind == request.operation.output.kind
        assert artifact.producer == f"sparselab {kind}"
        return artifact.model_dump(mode="json")

    monkeypatch.setattr(preparation, "verify_artifact", verify)
    result = preparation.dispatch(request, observer="observer")
    preparation.verify(request, result)
    assert calls == (["train"] if kind == "tokenizer_train" else ["pack"]) + [
        "verify",
        "verify",
    ]


@pytest.mark.parametrize("reuse", [False, True])
def test_fresh_process_compilation_has_no_optional_imports_or_effects(tmp_path, reuse):
    source = tmp_path / "campaign.yaml"
    source.write_text(yaml.safe_dump(declaration(reuse)))
    code = """
import importlib.abc
import os
import sys
class Fence(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','mlx','datasets','tokenizers','huggingface_hub'} or fullname in {'sparselab.campaign.engine','sparselab.campaign.state','sparselab.campaign.preparation','sparselab.data','sparselab.runtime','sparselab.corpus.release','sparselab.corpus.token_denominator'}:
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, Fence())
def guard(event,args):
    if event in {'subprocess.Popen','socket.connect','os.system','os.mkdir','os.rename','os.remove'}:
        raise AssertionError('unexpected effect: '+event)
    if event == 'open':
        path,mode,flags=args
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT):
            raise AssertionError('write: '+str(path))
        if str(path).endswith(('run.yaml','tokenizer.yaml','tokenizer.json')):
            raise AssertionError('config/artifact resolution during compile')
sys.addaudithook(guard)
from sparselab.campaign.plan import load_campaign
from sparselab.campaign.preparation_plan import compile_preparation
result=compile_preparation(load_campaign(__import__('pathlib').Path(sys.argv[1])))
assert result.operation('pack').output.kind == 'prepared_data'
assert 'sparselab.data' not in sys.modules and 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(source)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_preparation_adapter_has_no_graph_or_downstream_dependency():
    source = Path(__file__).parents[1] / "src/sparselab/campaign/preparation.py"
    tree = ast.parse(source.read_text())
    forbidden = {
        "sparselab.campaign.engine",
        "sparselab.campaign.state",
        "sparselab.workers",
        "sparselab.training.trainer",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not any((node.module or "").startswith(name) for name in forbidden)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert not {"engine", "stage", "rows"}.intersection(
                arg.arg for arg in node.args.args
            )


def test_native_plan_declaration_hashes_are_unchanged(tmp_path):
    from sparselab.campaign.state import CampaignStore

    # Captured with CampaignPlan from merged b264f41, before this projection.
    expected = {
        False: "da902f5c7c60c4a6de1f858d8b3ef8b261a4eef68451d731ae1d4aba6b2ca5fd",
        True: "889ac318bedec1b9d7a0b41e38690bd492550104ca1a69ca7a3b9bf0101d03da",
    }
    for reuse, digest in expected.items():
        plan = CampaignPlan.model_validate(declaration(reuse))
        compiled = compile_preparation(plan)
        assert compiled.operations
        assert CampaignStore(plan, tmp_path).declaration_sha == digest
    assert list(tmp_path.iterdir()) == []


def test_snapshot_ports_bind_only_the_declared_manifest(tmp_path):
    raw = declaration()
    raw["stages"].append(
        {
            "id": "source",
            "kind": "dataset_snapshot",
            "scope": "corpus",
            "lock": "source.lock.json",
            "output": "snapshot",
            "cache_dir": "hub",
        }
    )
    for stage in raw["stages"][:2]:
        stage["snapshot"] = "source"
        stage.setdefault("requires", []).append("source")
    compiled = compile_preparation(CampaignPlan.model_validate(raw))
    assert compiled.ordered_stage_ids == ("source", "train", "pack")
    _, config = configs(tmp_path)
    dataset = config.dataset.model_dump(mode="python")
    dataset.update(
        source="snapshot",
        revision="a" * 40,
        license="MIT",
        train_path=tmp_path / "snapshot/train.jsonl",
        validation_path=tmp_path / "snapshot/validation.jsonl",
        source_manifest_path=tmp_path / "snapshot/manifest.json",
    )
    config = TokenizerTrainConfig.model_validate(
        {**config.model_dump(mode="python"), "dataset": dataset}
    )
    operation = compiled.operation("train")
    bound = BoundPreparationInput(
        reference=operation.inputs[0],
        path=dataset["source_manifest_path"],
        artifact=PreparationArtifact(
            kind="dataset_snapshot", identifier="snapshot", sha256="a" * 64
        ),
    )
    request = PreparationRequest(
        operation=operation,
        source=tmp_path / "campaign.yaml",
        config=config,
        inputs=(bound,),
    )
    assert request.inputs[0].reference.producer == "source"
    with pytest.raises(ValueError, match="configured dataset differs"):
        PreparationRequest(
            operation=operation,
            source=request.source,
            config=config,
            inputs=(bound.model_copy(update={"path": tmp_path / "decoy.json"}),),
        )


def test_tiny_native_train_and_reuse_graphs_have_equal_preparation_and_replay(
    tmp_path, monkeypatch
):
    from sparselab.campaign.engine import CampaignEngine
    from sparselab.data.tokenizer import train_tokenizer
    from sparselab.training.manifest import sha256_file

    outputs = []
    for reuse in (False, True):
        root = tmp_path / ("reuse" if reuse else "train")
        root.mkdir()
        run, tokenizer = configs(root)
        raw = declaration(reuse)
        if reuse:
            path = train_tokenizer(tokenizer)
            raw["stages"][1]["artifact"]["sha256"] = sha256_file(path)
        for name, config in (("run.yaml", run), ("tokenizer.yaml", tokenizer)):
            (root / name).write_text(yaml.safe_dump(config.model_dump(mode="json")))
        source = root / "campaign.yaml"
        source.write_text(yaml.safe_dump(raw))
        engine = CampaignEngine(source, root / "work")
        result = engine.apply(allow_uncommitted_declaration=True)
        rows = {row["id"]: row for row in result["stages"]}
        assert rows["pack"]["state"] == "COMPLETE", result
        outputs.append(rows["pack"]["outputs"])
        monkeypatch.setattr(
            engine,
            "dispatch",
            lambda *a, **kw: pytest.fail("completed stage redispatched"),
        )
        replay = engine.apply(allow_uncommitted_declaration=True)
        assert {row["id"]: row for row in replay["stages"]}["pack"]["outputs"] == rows[
            "pack"
        ]["outputs"]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("kind", ["tokenizer_train", "data_prepare"])
def test_replay_rejects_reverified_identity_drift(tmp_path, monkeypatch, kind):
    from sparselab.campaign import preparation

    request = request_for(tmp_path, kind)
    recorded = PreparationResult(
        path=tmp_path
        / ("tok/tokenizer.json" if kind == "tokenizer_train" else "cache/packed"),
        artifact=PreparationArtifact(
            kind=request.operation.output.kind, identifier="recorded", sha256="a" * 64
        ),
    )
    calls = []

    def reverify(artifact, source, **options):
        calls.append(artifact)
        assert source == request.source
        assert artifact.sha256 == recorded.artifact.sha256
        assert options["dataset"] == request.config.dataset
        return {**artifact.model_dump(mode="json"), "sha256": "b" * 64}

    monkeypatch.setattr(preparation, "verify_artifact", reverify)
    with pytest.raises(ValueError, match="^preparation output identity changed$"):
        preparation.verify(request, recorded)
    assert len(calls) == 1
    assert recorded.artifact.sha256 == "a" * 64
    assert not recorded.path.exists()


def test_replay_rejects_prepared_location_before_artifact_verification(
    tmp_path, monkeypatch
):
    from sparselab.campaign import preparation

    request = request_for(tmp_path)
    recorded = PreparationResult(
        path=tmp_path / "outside/packed",
        artifact=PreparationArtifact(
            kind="prepared_data", identifier="recorded", sha256="a" * 64
        ),
    )
    monkeypatch.setattr(
        preparation,
        "verify_artifact",
        lambda *a, **kw: pytest.fail("artifact opened before location validation"),
    )
    with pytest.raises(ValueError, match="^prepared output differs from declaration$"):
        preparation.verify(request, recorded)
    assert not recorded.path.exists()


@pytest.mark.parametrize("count", [0, 2])
def test_engine_request_rejects_upstream_output_cardinality(
    tmp_path, monkeypatch, count
):
    from sparselab.campaign.engine import CampaignEngine

    source = tmp_path / "campaign.yaml"
    source.write_text(yaml.safe_dump(declaration(True)))
    run, _ = configs(tmp_path)
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(run.model_dump(mode="json")))
    engine = CampaignEngine(source, tmp_path / "work")
    artifact = {"kind": "tokenizer", "identifier": "tok", "sha256": "a" * 64}
    row = {
        "outputs": [dict(artifact) for _ in range(count)],
        "availability": {"path": str(run.tokenizer.path)},
    }
    rows = {"reuse": row}
    calls = []

    def upstream(actual_rows, producer):
        assert actual_rows is rows
        calls.append(producer)
        return row

    # Isolate the request boundary from the upstream verifier's own refusals.
    # Both artifacts are otherwise valid, so accepting the first is observable.
    monkeypatch.setattr(engine, "_upstream", upstream)
    with pytest.raises(
        ValueError, match="^preparation input requires one immutable output$"
    ):
        engine._preparation_request("pack", rows)
    assert calls == ["reuse"]
    assert not engine.work_dir.exists()


def test_campaign_without_preparation_loads_without_resolving_artifacts(tmp_path):
    from sparselab.campaign.engine import CampaignEngine

    raw = declaration(True)
    raw["stages"] = [raw["stages"][1]]
    source = tmp_path / "campaign.yaml"
    source.write_text(yaml.safe_dump(raw))
    engine = CampaignEngine(source, tmp_path / "work")
    assert engine.preparation_plan.operations == ()
    assert engine.preparation_plan.ordered_stage_ids == ("reuse",)
    assert tuple(stage.id for stage in engine.plan.ordered_stages()) == ("reuse",)
    assert set(engine.stages) == {"reuse"}
    assert not engine.work_dir.exists()
    assert not (tmp_path / "tok/tokenizer.json").exists()
