from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from sparselab.campaign.plan import CampaignPlan, load_campaign


def declaration() -> dict:
    return {
        "campaign_version": 1,
        "id": "tiny",
        "stages": [
            {
                "id": "corpus",
                "kind": "corpus_release",
                "scope": "corpus",
                "project": "recipe",
            },
            {
                "id": "ready",
                "kind": "corpus_readiness",
                "scope": "corpus",
                "requires": ["corpus"],
                "corpus": "corpus",
                "policy": {"min_heldout_families": 1},
            },
            {
                "id": "gate",
                "kind": "approval",
                "scope": "release",
                "requires": ["ready"],
                "bind": ["corpus", "ready"],
            },
        ],
    }


@pytest.mark.parametrize(
    "suffix,document",
    [
        ("yaml", "campaign_version: 1\ncampaign_version: 1\n"),
        ("json", '{"campaign_version":1,"campaign_version":1}'),
        ("yaml", "campaign_version: !!python/object/apply:os.system ['true']"),
        ("json", '{"campaign_version":NaN}'),
        ("yaml", "campaign_version: .nan"),
    ],
)
def test_reject_ambiguous_documents(tmp_path: Path, suffix: str, document: str) -> None:
    path = tmp_path / f"campaign.{suffix}"
    path.write_text(document)
    with pytest.raises((ValueError, TypeError)):
        load_campaign(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_id",
        "duplicate_requirement",
        "unknown_requirement",
        "self_dependency",
        "cycle",
        "missing_typed_requirement",
        "wrong_kind",
        "wrong_scope",
        "unknown_field",
        "absolute_path",
        "parent_path",
        "nan",
        "empty_policy",
    ],
)
def test_reject_invalid_dag(tmp_path: Path, mutation: str) -> None:
    value = copy.deepcopy(declaration())
    corpus, ready, gate = value["stages"]
    if mutation == "duplicate_id":
        gate["id"] = ready["id"]
    elif mutation == "duplicate_requirement":
        ready["requires"] = ["corpus", "corpus"]
    elif mutation == "unknown_requirement":
        ready["requires"] = ["missing"]
    elif mutation == "self_dependency":
        ready["requires"] = ["ready"]
    elif mutation == "cycle":
        corpus["requires"] = ["gate"]
    elif mutation == "missing_typed_requirement":
        ready["requires"] = []
    elif mutation == "wrong_kind":
        ready["corpus"] = "gate"
        ready["requires"] = ["gate"]
        gate["requires"] = ["corpus"]
        gate["bind"] = ["corpus"]
    elif mutation == "wrong_scope":
        corpus["scope"] = "model"
    elif mutation == "unknown_field":
        value["execute"] = "arbitrary code"
    elif mutation == "absolute_path":
        corpus["project"] = "/etc"
    elif mutation == "parent_path":
        corpus["project"] = "../recipe"
    elif mutation == "nan":
        ready["policy"]["passes"] = {
            "basis": "bytes",
            "requested_total": 1,
            "mixture": {"developer": float("nan")},
            "max_required": 1,
        }
    elif mutation == "empty_policy":
        ready["policy"] = {}
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises((ValueError, TypeError)):
        load_campaign(path)


def test_symlink_path_rejected(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "recipe").symlink_to(tmp_path / "real", target_is_directory=True)
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(declaration()))
    with pytest.raises(ValueError, match="symlink"):
        load_campaign(path)


def test_operational_artifact_accepts_external_absolute_location(
    tmp_path: Path,
) -> None:
    from sparselab.campaign.engine import CampaignEngine

    artifact = {
        "kind": "tokenizer",
        "version": 1,
        "producer": "fixture",
        "identifier": "selected",
        "sha256": "a" * 64,
        "path": str(tmp_path / "external-state/tokenizer/tokenizer.json"),
    }
    source = tmp_path / "campaign.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "campaign_version": 1,
                "id": "external-input",
                "stages": [
                    {
                        "id": "tokenizer",
                        "kind": "tokenizer_reference",
                        "scope": "tokenizer",
                        "artifact": artifact,
                    }
                ],
            }
        )
    )
    assert load_campaign(source).stages[0].artifact.path == artifact["path"]
    work = tmp_path / "workspace"
    status = CampaignEngine(source, work).inspect("status")
    assert by_id(status)["tokenizer"]["state"] == "BLOCKED"
    assert artifact["path"] in by_id(status)["tokenizer"]["reason"]
    assert not work.exists()

    artifact["path"] = str(tmp_path / "external-state/../escape.json")
    source.write_text(
        yaml.safe_dump(
            {
                "campaign_version": 1,
                "id": "external-input",
                "stages": [
                    {
                        "id": "tokenizer",
                        "kind": "tokenizer_reference",
                        "scope": "tokenizer",
                        "artifact": artifact,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="unsafe operational path"):
        load_campaign(source)


def test_author_order_and_schema(tmp_path: Path) -> None:
    value = declaration()
    value["stages"] = [value["stages"][2], value["stages"][1], value["stages"][0]]
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(value))
    plan = load_campaign(path)
    assert [stage.id for stage in plan.ordered_stages()] == ["corpus", "ready", "gate"]
    schema = (
        Path(__file__).resolve().parents[1] / "schemas/campaign-plan-v1.schema.json"
    )
    assert json.loads(schema.read_text()) == CampaignPlan.model_json_schema()


def make_full_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Real synthetic model inputs, independent of the descriptive corpus release."""
    import shutil

    from sparselab.config.models import DatasetConfig, RunConfig, TokenizerTrainConfig
    from sparselab.data.packing import prepare_data
    from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
    from sparselab.training.manifest import sha256_file
    from sparselab.workdir import ensure_work_dir

    root = Path(__file__).resolve().parents[1]
    recipe = tmp_path / "recipe"
    shutil.copytree(root / "examples/tiny-campaign", recipe)
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    ensure_work_dir(tmp_path / "work")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    dataset = DatasetConfig(
        source="synthetic",
        cache_dir=tmp_path / "data",
        synthetic_seed=7,
        train_max_documents=100,
        validation_max_documents=100,
        train_max_tokens=129,
        validation_max_tokens=81,
    )
    tokenizer = train_tokenizer(
        TokenizerTrainConfig(
            schema_version=1,
            vocab_size=260,
            max_documents=100,
            output_dir=tmp_path / "tokenizer",
            dataset=dataset,
        )
    )
    raw = yaml.safe_load((root / "configs/runtime_smoke_cpu.yaml").read_text())
    raw["name"] = "campaign-fixture"
    raw["model"]["vocab_size"] = 260
    raw["tokenizer"]["path"] = str(tokenizer)
    raw["dataset"] = dataset.model_dump(mode="json")
    raw["training"].update(gradient_accumulation=1, max_steps=2, max_tokens=32)
    raw["optimizer"]["warmup_steps"] = 1
    raw["checkpoint"]["every_steps"] = 1
    raw["evaluation"]["every_steps"] = 1
    raw["logging"]["root_dir"] = str(tmp_path / "work/runs")
    config = RunConfig.model_validate(raw)
    prepared = prepare_data(config, load_tokenizer(tokenizer))
    manifest = json.loads((prepared.root / "manifest.json").read_text())
    tokenizer_artifact = {
        "kind": "tokenizer",
        "version": 1,
        "producer": "fixture",
        "identifier": "tokenizer",
        "sha256": sha256_file(tokenizer),
        "path": str(tokenizer.relative_to(tmp_path)),
    }
    prepared_artifact = {
        "kind": "prepared_data",
        "version": 1,
        "producer": "fixture",
        "identifier": manifest["settings_sha256"],
        "sha256": manifest["manifest_sha256"],
        "path": str(prepared.root.relative_to(tmp_path)),
    }
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")))
    (tmp_path / "evaluation-suite.yaml").write_text(
        yaml.safe_dump(
            {
                "evaluation_suite_version": 1,
                "id": "fixture-heldout",
                "evaluations": [
                    {"id": "heldout", "role": "gate", "kind": "heldout_lm"}
                ],
            }
        )
    )
    (tmp_path / "model-readiness.yaml").write_text(
        yaml.safe_dump(
            {
                "readiness_version": 1,
                "id": "fixture-readiness",
                "require_verified_checkpoint": True,
                "required_gate_ids": ["heldout"],
                "min_completed_evaluations": 1,
                "max_heldout_loss": {"evaluation_id": "heldout", "value": 100.0},
                "require_human_review": False,
            }
        )
    )
    experiment = {
        "plan_version": 1,
        "id": "fixture",
        "base_run": "run.yaml",
        "evaluation_suite": "evaluation-suite.yaml",
        "artifacts": {"tokenizer": tokenizer_artifact, "prepared": prepared_artifact},
        "inputs": {"tokenizer": "tokenizer", "prepared_data": "prepared"},
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment))
    stages = declaration()["stages"][:2]
    stages[0]["project"] = "recipe/corpus.yaml"
    stages.extend(
        [
            {
                "id": "tokenizer",
                "kind": "tokenizer_reference",
                "scope": "tokenizer",
                "requires": ["ready"],
                "artifact": tokenizer_artifact,
            },
            {
                "id": "prepared",
                "kind": "artifact_reference",
                "scope": "model",
                "artifact": prepared_artifact,
            },
            {
                "id": "measurement",
                "kind": "token_measurement",
                "scope": "tokenizer",
                "requires": ["corpus", "tokenizer"],
                "corpus": "corpus",
                "tokenizer": "tokenizer",
            },
            {
                "id": "plan",
                "kind": "experiment_plan",
                "scope": "model",
                "requires": ["tokenizer", "prepared", "measurement"],
                "source": "experiment.yaml",
                "mode": "lock",
                "tokenizer": "tokenizer",
                "prepared": "prepared",
            },
            {
                "id": "runtime",
                "kind": "runtime_acceptance",
                "scope": "runtime",
                "requires": ["plan"],
                "plan": "plan",
            },
            {
                "id": "gate",
                "kind": "approval",
                "scope": "model",
                "requires": ["runtime"],
                "bind": [
                    "corpus",
                    "ready",
                    "tokenizer",
                    "prepared",
                    "measurement",
                    "plan",
                    "runtime",
                ],
            },
            {
                "id": "run",
                "kind": "experiment_run",
                "scope": "model",
                "requires": ["plan", "runtime", "gate"],
                "plan": "plan",
                "runtime": "runtime",
                "cell": "main:single",
            },
            {
                "id": "collect",
                "kind": "experiment_collect",
                "scope": "evaluation",
                "requires": ["plan", "run"],
                "plan": "plan",
                "run": "run",
            },
            {
                "id": "evaluation",
                "kind": "evaluation",
                "scope": "evaluation",
                "requires": ["collect"],
                "collect": "collect",
                "suite": "evaluation-suite.yaml",
            },
            {
                "id": "model",
                "kind": "model_readiness",
                "scope": "model",
                "requires": ["evaluation"],
                "evaluation": "evaluation",
                "policy": "model-readiness.yaml",
            },
        ]
    )
    path = tmp_path / "campaign.yaml"
    path.write_text(
        yaml.safe_dump(
            {"campaign_version": 1, "id": "fixture", "stages": stages}, sort_keys=False
        )
    )
    return path


def frozen_corpus(tmp_path: Path) -> Path:
    import shutil

    from sparselab.corpus.acquisition import acquire
    from sparselab.corpus.pipeline import build
    from sparselab.corpus.project import load_project
    from sparselab.corpus.release import freeze

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "examples/tiny-campaign", tmp_path / "recipe")
    project = load_project(tmp_path / "recipe/corpus.yaml")
    acquire(project, tmp_path / "work", offline=False)
    return freeze(build(project, tmp_path / "work", offline=True), tmp_path / "work")


def test_verified_readiness_denominator_and_deficits(tmp_path: Path) -> None:
    from sparselab.campaign.policy import CorpusReadinessPolicy, measure_readiness

    release = frozen_corpus(tmp_path)
    documents = [
        json.loads(line)
        for line in (release / "documents.jsonl").read_text().splitlines()
    ]
    expected = {}
    for domain in ("technical_docs", "developer"):
        unique = {
            row["content_sha256"]: row["text"]
            for row in documents
            if row["split"] == "train"
            and row["drop_reason"] is None
            and domain in row["domains"]
        }
        expected[domain] = sum(len(text.encode()) for text in unique.values())
    result = measure_readiness(
        release,
        CorpusReadinessPolicy(
            min_unique_train_bytes_by_domain=expected,
            required_nonzero_languages=("en",),
            required_nonzero_shapes=("raw_document",),
            min_heldout_families=1,
        ),
    )
    assert result["state"] == "COMPLETE"
    assert result["measurements"]["unique_train_bytes_by_domain"] == expected
    assert result["measurements"]["heldout_families"] == 1
    blocked = measure_readiness(
        release,
        CorpusReadinessPolicy(
            min_unique_train_bytes_by_domain={"absent": 1},
            required_nonzero_languages=("xx",),
            min_heldout_families=2,
        ),
    )
    assert blocked["state"] == "BLOCKED"
    assert blocked["outcome"] == "EXPAND_MORE"
    assert blocked["deficits"] == [
        {"dimension": "heldout_families", "observed": 1, "required": 2},
        {"dimension": "train_language", "observed": 0, "required": 1, "domain": "xx"},
        {
            "dimension": "unique_train_bytes",
            "observed": None,
            "required": 1,
            "domain": "absent",
        },
    ]


def test_source_pass_exact_ceil_and_zero_availability(tmp_path: Path) -> None:
    from sparselab.campaign.policy import CorpusReadinessPolicy, measure_readiness

    release = frozen_corpus(tmp_path)
    facts = measure_readiness(release, CorpusReadinessPolicy(min_heldout_families=1))[
        "measurements"
    ]
    amount = facts["unique_train_bytes_by_domain"]["developer"]
    for requested, expected in ((amount * 2, 1), (amount * 2 + 1, 2)):
        result = measure_readiness(
            release,
            CorpusReadinessPolicy(
                passes={
                    "basis": "bytes",
                    "requested_total": requested,
                    "mixture": {"developer": 0.5, "technical_docs": 0.5},
                    "max_required": 1,
                }
            ),
        )
        assert (
            result["measurements"]["projected_source_passes_by_domain"]["developer"]
            == expected
        )
        assert any(d["domain"] == "developer" for d in result["deficits"]) == (
            expected > 1
        )
    blocked = measure_readiness(
        release,
        CorpusReadinessPolicy(
            passes={
                "basis": "bytes",
                "requested_total": 1,
                "mixture": {"missing": 1.0},
                "max_required": 1,
            }
        ),
    )
    assert blocked["deficits"] == [
        {"dimension": "passes", "observed": None, "required": 1, "domain": "missing"}
    ]


def test_readiness_actual_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign.policy import CorpusReadinessPolicy, measure_readiness
    from sparselab.config.models import DatasetConfig, TokenizerTrainConfig
    from sparselab.data.tokenizer import load_tokenizer, train_tokenizer

    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    release = frozen_corpus(tmp_path)
    path = train_tokenizer(
        TokenizerTrainConfig(
            schema_version=1,
            vocab_size=260,
            max_documents=100,
            output_dir=tmp_path / "tokenizer",
            dataset=DatasetConfig(
                source="synthetic",
                cache_dir=tmp_path / "data",
                synthetic_seed=7,
                train_max_documents=100,
                validation_max_documents=100,
                train_max_tokens=129,
                validation_max_tokens=81,
            ),
        )
    )
    tokenizer = load_tokenizer(path)
    docs = [
        json.loads(line)
        for line in (release / "documents.jsonl").read_text().splitlines()
    ]
    expected = sum(
        len(tokenizer.encode(row["text"]).ids)
        for row in docs
        if row["split"] == "train"
        and row["drop_reason"] is None
        and "developer" in row["domains"]
    )
    policy = CorpusReadinessPolicy(
        min_unique_train_tokens_by_domain={"developer": expected}
    )
    assert measure_readiness(release, policy)["state"] == "BLOCKED"
    result = measure_readiness(release, policy, path)
    assert result["state"] == "COMPLETE"
    assert (
        result["measurements"]["unique_train_tokens_by_domain"]["developer"] == expected
    )


def by_id(result: dict) -> dict:
    return {stage["id"]: stage for stage in result["stages"]}


def invoke_cli(
    source: Path, work: Path, command: str, *arguments: str, expected_code: int = 0
) -> dict:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sparselab",
            "--work-dir",
            str(work),
            "campaign",
            command,
            str(source),
            *arguments,
            *(
                ("--allow-uncommitted-declaration",)
                if command in {"apply", "resume", "reconstruct"}
                else ()
            ),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    envelope = json.loads(result.stdout)
    assert envelope["format"] == "sparselab-campaign-command-v1"
    assert envelope["command"] == command
    return envelope


def test_engine_full_cpu_and_no_redispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign.engine import CampaignEngine
    from sparselab.evaluation.evidence import experiment_evidence
    from sparselab.workers.controller import Controller

    path = make_full_campaign(tmp_path, monkeypatch)
    engine = CampaignEngine(path, tmp_path / "work")
    waiting = by_id(invoke_cli(path, tmp_path / "work", "apply"))
    assert waiting["gate"]["state"] == "AWAITING_APPROVAL"
    assert waiting["run"]["state"] == "BLOCKED"
    assert waiting["corpus"]["outputs"][0]["sha256"]
    assert waiting["tokenizer"]["outputs"][0]["sha256"]
    assert waiting["plan"]["outputs"][0]["sha256"]
    invoke_cli(
        path,
        tmp_path / "work",
        "approve",
        "gate",
        "--note",
        "fixture-only wiring acceptance",
    )
    stopped = invoke_cli(path, tmp_path / "work", "apply")
    assert stopped["next_action"]["action"] == "execute_run"
    assert by_id(stopped)["run"]["state"] == "READY"
    running = invoke_cli(
        path,
        tmp_path / "work",
        "apply",
        "--execute-runs",
        "--max-wait-seconds",
        "0",
    )
    assert by_id(running)["run"]["state"] == "RUNNING"
    result = invoke_cli(path, tmp_path / "work", "resume", "--max-wait-seconds", "600")
    rows = by_id(result)
    assert {row["state"] for row in rows.values()} == {"COMPLETE"}
    evidence = experiment_evidence(Path(rows["run"]["availability"]["path"]))
    assert evidence["evidence_level"] == "checkpointed_held_out"
    assert (
        max(observation["step"] for observation in evidence["quality_observations"])
        == 2
    )
    assert rows["model"]["outcome"] == "READY_FOR_NEXT_STAGE"
    from sparselab.evaluation.readiness import verify_readiness_result
    from sparselab.evaluation.suite import verify_evaluation_index

    index = verify_evaluation_index(Path(rows["evaluation"]["availability"]["path"]))
    readiness = verify_readiness_result(Path(rows["model"]["availability"]["path"]))
    assert readiness["index_sha256"] == index["index_sha256"]
    assert readiness["state"] == "READY_FOR_NEXT_STAGE"
    assert index["checkpoint_sha256"] == readiness["checkpoint_sha256"]
    controller = Controller(
        Path(rows["run"]["availability"]["workspace"]) / "controller", read_only=True
    )
    attempts = controller.list_experiments()
    assert len(attempts) == 1
    assert attempts[0]["status"] == attempts[0]["ingestion_status"] == "COMPLETE"
    assert attempts[0]["spec"]["declaration_provenance"]["status"] == "UNKNOWN"
    assert (
        attempts[0]["spec"]["declaration_provenance"]["allow_uncommitted_declaration"]
        is True
    )
    assert rows["run"]["declaration_provenance"]["status"] == "UNKNOWN"
    receipt = next((engine.store.root / "receipts/run").glob("*.json"))
    assert (
        json.loads(receipt.read_text())["data"]["declaration_provenance"][
            "allow_uncommitted_declaration"
        ]
        is True
    )
    before = {
        str(p.relative_to(engine.store.root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in engine.store.root.rglob("*.json")
        if "receipts" in p.parts
    }
    index_before = (engine.store.root / "state.json").read_bytes()
    invoke_cli(path, tmp_path / "work", "apply")
    assert (engine.store.root / "state.json").read_bytes() == index_before
    assert before == {
        str(p.relative_to(engine.store.root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in engine.store.root.rglob("*.json")
        if "receipts" in p.parts
    }
    assert [r["attempt_id"] for r in controller.list_experiments()] == [
        r["attempt_id"] for r in attempts
    ]
    assert (
        sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 100_000_000
    )


def test_evaluation_stays_bound_to_collected_checkpoint_when_latest_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign.engine import CampaignEngine
    from sparselab.evaluation.readiness import issue_review, verify_readiness_result
    from sparselab.evaluation.suite import verify_evaluation_index
    from sparselab.training.manifest import canonical_json

    source = make_full_campaign(tmp_path, monkeypatch)
    value = yaml.safe_load(source.read_text())
    next(stage for stage in value["stages"] if stage["id"] == "model")["review"] = (
        "model-review.json"
    )
    source.write_text(yaml.safe_dump(value))
    policy_path = tmp_path / "model-readiness.yaml"
    policy = yaml.safe_load(policy_path.read_text())
    policy["require_human_review"] = True
    policy_path.write_text(yaml.safe_dump(policy))
    work = tmp_path / "work"

    def after_commit(stage_id: str, state: dict) -> None:
        if stage_id == "collect":
            raise KeyboardInterrupt("pause after immutable collection")

    engine = CampaignEngine(source, work, after_commit=after_commit)
    engine.apply(allow_uncommitted_declaration=True)
    engine.approve("gate")
    with pytest.raises(KeyboardInterrupt, match="pause after immutable collection"):
        engine.apply(
            max_wait_seconds=600,
            execute_runs=True,
            allow_uncommitted_declaration=True,
        )
    collected = by_id(engine.inspect("status"))["collect"]["measurements"]

    def after_evaluation(stage_id: str, state: dict) -> None:
        if stage_id == "evaluation":
            raise KeyboardInterrupt("pause before named model review")

    with pytest.raises(KeyboardInterrupt, match="pause before named model review"):
        CampaignEngine(source, work, after_commit=after_evaluation).apply(
            resume=True, allow_uncommitted_declaration=True
        )
    index_path = Path(
        by_id(engine.inspect("status"))["evaluation"]["availability"]["path"]
    )
    index = verify_evaluation_index(index_path)
    assert index["checkpoint"] == f"checkpoints/{collected['generation']}"
    assert index["checkpoint_sha256"] == collected["sha256"]
    review = issue_review(
        index_path,
        reviewer="fixture-reviewer",
        decision="approve",
        note="I reviewed this exact heldout index",
        output=tmp_path / "model-review.json",
    )
    result = CampaignEngine(source, work).apply(
        resume=True, allow_uncommitted_declaration=True
    )
    rows = by_id(result)
    readiness = verify_readiness_result(Path(rows["model"]["availability"]["path"]))
    assert readiness["review_sha256"] == review["receipt_sha256"]
    assert rows["model"]["outcome"] == "READY_FOR_NEXT_STAGE"
    run = Path(by_id(engine.inspect("status"))["run"]["availability"]["path"])
    earliest = min(
        (path for path in (run / "checkpoints").glob("step_*_gen_*")),
        key=lambda path: json.loads((path / "manifest.json").read_text())["step"],
    )
    assert earliest.name != collected["generation"]
    pointer = run / "checkpoints/latest.json"
    value = json.loads(pointer.read_text())
    value["relative_path"] = earliest.name
    value["manifest_sha256"] = json.loads((earliest / "manifest.json").read_text())[
        "sha256"
    ]
    pointer.write_bytes(canonical_json(value) + b"\n")
    assert (
        verify_evaluation_index(index_path)["checkpoint_sha256"] == collected["sha256"]
    )
    with pytest.raises((ValueError, OSError)):
        CampaignEngine(source, work).inspect("plan")


def test_crash_after_commit_replays_receipt_not_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from sparselab.campaign.engine import CampaignEngine

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "examples/tiny-campaign", tmp_path / "recipe")
    path = tmp_path / "campaign.yaml"
    value = declaration()
    value["stages"][0]["project"] = "recipe/corpus.yaml"
    path.write_text(yaml.safe_dump(value))
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))

    def interrupt(stage_id: str, state: dict) -> None:
        if stage_id == "corpus":
            raise KeyboardInterrupt("controlled post-commit crash")

    engine = CampaignEngine(path, tmp_path / "work", after_commit=interrupt)
    with pytest.raises(KeyboardInterrupt):
        engine.apply(allow_uncommitted_declaration=True)
    receipts = list((engine.store.root / "receipts/corpus").glob("*.json"))
    assert len(receipts) == 1
    before = (receipts[0].read_bytes(), receipts[0].stat().st_mtime_ns)
    recovered = CampaignEngine(path, tmp_path / "work")
    original_dispatch = recovered.dispatch
    dispatched = []

    def recording_dispatch(stage, rows, max_wait_seconds, *, execute_runs=False):
        dispatched.append(stage.id)
        return original_dispatch(
            stage, rows, max_wait_seconds, execute_runs=execute_runs
        )

    monkeypatch.setattr(recovered, "dispatch", recording_dispatch)
    resumed = by_id(recovered.apply(resume=True, allow_uncommitted_declaration=True))
    assert "corpus" not in dispatched
    assert resumed["ready"]["state"] == "COMPLETE"
    assert resumed["gate"]["state"] == "AWAITING_APPROVAL"
    assert (receipts[0].read_bytes(), receipts[0].stat().st_mtime_ns) == before
    recovered.approve("gate")
    terminal = recovered.apply(resume=True, allow_uncommitted_declaration=True)
    independent = tmp_path / "independent"
    independent.mkdir()
    shutil.copytree(tmp_path / "recipe", independent / "recipe")
    independent_source = independent / "campaign.yaml"
    independent_source.write_text(path.read_text())
    uninterrupted = CampaignEngine(independent_source, independent / "work")
    uninterrupted.apply(allow_uncommitted_declaration=True)
    uninterrupted.approve("gate")
    other = uninterrupted.apply(allow_uncommitted_declaration=True)

    def stable(result):
        return [
            {
                key: stage.get(key)
                for key in ("id", "kind", "scope", "state", "outcome", "outputs")
            }
            for stage in result["stages"]
        ]

    assert stable(terminal) == stable(other)


def test_receipt_tampering_never_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign.engine import CampaignEngine

    path = make_full_campaign(tmp_path, monkeypatch)
    engine = CampaignEngine(path, tmp_path / "work")
    engine.apply(allow_uncommitted_declaration=True)
    receipt = next((engine.store.root / "receipts/corpus").glob("*.json"))
    receipt.write_text("{}\n")
    with pytest.raises((ValueError, TypeError, KeyError)):
        engine.apply(resume=True, allow_uncommitted_declaration=True)


def test_deficit_new_declaration_does_not_mutate_old_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from sparselab.campaign.engine import CampaignEngine

    path = make_full_campaign(tmp_path, monkeypatch)
    value = yaml.safe_load(path.read_text())
    value["stages"][1]["policy"] = {"min_unique_train_bytes_by_domain": {"absent": 1}}
    path.write_text(yaml.safe_dump(value))
    first = CampaignEngine(path, tmp_path / "work")
    blocked = by_id(first.apply(allow_uncommitted_declaration=True))
    assert blocked["ready"]["state"] == "BLOCKED"
    assert blocked["ready"]["outcome"] == "EXPAND_MORE"
    assert blocked["tokenizer"]["state"] == "BLOCKED"
    assert blocked["plan"]["state"] == "BLOCKED"
    original = (first.store.root / "state.json").read_bytes()
    shutil.copytree(tmp_path / "recipe", tmp_path / "expanded")
    source = tmp_path / "expanded/sources/tiny_developer.yaml"
    expanded = yaml.safe_load(source.read_text())
    expanded["domains"] = ["absent", "developer"]
    source.write_text(yaml.safe_dump(expanded))
    value["stages"][0]["project"] = "expanded/corpus.yaml"
    second_path = tmp_path / "expanded.yaml"
    second_path.write_text(yaml.safe_dump(value))

    def interrupt(stage_id: str, state: dict) -> None:
        if stage_id == "ready":
            raise KeyboardInterrupt

    second = CampaignEngine(second_path, tmp_path / "work", after_commit=interrupt)
    with pytest.raises(KeyboardInterrupt):
        second.apply(allow_uncommitted_declaration=True)
    projected = by_id(CampaignEngine(second_path, tmp_path / "work").inspect())
    assert projected["ready"]["state"] == "COMPLETE"
    assert projected["tokenizer"]["state"] == "READY"
    assert first.store.declaration_sha != second.store.declaration_sha
    assert (first.store.root / "state.json").read_bytes() == original


def test_approval_replay_rejection_and_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign.engine import CampaignEngine

    path = make_full_campaign(tmp_path, monkeypatch)
    engine = CampaignEngine(path, tmp_path / "work")
    engine.apply(allow_uncommitted_declaration=True)
    approved = by_id(engine.approve("gate", note="original authorization"))
    assert approved["gate"]["state"] == "COMPLETE"
    approval = next((engine.store.root / "approvals/gate").glob("*.json"))
    before = approval.read_bytes()
    engine.approve("gate", note="replayed note must not overwrite")
    assert approval.read_bytes() == before
    with pytest.raises(ValueError):
        engine.approve("gate", decision="reject")
    value = yaml.safe_load(path.read_text())
    value["stages"][1]["policy"]["min_heldout_families"] = 2
    path.write_text(yaml.safe_dump(value))
    changed = CampaignEngine(path, tmp_path / "work")
    assert changed.store.declaration_sha != engine.store.declaration_sha
    assert by_id(changed.inspect("status"))["gate"]["state"] != "COMPLETE"


def test_changed_reference_digest_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign.engine import CampaignEngine

    path = make_full_campaign(tmp_path, monkeypatch)
    value = yaml.safe_load(path.read_text())
    value["stages"][2]["artifact"]["sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(value))
    result = by_id(
        CampaignEngine(path, tmp_path / "work").apply(
            allow_uncommitted_declaration=True
        )
    )
    assert result["tokenizer"]["state"] == "FAILED"
    assert result["plan"]["state"] == "BLOCKED"


@pytest.mark.parametrize("command", ["plan", "status", "next", "explain"])
def test_engine_readonly_never_creates_workspace(tmp_path: Path, command: str) -> None:
    from sparselab.campaign.engine import CampaignEngine

    source = Path(__file__).resolve().parents[1] / "examples/tiny-campaign.yaml"
    work = tmp_path / "nonexistent"
    result = CampaignEngine(source, work).inspect(command)
    assert not work.exists()
    rows = by_id(result)
    assert rows["corpus"]["state"] in {"NOT_STARTED", "READY"}
    assert rows["readiness"]["state"] in {"NOT_STARTED", "BLOCKED"}
    assert rows["corpus-approval"]["state"] in {"NOT_STARTED", "BLOCKED"}
    assert result["next_action"]


def test_receipt_reconciles_crash_before_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from sparselab.campaign.engine import CampaignEngine

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "examples/tiny-campaign", tmp_path / "recipe")
    value = declaration()
    value["stages"][0]["project"] = "recipe/corpus.yaml"
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(value))
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    engine = CampaignEngine(path, tmp_path / "work")
    original_save = engine.store.save

    def fail_after_receipt(state):
        if list((engine.store.root / "receipts/corpus").glob("*.json")):
            raise KeyboardInterrupt("crash between receipt and state index")
        return original_save(state)

    monkeypatch.setattr(engine.store, "save", fail_after_receipt)
    with pytest.raises(KeyboardInterrupt):
        engine.apply(allow_uncommitted_declaration=True)
    receipt = next((engine.store.root / "receipts/corpus").glob("*.json"))
    before = receipt.read_bytes()
    resumed = CampaignEngine(path, tmp_path / "work")
    result = by_id(resumed.apply(resume=True, allow_uncommitted_declaration=True))
    assert result["corpus"]["state"] == "COMPLETE"
    assert result["ready"]["state"] == "COMPLETE"
    assert result["gate"]["state"] == "AWAITING_APPROVAL"
    assert receipt.read_bytes() == before


def test_synthetic_plan_rejects_claimed_corpus_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign.engine import CampaignEngine

    path = make_full_campaign(tmp_path, monkeypatch)
    value = yaml.safe_load(path.read_text())
    stage = next(stage for stage in value["stages"] if stage["id"] == "plan")
    stage["corpus"] = "corpus"
    stage["requires"].append("corpus")
    path.write_text(yaml.safe_dump(value))
    rows = by_id(
        CampaignEngine(path, tmp_path / "work").apply(
            allow_uncommitted_declaration=True
        )
    )
    assert rows["plan"]["state"] == "FAILED"
    assert rows["runtime"]["state"] == "BLOCKED"


def test_plan_rejects_different_verified_prepared_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign.engine import CampaignEngine
    from sparselab.config.models import RunConfig
    from sparselab.data.packing import prepare_data
    from sparselab.data.tokenizer import load_tokenizer

    path = make_full_campaign(tmp_path, monkeypatch)
    raw = yaml.safe_load((tmp_path / "run.yaml").read_text())
    raw["dataset"]["synthetic_seed"] = 8
    config = RunConfig.model_validate(raw)
    different = prepare_data(config, load_tokenizer(config.tokenizer.path))
    manifest = json.loads((different.root / "manifest.json").read_text())
    value = yaml.safe_load(path.read_text())
    artifact = next(stage for stage in value["stages"] if stage["id"] == "prepared")[
        "artifact"
    ]
    artifact.update(
        identifier=manifest["settings_sha256"],
        sha256=manifest["manifest_sha256"],
        path=str(different.root.relative_to(tmp_path)),
    )
    path.write_text(yaml.safe_dump(value))
    rows = by_id(
        CampaignEngine(path, tmp_path / "work").apply(
            allow_uncommitted_declaration=True
        )
    )
    assert rows["prepared"]["state"] == "COMPLETE"
    assert rows["plan"]["state"] == "FAILED"


def test_status_is_historical_but_plan_reverifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign.engine import CampaignEngine

    path = make_full_campaign(tmp_path, monkeypatch)
    engine = CampaignEngine(path, tmp_path / "work")
    engine.apply(allow_uncommitted_declaration=True)
    (tmp_path / "tokenizer/tokenizer.json").unlink()
    history = by_id(engine.inspect("status"))
    assert history["tokenizer"]["state"] == "COMPLETE"
    assert history["tokenizer"]["verification"] == "last_committed"
    assert history["tokenizer"]["verified_at"]
    assert (
        next(
            row
            for row in engine.inspect("status")["recoverability"]
            if row["id"] == "tokenizer"
        )["classification"]
        == "MISSING_EXTERNAL"
    )
    with pytest.raises((ValueError, OSError)):
        engine.inspect("plan")


@pytest.mark.parametrize("command", ["validate", "plan", "status", "next", "explain"])
def test_cli_readonly_on_nonexistent_workspace(tmp_path: Path, command: str) -> None:
    source = Path(__file__).resolve().parents[1] / "examples/tiny-campaign.yaml"
    work = tmp_path / "nonexistent"
    result = invoke_cli(source, work, command)
    assert not work.exists()
    assert result["id"] == "tiny-campaign"
    if command != "validate":
        assert result["next_action"]["action"] == "apply"
    if command == "explain":
        assert (
            by_id(result)["readiness"]["declaration"]["policy"]["min_heldout_families"]
            == 1
        )


def test_missing_declaration_input_blocks_before_workspace(tmp_path: Path) -> None:
    value = declaration()
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(value))
    work = tmp_path / "work"
    result = invoke_cli(path, work, "apply", expected_code=1)
    assert result["state"] == "FAILED"
    assert "recipe" in result["error"]["reason"]
    assert not work.exists()
    historical = invoke_cli(path, work, "status")
    assert by_id(historical)["corpus"]["state"] == "BLOCKED"
    assert by_id(historical)["gate"]["state"] == "BLOCKED"


def test_stale_running_records_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from sparselab.campaign.engine import CampaignEngine

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "examples/tiny-campaign", tmp_path / "recipe")
    path = tmp_path / "campaign.yaml"
    value = declaration()
    value["stages"][0]["project"] = "recipe/corpus.yaml"
    path.write_text(yaml.safe_dump(value))
    engine = CampaignEngine(path, tmp_path / "work")

    def interrupt(stage, rows, max_wait_seconds, *, execute_runs=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "dispatch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        engine.apply(allow_uncommitted_declaration=True)
    assert by_id(engine.inspect("status"))["corpus"]["state"] == "RUNNING"
    recovered = CampaignEngine(path, tmp_path / "work")
    rows = by_id(recovered.apply(resume=True, allow_uncommitted_declaration=True))
    assert rows["corpus"]["state"] == "COMPLETE"
    assert [a["state"] for a in rows["corpus"]["attempts"]] == [
        "INTERRUPTED",
        "COMPLETE",
    ]
    assert rows["corpus"]["observations"][0]["state"] == "INTERRUPTED"


@pytest.mark.parametrize(
    "decision,expected_state", [("approve", "COMPLETE"), ("reject", "BLOCKED")]
)
def test_resume_recovers_committed_approval_before_gate_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str, expected_state: str
) -> None:
    import shutil

    from sparselab.campaign.engine import CampaignEngine

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "examples/tiny-campaign", tmp_path / "recipe")
    path = tmp_path / "campaign.yaml"
    value = declaration()
    value["stages"][0]["project"] = "recipe/corpus.yaml"
    path.write_text(yaml.safe_dump(value))
    engine = CampaignEngine(path, tmp_path / "work")
    engine.apply(allow_uncommitted_declaration=True)

    def interrupt(stage, row):
        raise KeyboardInterrupt("approval durable; gate receipt not published")

    monkeypatch.setattr(engine.store, "commit", interrupt)
    with pytest.raises(KeyboardInterrupt):
        engine.approve("gate", decision=decision, note="original")
    receipt = next((engine.store.root / "approvals/gate").glob("*.json"))
    original = receipt.read_bytes()
    result = invoke_cli(path, tmp_path / "work", "resume")
    assert by_id(result)["gate"]["state"] == expected_state
    assert receipt.read_bytes() == original


def test_state_receipt_conflict_is_rejected(tmp_path: Path) -> None:
    from sparselab.campaign.engine import CampaignEngine

    source = Path(__file__).resolve().parents[1] / "examples/tiny-campaign.yaml"
    engine = CampaignEngine(source, tmp_path / "work")
    rows = by_id(engine.apply(allow_uncommitted_declaration=True))
    changed = copy.deepcopy(rows["corpus"])
    changed["outputs"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="conflicting committed"):
        engine.store.commit(engine.stages["corpus"], changed)


def test_controller_attempt_cannot_spoof_locked_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign.engine import CampaignEngine
    from sparselab.experiments.cli import locked_cell_request
    from sparselab.experiments.lock import open_lock
    from sparselab.workers.controller import Controller

    path = make_full_campaign(tmp_path, monkeypatch)
    engine = CampaignEngine(path, tmp_path / "work")
    rows = by_id(engine.apply(allow_uncommitted_declaration=True))
    engine.approve("gate")
    lock = open_lock(Path(rows["plan"]["availability"]["path"]))
    workspace = Path(rows["plan"]["availability"]["workspace"])
    controller = Controller(workspace / "controller")
    request = locked_cell_request(
        lock, lock.cells[0], "unregistered", workspace, controller
    )
    request["config"] = request["config"].model_copy(
        update={"seed": request["config"].seed + 1}
    )
    controller.submit_many([request])
    result = by_id(engine.apply(max_wait_seconds=0, allow_uncommitted_declaration=True))
    assert result["run"]["state"] == "FAILED"
    assert "locked cell" in result["run"]["reason"]
    assert result["collect"]["state"] == "BLOCKED"
    assert len(controller.list_experiments()) == 1


def test_terminal_interrupted_worker_is_not_runnable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign.engine import CampaignEngine
    from sparselab.workers.controller import Controller

    path = make_full_campaign(tmp_path, monkeypatch)
    engine = CampaignEngine(path, tmp_path / "work")
    engine.apply(allow_uncommitted_declaration=True)
    engine.approve("gate")
    queued = by_id(
        engine.apply(
            max_wait_seconds=0, execute_runs=True, allow_uncommitted_declaration=True
        )
    )["run"]
    controller = Controller(Path(queued["availability"]["workspace"]) / "controller")
    assert controller.cancel(queued["measurements"]["run_id"])["status"] == "CANCELLED"
    result = engine.apply(
        resume=True, max_wait_seconds=0, allow_uncommitted_declaration=True
    )
    assert by_id(result)["run"]["state"] == "INTERRUPTED"
    assert by_id(result)["collect"]["state"] == "BLOCKED"
    assert result["next_action"]["action"] == "wait"
    assert (
        engine.apply(allow_uncommitted_declaration=True)["next_action"]["action"]
        == "wait"
    )
    assert len(controller.list_experiments()) == 1


def test_lost_submitted_attempt_never_enqueues_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from sparselab.campaign.engine import CampaignEngine

    path = make_full_campaign(tmp_path, monkeypatch)
    engine = CampaignEngine(path, tmp_path / "work")
    engine.apply(allow_uncommitted_declaration=True)
    engine.approve("gate")
    queued = by_id(
        engine.apply(
            max_wait_seconds=0,
            execute_runs=True,
            allow_uncommitted_declaration=True,
        )
    )["run"]
    assert set(queued["measurements"]) == {"experiment_id", "attempt_id", "run_id"}
    controller = Path(queued["availability"]["workspace"]) / "controller"
    shutil.rmtree(controller)

    result = by_id(
        engine.apply(
            resume=True,
            max_wait_seconds=0,
            execute_runs=True,
            allow_uncommitted_declaration=True,
        )
    )
    assert result["run"]["state"] == "FAILED"
    assert "LOST_SUBMISSION" in result["run"]["reason"]
    assert result["run"]["measurements"] == queued["measurements"]
    assert not controller.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("--max-wait-seconds", "-1"),
        ("--max-wait-seconds", "nan"),
        ("--max-wait-seconds", "nonsense"),
        ("--unknown-option",),
    ],
)
def test_cli_argument_errors_are_json_without_mutation(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    source = Path(__file__).resolve().parents[1] / "examples/tiny-campaign.yaml"
    work = tmp_path / "missing-work"
    result = invoke_cli(source, work, "apply", *arguments, expected_code=2)
    assert result["state"] == "FAILED"
    assert result["error"]["type"] == "ArgumentError"
    assert not work.exists()


@pytest.mark.skipif(
    not Path("/proc").is_dir(), reason="requires a zero-capacity procfs mount"
)
def test_runtime_uses_explicit_workspace_not_unrelated_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign.engine import CampaignEngine

    path = make_full_campaign(tmp_path, monkeypatch)
    unrelated = tmp_path / "unrelated-default"
    unrelated.symlink_to("/proc", target_is_directory=True)
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(unrelated))
    engine = CampaignEngine(path, tmp_path / "work")
    rows = by_id(engine.apply(allow_uncommitted_declaration=True))
    assert rows["runtime"]["state"] == "COMPLETE"
    assert rows["gate"]["state"] == "AWAITING_APPROVAL"


def test_cli_campaign_source_accepts_parent_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    path = make_full_campaign(tmp_path, monkeypatch)
    repo = Path(__file__).resolve().parents[1]
    source = Path(os.path.relpath(path, repo))
    result = invoke_cli(source, tmp_path / "work", "apply")
    assert by_id(result)["tokenizer"]["state"] == "COMPLETE"
    assert by_id(result)["gate"]["state"] == "AWAITING_APPROVAL"


def test_initial_status_does_not_claim_missing_input_is_ready(tmp_path: Path) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(declaration()))
    work = tmp_path / "missing-work"
    result = invoke_cli(path, work, "status")
    assert by_id(result)["corpus"]["state"] == "BLOCKED"
    assert by_id(result)["ready"]["blocked_by"] == ["corpus"]
    assert not work.exists()


def test_uncommitted_campaign_preflight_blocks_before_store_creation(
    tmp_path: Path,
) -> None:
    from sparselab.campaign.engine import CampaignEngine

    source = tmp_path / "campaign.yaml"
    source.write_text(yaml.safe_dump(declaration()))
    work = tmp_path / "persistent-state"
    engine = CampaignEngine(source, work)
    with pytest.raises(ValueError, match="UNKNOWN"):
        engine.apply()
    assert not work.exists()
    with pytest.raises(ValueError, match="linked recovery manifest"):
        engine.reconstruct(allow_uncommitted_declaration=True)
    assert not work.exists()


def test_committed_campaign_checkout_warning_and_shared_external_root(
    tmp_path: Path,
) -> None:
    import shutil
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    first = tmp_path / "first-checkout"
    first.mkdir()
    subprocess.run(["git", "init", "-q", str(first)], check=True)
    subprocess.run(
        ["git", "-C", str(first), "config", "user.email", "fixture@example.test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(first), "config", "user.name", "Fixture"],
        check=True,
    )
    shutil.copytree(repo / "examples/tiny-campaign", first / "tiny-campaign")
    (first / "campaign.yaml").write_bytes(
        (repo / "examples/tiny-campaign.yaml").read_bytes()
    )
    subprocess.run(["git", "-C", str(first), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(first), "commit", "-qm", "Declare tiny campaign"],
        check=True,
    )

    def command(checkout: Path, work: Path, verb: str) -> tuple[dict, str]:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "sparselab",
                "--work-dir",
                str(work),
                "campaign",
                verb,
                str(checkout / "campaign.yaml"),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        return json.loads(process.stdout), process.stderr

    inside = first / "legacy-state"
    selected, warnings = command(first, inside, "apply")
    assert "STORAGE_INSIDE_GIT_CHECKOUT" in warnings
    assert selected["storage_checks"][0]["git_root"] == str(first)
    assert by_id(selected)["corpus"]["storage_checks"][0]["reason_code"] == (
        "STORAGE_INSIDE_GIT_CHECKOUT"
    )
    outside = tmp_path / "persistent-state"
    external, no_warning = command(first, outside, "apply")
    assert "STORAGE_INSIDE_GIT_CHECKOUT" not in no_warning
    assert external["storage_checks"] == []
    second = tmp_path / "second-checkout"
    subprocess.run(["git", "clone", "-q", str(first), str(second)], check=True)
    current, no_warning = command(second, outside, "status")
    assert "STORAGE_INSIDE_GIT_CHECKOUT" not in no_warning
    assert current["declaration_sha256"] == external["declaration_sha256"]
    assert by_id(current)["corpus"]["outputs"] == by_id(external)["corpus"]["outputs"]
    assert (
        by_id(current)["corpus"]["availability"]
        == by_id(external)["corpus"]["availability"]
    )
    assert not (second / "legacy-state").exists()
    from sparselab.campaign.engine import CampaignEngine

    state_path = CampaignEngine(first / "campaign.yaml", outside).store.index
    before = state_path.read_bytes()
    (first / "unrelated.log").write_text("not scientific intent\n")
    command(first, outside, "apply")
    assert state_path.read_bytes() == before

    source_file = first / "tiny-campaign/texts/tiny_docs-0.txt"
    source_file.write_bytes(source_file.read_bytes() + b"\nrevised scientific input\n")
    subprocess.run(["git", "-C", str(first), "add", str(source_file)], check=True)
    subprocess.run(
        ["git", "-C", str(first), "commit", "-qm", "Revise one corpus source"],
        check=True,
    )
    changed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sparselab",
            "--work-dir",
            str(outside),
            "campaign",
            "apply",
            str(first / "campaign.yaml"),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert changed.returncode == 1
    assert (
        "DECLARATION_IDENTITY_CHANGED" in json.loads(changed.stdout)["error"]["reason"]
    )
    assert state_path.read_bytes() == before


def test_committed_recovery_output_inventory_can_extend_without_changing_intent(
    tmp_path: Path,
) -> None:
    import shutil
    import subprocess

    from sparselab.campaign.engine import CampaignEngine
    from sparselab.recovery.evidence import export_evidence

    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for key, value in (
        ("user.email", "fixture@example.test"),
        ("user.name", "Fixture"),
    ):
        subprocess.run(["git", "-C", str(repo), "config", key, value], check=True)
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "examples/tiny-campaign",
        repo / "tiny-campaign",
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "Pin corpus inputs"], check=True
    )
    source_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    recovery = {
        "recovery_version": 1,
        "id": "tiny-recovery",
        "source_commit": source_commit,
        "steps": [
            {
                "id": "later",
                "kind": "external_required",
                "role": "human_decision",
                "reason": "not selected",
            },
        ],
    }
    (repo / "recovery.yaml").write_text(yaml.safe_dump(recovery))
    campaign = {
        "campaign_version": 1,
        "id": "tiny-recovery-campaign",
        "recovery": "recovery.yaml",
        "stages": [
            {
                "id": "corpus",
                "kind": "corpus_release",
                "scope": "corpus",
                "project": "tiny-campaign/corpus.yaml",
            }
        ],
    }
    (repo / "campaign.yaml").write_text(yaml.safe_dump(campaign))
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "Declare recovery"], check=True
    )
    engine = CampaignEngine(repo / "campaign.yaml", tmp_path / "persistent-state")
    initial = by_id(engine.apply())["corpus"]
    assert initial["state"] == "COMPLETE"

    probe = tmp_path / "runtime-probe.json"
    probe.write_text('{"format_version":1,"status":"observation"}')
    (repo / "evidence").mkdir()
    reference = repo / "evidence/probe.json"
    export_evidence(
        "runtime_probe",
        probe,
        reference,
        source_commit=source_commit,
        declaration_hashes=[],
    )
    recovery["evidence"] = ["evidence/probe.json"]
    (repo / "recovery.yaml").write_text(yaml.safe_dump(recovery))
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "Publish verified output"], check=True
    )
    updated = by_id(engine.apply())["corpus"]
    assert updated["outputs"] == initial["outputs"]

    recovery["steps"][0]["reason"] = "different scientific decision"
    (repo / "recovery.yaml").write_text(yaml.safe_dump(recovery))
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "Change source intent"], check=True
    )
    with pytest.raises(ValueError, match="DECLARATION_IDENTITY_CHANGED"):
        engine.apply()


def _accelerator_campaign(tmp_path, monkeypatch):
    from test_runtime_profiles import fake_runtime, profile_value

    from sparselab.campaign.engine import CampaignEngine
    from sparselab.config.models import RunConfig
    from sparselab.runtime_profile import RuntimeProfile

    source = make_full_campaign(tmp_path, monkeypatch)
    run_path = tmp_path / "run.yaml"
    config = RunConfig.model_validate(yaml.safe_load(run_path.read_text()))
    probe, config = fake_runtime(monkeypatch, config, "rocm")
    from sparselab import runtime

    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/contracts/worker_capabilities_v1.json"
        ).read_text()
    )
    tested = runtime.RuntimeInfo.from_dict(
        {
            **fixture["runtime"],
            "backend": "rocm",
            "torch_device": "cuda:0",
            "device_name": "Fixture Accelerator",
            "runtime_version": "fixture-hip",
            "tested_precisions": ["fp32", "bf16"],
            "tested_features": [
                "forward_backward_optimizer",
                "activation_checkpointing",
            ],
        }
    )
    monkeypatch.setattr(runtime, "validate_runtime", lambda *args, **kwargs: tested)
    run_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    plan_path = tmp_path / "experiment.yaml"
    plan = yaml.safe_load(plan_path.read_text())
    plan["execution"] = {"backend": "rocm"}
    plan_path.write_text(yaml.safe_dump(plan))
    campaign = yaml.safe_load(source.read_text())
    next(stage for stage in campaign["stages"] if stage["id"] == "runtime")[
        "profile_id"
    ] = "offline-cpu"
    source.write_text(yaml.safe_dump(campaign))
    profile = RuntimeProfile.model_validate(profile_value(backend="rocm"))
    return source, profile, probe, CampaignEngine


@pytest.mark.parametrize("backend", [None, "cpu", "cuda", "rocm"])
def test_campaign_accelerator_acceptance_requires_matching_source(
    tmp_path, monkeypatch, backend
):
    source, profile, _, engine_type = _accelerator_campaign(tmp_path, monkeypatch)
    selected = (
        None if backend is None else profile.model_copy(update={"backend": backend})
    )
    engine = engine_type(source, tmp_path / "work", runtime_profile=selected)
    rows = by_id(engine.apply(allow_uncommitted_declaration=True))
    assert rows["runtime"]["state"] == ("COMPLETE" if backend == "rocm" else "BLOCKED")
    assert rows["run"]["state"] == "BLOCKED"
    if backend == "rocm":
        entry = rows["runtime"]["availability"]["runtime_bindings"]["main:single"]
        assert entry["tested_runtime"]["backend"] == "rocm"
        assert (
            rows["runtime"]["measurements"]["runtime_bindings"]["main:single"]
            == entry["binding_sha256"]
        )
    from sparselab.workers.controller import Controller

    workspace = Path(rows["plan"]["availability"]["workspace"])
    if backend is None:
        assert not (workspace / "controller").exists()
    else:
        assert (
            Controller(workspace / "controller", read_only=True).list_experiments()
            == []
        )


def test_campaign_runtime_drift_blocks_without_ambient_worker(tmp_path, monkeypatch):
    from sparselab.workers.controller import Controller

    source, profile, probe, engine_type = _accelerator_campaign(tmp_path, monkeypatch)
    engine = engine_type(source, tmp_path / "work", runtime_profile=profile)
    initial = by_id(engine.apply(allow_uncommitted_declaration=True))
    assert initial["runtime"]["state"] == "COMPLETE"
    engine.approve("gate")
    probe["device_name"] = "Changed accelerator"
    rows = by_id(engine.apply(execute_runs=True, allow_uncommitted_declaration=True))
    assert rows["run"]["state"] == "BLOCKED"
    assert "fresh" in rows["run"]["reason"] or "changed" in rows["run"]["reason"]
    workspace = Path(rows["plan"]["availability"]["workspace"])
    controller = Controller(workspace / "controller", read_only=True)
    assert controller.list_experiments() == []
    assert controller.store.worker_records() == []


def test_campaign_bound_profile_dispatch_persists_operational_identity(
    tmp_path, monkeypatch
):
    from sparselab.training.manifest import source_identity
    from sparselab.workers.controller import Controller

    source, profile, _, engine_type = _accelerator_campaign(tmp_path, monkeypatch)
    engine = engine_type(source, tmp_path / "work", runtime_profile=profile)
    initial = by_id(engine.apply(allow_uncommitted_declaration=True))
    entry = initial["runtime"]["availability"]["runtime_bindings"]["main:single"]
    engine.approve("gate")
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/contracts/worker_capabilities_v1.json"
        ).read_text()
    )

    def discovery(self, definition, op, payload, **kwargs):
        assert op == "discover"
        capability = {
            **fixture,
            **definition.model_dump(mode="json"),
            "source_identity_sha256": source_identity()["sha256"],
            "runtime": {
                **fixture["runtime"],
                "backend": "rocm",
                "torch_device": "cuda:0",
                "device_name": "Fixture Accelerator",
            },
        }
        return {"capabilities": capability}

    monkeypatch.setattr(Controller, "_rpc_result", discovery)
    rows = by_id(
        engine.apply(
            execute_runs=True, max_wait_seconds=0, allow_uncommitted_declaration=True
        )
    )
    assert rows["run"]["state"] == "RUNNING"
    workspace = Path(rows["plan"]["availability"]["workspace"])
    controller = Controller(workspace / "controller", read_only=True)
    (attempt,) = controller.list_experiments()
    (registered,) = controller.store.worker_records()
    assert registered["definition"]["python"] == str(profile.python)
    assert registered["definition"]["backend"] == "rocm"
    assert attempt["spec"]["config"]["runtime"]["backend"] == "rocm"
    assert attempt["spec"]["plan"]["runtime_binding_sha256"] == entry["binding_sha256"]
    assert attempt["spec"]["plan"]["execution_binding_sha256"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"profile_id": "../bad"},
        {"worker": "/absolute/path"},
        {"profile_id": "profile", "worker": "worker"},
    ],
)
def test_runtime_acceptance_rejects_unsafe_or_ambiguous_source(changes):
    from sparselab.campaign.plan import RuntimeAcceptance

    with pytest.raises(ValueError):
        RuntimeAcceptance(
            id="runtime",
            kind="runtime_acceptance",
            scope="runtime",
            requires=("plan",),
            plan="plan",
            **changes,
        )


def test_cpu_evaluation_cannot_inherit_accelerator_authorization():
    from sparselab.campaign.plan import Evaluation

    with pytest.raises(ValueError, match="CPU"):
        Evaluation(
            id="evaluate",
            kind="evaluation",
            scope="evaluation",
            collect="collect",
            suite="suite.yaml",
            backend="cpu",
            runtime="runtime",
        )


def test_runtime_binding_preserves_science_and_prior_blocked_receipt(
    tmp_path, monkeypatch
):
    from sparselab.experiments.binding import bind_runtime, open_runtime_binding
    from sparselab.experiments.lock import open_lock

    source, profile, _, engine_type = _accelerator_campaign(tmp_path, monkeypatch)
    blocked = engine_type(source, tmp_path / "work")
    before_rows = by_id(blocked.apply(allow_uncommitted_declaration=True))
    lock_path = Path(before_rows["plan"]["availability"]["path"])
    original = open_lock(lock_path)
    lock_bytes = lock_path.read_bytes()
    availability_path = lock_path.with_suffix(".availability.json")
    availability_bytes = availability_path.read_bytes()
    old_receipts = {
        path: path.read_bytes()
        for path in (blocked.store.root / "receipts/runtime").glob("*.json")
    }
    assert before_rows["runtime"]["state"] == "BLOCKED"
    workspace = Path(before_rows["plan"]["availability"]["workspace"])
    path = bind_runtime(original, original.cells[0], workspace, profile=profile)
    binding = open_runtime_binding(path, original, original.cells[0])
    assert binding["scientific_sha256"] == original.scientific_sha256
    assert binding["tested_runtime"]["backend"] == "rocm"
    interpreter_alias = tmp_path / "vendor-python"
    interpreter_alias.symlink_to(profile.python)
    relocated_profile = profile.model_copy(update={"python": interpreter_alias})
    relocated_path = bind_runtime(
        original, original.cells[0], workspace, profile=relocated_profile
    )
    relocated = open_runtime_binding(relocated_path, original, original.cells[0])
    assert relocated["binding_sha256"] != binding["binding_sha256"]
    assert relocated["scientific_sha256"] == binding["scientific_sha256"]
    # A new operational receipt must never replace a committed Campaign outcome.
    replay = engine_type(source, tmp_path / "work", runtime_profile=profile)
    assert (
        by_id(replay.apply(allow_uncommitted_declaration=True))["runtime"]["state"]
        == "BLOCKED"
    )
    assert lock_path.read_bytes() == lock_bytes
    assert availability_path.read_bytes() == availability_bytes
    assert open_lock(lock_path).scientific_sha256 == original.scientific_sha256
    assert all(path.read_bytes() == content for path, content in old_receipts.items())


def test_mixed_cells_accept_only_their_declared_runtime_source(tmp_path, monkeypatch):
    source, profile, _, engine_type = _accelerator_campaign(tmp_path, monkeypatch)
    plan_path = tmp_path / "experiment.yaml"
    plan = yaml.safe_load(plan_path.read_text())
    plan.pop("execution")
    plan["axes"] = [
        {
            "name": "target",
            "choices": [
                {"label": "cpu", "set": {"runtime.backend": "cpu"}},
                {"label": "rocm", "set": {"runtime.backend": "rocm"}},
            ],
        }
    ]
    plan_path.write_text(yaml.safe_dump(plan))
    campaign = yaml.safe_load(source.read_text())
    stages = campaign["stages"]
    position = next(i for i, stage in enumerate(stages) if stage["id"] == "runtime")
    stages.insert(
        position,
        {
            "id": "cpu-runtime",
            "kind": "runtime_acceptance",
            "scope": "runtime",
            "requires": ["plan"],
            "plan": "plan",
        },
    )
    next(stage for stage in stages if stage["id"] == "run")["cell"] = "main:target=rocm"
    stages.append(
        {
            "id": "cpu-run",
            "kind": "experiment_run",
            "scope": "model",
            "requires": ["plan", "cpu-runtime", "gate"],
            "plan": "plan",
            "runtime": "cpu-runtime",
            "cell": "main:target=cpu",
        }
    )
    source.write_text(yaml.safe_dump(campaign))
    engine = engine_type(source, tmp_path / "work", runtime_profile=profile)
    rows = by_id(engine.apply(allow_uncommitted_declaration=True))
    assert rows["cpu-runtime"]["state"] == rows["runtime"]["state"] == "COMPLETE"
    assert rows["cpu-runtime"]["measurements"]["cells"] == ["main:target=cpu"]
    assert rows["runtime"]["measurements"]["cells"] == ["main:target=rocm"]
    assert set(rows["runtime"]["availability"]["runtime_bindings"]) == {
        "main:target=rocm"
    }


def test_profile_campaign_evaluation_rederives_for_checkpoint_runtime(
    tmp_path, monkeypatch
):
    from test_runtime_profiles import profile_value

    from sparselab.evaluation.suite import verify_evaluation_index
    from sparselab.workers.controller import Controller

    source = make_full_campaign(tmp_path, monkeypatch)
    value = yaml.safe_load(source.read_text())
    next(stage for stage in value["stages"] if stage["id"] == "runtime")[
        "profile_id"
    ] = "offline-cpu"
    evaluation = next(stage for stage in value["stages"] if stage["id"] == "evaluation")
    evaluation["runtime"] = "runtime"
    evaluation["requires"].append("runtime")
    source.write_text(yaml.safe_dump(value))
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump(profile_value()))
    work = tmp_path / "work"
    accepted = by_id(
        invoke_cli(source, work, "apply", "--runtime-profile", str(profile))
    )
    assert accepted["runtime"]["state"] == "COMPLETE"
    invoke_cli(source, work, "approve", "gate")
    rows = by_id(
        invoke_cli(
            source,
            work,
            "apply",
            "--execute-runs",
            "--runtime-profile",
            str(profile),
            "--max-wait-seconds",
            "600",
        )
    )
    assert (
        rows["run"]["state"]
        == rows["collect"]["state"]
        == rows["evaluation"]["state"]
        == "COMPLETE"
    )
    index = verify_evaluation_index(Path(rows["evaluation"]["availability"]["path"]))
    assert index["evaluation_runtime"]["backend"] == "cpu"
    assert index["evaluation_runtime"]["observed"]["precision"] == "fp32"
    assert index["runtime_authorization"]["kind"] == "profile"
    controller = Controller(
        Path(rows["run"]["availability"]["workspace"]) / "controller", read_only=True
    )
    (attempt,) = controller.list_experiments()
    assert attempt["status"] == attempt["ingestion_status"] == "COMPLETE"
    assert (
        attempt["spec"]["plan"]["runtime_binding_sha256"]
        == rows["runtime"]["availability"]["runtime_bindings"]["main:single"][
            "binding_sha256"
        ]
    )
