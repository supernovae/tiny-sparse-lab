"""Frozen Corpus Forge tokenizer reuse across release variants and promotion."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.config.models import RunConfig
from sparselab.corpus.acquisition import acquire
from sparselab.corpus.export import export_release
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project
from sparselab.corpus.release import freeze
from sparselab.data.conversations import iter_rendered_conversations
from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
from sparselab.experiments.lock import resolve_plan
from sparselab.experiments.plan import CorpusVariant, ExperimentPlan
from sparselab.experiments.prepare import prepare_plan
from sparselab.training.manifest import config_sha256, sha256_file
from sparselab.training.trainer import train
from sparselab.workers.bundles import prepare_dispatch_bundle


@pytest.fixture(scope="module")
def authored(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    root = tmp_path_factory.mktemp("frozen-tokenizer-reuse")
    project_root = root / "project"
    shutil.copytree("corpora/devmind-sample-v0", project_root)
    project_path = project_root / "corpus.yaml"
    base = root / "run.yaml"
    shutil.copyfile("configs/runtime_smoke_cpu.yaml", base)
    workspace = root / "workspace"
    project = load_project(project_path)
    acquire(project, workspace)
    release = freeze(build(project, workspace, offline=True), workspace)
    export = export_release(release, "lm", base, 300, workspace)
    tokenizer = train_tokenizer(load_tokenizer_config(export / "tokenizer.yaml"))
    pinned = project_root / "pinned" / "shared-tokenizer.json"
    pinned.parent.mkdir(exist_ok=True)
    shutil.copyfile(tokenizer, pinned)
    digest = sha256_file(tokenizer)
    fraction = {
        "generated_share": 0.5640625,
        "train_tokens": 640,
        "tokenizer_path": "pinned/shared-tokenizer.json",
        "tokenizer_sha256": digest,
    }
    spec = {
        "kind": "tokenizer",
        "version": 1,
        "producer": "sparselab",
        "identifier": tokenizer.parent.name,
        "sha256": digest,
        "path": str(tokenizer),
    }
    plan = {
        "plan_version": 1,
        "id": "frozen-tokenizer-comparison",
        "base_run": "run.yaml",
        "artifacts": {"shared_tokenizer": spec},
        "corpus_variants": [
            {
                "id": "full",
                "project": "project/corpus.yaml",
                "tokenizer_artifact": "shared_tokenizer",
                "release_set": {"fraction": fraction},
            },
            {
                "id": "filtered",
                "project": "project/corpus.yaml",
                "tokenizer_artifact": "shared_tokenizer",
                "release_set": {
                    "fraction": fraction,
                    "include_shapes": ["raw_document", "tool_trace"],
                },
            },
        ],
        "axes": [
            {
                "name": "corpus",
                "choices": [
                    {"label": "full", "set": {"inputs.corpus_variant": "full"}},
                    {"label": "filtered", "set": {"inputs.corpus_variant": "filtered"}},
                ],
            }
        ],
        "execution": {"backend": "cpu"},
    }
    plan["comparisons"] = [
        {
            "id": "exact-shape",
            "baseline": {"corpus": "full"},
            "variant": {"corpus": "filtered"},
            "mode": "multi_factor",
            "confounders": ["filtered validation rows", "small selected train subset"],
            "interventions": [
                "artifacts.corpus_export.identifier",
                "artifacts.corpus_export.sha256",
                "artifacts.corpus_release.identifier",
                "artifacts.corpus_release.sha256",
                "artifacts.prepared_data.identifier",
                "artifacts.prepared_data.sha256",
                "dataset.revision",
            ],
            "invariants": [
                "artifacts.tokenizer.sha256",
                "training.max_tokens",
                "seed",
                "model.vocab_size",
            ],
        }
    ]
    path = root / "plan.json"
    path.write_text(json.dumps(plan))
    return path, workspace, plan


def test_same_frozen_tokenizer_across_distinct_releases(
    authored: tuple[Path, Path, dict],
) -> None:
    path, workspace, raw = authored
    plan = ExperimentPlan.model_validate(raw)
    prepared = prepare_plan(plan, path, workspace)
    full, filtered = prepared["variants"]
    assert (
        full["tokenizer_artifact"]
        == filtered["tokenizer_artifact"]
        == "shared_tokenizer"
    )
    assert full["tokenizer_sha256"] == filtered["tokenizer_sha256"]
    assert full["release_id"] != filtered["release_id"]
    assert full["export_path"] != filtered["export_path"]
    assert full["prepared_manifest_sha256"] != filtered["prepared_manifest_sha256"]
    assert not (
        Path(filtered["export_path"]) / "tokenizer" / "tokenizer_manifest.json"
    ).exists()
    assert load_config(Path(full["export_path"]) / "run.yaml").tokenizer.path != Path(
        full["tokenizer_path"]
    )
    tokenizer = load_tokenizer(Path(full["tokenizer_path"]))
    releases = [Path(record["release_path"]) for record in (full, filtered)]
    rows = [
        [
            json.loads(line)["text"]
            for line in (release / "lm/train.jsonl").read_text().splitlines()
        ]
        for release in releases
    ]
    assert rows[0] == rows[1]
    assert [len(tokenizer.encode(text).ids) for text in rows[0]] == [279]
    chat_rows = [
        [row.text for row in iter_rendered_conversations(release / "chat/train.jsonl")]
        for release in releases
    ]
    assert chat_rows[0] == chat_rows[1]
    assert [len(tokenizer.encode(text).ids) for text in chat_rows[0]] == [361]
    assert (
        sum(len(tokenizer.encode(text).ids) for text in rows[0] + chat_rows[0]) == 640
    )
    assert (releases[0] / "chat/validation.jsonl").read_bytes() != (
        releases[1] / "chat/validation.jsonl"
    ).read_bytes()
    locked = resolve_plan(plan, path, prepared=prepared)
    left, right = locked.cells
    assert left.artifacts["tokenizer"] == right.artifacts["tokenizer"]
    assert left.artifacts["corpus_release"] != right.artifacts["corpus_release"]
    assert left.artifacts["prepared_data"] != right.artifacts["prepared_data"]
    assert left.config.training.max_tokens == right.config.training.max_tokens
    assert locked.comparisons[0].mode == "multi_factor"
    assert {item["path"] for item in locked.comparisons[0].differences} == set(
        raw["comparisons"][0]["interventions"]
    )
    # This tiny exact-budget pair filters an unselected candidate and alters validation;
    # it proves identity handling, not a model-quality corpus-shape effect.


def test_relocated_corpus_rejects_forged_inner_release_binding(
    authored: tuple[Path, Path, dict], tmp_path: Path
) -> None:
    from sparselab.training.manifest import canonical_json
    from sparselab.workers.bundles import verify_portable_corpus_binding

    path, workspace, raw = authored
    plan = ExperimentPlan.model_validate(raw)
    prepared = prepare_plan(plan, path, workspace)
    cell = resolve_plan(plan, path, prepared=prepared).cells[0]
    bundle = tmp_path / "dispatch"
    prepare_dispatch_bundle(cell.config, bundle)
    assets = bundle / "assets"
    verified = verify_portable_corpus_binding(cell.config, assets)
    assert verified["corpus_export"]["export_sha256"] == sha256_file(
        assets / "corpus/export.json"
    )

    export_path = assets / "corpus/export.json"
    export = json.loads(export_path.read_text())
    export["release_manifest_sha256"] = "0" * 64
    export_path.write_bytes(canonical_json(export) + b"\n")
    record_path = assets / "corpus/binding.json"
    record = json.loads(record_path.read_text())
    forged_sha = sha256_file(export_path)
    record["files"]["export.json"] = forged_sha
    record["corpus_export"]["export_sha256"] = forged_sha
    record_path.write_bytes(canonical_json(record) + b"\n")
    with pytest.raises(ValueError, match="inner identity"):
        verify_portable_corpus_binding(cell.config, assets)


def test_reuse_rejects_bad_strategy_and_fraction_pin(
    authored: tuple[Path, Path, dict],
) -> None:
    path, workspace, raw = authored
    with pytest.raises(ValueError, match="exactly one"):
        CorpusVariant(id="none", project="p")
    with pytest.raises(ValueError, match="exactly one"):
        CorpusVariant(
            id="both", project="p", vocab_size=300, tokenizer_artifact="shared"
        )
    changed = json.loads(json.dumps(raw))
    changed["corpus_variants"][0]["release_set"]["fraction"]["tokenizer_sha256"] = (
        "a" * 64
    )
    with pytest.raises(ValueError, match="fraction tokenizer"):
        prepare_plan(ExperimentPlan.model_validate(changed), path, workspace)
    changed = json.loads(json.dumps(raw))
    changed["artifacts"]["shared_tokenizer"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="invalid tokenizer artifact"):
        prepare_plan(ExperimentPlan.model_validate(changed), path, workspace)
    changed = json.loads(json.dumps(raw))
    changed["artifacts"].pop("shared_tokenizer")
    with pytest.raises(ValueError, match="external tokenizer artifact"):
        ExperimentPlan.model_validate(changed)
    changed = json.loads(json.dumps(raw))
    changed["artifacts"]["shared_tokenizer"]["kind"] = "prompt_set"
    with pytest.raises(ValueError, match="external tokenizer artifact"):
        ExperimentPlan.model_validate(changed)
    manifest = Path(raw["artifacts"]["shared_tokenizer"]["path"]).with_name(
        "tokenizer_manifest.json"
    )
    original = manifest.read_bytes()
    try:
        provenance = json.loads(original)
        provenance["vocab_size"] = 301
        manifest.write_text(json.dumps(provenance))
        with pytest.raises(ValueError, match="invalid tokenizer artifact"):
            prepare_plan(ExperimentPlan.model_validate(raw), path, workspace)
    finally:
        manifest.write_bytes(original)


def test_fraction_override_and_lock_revalidation(
    authored: tuple[Path, Path, dict],
) -> None:
    path, workspace, raw = authored
    changed = json.loads(json.dumps(raw))
    changed["artifacts"]["selector"] = dict(changed["artifacts"]["shared_tokenizer"])
    changed["corpus_variants"][0]["fraction_tokenizer"] = "selector"
    plan = ExperimentPlan.model_validate(changed)
    prepared = prepare_plan(plan, path, workspace)
    resolve_plan(plan, path, prepared=prepared)
    broken = json.loads(json.dumps(prepared))
    broken["variants"][0]["tokenizer_artifact"] = "selector"
    with pytest.raises(ValueError, match="declared tokenizer artifact"):
        resolve_plan(plan, path, prepared=broken)
    broken = json.loads(json.dumps(prepared))
    broken["variants"][0]["config"]["model"]["vocab_size"] = 301
    broken["variants"][0]["config_sha256"] = config_sha256(
        broken["variants"][0]["config"]
    )
    with pytest.raises(ValueError, match="vocabulary"):
        resolve_plan(plan, path, prepared=broken)
    changed["corpus_variants"][0]["fraction_tokenizer"] = "missing"
    with pytest.raises(ValueError, match="external tokenizer artifact"):
        ExperimentPlan.model_validate(changed)
    changed["corpus_variants"][0]["fraction_tokenizer"] = "selector"
    changed["corpus_variants"][0]["release_set"].pop("fraction")
    with pytest.raises(ValueError, match="requires a fractional release"):
        prepare_plan(ExperimentPlan.model_validate(changed), path, workspace)


def test_status_filters_bind_distinct_releases(
    authored: tuple[Path, Path, dict],
) -> None:
    path, workspace, raw = authored
    changed = json.loads(json.dumps(raw))
    changed.pop("comparisons")
    changed["corpus_variants"][0]["release_set"].pop("fraction")
    changed["corpus_variants"][1]["release_set"] = {
        "accepted_generation_statuses": ["source_entailed", "oracle_verified"]
    }
    plan = ExperimentPlan.model_validate(changed)
    prepared = prepare_plan(plan, path, workspace)
    locked = resolve_plan(plan, path, prepared=prepared)
    assert (
        prepared["variants"][0]["release_id"] != prepared["variants"][1]["release_id"]
    )
    assert (
        locked.cells[0].artifacts["corpus_release"]
        != locked.cells[1].artifacts["corpus_release"]
    )
    changed["corpus_variants"][1]["release_set"]["accepted_generation_statuses"] = [
        "ranked"
    ]
    with pytest.raises(ValueError, match="accepted_generation_statuses"):
        prepare_plan(ExperimentPlan.model_validate(changed), path, workspace)


def test_unfractioned_chat_shapes_change_actual_training_rows(
    authored: tuple[Path, Path, dict],
) -> None:
    path, workspace, raw = authored
    changed = json.loads(json.dumps(raw))
    changed.pop("comparisons")
    changed["corpus_variants"][0].update(
        {
            "view": "chat",
            "release_set": {
                "include_shapes": ["raw_document", "troubleshooting_scenario"]
            },
        }
    )
    changed["corpus_variants"][1].update(
        {
            "view": "chat",
            "release_set": {
                "include_shapes": [
                    "raw_document",
                    "troubleshooting_scenario",
                    "tool_trace",
                ]
            },
        }
    )
    records = prepare_plan(ExperimentPlan.model_validate(changed), path, workspace)[
        "variants"
    ]
    left, right = [
        Path(record["release_path"]) / "chat/train.jsonl" for record in records
    ]
    assert left.read_bytes() != right.read_bytes()
    tokenizer = load_tokenizer(Path(records[0]["tokenizer_path"]))
    available = [
        sum(
            len(tokenizer.encode(row.text).ids)
            for row in iter_rendered_conversations(file)
        )
        for file in (left, right)
    ]
    assert available[0] != available[1]
    assert (
        records[0]["config"]["training"]["max_tokens"]
        == records[1]["config"]["training"]["max_tokens"]
    )


def test_external_pretraining_promotes_chat_variant(
    authored: tuple[Path, Path, dict],
) -> None:
    path, workspace, raw = authored
    plan = ExperimentPlan.model_validate(raw)
    parent_record = prepare_plan(plan, path, workspace)["variants"][0]
    parent_values = parent_record["config"]
    parent_values["training"].update({"max_steps": 1, "max_tokens": 32})
    parent_values["optimizer"]["warmup_steps"] = 0
    parent_values["logging"]["root_dir"] = str(workspace / "runs")

    parent = RunConfig.model_validate(
        {
            **parent_values,
            "tokenizer": {
                **parent_values["tokenizer"],
                "path": parent_record["tokenizer_path"],
            },
        }
    )
    run_id = train(parent, run_id="frozen-parent")
    run = parent.logging.root_dir / run_id
    pointer = json.loads((run / "checkpoints/latest.json").read_text())
    generation = run / "checkpoints" / pointer["relative_path"]
    manifest = json.loads((generation / "manifest.json").read_text())
    chat = json.loads(json.dumps(raw))
    chat["id"] = "frozen-chat-promotion"
    chat["corpus_variants"] = [
        {
            "id": "chat",
            "project": raw["corpus_variants"][0]["project"],
            "view": "chat",
            "tokenizer_artifact": "shared_tokenizer",
        }
    ]
    chat.pop("axes")
    chat.pop("comparisons")
    chat["artifacts"]["parent"] = {
        "kind": "checkpoint",
        "version": manifest["format_version"],
        "state": "full",
        "producer": "sparselab",
        "identifier": generation.name,
        "sha256": manifest["sha256"],
        "path": str(generation),
    }
    chat["phases"] = [{"id": "sft", "transition": "promote", "checkpoint": "parent"}]
    chat_plan = ExperimentPlan.model_validate(chat)
    child_record = prepare_plan(chat_plan, path, workspace)["variants"][0]
    resolved = resolve_plan(
        chat_plan,
        path,
        prepared={
            "format": "experiment-preparation-v1",
            "id": chat_plan.id,
            "variants": [child_record],
        },
    )
    child = resolved.cells[0]
    assert child.config.dataset.source == "local_chat"
    assert child.artifacts["corpus_release"]["sha256"] != parent_record["release_id"]
    assert (
        child.artifacts["prepared_data"]["sha256"]
        != parent_record["prepared_manifest_sha256"]
    )
    assert child.artifacts["tokenizer"]["sha256"] == parent_record["tokenizer_sha256"]
    assert child.config.model == parent.model
    assert child.artifacts["parent_checkpoint"]["sha256"] == manifest["sha256"]
    bundle = prepare_dispatch_bundle(
        child.config, workspace / "chat-promotion-bundle", promote=generation
    )
    assert bundle.continuation.checkpoint_sha256 == manifest["sha256"]
    alternate_export = export_release(
        Path(parent_record["release_path"]),
        "lm",
        path.parent / raw["base_run"],
        301,
        workspace,
    )
    alternate = train_tokenizer(
        load_tokenizer_config(alternate_export / "tokenizer.yaml")
    )
    mismatch = json.loads(json.dumps(chat))
    mismatch["artifacts"]["other"] = {
        **mismatch["artifacts"]["shared_tokenizer"],
        "path": str(alternate),
        "identifier": alternate.parent.name,
        "sha256": sha256_file(alternate),
    }
    mismatch["corpus_variants"][0]["tokenizer_artifact"] = "other"
    mismatch_plan = ExperimentPlan.model_validate(mismatch)
    mismatch_record = prepare_plan(mismatch_plan, path, workspace)["variants"][0]
    with pytest.raises(
        ValueError, match="promotion changes external parent architecture/tokenizer"
    ):
        resolve_plan(
            mismatch_plan,
            path,
            prepared={
                "format": "experiment-preparation-v1",
                "id": mismatch_plan.id,
                "variants": [mismatch_record],
            },
        )
