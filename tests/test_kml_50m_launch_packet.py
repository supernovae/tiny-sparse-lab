"""Static, zero-model review of the prospective 50M declaration packet."""

import json
from pathlib import Path

import pytest
import yaml

from sparselab.corpus.acquisition import _project_sha, declaration_sha256
from sparselab.corpus.declaration_render import render_declaration
from sparselab.corpus.project import load_project
from sparselab.operational_monitor import MonitorPolicy
from sparselab.resource_envelope import ResourceEnvelope
from sparselab.training.attempt_commands import phase_output_paths
from sparselab.training.manifest import sha256_file

PACKET = (
    Path(__file__).resolve().parents[1]
    / "experiments/research/kernel-memory-lab/card05-base-50m"
)
OLD = PACKET.parent / "card05-base"


def test_prospective_acquisition_changes_only_declared_selection() -> None:
    project = load_project(
        PACKET / "project-acquire-reuse-v2.yaml", verify_inputs=False
    )
    prior = load_project(OLD / "project-acquire.yaml", verify_inputs=False)
    historical = load_project(PACKET / "project-acquire.yaml", verify_inputs=False)
    assert historical.config.id == "kernel-memory-lab-card05-base-50m-v1"
    assert "source_effects" not in historical.config.model_dump(mode="json")
    assert _project_sha(historical) == (
        "cc992009c68aaf8e40bfbe070dc09f80c8649b55834ef41aa8995a64f146132a"
    )
    assert project.config.id == "kernel-memory-lab-card05-base-50m-v2"
    assert project.config.transport_budget.max_source_body_bytes == 960485500
    assert project.config.transport_budget.max_metadata_body_bytes == 4194304
    assert project.config.transport_budget.max_retries_per_shard == 1
    assert project.release.lm.selected is False
    assert _project_sha(project) == (
        "12604ffb8ccfe0351f6379ea0f781b3f9b45f6fb4ad9c8cfa80dd0b744f8b83c"
    )
    for source, previous in zip(project.sources[:3], prior.sources[:3], strict=True):
        assert declaration_sha256(source) == declaration_sha256(previous)
    source = project.sources[3]
    previous = prior.sources[3]
    assert source.id == previous.id
    assert source.revision == previous.revision
    assert source.rights == previous.rights
    assert source.acquisition.bounded_shards[0].expected_sha256 == (
        previous.acquisition.bounded_shards[0].expected_sha256
    )
    assert source.acquisition.bounded_shards[0].hash_remainders == (0, 1, 2)
    assert source.acquisition.max_bytes == 402653184


def test_unrendered_admission_cannot_be_mistaken_for_an_admitted_release() -> None:
    template = json.loads(
        (PACKET / "application-template-reuse-v2.template.json").read_text()
    )
    previous = json.loads((OLD / "application-template-v1.json").read_text())
    assert template["sources"] == previous["sources"]
    assert template["policy_sha256"] == sha256_file(
        PACKET / "APPLICATION_POLICY_PROSPECTIVE.md"
    )
    assert template["application_binding"]["project_sha256"] == (
        "12604ffb8ccfe0351f6379ea0f781b3f9b45f6fb4ad9c8cfa80dd0b744f8b83c"
    )
    assert "${" in template["application_binding"]["acquisition_lock_sha256"]
    assert (
        "${"
        in template["application_binding"]["sources"]["kml_scale_wikimedia"][
            "snapshot_sha256"
        ]
    )
    release = yaml.safe_load((PACKET / "release-reviewed.template.yaml").read_text())
    assert release["normalizer"] == "normalizer-structure-v3"
    assert "${" in release["record_admission"]["sha256"]


def test_checked_in_preparation_templates_resolve_map_inputs(tmp_path: Path) -> None:
    """Exercise exact packet templates and native paths, without corpus execution."""
    phases = dict(
        json.loads((PACKET / "preparation-phase-paths-v2.json").read_text())["phases"]
    )
    prep = tmp_path / "attempt" / "prep"

    def leaf(label: str) -> Path:
        return phase_output_paths(prep.parent, label, phases[label])["leaf"]

    def render(
        template: Path, label: str, bindings: dict[str, str] | None = None
    ) -> Path:
        output = leaf(label)
        render_declaration(template, json.dumps(bindings or {}), output, tmp_path)
        return output

    def required(path: Path) -> None:
        if not path.is_file():
            raise ValueError(f"missing required phase input: {path}")

    acquisition = load_project(
        PACKET / "project-acquire-reuse-v2.yaml", verify_inputs=False
    )
    policy = PACKET / "APPLICATION_POLICY_PROSPECTIVE.md"
    selection = PACKET / "admission-inspection-selection-v1.json"
    contract = json.loads(
        (PACKET / "attempt-contract-preparation-only.template.json").read_text()
    )
    for path, expected in (
        (policy, contract["admission_policy_sha256"]),
        (selection, contract["admission_selection_sha256"]),
    ):
        required(path)
        assert sha256_file(path) == expected
    application = render(
        PACKET / "application-template-reuse-v2.template.json",
        "application-declaration",
        {
            "NEW_ACQUISITION_LOCK_SHA256": "a" * 64,
            "NEW_WIKIMEDIA_SNAPSHOT_SHA256": "b" * 64,
        },
    )
    assert json.loads(application.read_text())["policy_sha256"] == sha256_file(policy)

    # These fixture payloads stand in only for future receipts and judgments.
    # They are never passed to corpus admission, inspection or release gates.
    draft = leaf("admission-draft")
    draft.write_text('{"fixture": "draft"}\n')
    inspection = leaf("admission-inspection")
    inspection.write_text('{"fixture": "inspection"}\n')
    for path in (application, draft, inspection, policy, selection):
        required(path)
    admission = render(draft, "reviewed-admission")
    review = render(
        PACKET / "admission-review-v2.template.json",
        "admission-review",
        {
            "REVIEWER": "fixture-only",
            "REVIEWED_ON": "fixture-only",
            "DRAFT_PATH": str(draft),
            "DRAFT_SHA256": sha256_file(draft),
            "ADMISSION_SHA256": sha256_file(admission),
            "INSPECTION_PATH": str(inspection),
            "INSPECTION_SHA256": sha256_file(inspection),
            "ITEM_DECISIONS_JSON": "[]",
        },
    )
    review_payload = json.loads(review.read_text())
    assert review_payload["draft_path"] == str(draft)
    assert review_payload["inspection_path"] == str(inspection)
    for key, path in (("draft_sha256", draft), ("inspection_sha256", inspection)):
        assert review_payload[key] == sha256_file(path)
    required(admission)
    required(review)
    release = render(
        PACKET / "release-reviewed.template.yaml",
        "reviewed-release-declaration",
        {
            "REVIEWED_ADMISSION_FILE": "admission.json",
            "REVIEWED_ADMISSION_SHA256": sha256_file(admission),
        },
    )

    source_templates = {
        "source-pagerduty": "pagerduty.yaml",
        "source-gutenberg": "project-gutenberg.yaml",
        "source-scoutflo": "scoutflo.yaml",
        "source-wikimedia": "wikimedia.yaml",
    }
    for label, name in source_templates.items():
        assert sha256_file(render(PACKET / "sources" / name, label)) == sha256_file(
            PACKET / "sources" / name
        )
    transform = render(PACKET / "transforms/lm.yaml", "transform-declaration")
    placeholder = render(
        PACKET / "splits-acquire.yaml", "placeholder-split-declaration"
    )
    assert sha256_file(transform) == sha256_file(PACKET / "transforms/lm.yaml")
    assert sha256_file(placeholder) == sha256_file(PACKET / "splits-acquire.yaml")

    prefreeze_path = render(
        PACKET / "project-pre-freeze-reuse-v2.template.yaml", "pre-freeze-declaration"
    )
    prefreeze = load_project(prefreeze_path, verify_inputs=False)
    assert prefreeze.config.sources == tuple(
        sorted(str(leaf(label).relative_to(prep)) for label in source_templates)
    )
    assert prefreeze.config.transforms == ()
    assert prefreeze.config.splits == str(placeholder.relative_to(prep))
    assert prefreeze.config.release == str(release.relative_to(prep))
    for reference in (
        *prefreeze.config.sources,
        prefreeze.config.splits,
        prefreeze.config.release,
    ):
        required(prep / reference)
    assert (prep / prefreeze.release.record_admission.path) == admission
    assert prefreeze.release.record_admission.sha256 == sha256_file(admission)
    assert _project_sha(prefreeze) == _project_sha(acquisition)
    assert prefreeze.release.schema_version == 2
    assert prefreeze.release.normalizer == "normalizer-structure-v3"

    reviewed_splits = render(PACKET / "splits-acquire.yaml", "family-freeze")
    build_path = render(
        PACKET / "project-build-reuse-v2.template.yaml",
        "final-build-declaration",
        {"REVIEWED_SPLITS_FILE": str(reviewed_splits.relative_to(prep))},
    )
    build = load_project(build_path, verify_inputs=False)
    assert build.config.sources == prefreeze.config.sources
    assert build.config.transforms == (str(transform.relative_to(prep)),)
    assert build.config.splits == str(reviewed_splits.relative_to(prep))
    assert build.config.release == str(release.relative_to(prep))
    for reference in (
        *build.config.sources,
        *build.config.transforms,
        build.config.splits,
        build.config.release,
    ):
        required(prep / reference)
    assert _project_sha(build) == _project_sha(acquisition)

    # Old B17 names cannot resolve in this map-only layout; no alias is made.
    for old in (
        "sources/pagerduty.yaml",
        "sources/project-gutenberg.yaml",
        "sources/scoutflo.yaml",
        "sources/wikimedia.yaml",
        "splits-acquire.yaml",
    ):
        with pytest.raises(ValueError, match="missing required phase input"):
            required(prep / old)
    with pytest.raises(ValueError, match="missing required phase input"):
        required(prep / "sources/removed.yaml")
    with pytest.raises(ValueError, match="declaration bindings do not exactly cover"):
        render_declaration(
            PACKET / "project-build-reuse-v2.template.yaml",
            "{}",
            prep / "unresolved-build.yaml",
            tmp_path,
        )
    review.unlink()
    with pytest.raises(ValueError, match="missing required phase input"):
        required(review)
    leaf("source-pagerduty").unlink()
    with pytest.raises(ValueError, match="missing or symlinked declaration"):
        load_project(prefreeze_path, verify_inputs=False)


def test_caps_scientific_settings_and_dynamic_identity_slots() -> None:
    mixture = yaml.safe_load((PACKET / "mixture.template.yaml").read_text())
    assert mixture["target_quotas"] == {
        "general_prose": 40500000,
        "explanatory_prose": 9000000,
        "incident_response_docs": 500000,
    }
    assert mixture["max_exposures"] == 2
    assert mixture["tokenizer_origin_release_id"] == (
        "e893cb2e3c65f7f6360f73e4daa159dcec5da1b4c2c6f911ff5d9d549b815818"
    )
    floors = yaml.safe_load((PACKET / "token-floors.yaml").read_text())
    assert floors["min_unique_train_tokens_by_domain"] == {
        "general_prose": 20250000,
        "explanatory_prose": 4500000,
        "incident_response_docs": 250000,
    }
    template = (PACKET / "RUN_CONFIG_TEMPLATE_REUSE_V2.md").read_text()
    config = yaml.safe_load(template.split("```yaml\n", 1)[1].split("\n```", 1)[0])
    assert (config["seed"], config["runtime"]["backend"]) == (17, "rocm")
    assert config["training"]["max_steps"] == 48829
    assert config["training"]["max_tokens"] == 50000000
    assert config["optimizer"]["decay_steps"] == 48829
    assert config["optimizer"]["warmup_steps"] == 500
    assert config["checkpoint"]["every_steps"] == 5000
    assert config["evaluation"]["every_steps"] == 5000
    assert "${" in config["dataset"]["revision"]
    selection = json.loads((PACKET / "selection.template.json").read_text())
    assert selection["steps"] == [
        0,
        5000,
        10000,
        15000,
        20000,
        25000,
        30000,
        35000,
        40000,
        45000,
        48829,
    ]
    assert selection["target_positions"] == 50000000
    contract = json.loads(
        (PACKET / "attempt-contract-reuse-v2.template.json").read_text()
    )
    assert contract["max_optimizer_updates"] == 48829
    assert contract["max_actual_target_positions"] == 50000000
    assert contract["max_fixed_profile_forward_positions"] == 55000
    assert contract["max_nontraining_forward_positions"] == 170000
    assert contract["max_generation_calls"] == 16
    assert contract["max_generated_tokens"] == 1024
    assert contract["max_wall_seconds"] == 36000
    assert contract["workspace_baseline_sha256"] == (
        "${VERIFIED_WORKSPACE_BASELINE_IDENTITY_SHA256}"
    )
    assert contract["require_preledger_monitor_binding"] is True
    preparation = MonitorPolicy.model_validate(
        yaml.safe_load((PACKET / "monitor-preparation.yaml").read_text())
    )
    whole = MonitorPolicy.model_validate(
        yaml.safe_load((PACKET / "monitor-whole.yaml").read_text())
    )
    eval_output = MonitorPolicy.model_validate(
        yaml.safe_load((PACKET / "monitor-eval-output.yaml").read_text())
    )
    envelope = ResourceEnvelope.model_validate(
        yaml.safe_load((PACKET / "resource-envelope.yaml").read_text())
    )
    assert preparation.max_added_workspace_bytes == 8 * 1024**3
    assert whole.max_added_workspace_bytes == 72 * 1024**3
    assert whole.max_device_memory_bytes == 20 * 1024**3
    assert whole.max_tree_rss_bytes == 24 * 1024**3
    assert eval_output.max_added_workspace_bytes == 64 * 1024**2
    assert envelope.min_disk_bytes == 74 * 1024**3
