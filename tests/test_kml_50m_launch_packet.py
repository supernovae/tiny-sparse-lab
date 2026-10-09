"""Static, zero-model review of the prospective 50M declaration packet."""

import json
import shutil
from pathlib import Path

import yaml

from sparselab.corpus.acquisition import _project_sha, declaration_sha256
from sparselab.corpus.project import load_project
from sparselab.operational_monitor import MonitorPolicy
from sparselab.resource_envelope import ResourceEnvelope
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


def test_prefreeze_schema_normalizer_and_acquisition_identity(tmp_path: Path) -> None:
    copy = tmp_path / "packet"
    shutil.copytree(PACKET, copy)
    release = (copy / "release-reviewed.template.yaml").read_text()
    (copy / "release-reviewed.yaml").write_text(
        release.replace(
            "${REVIEWED_ADMISSION_FILE}", "reviewed-admission.json"
        ).replace("${REVIEWED_ADMISSION_SHA256}", "a" * 64)
    )
    prefreeze = load_project(
        copy / "project-pre-freeze-reuse-v2.template.yaml", verify_inputs=False
    )
    acquisition = load_project(
        copy / "project-acquire-reuse-v2.yaml", verify_inputs=False
    )
    assert _project_sha(prefreeze) == _project_sha(acquisition)
    assert prefreeze.release.schema_version == 2
    assert prefreeze.release.normalizer == "normalizer-structure-v3"
    assert prefreeze.release.record_admission is not None
    assert prefreeze.config.splits == "splits-acquire.yaml"


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
