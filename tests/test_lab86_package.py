"""Production LAB86 declarations; metadata and tiny fixtures only, no HTTP."""

import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from sparselab.cli.main import build_parser
from sparselab.config.models import TokenizerSampling
from sparselab.corpus.acquisition import _bounded_hf_rows, _project_sha
from sparselab.corpus.admission_inspection import _selection
from sparselab.corpus.declaration_render import render_declaration
from sparselab.corpus.mixture import MixtureDeclaration
from sparselab.corpus.project import ReleaseDeclaration, load_project
from sparselab.corpus.row_qualification import exceptions
from sparselab.operational_monitor import MonitorPolicy
from sparselab.training.attempt_budget import AttemptContract
from sparselab.training.manifest import canonical_json
from sparselab.training.preparation import compile_phase, load_preparation

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/research/exp1-preparation/lab86"
PLAN = PACKAGE.parent / "preparation.json"
GIB = 1024**3


def read(name):
    return yaml.safe_load((PACKAGE / name).read_text())


def test_full_scope_selection_and_first_gate():
    manifest = read("shards.json")
    catalog = read("evidence/candidate-catalog.json")
    rows = manifest["shards"]
    assert [len(catalog[k]) for k in ("web", "synthetic", "code")] == [234, 104, 880]
    expected = []
    remainder = []
    for domain, candidates in catalog.items():
        pilot = next(r for r in rows if r["domain"] == domain and r["pilot"])
        assert (
            pilot["path"]
            == min(candidates, key=lambda r: (r["bytes"], r["path"]))["path"]
        )
        total = pilot["bytes"]
        expected.append(pilot["path"])
        ranked = sorted(
            (r for r in candidates if r["path"] != pilot["path"]),
            key=lambda r: hashlib.sha256(
                (
                    manifest["selection_seed"]
                    + "\n"
                    + pilot["repository"]
                    + "\n"
                    + pilot["revision"]
                    + "\n"
                    + r["path"]
                ).encode()
            ).hexdigest(),
        )
        for row in ranked:
            if total + row["bytes"] > manifest["compressed_caps_bytes"][domain]:
                break
            remainder.append(row["path"])
            total += row["bytes"]
    assert [r["path"] for r in rows] == expected + remainder
    assert len(rows) == 84
    assert sum(r["bytes"] for r in rows) == manifest["selected_bytes"] == 85077359484
    assert sum(r["bytes"] for r in rows[:3]) == manifest["pilot_bytes"] == 3771713555
    assert rows[3]["bytes"] > 4 * GIB - manifest["pilot_bytes"]
    assert sorted(r["source_id"] for r in rows) == [r["source_id"] for r in rows]


def test_real_project_sources_budget_and_policy():
    project = load_project(PACKAGE / "project-acquire.yaml")
    rows = read("shards.json")["shards"]
    assert len(project.sources) == len(rows)
    for source, row in zip(project.sources, rows, strict=True):
        shard = source.acquisition.bounded_shards[0]
        assert source.id == row["source_id"]
        assert source.revision == row["revision"]
        assert shard.path == row["path"]
        assert shard.expected_sha256 == row["sha256"]
        assert shard.max_shard_bytes == row["bytes"]
        assert source.acquisition.max_decompressed_bytes is None
        assert source.rights.training_eligibility == "review_required"
        assert source.acquisition.qualification is not None
    budget = project.config.transport_budget
    assert budget.initial_body_bytes == 4 * GIB
    assert budget.max_total_body_bytes == 100 * GIB
    assert budget.max_source_body_bytes + budget.max_metadata_body_bytes == 100 * GIB
    assert budget.max_disk_bytes == 300 * GIB
    assert budget.max_wall_seconds == 172800
    assert not project.release.lm.selected
    for name, count in [
        ("application-policy.json", 84),
        ("pilot-application-policy.json", 3),
    ]:
        policy = read(name)
        assert len(policy["sources"]) == count
        assert (
            policy["policy_sha256"]
            == hashlib.sha256((PACKAGE / "POLICY.md").read_bytes()).hexdigest()
        )
        assert {s["rights"]["training_eligibility"] for s in policy["sources"]} == {
            "review_required"
        }


def test_zero_model_contract_and_resource_policies():
    raw = read("attempt-contract.template.json")
    for key, value in list(raw.items()):
        if isinstance(value, str) and value.startswith("${"):
            raw[key] = "a" * 64  # Fixture identity only; never written to production.
    contract = AttemptContract.model_validate(raw)
    assert all(
        value == 0
        for key, value in raw.items()
        if key.startswith("max_") and key != "max_wall_seconds"
    )
    assert contract.preparation_only and contract.require_release_acceptance_binding
    assert (
        contract.acquisition_project_sha256
        == hashlib.sha256((PACKAGE / "project-acquire.yaml").read_bytes()).hexdigest()
    )
    assert contract.preparation_acquisition_identity_sha256 == _project_sha(
        load_project(PACKAGE / "project-acquire.yaml")
    )
    policy = MonitorPolicy.model_validate(read("monitor.yaml"))
    assert policy.max_tree_rss_bytes == 16 * GIB
    assert policy.max_added_workspace_bytes == 300 * GIB
    assert (
        contract.monitor_policy_sha256
        == hashlib.sha256((PACKAGE / "monitor.yaml").read_bytes()).hexdigest()
    )
    sampling = TokenizerSampling.model_validate(read("tokenizer-sampling.yaml"))
    assert sampling.byte_weights == {"web": 60, "synthetic": 20, "code": 20}
    assert sampling.max_fit_bytes == 512 * 1024**2
    mixture = read("mixture.template.yaml")
    mixture["tokenizer_origin_release_id"] = "a" * 64
    assert MixtureDeclaration.model_validate(mixture).max_exposures == 1
    assert read("family-clusters.template.json")["split_weights"] == {
        "train": 98,
        "validation": 1,
        "test": 1,
    }


def test_every_production_phase_compiles_and_has_matching_job(tmp_path):
    plan = load_preparation(PLAN)
    session = read("session.template.json")
    assert len(session["slots"]) == len(plan["phases"]) == 24
    for slot, (label, phase) in zip(
        session["slots"], plan["phases"].items(), strict=True
    ):
        assert slot["id"] == label
        needed = {a[2:-1] for a in phase["args"] if a.startswith("${")} - plan[
            "defaults"
        ].keys()
        bindings = {key: str(tmp_path / key) for key in needed}
        compiled = compile_phase(PLAN, tmp_path, label, bindings)
        build_parser().parse_args(compiled["args"])
        job = read(f"jobs/{label}.template.json")
        assert job["argv"] == slot["argv"]
        assert job["environment_profile"] == slot["environment_profile"]
        assert "--execute-runs" not in job["argv"]
        if needed:
            with pytest.raises(ValueError, match="missing preparation bindings"):
                compile_phase(PLAN, tmp_path, label, {})


def test_native_timestamp_support_and_code_language_boundary(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "tiny.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"text": "tiny", "metadata": {"date": datetime(2024, 1, 1, tzinfo=UTC)}}]
        ),
        path,
    )
    row = next(_bounded_hf_rows(path, limit=1, max_line_bytes=1024))[1]
    assert row["metadata"]["date"] == "2024-01-01T00:00:00+00:00"
    assert canonical_json(row)
    code = load_project(PACKAGE / "project-acquire.yaml").sources[2]
    reason, _ = exceptions(
        {
            "code": "print(1)",
            "repo_name": "example/project",
            "path": "main.py",
            "license": "mit",
        },
        code.acquisition.qualification,
        "code",
    )
    assert reason == "record outside declared language policy"


def test_package_input_hashes():
    manifest = read("package-sha256.json")
    expected = {
        p.relative_to(PACKAGE).as_posix()
        for p in PACKAGE.rglob("*")
        if p.is_file() and p.name != "package-sha256.json"
    } | {"../preparation.json"}
    assert set(manifest) == expected
    for relative, digest in manifest.items():
        assert hashlib.sha256((PACKAGE / relative).read_bytes()).hexdigest() == digest


def test_pilot_boundary_has_only_host_bindings_and_no_future_evidence():
    session = read("pilot-session.template.json")
    assert [s["id"] for s in session["slots"]] == [
        "attempt-init",
        "transport-init",
        "pilot-acquire",
        "pilot-budget-status",
    ]
    identity = hashlib.sha256((PACKAGE / "pilot-content.json").read_bytes()).hexdigest()
    contract = read("pilot-contract.template.json")
    assert contract["content_identity_sha256"] == identity
    assert [
        v for v in contract.values() if isinstance(v, str) and v.startswith("${")
    ] == ["${HOST_WORKSPACE_BASELINE_SHA256}"]
    for slot in session["slots"]:
        job = read(f"pilot-jobs/{slot['id']}.template.json")
        assert job["argv"] == slot["argv"]
        assert "GENERATED_" not in str(job)
        assert "BASE_RUN_CONFIG" not in str(job)
        if slot["id"] != "attempt-init":
            assert job["argv"][-1] == identity


def _render_package_templates(package, output_root):
    """Render production text with fixture bindings, never execute generated argv."""
    output_root.mkdir(parents=True, exist_ok=True)
    fixture = output_root / "package"
    shutil.copytree(package, fixture)
    outputs = {}
    for template in sorted(fixture.rglob("*.template.*")):
        slots = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", template.read_text()))
        values = {}
        for key in slots:
            if key.endswith("_JSON"):
                value = "[]"
            elif "SHA256" in key or key == "GENERATED_RELEASE_ID":
                value = "a" * 64
            elif key == "HOST_REVIEWED_LAB_COMMIT":
                value = "b" * 40
            else:
                value = "/fixture/" + key
            values[key] = value
        relative = {
            "GENERATED_REVIEWED_RELEASE_DECLARATION": "release-reviewed.yaml",
            "GENERATED_FROZEN_SPLITS": "splits-acquire.yaml",
            "GENERATED_REVIEWED_ADMISSION_PATH": "fixture-admission.json",
        }
        values.update({key: value for key, value in relative.items() if key in slots})
        destination = template.with_name(template.name.replace(".template.", "."))
        render_declaration(template, json.dumps(values), destination, output_root)
        outputs[template.relative_to(fixture).as_posix()] = destination
    return fixture, outputs


def _assert_downstream_semantics(package, output_root):
    fixture, outputs = _render_package_templates(package, output_root)
    full = load_project(fixture / "project-acquire.yaml")
    pilot = load_project(outputs["project-pilot-reuse.template.yaml"])
    assert pilot.config.sources == full.config.sources[:3]
    assert pilot.config.transport_budget is None
    assert [s.id for s in pilot.sources] == [s.id for s in full.sources[:3]]
    for effect, original in zip(
        pilot.config.source_effects, full.config.source_effects[:3], strict=True
    ):
        assert effect.effect == "reuse_only"
        assert effect.origin_project_id == full.config.id
        assert effect.source_id == original.source_id
        assert effect.declaration_sha256 == original.declaration_sha256
        assert effect.snapshot_sha256 == "a" * 64
    for name in ("project-pre-freeze.template.yaml", "project-build.template.yaml"):
        project = load_project(outputs[name])
        assert project.config.id == full.config.id
        assert project.config.sources == full.config.sources
        assert project.config.source_effects == full.config.source_effects
        assert project.config.transport_budget == full.config.transport_budget
        assert _project_sha(project) == _project_sha(full)
        assert project.release.record_admission is not None
        assert project.release.normalizer == "normalizer-structure-v3"
    release = ReleaseDeclaration.model_validate(
        yaml.safe_load(outputs["release-reviewed.template.yaml"].read_text())
    )
    assert release.lm.training_splits == ("train", "validation")
    assert release.lm.selected and not release.chat.selected
    assert release.mixture == {"web": 0.6, "synthetic": 0.2, "code": 0.2}
    clusters = json.loads(outputs["family-clusters.template.json"].read_text())
    assert isinstance(clusters["merges"], list)
    for name, project in (
        ("inspection-selection.json", full),
        ("pilot-inspection-selection.json", pilot),
    ):
        _selection(fixture / name, fixture, {s.id for s in project.sources})
    plan = load_preparation(PLAN)
    session = json.loads(outputs["session.template.json"].read_text())
    assert [s["id"] for s in session["slots"]] == list(plan["phases"])
    for label in plan["phases"]:
        job = yaml.safe_load(outputs[f"jobs/{label}.template.json"].read_text())
        slot = next(s for s in session["slots"] if s["id"] == label)
        assert job["argv"] == slot["argv"]
    return outputs


def test_all_templates_render_and_downstream_semantics_hold(tmp_path):
    outputs = _assert_downstream_semantics(PACKAGE, tmp_path)
    assert len(outputs) == len(list(PACKAGE.rglob("*.template.*")))


@pytest.mark.parametrize(
    "mutation",
    [
        "pilot_digest",
        "pilot_origin",
        "pre_freeze_source",
        "build_source",
        "build_budget",
        "test_training_split",
        "inspection_id",
        "slot_id",
        "lowercase_slot",
    ],
)
def test_coordinated_template_tampering_is_rejected(tmp_path, mutation):
    package = tmp_path / "mutated"
    shutil.copytree(PACKAGE, package)
    name = {
        "pilot_digest": "project-pilot-reuse.template.yaml",
        "pilot_origin": "project-pilot-reuse.template.yaml",
        "pre_freeze_source": "project-pre-freeze.template.yaml",
        "build_source": "project-build.template.yaml",
        "build_budget": "project-build.template.yaml",
        "test_training_split": "release-reviewed.template.yaml",
        "inspection_id": "inspection-selection.json",
        "slot_id": "session.template.json",
        "lowercase_slot": "project-pilot-reuse.template.yaml",
    }[mutation]
    path = package / name
    value = yaml.safe_load(path.read_text())
    if mutation == "pilot_digest":
        value["source_effects"][0]["declaration_sha256"] = "b" * 64
    elif mutation == "pilot_origin":
        value["source_effects"][0]["origin_project_id"] = "different-origin"
    elif mutation in {"pre_freeze_source", "build_source"}:
        value["sources"].pop()
        value["source_effects"].pop()  # Consistent lists still violate full scope.
    elif mutation == "build_budget":
        value["transport_budget"]["max_total_body_bytes"] *= 2
    elif mutation == "test_training_split":
        value["lm"]["training_splits"].append("test")
    elif mutation == "inspection_id":
        value["sources"][0]["source_id"] = "unknown-source"
    elif mutation == "slot_id":
        value["slots"][0]["id"] = "wrong-phase"
    else:
        value["source_effects"][0]["snapshot_sha256"] = (
            "${GENERATED_SNAPSHOT_SHA256_lowercase}"
        )
    path.write_text(
        json.dumps(value, indent=2)
        if path.suffix == ".json"
        else yaml.safe_dump(value, sort_keys=False)
    )
    manifest = json.loads((package / "package-sha256.json").read_text())
    manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (package / "package-sha256.json").write_text(json.dumps(manifest))
    # Manifest is deliberately consistent; semantic validation must kill the mutant.
    with pytest.raises((AssertionError, ValueError)):
        _assert_downstream_semantics(package, tmp_path / "rendered")


def test_verbatim_source_evidence_hashes():
    for name in ("code-README.md.json", "smollm-README.md.json", "code-loader.py.json"):
        evidence = read("evidence/" + name)
        assert (
            hashlib.sha256(evidence["text"].encode()).hexdigest()
            == evidence["raw_sha256"]
        )


def test_host_object_binding_is_data_not_a_quoted_mapping(tmp_path):
    _, outputs = _render_package_templates(PACKAGE, tmp_path)
    job = json.loads(outputs["pilot-jobs/pilot-acquire.template.json"].read_text())
    runtime = {"python": "3.14.4", "executable_sha256": "a" * 64}
    required = {"/fixture/project-acquire.yaml": "b" * 64}
    for field, value in (("runtime", runtime), ("required_files", required)):
        path = tmp_path / (field + ".json")
        path.write_text(json.dumps(value))
        job[field] = str(path)
        parsed = json.loads(Path(job[field]).read_text())
        assert isinstance(parsed, dict) and parsed
        job[field] = parsed
    assert job["runtime"] == runtime
    assert job["required_files"] == required
