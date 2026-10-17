"""Production LAB86 declarations; metadata and tiny fixtures only, no HTTP."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from sparselab.cli.main import build_parser
from sparselab.config.models import TokenizerSampling
from sparselab.corpus.acquisition import _bounded_hf_rows, _project_sha
from sparselab.corpus.mixture import MixtureDeclaration
from sparselab.corpus.project import load_project
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
    for relative, digest in read("package-sha256.json").items():
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
