"""Supervised native tiny BPE preparation, no model or network workload."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from test_attempt_packet_dispatch import _fixture, _wrapped
from test_kml_50m_preparation_sequence import _prior

from sparselab.config.loading import load_tokenizer_config
from sparselab.config.models import TokenizerSampling
from sparselab.corpus.export import export_release
from sparselab.data.tokenizer import verify_tokenizer_artifact
from sparselab.training.attempt_budget import AttemptBudget, AttemptBudgetError


def setup(tmp_path, *, require_acceptance=False):
    paths = _fixture(tmp_path, initialize=False)
    root = paths["root"]
    contract = json.loads(paths["contract"].read_text())
    contract.update(
        preparation_only=True,
        offline_retained_sources_only=True,
        require_release_acceptance_binding=require_acceptance,
    )
    paths["contract"].write_text(json.dumps(contract))
    AttemptBudget.create_contract(
        paths["ledger"],
        contract_path=paths["contract"],
        expected_sha256=hashlib.sha256(paths["contract"].read_bytes()).hexdigest(),
    )
    release, _, strata = _prior(root)
    exported = export_release(
        release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 260, root
    )
    config = load_tokenizer_config(exported / "tokenizer.yaml")
    sampling = TokenizerSampling(
        source_strata=strata,
        byte_weights=dict.fromkeys(strata.values(), 1),
        max_fit_bytes=min(6000, config.dataset.train_max_tokens),
        audit_documents_per_stratum=1,
    )
    config = config.model_copy(update={"sampling": sampling})
    config_path = root / "sampled-tokenizer.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    return paths, config, config_path


def run(paths, config_path):
    from sparselab.training.preparation import compile_phase

    plan = Path("experiments/research/exp1-preparation/preparation.json")
    compiled = compile_phase(
        plan, paths["root"], "fit-tokenizer", {"TOKENIZER_CONFIG": str(config_path)}
    )
    return AttemptBudget(paths["ledger"]).run_contract(
        _wrapped(paths, *compiled["args"]),
        activity="inspect",
        label="tokenizer-fit",
        content_identity_sha256=str(paths["identity"]),
        monitor_policy_path=paths["whole"],
        workspace_baseline_path=paths["baseline"],
        workspace_root=paths["root"],
        completion=paths["root"] / "tokenizer-completion.json",
    )


def test_native_sampled_tokenizer_runs_under_zero_model_preparation_contract(tmp_path):
    paths, config, config_path = setup(tmp_path)
    result = run(paths, config_path)
    path = config.output_dir / "tokenizer.json"
    assert path.is_file(), result
    manifest = verify_tokenizer_artifact(
        path,
        source="local_text",
        revision=config.dataset.revision,
        vocab_size=260,
        dataset=config.dataset,
    )
    assert manifest["sampling_audit"]["disjoint_from_fit"]
    status = AttemptBudget(paths["ledger"]).status()
    assert len(status["reservations"]) == 1
    assert status["reservations"][0]["updates"] == 0


@pytest.mark.parametrize("change", ["outside", "existing", "unsampled"])
def test_invalid_tokenizer_output_or_unbounded_input_refuses_before_reservation(
    tmp_path, change
):
    paths, config, config_path = setup(tmp_path)
    payload = config.model_dump(mode="json")
    if change == "outside":
        payload["output_dir"] = str(tmp_path / "outside")
    elif change == "existing":
        config.output_dir.mkdir(parents=True)
    else:
        payload.pop("sampling")
    config_path.write_text(yaml.safe_dump(payload))
    with pytest.raises(AttemptBudgetError, match="tokenizer"):
        run(paths, config_path)
    assert AttemptBudget(paths["ledger"]).status()["reservations"] == []


def test_exp1_plan_compiles_every_public_leaf_and_refuses_missing_bindings(tmp_path):
    from sparselab.cli.main import build_parser
    from sparselab.training.preparation import compile_phase, load_preparation

    path = Path("experiments/research/exp1-preparation/preparation.json")
    plan = load_preparation(path)
    for label, phase in plan["phases"].items():
        required = {word[2:-1] for word in phase["args"] if word.startswith("${")}
        bindings = {
            key: str(tmp_path / key) for key in required - plan["defaults"].keys()
        }
        compiled = compile_phase(path, tmp_path, label, bindings)
        build_parser().parse_args(compiled["args"])
        from sparselab.training.attempt_commands import classify_attempt_command

        classified = classify_attempt_command(
            _wrapped(
                {
                    "root": tmp_path,
                    "prep": tmp_path / "policy",
                    "baseline": tmp_path / "baseline",
                },
                *compiled["args"],
            )
        )
        assert classified.effect in {"preparation", "inspection"}
        assert "budget-release-gate" not in compiled["args"]
    with pytest.raises(ValueError, match="missing preparation bindings"):
        compile_phase(path, tmp_path, "pilot-acquire", {})


def test_tokenizer_cannot_bypass_required_release_acceptance(tmp_path):
    paths, _, config_path = setup(tmp_path, require_acceptance=True)
    with pytest.raises(AttemptBudgetError, match="exact release has not been accepted"):
        run(paths, config_path)
    assert AttemptBudget(paths["ledger"]).status()["reservations"] == []


def run_words(paths, *words):
    return AttemptBudget(paths["ledger"]).run_contract(
        _wrapped(paths, *words),
        activity="inspect",
        label="guard-fixture",
        content_identity_sha256=str(paths["identity"]),
        monitor_policy_path=paths["whole"],
        workspace_baseline_path=paths["baseline"],
        workspace_root=paths["root"],
        completion=paths["root"] / "guard-completion.json",
    )


@pytest.mark.parametrize(
    "change", ["relative", "dangling", "monitor", "tampered_export"]
)
def test_tokenizer_preflight_guards_before_reservation(tmp_path, change):
    paths, config, config_path = setup(tmp_path)
    payload = config.model_dump(mode="json")
    reason = "tokenizer"
    if change == "relative":
        payload["output_dir"] = "relative-tokenizer"
    elif change == "dangling":
        config.output_dir.symlink_to(
            paths["root"] / "missing-output", target_is_directory=True
        )
    elif change == "monitor":
        payload["output_dir"] = str(paths["root"] / "preparation-monitor")
        reason = "collision"
    else:
        config.dataset.train_path.write_text("tampered fixture export\n")
        reason = "tampered artifact"
    config_path.write_text(yaml.safe_dump(payload))
    with pytest.raises((ValueError, AttemptBudgetError), match=reason):
        run(paths, config_path)
    assert AttemptBudget(paths["ledger"]).status()["reservations"] == []


def test_export_base_config_is_protected_input(tmp_path):
    from sparselab.training.attempt_budget import validate_attempt_paths
    from sparselab.training.attempt_commands import classify_attempt_command

    command = classify_attempt_command(
        _wrapped(
            {
                "root": tmp_path,
                "prep": tmp_path / "policy",
                "baseline": tmp_path / "baseline",
            },
            "corpus",
            "export",
            str(tmp_path / "release"),
            "--view",
            "lm",
            "--base-run-config",
            str(tmp_path / "completion.json"),
            "--vocab-size",
            "260",
        )
    )
    with pytest.raises(AttemptBudgetError, match="aliases protected input"):
        validate_attempt_paths(
            command,
            ledger=tmp_path / "ledger",
            completion=tmp_path / "completion.json",
            monitor_log_dir=tmp_path / "outer-monitor",
            monitor_policy=tmp_path / "whole",
            baseline=tmp_path / "baseline",
        )


@pytest.mark.parametrize("missing", ["--view", "--base-run-config", "--vocab-size"])
def test_export_requires_native_options(tmp_path, missing):
    from sparselab.training.attempt_commands import classify_attempt_command

    args = ["corpus", "export", str(tmp_path / "release")]
    for option, value in (
        ("--view", "lm"),
        ("--base-run-config", str(tmp_path / "config")),
        ("--vocab-size", "260"),
    ):
        if option != missing:
            args.extend([option, value])
    with pytest.raises(ValueError, match="required|missing|requires"):
        classify_attempt_command(
            _wrapped(
                {
                    "root": tmp_path,
                    "prep": tmp_path / "policy",
                    "baseline": tmp_path / "baseline",
                },
                *args,
            )
        )


@pytest.mark.parametrize("operation", ["tokenizer", "export"])
@pytest.mark.parametrize("acceptance", ["missing", "mismatching"])
def test_exact_release_acceptance_before_reservation(
    tmp_path, monkeypatch, operation, acceptance
):
    paths, config, config_path = setup(tmp_path, require_acceptance=True)
    if acceptance == "mismatching":
        # Isolate matching against an already-resolved acceptance. Native receipt
        # authentication is independently covered by the connected sequence.
        accepted = paths["root"] / "accepted-fixture.json"
        accepted.write_text(
            json.dumps({"release_path": str(paths["root"] / "different-release")})
        )
        monkeypatch.setattr(
            AttemptBudget,
            "_resolved_artifact",
            lambda self, kind, **kw: (
                (accepted, "fixture") if kind == "release_acceptance" else None
            ),
        )
    reason = (
        "not been accepted"
        if acceptance == "missing"
        else "release differs from acceptance"
    )
    with pytest.raises(AttemptBudgetError, match=reason):
        if operation == "tokenizer":
            run(paths, config_path)
        else:
            run_words(
                paths,
                "corpus",
                "export",
                str(config.dataset.corpus_release_path),
                "--view",
                "lm",
                "--base-run-config",
                str(Path("configs/runtime_smoke_cpu.yaml").resolve()),
                "--vocab-size",
                "260",
            )
    assert AttemptBudget(paths["ledger"]).status()["reservations"] == []


@pytest.mark.parametrize("change", [None, "acquire", "allocation", "unbound"])
def test_pilot_dispatch_enforces_bound_reuse_only_without_allocation(tmp_path, change):
    import shutil

    from test_kml_50m_preparation_sequence import _sources

    from sparselab.corpus.acquisition import verify_acquisition
    from sparselab.corpus.project import load_project

    paths = _fixture(tmp_path, initialize=False)
    root = paths["root"]
    origin, _ = _sources(root, "origin")
    lock = verify_acquisition(load_project(origin / "acquire.yaml"), root)
    pilot = root / "pilot"
    shutil.copytree(origin, pilot)
    payload = yaml.safe_load((pilot / "acquire.yaml").read_text())
    payload["id"] = "pilot"
    payload["source_effects"] = [
        {
            "source_id": source_id,
            "effect": "reuse_only",
            "origin_project_id": "origin",
            "declaration_sha256": item["declaration_sha256"],
            "snapshot_sha256": item["snapshot_sha256"],
        }
        for source_id, item in lock["sources"].items()
    ]
    if change == "acquire":
        effect = payload["source_effects"][0]
        effect.update(effect="acquire")
        effect.pop("origin_project_id")
        effect.pop("snapshot_sha256")
    elif change == "allocation":
        payload["transport_budget"] = {
            "attempt_id": "fixture",
            "max_source_body_bytes": 1000,
            "max_metadata_body_bytes": 1000,
            "max_transfers": 10,
            "max_retries_per_shard": 1,
            "max_wall_seconds": 100,
            "max_disk_bytes": 10000,
        }
    recipe = pilot / "acquire.yaml"
    recipe.write_text(yaml.safe_dump(payload))
    contract = json.loads(paths["contract"].read_text())
    contract.update(
        preparation_only=True,
        offline_retained_sources_only=True,
        acquisition_project_sha256=None
        if change == "unbound"
        else hashlib.sha256(recipe.read_bytes()).hexdigest(),
    )
    paths["contract"].write_text(json.dumps(contract))
    AttemptBudget.create_contract(
        paths["ledger"],
        contract_path=paths["contract"],
        expected_sha256=hashlib.sha256(paths["contract"].read_bytes()).hexdigest(),
    )
    if change is None:
        run_words(paths, "corpus", "acquire", str(recipe))
        assert verify_acquisition(load_project(recipe), root)["project_id"] == "pilot"
        assert not (root / "corpora/pilot/transport-budget.sqlite").exists()
    else:
        with pytest.raises(AttemptBudgetError, match="verified offline reuse only"):
            run_words(paths, "corpus", "acquire", str(recipe))
        assert AttemptBudget(paths["ledger"]).status()["reservations"] == []
