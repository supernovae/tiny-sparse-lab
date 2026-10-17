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
        assert "budget-release-gate" not in compiled["args"]
    with pytest.raises(ValueError, match="missing preparation bindings"):
        compile_phase(path, tmp_path, "pilot-acquire", {})


def test_tokenizer_cannot_bypass_required_release_acceptance(tmp_path):
    paths, _, config_path = setup(tmp_path, require_acceptance=True)
    with pytest.raises(AttemptBudgetError, match="exact release has not been accepted"):
        run(paths, config_path)
    assert AttemptBudget(paths["ledger"]).status()["reservations"] == []
