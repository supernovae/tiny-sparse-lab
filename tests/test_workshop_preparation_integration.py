"""Optional connected Workshop authoring -> native validated stage, no model run.

Fixture substitutions are a tiny synthetic corpus and prebuilt WordLevel
vocabulary. The Workshop model, 2048-token context, effective batch and declared
12B run budget remain unchanged; this is wiring evidence, not scientific supply.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.corpus import mixture, pipeline
from sparselab.corpus.release import freeze
from sparselab.model.inspection import parameter_inventory
from sparselab.staging import materialize_prepared_inputs, stage, verify_stage_bundle


def test_workshop_binder_to_native_stage_bundle(tmp_path, monkeypatch):
    checkout = os.environ.get("WORKSHOP_CHECKOUT")
    if checkout is None:
        pytest.skip("set WORKSHOP_CHECKOUT to the reviewed Workshop checkout")
    from test_completed_preparation import fixture
    from test_kml_50m_preparation_sequence import _authenticated_fixture_tokenizer

    workshop = Path(checkout)
    module_spec = importlib.util.spec_from_file_location(
        "workshop_binder", workshop / "scripts/bind-experiment.py"
    )
    binder = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(binder)
    plan = workshop / "experiments/devmind/experiment.json"
    root, _, project = fixture(tmp_path, monkeypatch, text_repetitions=300)
    built = pipeline.build(project, root, reuse_completed_preparation=True)
    release = freeze(built, root)
    tokenizer_config, tokenizer_path = _authenticated_fixture_tokenizer(
        root, release, vocab_size=32768
    )
    fit = load_tokenizer_config(tokenizer_config)
    declaration = root / "mixture.yaml"
    declaration.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "release_path": str(release),
                "family_inventory": str(release / "family-inventory.jsonl"),
                "tokenizer_config": str(tokenizer_config),
                "source_strata": {project.sources[0].id: "general_prose"},
                "target_quotas": {"general_prose": 4096},
                "max_exposures": 1,
                "seed": 42,
            }
        )
    )
    output = root / "mixture"
    receipt = mixture.materialize_mixture(declaration, output)
    bindings = binder.read(plan.with_name("binding.example.json"))
    bindings["dataset"].update(
        revision=release.name,
        mixture_declaration_path=str(declaration),
        mixture_output_path=str(output),
        cache_dir=str(root / "packed"),
        train_path=str(output / "train.tokens.jsonl"),
        validation_path=str(release / "lm/validation.jsonl"),
        train_max_documents=receipt["records"],
        validation_max_documents=1,
        train_max_tokens=4096,
        validation_max_tokens=fit.dataset.validation_max_tokens,
        license=fit.dataset.license,
    )
    bindings["tokenizer"]["path"] = str(tokenizer_path)
    bindings["runtime"]["backend"] = "cpu"
    bindings["logging"]["root_dir"] = str(root / "runs")
    binding_path = root / "bindings.json"
    binding_path.write_text(json.dumps(bindings))
    config_path = root / "run.json"
    binder.bind(plan, binding_path, config_path)
    config = load_config(config_path)
    assert parameter_inventory(config).total == 341885952
    assert config.training.seq_len == 2048
    assert config.training.max_tokens == 12000000000
    assert (
        config.model.model_dump(mode="json").items()
        >= binder.read(plan.with_name("model.json")).items()
    )
    inputs = materialize_prepared_inputs(config, root / "prepared-inputs")
    bundle = stage(config, root / "stage", through="validate", prepared_inputs=inputs)
    assert verify_stage_bundle(bundle, config)
    assert not (bundle / "pilots").exists()
    assert not list(root.rglob("*.safetensors"))
    # The actual training consumer reuses this verifier; parse its public option
    # without invoking the trainer or creating a model/checkpoint.
    from sparselab.cli.main import build_parser

    args = build_parser(work_dir=root).parse_args(
        ["train", str(config_path), "--stage-bundle", str(bundle), "--run-id", "held"]
    )
    assert args.stage_bundle == str(bundle)
