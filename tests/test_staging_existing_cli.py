"""The public stage adapter consumes authenticated inputs without preparing."""

import argparse

import pytest
import yaml
from test_training import config as fixture_config

from sparselab import staging
from sparselab.cli.main import _stage, build_parser


@pytest.mark.parametrize("through", ["inspect", "validate", "smoke", "warmup"])
def test_existing_inputs_parser(tmp_path, through):
    args = build_parser().parse_args(
        [
            "stage",
            "run.yaml",
            "--output",
            str(tmp_path / "stage"),
            "--prepared-inputs",
            str(tmp_path / "inputs"),
            "--through",
            through,
        ]
    )
    assert args.prepared_inputs == tmp_path / "inputs"
    assert args.through == through


@pytest.mark.parametrize(
    "damage", [None, "inventory", "config", "source", "symlink", "output"]
)
def test_existing_inputs_handler_never_prepares(tmp_path, monkeypatch, damage):
    config = fixture_config(tmp_path)
    bundle = staging.materialize_prepared_inputs(config, tmp_path / "inputs")
    raw = config.model_dump(mode="json")
    if damage == "config":
        raw["dataset"]["synthetic_seed"] += 1
    elif damage == "inventory":
        array = bundle / "assets/data/train.npy"
        array.write_bytes(array.read_bytes() + b"tampered")
    elif damage == "source":
        monkeypatch.setattr(staging, "source_identity", lambda: {"sha256": "0" * 64})
    elif damage == "symlink":
        link = tmp_path / "input-link"
        link.symlink_to(bundle, target_is_directory=True)
        bundle = link
    run = tmp_path / "run.yaml"
    run.write_text(yaml.safe_dump(raw))

    def forbidden(*args, **kwargs):
        pytest.fail("existing input staging must not prepare or download")

    monkeypatch.setattr(staging, "prepare_data", forbidden)
    monkeypatch.setattr(staging, "materialize_prepared_inputs", forbidden)
    output = tmp_path / "stage"
    if damage == "output":
        output.mkdir()
    args = argparse.Namespace(
        config=str(run),
        output=str(output),
        prepared_inputs=bundle,
        through="validate",
        work_dir=tmp_path / "work",
        cold_verify=True,
        pilot_deadline_policy=None,
        runtime_authorization=None,
        resource_envelope_value=None,
        tokenizer_batch_documents=32,
        tokenizer_batch_source_bytes=1024 * 1024,
    )
    if damage:
        with pytest.raises(FileExistsError if damage == "output" else ValueError):
            _stage(args)
    else:
        _stage(args)
        staging.verify_stage_bundle(output, config)
