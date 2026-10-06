"""Existing-input API checks run before inspection, without model training."""

import json
from pathlib import Path

import pytest
from test_staging import _config

from sparselab import staging
from sparselab.training.manifest import sha256_file, source_identity


@pytest.fixture
def prepared(tmp_path):
    config = _config(tmp_path)
    inputs = staging.materialize_prepared_inputs(config, tmp_path / "inputs")
    return config, inputs


@pytest.mark.parametrize("through", ["inspect", "validate"])
@pytest.mark.parametrize(
    "damage", ["source", "config", "tokenizer", "array", "inventory", "symlink"]
)
def test_prepared_inputs_reject_before_inspection(
    prepared, tmp_path, monkeypatch, through, damage
):
    config, inputs = prepared
    if damage == "source":
        sealed = staging._read_sealed(inputs / "inputs.json")
        staging._seal(
            inputs / "inputs.json",
            {
                **{key: value for key, value in sealed.items() if key != "sha256"},
                "source_identity_sha256": "0" * 64,
            },
        )
    elif damage == "config":
        config = config.model_copy(update={"seed": config.seed + 1})
    elif damage == "tokenizer":
        tokenizer = inputs / "assets/tokenizer.json"
        tokenizer.write_bytes(tokenizer.read_bytes() + b" ")
    elif damage == "array":
        array = inputs / "assets/data/train.npy"
        data = array.read_bytes()
        array.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    elif damage == "inventory":
        (inputs / "assets/unadvertised.txt").write_text("unexpected")
    else:
        tokenizer = inputs / "assets/tokenizer.json"
        tokenizer.unlink()
        tokenizer.symlink_to(config.tokenizer.path)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid inputs must fail before inspection or preparation")

    monkeypatch.setattr(staging, "inspect_runtime", forbidden)
    monkeypatch.setattr(staging, "materialize_prepared_inputs", forbidden)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        staging.stage(config, output, through=through, prepared_inputs=inputs)
    assert not output.exists()


@pytest.mark.parametrize("through", ["inspect", "validate"])
def test_prepared_mode_rejects_generic_drift(prepared, tmp_path, through):
    config, inputs = prepared
    output = tmp_path / "drift"
    with pytest.raises(ValueError, match="cannot allow source/runtime drift"):
        staging.stage(
            config,
            output,
            through=through,
            prepared_inputs=inputs,
            allow_runtime_drift=True,
        )
    assert not output.exists()


def test_prepared_mode_does_not_reuse_completed_output(prepared, tmp_path):
    config, inputs = prepared
    output = staging.stage(config, tmp_path / "inspect", through="inspect")
    before = (output / "bundle.json").read_bytes()
    # Ordinary staging keeps its existing verified reuse contract.
    assert staging.stage(config, output, through="inspect") == output
    with pytest.raises(FileExistsError, match="new output"):
        staging.stage(config, output, through="inspect", prepared_inputs=inputs)
    assert (output / "bundle.json").read_bytes() == before


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("symlink", [False, True])
def test_prepared_mode_preserves_competing_output(
    prepared, tmp_path, monkeypatch, late, symlink
):
    config, inputs = prepared
    output = tmp_path / "contended"
    target = tmp_path / "missing-target"

    def occupy():
        if symlink:
            output.symlink_to(target, target_is_directory=True)
        else:
            output.mkdir()
            (output / "owner.txt").write_bytes(b"other writer")

    if late:
        flock = staging.fcntl.flock

        def lock_and_occupy(*args):
            result = flock(*args)
            occupy()
            return result

        monkeypatch.setattr(staging.fcntl, "flock", lock_and_occupy)
    else:
        occupy()
    with pytest.raises(FileExistsError, match="new output"):
        staging.stage(config, output, through="inspect", prepared_inputs=inputs)
    if symlink:
        assert output.is_symlink() and output.readlink() == target
        assert not target.exists()
    else:
        assert sorted(path.name for path in output.iterdir()) == ["owner.txt"]
        assert (output / "owner.txt").read_bytes() == b"other writer"


@pytest.mark.parametrize("assets_path", [False, True])
def test_prepared_reuse_preserves_bytes_and_provenance(
    prepared, tmp_path, monkeypatch, assets_path
):
    config, inputs = prepared
    original = {
        p.relative_to(inputs): sha256_file(p) for p in inputs.rglob("*") if p.is_file()
    }
    old_inputs = staging._read_sealed(inputs / "inputs.json")
    prepared_manifest = json.loads((inputs / "assets/data/manifest.json").read_text())

    def forbidden(*args, **kwargs):
        pytest.fail("existing-input reuse must not prepare data or execute a pilot")

    monkeypatch.setattr(staging, "materialize_prepared_inputs", forbidden)
    monkeypatch.setattr(staging, "prepare_data", forbidden)
    monkeypatch.setattr(staging, "_run_pilot", forbidden)
    first = staging.stage(
        config,
        tmp_path / "first",
        through="validate",
        prepared_inputs=inputs / "assets" if assets_path else inputs,
    )
    second = staging.stage(
        config,
        tmp_path / "second",
        through="validate",
        prepared_inputs=first,
    )
    parent = old_inputs
    for output in (first, second):
        staging.verify_stage_bundle(output, config)
        receipt = staging._read_sealed(output / "inputs.json")
        assert receipt["parent_inputs_sha256"] == parent["sha256"]
        assert (
            receipt["producer_source_identity_sha256"]
            == old_inputs["source_identity_sha256"]
        )
        assert (
            receipt["prepared_production_source_identity_sha256"]
            == prepared_manifest["cache_identity"]["source_identity_sha256"]
        )
        assert (
            receipt["verification_source_identity_sha256"]
            == source_identity()["sha256"]
        )
        assert receipt["source_identity_sha256"] == source_identity()["sha256"]
        assert receipt["artifacts"] == old_inputs["artifacts"]
        assert not (output / "prepared").exists()
        parent = receipt
    assert {
        p.relative_to(inputs): sha256_file(p) for p in inputs.rglob("*") if p.is_file()
    } == original


def test_existing_inputs_cannot_replace_late_publication(
    prepared, tmp_path, monkeypatch
):
    config, inputs = prepared
    output = tmp_path / "publication"
    publish = staging._publish_prepared_directory

    def competing_writer(source: Path, target: Path):
        if target == output:
            target.mkdir()
            (target / "owner.txt").write_text("other writer")
        return publish(source, target)

    monkeypatch.setattr(staging, "_publish_prepared_directory", competing_writer)
    with pytest.raises(FileExistsError):
        staging.stage(config, output, through="validate", prepared_inputs=inputs)
    assert sorted(path.name for path in output.iterdir()) == ["owner.txt"]
    assert (output / "owner.txt").read_text() == "other writer"


@pytest.mark.parametrize("through", ["inspect", "validate"])
@pytest.mark.parametrize("damage", ["payload", "extra"])
def test_ordinary_stage_reuse_checks_payload_and_inventory(tmp_path, through, damage):
    config = _config(tmp_path)
    output = staging.stage(config, tmp_path / "reusable", through=through)
    before = (output / "bundle.json").read_bytes()
    assert staging.stage(config, output, through=through) == output
    assert (output / "bundle.json").read_bytes() == before
    if damage == "payload":
        report = output / "stage.json"
        report.write_bytes(report.read_bytes() + b" ")
    else:
        (output / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="length mismatch|differs from actual files"):
        staging.stage(config, output, through=through)


def test_provenance_uses_verified_input_and_copied_manifest(
    prepared, tmp_path, monkeypatch
):
    config, inputs = prepared
    original = staging._read_sealed(inputs / "inputs.json")
    manifest_path = inputs / "assets/data/manifest.json"
    original_manifest = manifest_path.read_bytes()
    copy_tree = staging._copy_tree

    def copy_then_change_source(source, destination, **kwargs):
        result = copy_tree(source, destination, **kwargs)
        if source == inputs / "assets":
            staging._seal(
                inputs / "inputs.json",
                {
                    **{
                        key: value for key, value in original.items() if key != "sha256"
                    },
                    "producer_source_identity_sha256": "f" * 64,
                },
            )
            changed = json.loads(original_manifest)
            changed["cache_identity"]["source_identity_sha256"] = "f" * 64
            manifest_path.write_text(json.dumps(changed))
        return result

    monkeypatch.setattr(staging, "_copy_tree", copy_then_change_source)
    output = staging.stage(
        config, tmp_path / "isolated", through="validate", prepared_inputs=inputs
    )
    staging.verify_stage_bundle(output, config)
    receipt = staging._read_sealed(output / "inputs.json")
    assert receipt["parent_inputs_sha256"] == original["sha256"]
    assert (
        receipt["producer_source_identity_sha256"] == original["source_identity_sha256"]
    )
    assert (
        receipt["prepared_production_source_identity_sha256"]
        == json.loads(original_manifest)["cache_identity"]["source_identity_sha256"]
    )
    assert (output / "assets/data/manifest.json").read_bytes() == original_manifest
