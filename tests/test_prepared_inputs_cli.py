"""Data-only CLI delegation to the native sealed prepared-input APIs."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sparselab.cli import main as cli


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = SimpleNamespace(
        dataset=SimpleNamespace(source="local_token_mixture"),
        training=SimpleNamespace(max_tokens=3),
    )
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr("sparselab.data.legacy.require_current_dataset", lambda _: None)
    root = tmp_path / "inputs"
    data = root / "assets" / "data"
    data.mkdir(parents=True)
    np.save(data / "train_supervision.npy", np.array([False, True, True, True, False]))
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_sha256": "m" * 64,
                "supervision": {"kind": "token-loss-mask-v1"},
                "train": {
                    "supervised_target_positions": 3,
                    "scheduled_target_positions": 3,
                    "supervised_target_positions_by_stratum": {"prose": 3},
                },
                "token_mixture": {"actual_target_tokens": {"prose": 3}},
            }
        )
    )
    verified = {"sha256": "b" * 64, "source_identity_sha256": "s" * 64}
    return config, root, verified


def test_parser_exposes_only_publish_and_verify():
    parser = cli.build_parser()
    publish = parser.parse_args(
        ["data", "prepared-inputs", "publish", "run.yaml", "--output", "inputs"]
    )
    verify = parser.parse_args(
        ["data", "prepared-inputs", "verify", "run.yaml", "inputs"]
    )
    assert publish.inputs_command == "publish" and publish.output == Path("inputs")
    assert verify.inputs_command == "verify" and verify.bundle == Path("inputs")
    assert cli._read_only_command(verify)
    assert not cli._read_only_command(publish)


def test_publish_and_verify_bind_config_and_cold_identity(
    tmp_path, monkeypatch, capsys
):
    config, root, verified = _fixture(tmp_path, monkeypatch)
    calls = []

    def publish(actual_config, destination, **options):
        calls.append(("publish", actual_config, destination, options))

    def verify(destination, actual_config, **options):
        calls.append(("verify", actual_config, destination, options))
        return verified

    monkeypatch.setattr("sparselab.staging.materialize_prepared_inputs", publish)
    monkeypatch.setattr("sparselab.staging.verify_prepared_inputs", verify)
    common = {
        "config": str(tmp_path / "run.yaml"),
        "resource_envelope_value": None,
        "tokenizer_batch_documents": 2,
        "tokenizer_batch_source_bytes": 1024,
    }
    cli._data_prepared_inputs(
        argparse.Namespace(**common, inputs_command="publish", output=root, bundle=None)
    )
    published = json.loads(capsys.readouterr().out)
    assert published["bundle_manifest_sha256"] == verified["sha256"]
    assert published["source_identity_sha256"] == verified["source_identity_sha256"]
    assert published["prepared_data_manifest_sha256"] == "m" * 64
    assert published["supervised_target_positions"] == 3
    assert published["scheduled_target_positions"] == 3
    assert published["supervised_target_positions_by_stratum"] == {"prose": 3}
    assert [call[0] for call in calls] == ["publish", "verify"]
    assert all(call[1] is config and call[2] == root for call in calls)
    assert calls[0][3]["verification_mode"] == "cold"
    assert calls[1][3] == {"verification_mode": "cold"}

    calls.clear()
    cli._data_prepared_inputs(
        argparse.Namespace(**common, inputs_command="verify", bundle=root)
    )
    assert json.loads(capsys.readouterr().out) == published
    assert calls == [("verify", config, root, {"verification_mode": "cold"})]


@pytest.mark.parametrize(
    "damage", ["missing_mask", "mask_count", "wrong_count", "wrong_config", "drift"]
)
def test_verify_rejects_identity_or_supervision_drift(tmp_path, monkeypatch, damage):
    config, root, verified = _fixture(tmp_path, monkeypatch)
    manifest_path = root / "assets" / "data" / "manifest.json"
    if damage == "missing_mask":
        manifest = json.loads(manifest_path.read_text())
        manifest["supervision"]["kind"] = "all_tokens"
        manifest_path.write_text(json.dumps(manifest))
    elif damage == "mask_count":
        np.save(
            root / "assets" / "data" / "train_supervision.npy",
            np.array([False, True, False, True, False]),
        )
    elif damage == "wrong_count":
        config.training.max_tokens = 4

    def verify(destination, actual_config, **options):
        assert destination == root and actual_config is config
        assert options == {"verification_mode": "cold"}
        if damage in {"wrong_config", "drift"}:
            raise ValueError("native prepared-input identity mismatch")
        return verified

    monkeypatch.setattr("sparselab.staging.verify_prepared_inputs", verify)
    args = argparse.Namespace(
        config=str(tmp_path / "run.yaml"), inputs_command="verify", bundle=root
    )
    with pytest.raises(ValueError):
        cli._data_prepared_inputs(args)
