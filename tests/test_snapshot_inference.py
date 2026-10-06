"""Snapshot-trained runs remain self-contained across human inference surfaces."""

from __future__ import annotations

import json
import shutil
import sys
import threading

import pytest
from test_snapshot_preparation import snapshot_fixture  # noqa: F401

from sparselab.cli.main import build_parser, main
from sparselab.dashboard.surface_chat import generate_comparison, verified_checkpoints
from sparselab.evaluation.inference import load_run
from sparselab.evaluation.serving import LocalInference
from sparselab.training.trainer import train


@pytest.fixture
def snapshot_run(request):
    fixture = request.getfixturevalue("snapshot_fixture")
    config = fixture.run.model_copy(
        update={
            "training": fixture.run.training.model_copy(update={"max_steps": 2}),
            "optimizer": fixture.run.optimizer.model_copy(update={"warmup_steps": 0}),
        }
    )
    train(config, run_id="snapshot-model")
    return config


def _cli(monkeypatch, capsys, runs_dir, command):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            command,
            "snapshot-model",
            "--runs-dir",
            str(runs_dir),
            "--message" if command == "chat" else "--prompt",
            "Hi",
            "--max-new-tokens",
            "1",
            "--json",
        ],
    )
    main()
    return json.loads(capsys.readouterr().out)


def _cells(run):
    generations = sorted((run / "checkpoints").glob("step_*"))
    return [f"initial={generations[0]}", f"final={generations[-1]}"]


def test_relocated_snapshot_run_uses_owned_assets_on_every_surface(
    snapshot_run, tmp_path, monkeypatch, capsys
):
    config = snapshot_run
    original = load_run("snapshot-model", config.logging.root_dir)
    assert original.identity["step"] == 2
    store = tmp_path / "relocated-runs"
    relocated = store / "snapshot-model"
    shutil.copytree(original.run, relocated)
    prepared_manifest = json.loads((original.run / "data/manifest.json").read_text())
    assert prepared_manifest["source"] == "snapshot"
    assert prepared_manifest["cache_identity"]["snapshot_source_sha256"]
    source_manifest = json.loads(config.dataset.source_manifest_path.read_text())
    assert prepared_manifest["dataset_snapshot"] == source_manifest
    assert (
        json.loads((relocated / "data/manifest.json").read_text()) == prepared_manifest
    )
    capsys.readouterr()

    # Inference must never reopen acquisition/preparation inputs or the external
    # tokenizer. Restore each directory for the fixture's source-integrity teardown.
    hidden = []
    for path in (
        config.tokenizer.path.parent,
        config.dataset.cache_dir,
        config.dataset.source_manifest_path.parent,
    ):
        destination = path.with_name(path.name + "-unavailable")
        path.rename(destination)
        hidden.append((path, destination))
    try:

        def forbidden(*args, **kwargs):
            pytest.fail("inference tried to read source snapshot documents")

        monkeypatch.setattr("sparselab.data.sources.iter_snapshot", forbidden)
        loaded = load_run("snapshot-model", store)
        assert loaded.identity == original.identity
        for command in ("chat", "generate"):
            result = _cli(monkeypatch, capsys, store, command)
            assert result["identity"] == loaded.identity
            assert result["prompt_tokens"] > 0
        app = LocalInference(loaded, "snapshot-model")
        assert app.models()["data"][0]["sparselab"]["identity"] == loaded.identity
        result = app.complete(
            {"model": "snapshot-model", "prompt": "Hi", "max_tokens": 1},
            chat=False,
            cancellation=threading.Event(),
        )
        assert result["sparselab"]["identity"] == loaded.identity
        checkpoints = verified_checkpoints(_cells(relocated))
        assert checkpoints[-1][1].identity == loaded.identity
        comparison = generate_comparison(
            "Hi",
            checkpoints,
            max_new_tokens=1,
            temperature=0,
            top_k=0,
            seed=0,
            ordinal=0,
        )
        assert {card["identity"]["step"] for card in comparison["cards"]} == {0, 2}
        assert all(
            card["identity"]["source_identity_sha256"]
            == loaded.identity["source_identity_sha256"]
            for card in comparison["cards"]
        )
    finally:
        for path, destination in reversed(hidden):
            destination.rename(path)


@pytest.mark.parametrize(
    "asset",
    [
        "tokenizer.json",
        "tokenizer_manifest.json",
        "data/train.npy",
        "data/manifest.json",
    ],
)
@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_snapshot_owned_asset_damage_fails_all_load_gates(
    snapshot_run, tmp_path, monkeypatch, capsys, asset, damage
):
    config = snapshot_run
    store = tmp_path / "damaged-runs"
    run = store / "snapshot-model"
    shutil.copytree(config.logging.root_dir / "snapshot-model", run)
    target = run / asset
    if damage == "missing":
        target.unlink()
    else:
        content = target.read_bytes()
        target.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    capsys.readouterr()
    with pytest.raises((ValueError, OSError)):
        load_run("snapshot-model", store)
    for command in ("chat", "generate"):
        with pytest.raises(
            ValueError, match="artifact integrity failure|digest mismatch"
        ):
            _cli(monkeypatch, capsys, store, command)
        assert capsys.readouterr().out == ""
    with pytest.raises((ValueError, OSError)):
        verified_checkpoints(_cells(run))

    # The serve CLI must reject the run before constructing an API or binding a port.
    def forbidden(*args, **kwargs):
        pytest.fail("server initialized before run artifact validation")

    monkeypatch.setattr("sparselab.evaluation.serving.LocalInference", forbidden)
    args = build_parser().parse_args(
        [
            "serve",
            "snapshot-model",
            "--runs-dir",
            str(store),
            "--port",
            "0",
        ]
    )
    args.runtime_authorization = None
    with pytest.raises((ValueError, OSError)):
        args.handler(args)
