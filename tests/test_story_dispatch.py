"""Story inputs remain executable after controller sources are unavailable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparselab.data import datasets
from sparselab.data.local_stories import REVISION
from sparselab.experiments.lock import resolve_plan
from sparselab.staging import stage, verify_prepared_inputs, verify_stage_bundle
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import sha256_file
from sparselab.training.trainer import train
from sparselab.workers.bundles import (
    install_dispatch_bundle,
    materialize_dispatch_bundle,
    prepare_dispatch_bundle,
    verify_dispatch_bundle,
)


@pytest.mark.parametrize("source", ["tinystories", "local_stories"])
def test_story_dispatch_executes_without_controller_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    from test_experiment_story_inputs import story_plan

    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    controller = tmp_path / "controller"
    controller.mkdir()
    plan, declaration, _ = story_plan(controller, monkeypatch, source)
    # Logs belong on the worker; tokenizer/cache/snapshot paths remain on the controller.
    plan = plan.model_copy(
        update={
            "base_run": plan.base_run.model_copy(
                update={
                    "logging": plan.base_run.logging.model_copy(
                        update={"root_dir": tmp_path / "runs"}
                    )
                }
            )
        }
    )
    run = resolve_plan(plan, declaration).cells[0].config
    source_digest = (
        sha256_file(run.dataset.source_manifest_path)
        if run.dataset.source_manifest_path is not None
        else None
    )
    dispatch = tmp_path / "dispatch"
    bundle = prepare_dispatch_bundle(run, dispatch)
    assert verify_dispatch_bundle(dispatch).digest() == bundle.digest()
    controller.rename(tmp_path / "unavailable-controller")

    def offline(*_args, **_kwargs):
        raise AssertionError("worker attempted a dataset download")

    monkeypatch.setattr("datasets.load_dataset", offline)
    monkeypatch.setattr(datasets, "load_dataset", offline)
    worker = tmp_path / "worker"
    install_dispatch_bundle(
        worker,
        bundle,
        {
            f"assets/{item.sha256}": dispatch / item.relative_path
            for item in bundle.files
        },
        check_only=False,
    )
    private = tmp_path / "private"
    materialize_dispatch_bundle(worker, bundle.digest(), private)
    verify_prepared_inputs(private, run)
    changed = run.model_copy(
        update={
            "dataset": run.dataset.model_copy(
                update={"train_max_tokens": run.dataset.train_max_tokens + 1}
            )
        }
    )
    with pytest.raises(ValueError, match="configuration differs"):
        verify_prepared_inputs(private, changed)
    manifest = json.loads((private / "assets/data/manifest.json").read_text())
    assert manifest["source"] == source
    assert manifest["revision"] == REVISION
    if source == "local_stories":
        assert manifest["cache_identity"]["local_stories_source_sha256"]
        tokenizer_manifest = json.loads(
            (private / "assets/tokenizer_manifest.json").read_text()
        )
        assert tokenizer_manifest["source_manifest_sha256"] == source_digest
    staged = stage(
        run, tmp_path / "staged", through="validate", prepared_inputs=private
    )
    verify_stage_bundle(staged, run)
    run_id = train(run, run_id="offline-story", stage_bundle=staged)
    run_root = run.logging.root_dir / run_id
    checkpoint = CheckpointManager(run_root).load(run_root / "checkpoints/latest.json")
    assert checkpoint.tokens_seen == 64
    assert not run.tokenizer.path.exists()
    assert not any(path.is_file() for path in controller.rglob("*"))
    # The relocated evidence is still authenticated after sources disappear.
    metadata = private / "assets/tokenizer_manifest.json"
    metadata.write_bytes(metadata.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        verify_prepared_inputs(private, run)
