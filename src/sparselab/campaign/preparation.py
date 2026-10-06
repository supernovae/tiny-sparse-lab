"""Thin Campaign adapters over the native dataset and artifact operations."""

from pathlib import Path
from typing import Any

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.experiments.artifacts import verify_artifact
from sparselab.experiments.plan import Artifact
from sparselab.training.manifest import sha256_file

KINDS = {"dataset_snapshot", "tokenizer_train", "data_prepare"}


def config_for(engine: Any, stage: Any, rows: dict) -> Any:
    loader = load_tokenizer_config if stage.kind == "tokenizer_train" else load_config
    config = loader(engine._path(stage.config))
    if stage.snapshot is not None:
        snapshot = engine._upstream(rows, stage.snapshot)
        manifest = Path(snapshot["availability"]["path"])
        if (
            config.dataset.source != "snapshot"
            or config.dataset.source_manifest_path != manifest
        ):
            raise ValueError("configured dataset differs from upstream snapshot")
    if stage.kind == "data_prepare":
        tokenizer = engine._upstream(rows, stage.tokenizer)
        if config.tokenizer.path != Path(tokenizer["availability"]["path"]):
            raise ValueError("configured tokenizer differs from upstream tokenizer")
    return config


def _artifact(
    engine: Any, stage: Any, path: Path, config: Any, output: dict | None = None
) -> dict:
    import json

    if output is None:
        if stage.kind == "tokenizer_train":
            output = {
                "kind": "tokenizer",
                "identifier": path.parent.name,
                "sha256": sha256_file(path),
            }
        else:
            manifest = json.loads((path / "manifest.json").read_text())
            output = {
                "kind": "prepared_data",
                "identifier": manifest["settings_sha256"],
                "sha256": manifest["manifest_sha256"],
            }
    verified = verify_artifact(
        Artifact.model_validate(
            {
                **output,
                "version": 1,
                "producer": f"sparselab {stage.kind}",
                "path": str(path),
            }
        ),
        engine.source,
        dataset=config.dataset,
        **engine._verification(),
    )
    return {key: verified[key] for key in ("kind", "identifier", "sha256")}


def snapshot_identity(engine: Any, stage: Any, path: Path) -> dict:
    from sparselab.data.sources import verify_lock, verify_snapshot

    lock = verify_lock(engine._operational_path(stage.lock))
    manifest = verify_snapshot(path)
    if manifest["lock"] != lock:
        raise ValueError("snapshot differs from declared source lock")
    return {
        "kind": "dataset_snapshot",
        "identifier": lock["lock_sha256"],
        "sha256": manifest["manifest_sha256"],
    }


def verify(engine: Any, stage: Any, row: dict, rows: dict) -> None:
    path = engine._operational_path(row["availability"]["path"])
    if stage.kind == "dataset_snapshot":
        if path != engine._operational_path(stage.output) / "manifest.json":
            raise ValueError("snapshot output location differs from declaration")
        output = snapshot_identity(engine, stage, path)
    else:
        config = config_for(engine, stage, rows)
        if (
            stage.kind == "tokenizer_train"
            and path != config.output_dir / "tokenizer.json"
        ):
            raise ValueError("tokenizer output differs from declaration")
        if stage.kind == "data_prepare" and path.parent != config.dataset.cache_dir:
            raise ValueError("prepared output differs from declaration")
        output = _artifact(engine, stage, path, config, row["outputs"][0])
    if row["outputs"] != [output]:
        raise ValueError("preparation output identity changed")


def dispatch(
    engine: Any, stage: Any, rows: dict, *, resume_acquisition: bool = False
) -> dict:
    if stage.kind == "dataset_snapshot":
        from sparselab.data.sources import snapshot_source

        output = engine._operational_path(stage.output)
        # A published snapshot left by an interrupted receipt commit is verified,
        # never acquired again. Partial acquisition requires explicit resume.
        path = output / "manifest.json"
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            journal = output.with_name(f".{output.name}.work")
            try:
                path = snapshot_source(
                    engine._operational_path(stage.lock),
                    output,
                    engine._operational_path(stage.cache_dir),
                    resume=stage.resume or (resume_acquisition and journal.exists()),
                )
            except (OSError, RuntimeError) as error:
                if not journal.is_dir() or journal.is_symlink():
                    raise
                return engine._result(
                    "INTERRUPTED", "DO_NOT_ADVANCE", reason=str(error)
                )
        identity = snapshot_identity(engine, stage, path)
    else:
        from sparselab.data.legacy import require_current_dataset
        from sparselab.data.tokenizer import load_tokenizer, train_tokenizer

        config = config_for(engine, stage, rows)
        require_current_dataset(config.dataset)
        if stage.kind == "tokenizer_train":
            path = train_tokenizer(config)
        else:
            from sparselab.data.packing import prepare_data

            path = prepare_data(
                config, load_tokenizer(config.tokenizer.path), **engine._verification()
            ).root
        identity = _artifact(engine, stage, path, config)
    return engine._result(outputs=[identity], availability={"path": str(path)})
