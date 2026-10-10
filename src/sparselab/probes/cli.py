"""``sparselab probe``: run the battery on a run or checkpoint, write a record."""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sparselab.probes.runner import (
    RESULT_FORMAT,
    progress_writer,
    run_battery,
    seal,
    verify,
    write_json_atomic,
)


def _run_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / "manifest.json").is_file() and (
            candidate / "resolved_config.yaml"
        ).is_file():
            return candidate
    return None


def resolve_target(
    spec: str, *, lab_dir: Path, runs_dir: Path | None = None
) -> tuple[Path, str | None]:
    """RUN-OR-CHECKPOINT -> (run directory, checkpoint path or None).

    Accepts a run directory, a checkpoint file/folder or pointer inside a run,
    or a run id under ``LAB/runs`` (then ``--runs-dir``).
    """
    path = Path(spec)
    if path.exists():
        path = path.absolute()
        run = _run_root(path if path.is_dir() else path.parent)
        if run is None:
            raise ValueError(f"not inside a SparseLab run: {spec}")
        return run, None if path == run else str(path)
    for root in (lab_dir / "runs", runs_dir):
        if root is not None and _run_root(root / spec) == (root / spec):
            return (root / spec).absolute(), None
    raise ValueError(f"unknown run or checkpoint: {spec}")


def load_target(
    spec: str,
    *,
    lab_dir: Path,
    runs_dir: Path | None,
    backend: str | None,
    authorization: Any,
) -> Any:
    from sparselab.evaluation.inference import load_run

    run, checkpoint = resolve_target(spec, lab_dir=lab_dir, runs_dir=runs_dir)
    return load_run(
        run.name,
        run.parent,
        checkpoint,
        backend,
        authorization=authorization,
    )


def run_probe(
    target_spec: str,
    baseline_spec: str | None,
    *,
    lab_dir: Path,
    runs_dir: Path | None = None,
    tier: str = "fast",
    fast_fail: bool = True,
    backend: str | None = None,
    authorization: Any = None,
) -> tuple[dict[str, Any], Path]:
    """Run the battery and publish ``LAB/probes/<id>/probe.json`` (sealed)."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    probe_id = f"probe-{stamp}-{secrets.token_hex(4)}"
    folder = lab_dir / "probes" / probe_id
    folder.mkdir(parents=True, exist_ok=False)
    progress = progress_writer(folder / "progress.json")
    progress({"state": "loading", "tier": tier, "target": target_spec})
    try:
        target = load_target(
            target_spec,
            lab_dir=lab_dir,
            runs_dir=runs_dir,
            backend=backend,
            authorization=authorization,
        )
        baseline = (
            load_target(
                baseline_spec,
                lab_dir=lab_dir,
                runs_dir=runs_dir,
                backend=backend,
                authorization=authorization,
            )
            if baseline_spec is not None
            else None
        )
        result = run_battery(
            target, baseline, tier=tier, fast_fail=fast_fail, progress=progress
        )
    except BaseException as error:
        progress({"state": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    result = seal({**result, "probe_id": probe_id})
    path = folder / "probe.json"
    write_json_atomic(path, result, exclusive=True)
    return result, path


def read_probe(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "probe.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("format") != RESULT_FORMAT:
        raise ValueError(f"not a probe record: {path}")
    if not verify(result):
        raise ValueError(f"probe record digest mismatch: {path}")
    return result
