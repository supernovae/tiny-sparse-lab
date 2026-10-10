"""``sparselab probe``: run the battery on a run or checkpoint, write a record."""

from __future__ import annotations

import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sparselab.lab_context import LabContext, LabSignal
from sparselab.lab_records import read_lab_record, write_sealed
from sparselab.probes.runner import Arm, progress_writer, run_battery


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
    resource_envelope: Any = None,
) -> tuple[dict[str, Any], Path]:
    """Run the battery and publish ``LAB/probes/<id>/probe.json`` (sealed).

    The same :class:`LabContext` as ``sparselab try`` applies: touch
    ``LAB/probes/<id>/CANCEL`` (or Ctrl-C) to stop at a safe point; a resource
    envelope is checked before every probe. A stopped battery is still
    published, marked incomplete.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    probe_id = f"probe-{stamp}-{secrets.token_hex(4)}"
    folder = lab_dir / "probes" / probe_id
    folder.mkdir(parents=True, exist_ok=False)
    progress = progress_writer(folder / "progress.json")
    progress({"state": "loading", "tier": tier, "target": target_spec})
    context = LabContext(
        folder / "CANCEL", resource_envelope=resource_envelope, workspace=lab_dir
    )
    print(
        f"probe {probe_id}: touch {context.cancel_path} (or Ctrl-C) to stop",
        file=sys.stderr,
    )

    def arm(spec: str) -> Arm:
        # Resolve now (fail fast on a typo); load lazily, one arm at a time.
        resolve_target(spec, lab_dir=lab_dir, runs_dir=runs_dir)
        return Arm(
            load=lambda: load_target(
                spec,
                lab_dir=lab_dir,
                runs_dir=runs_dir,
                backend=backend,
                authorization=authorization,
            )
        )

    context.install()
    try:
        result = run_battery(
            arm(target_spec),
            arm(baseline_spec) if baseline_spec is not None else None,
            tier=tier,
            fast_fail=fast_fail,
            progress=progress,
            context=context,
        )
    except (LabSignal, KeyboardInterrupt, Exception) as error:
        context.finalizing = True
        name = error.name if isinstance(error, LabSignal) else type(error).__name__
        progress({"state": "failed", "error": f"{name}: {error}"})
        if isinstance(error, (LabSignal, KeyboardInterrupt)):
            raise KeyboardInterrupt from None
        raise
    finally:
        context.restore()
    path = folder / "probe.json"
    sealed = write_sealed(path, {**result, "probe_id": probe_id})
    return sealed, path


def read_probe(path: Path) -> dict[str, Any]:
    """Verified probe record (the shared lab-record reader)."""
    return read_lab_record(path, "probe")[1]
