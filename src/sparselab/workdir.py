"""Project-local scratch directory used for SparseLab-managed temporary files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

WORK_DIR_ENV = "SPARSELAB_WORK_DIR"
_TEMP_DIR_ENV_VARS = ("TMPDIR", "TEMP", "TMP")


def resolve_work_dir(
    path: str | Path | None = None, *, cwd: Path | None = None
) -> Path:
    """Resolve an explicit, configured, or project-default scratch directory."""
    current = (cwd if cwd is not None else Path.cwd()).expanduser().resolve()
    configured = path if path is not None else os.environ.get(WORK_DIR_ENV)
    if configured is None or not os.fspath(configured).strip():
        project_root = next(
            (
                candidate
                for candidate in (current, *current.parents)
                if (candidate / "pyproject.toml").is_file()
            ),
            current,
        )
        return (project_root / "sparselab-work").resolve()

    resolved = Path(configured).expanduser()
    if not resolved.is_absolute():
        resolved = current / resolved
    return resolved.resolve()


def ensure_work_dir(path: str | Path | None = None, *, cwd: Path | None = None) -> Path:
    """Create the scratch directory and direct Python and child-process temp files to it."""
    work_dir = resolve_work_dir(path, cwd=cwd)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not work_dir.is_dir():
        raise NotADirectoryError(work_dir)

    os.environ[WORK_DIR_ENV] = str(work_dir)
    for name in _TEMP_DIR_ENV_VARS:
        os.environ[name] = str(work_dir)
    tempfile.tempdir = str(work_dir)
    return work_dir
