"""Archived CI-ledger checks with zero model and zero optimizer work."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sqlite3
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from sparselab.training.attempt_budget import AttemptBudget, AttemptBudgetError

ROOT = Path(__file__).resolve().parents[1] / "tools/kernel-memory-lab/ci-guard"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT))
    init = _load("kml_ci_audit_init", "init_budget.py")
    audit = _load("kml_ci_audit", "audit_relocated_ledger.py")
    original = tmp_path / "original"
    original.mkdir()
    (original / "storage-baseline.json").write_text("{}\n")
    commit = "a" * 40
    init.initialize(original, "archive", commit, time.time_ns() + 120_000_000_000)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for name in (
        "attempt-budget.sqlite",
        "attempt-contract.json",
        "quota-policy.json",
        "storage-baseline.json",
    ):
        shutil.copy2(original / name, relocated / name)
    # The native live binding must remain strict when its original path goes away.
    (original / "attempt-contract.json").unlink()
    with pytest.raises(AttemptBudgetError, match="invalid attempt contract"):
        AttemptBudget(relocated / "attempt-budget.sqlite").status()
    return audit, relocated, commit


def _audit(audit, root: Path, commit: str):
    return audit.audit(
        root,
        kind="archive",
        commit=commit,
        expected_ledger_sha256=_sha(root / "attempt-budget.sqlite"),
        expected_contract_sha256=_sha(root / "attempt-contract.json"),
    )


def test_archived_ledger_audit_is_read_only_and_keeps_original_binding(archive) -> None:
    audit, root, commit = archive
    before = {
        path.name: (path.stat().st_mtime_ns, _sha(path)) for path in root.iterdir()
    }
    result = _audit(audit, root, commit)
    assert result["audit_kind"] == "relocated-read-only-not-live-authorization"
    assert result["sqlite_integrity"] == "ok"
    assert result["original_contract_path"].endswith("original/attempt-contract.json")
    assert result["quota"] == (2, 32, 0, 0)
    assert (
        result["charged"]
        == result["actual_from_completed_reservations"]
        == (0, 0, 0, 0)
    )
    assert result["reservation_count"] == 0
    assert before == {
        path.name: (path.stat().st_mtime_ns, _sha(path)) for path in root.iterdir()
    }


def test_archived_ledger_rejects_changed_identity_contract_and_counts(archive) -> None:
    audit, root, commit = archive
    with pytest.raises(ValueError, match="content identity"):
        _audit(audit, root, "b" * 40)
    with pytest.raises(ValueError, match="evidence inventory"):
        audit.audit(
            root,
            kind="archive",
            commit=commit,
            expected_ledger_sha256="0" * 64,
            expected_contract_sha256=_sha(root / "attempt-contract.json"),
        )
    with sqlite3.connect(root / "attempt-budget.sqlite") as connection:
        connection.execute("UPDATE budget SET used_updates=1 WHERE id=1")
    with pytest.raises(ValueError, match="reservation totals"):
        _audit(audit, root, commit)


def test_archived_ledger_rejects_embedded_or_archived_contract_drift(archive) -> None:
    audit, root, commit = archive
    with sqlite3.connect(root / "attempt-budget.sqlite") as connection:
        connection.execute("UPDATE budget SET contract_json='{}' WHERE id=1")
    with pytest.raises(ValueError, match="embedded contract digest"):
        _audit(audit, root, commit)
    (root / "attempt-contract.json").write_text("{}\n")
    with pytest.raises(ValidationError):
        _audit(audit, root, commit)


def test_archived_ledger_rejects_sidecar_and_symlink(archive) -> None:
    audit, root, commit = archive
    sidecar = root / "attempt-budget.sqlite-wal"
    sidecar.write_bytes(b"not a frozen archive")
    with pytest.raises(ValueError, match="sidecar"):
        _audit(audit, root, commit)
    sidecar.unlink()
    contract = root / "attempt-contract.json"
    contract.rename(root / "contract-original.json")
    contract.symlink_to(root / "contract-original.json")
    with pytest.raises(ValueError, match="symlinked"):
        _audit(audit, root, commit)


def test_archived_ledger_rejects_corrupt_sqlite(archive) -> None:
    audit, root, commit = archive
    (root / "attempt-budget.sqlite").write_bytes(b"not sqlite")
    with pytest.raises(sqlite3.DatabaseError):
        _audit(audit, root, commit)
