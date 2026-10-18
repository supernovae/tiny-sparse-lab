"""Local disposition checks; preservation was verified at the recorded Git pins."""

import json
from pathlib import Path


def test_personal_report_bundles_retired_as_complete_groups() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/report-retirement.json").read_text())
    reports = [
        row for row in manifest["files"] if row["source"].startswith("artifacts/")
    ]
    assert len(reports) == 163
    assert len({row["source"] for row in reports}) == 163
    assert len({Path(row["source"]).parts[2] for row in reports}) == 4
    for row in reports:
        assert row["destination"] == "provenance/lab/" + row["source"]
        assert len(row["sha256"]) == 64
        assert len(row["git_blob"]) == 40
        assert row["size_bytes"] > 0
        assert not (root / row["source"]).exists()
    guard = json.loads((root / "docs/research-resource-retirement.json").read_text())
    moved = guard["report_evidence_moved_to_workshop"]["previous_local_integrity_rows"]
    preserved = {row["source"]: row for row in manifest["files"]}
    assert len(moved) == 5
    for row in moved:
        original = preserved[row["relative_path"]]
        assert row["sha256"] == original["sha256"]
        assert row["size_bytes"] == original["size_bytes"]
