from __future__ import annotations

import hashlib
from pathlib import Path

from sparselab.dashboard.research import _learning_observations
from sparselab.training.manifest import canonical_json


def test_dashboard_accepts_only_content_addressed_learning_observations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "learning-observations"
    root.mkdir()
    payload = {
        "format": "sparselab_learning_observation_v1",
        "identity": "",
        "run_id": "seed42",
        "checkpoint": {"step": 200, "tokens_seen": 204800},
        "groups": {},
    }
    payload["identity"] = hashlib.sha256(
        canonical_json(
            {key: value for key, value in payload.items() if key != "identity"}
        )
    ).hexdigest()
    raw = canonical_json(payload) + b"\n"
    (root / f"{hashlib.sha256(raw.rstrip(b'\n')).hexdigest()}.json").write_bytes(raw)
    (root / "bad.json").write_text("{}")

    accepted, rejected = _learning_observations(tmp_path)
    assert len(accepted) == 1
    assert accepted[0]["identity"] == payload["identity"]
    assert rejected == ["bad.json: observation format or content address mismatch"]
