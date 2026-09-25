from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparselab.experiments.reporting import build_study_report, write_study_report
from sparselab.training.manifest import canonical_json

_REPORT_INPUTS = (
    Path(__file__).resolve().parents[1]
    / "artifacts/research-reports/fbdb00217f8e952e12bd07e053d97d85796f889748ee77d3bc81b21bb3c98c0b/inputs"
)
_STUDY = _REPORT_INPUTS / "research-scaffold" / "study.yaml"
_RECEIPT = _REPORT_INPUTS / "receipt.json"
_COLLECTED = _REPORT_INPUTS / "collected-report.json"


def _signed_evidence() -> tuple[dict[str, object], bytes]:
    payload: dict[str, object] = {
        "format": "sparselab-portability-evidence",
        "version": 1,
        "campaign_id": "engram-portability-v1-smoke",
        "scale": "smoke",
        "protocol_sha256": "a" * 64,
        "world_manifest_sha256": "b" * 64,
        "arms": [
            {
                "arm_id": "recipient-a-semantic-real",
                "recipient": "a",
                "representation": "semantic",
                "condition": "real",
                "seed": 17,
                "status": "blocked",
                "run_id": None,
            }
        ],
        "observations": [],
        "summary": {
            "planned_arms": 1,
            "completed_arms": 0,
            "failed_arms": 0,
            "censored_arms": 0,
        },
        "limitations": ["No measurements are claimed."],
    }
    payload["sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload, canonical_json(payload) + b"\n"


def _write_evidence(path: Path) -> tuple[dict[str, object], bytes]:
    payload, content = _signed_evidence()
    path.write_bytes(content)
    return payload, content


def test_report_retains_only_verified_portability_evidence_metadata(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "portability-evidence.json"
    expected, _ = _write_evidence(evidence_path)

    report = build_study_report(
        _STUDY,
        _RECEIPT,
        _COLLECTED,
        portability_evidence_path=evidence_path,
    )

    inputs = report["inputs"]
    assert isinstance(inputs, dict)
    expected_metadata = {
        key: expected[key]
        for key in (
            "format",
            "version",
            "campaign_id",
            "scale",
            "protocol_sha256",
            "world_manifest_sha256",
            "summary",
            "sha256",
        )
    }
    assert inputs["portability_evidence"] == expected_metadata
    assert "arms" not in inputs["portability_evidence"]
    assert "observations" not in inputs["portability_evidence"]


@pytest.mark.parametrize("kind", ["hash", "schema"])
def test_report_rejects_tampered_portability_evidence(
    tmp_path: Path, kind: str
) -> None:
    evidence_path = tmp_path / "portability-evidence.json"
    payload, _ = _write_evidence(evidence_path)
    if kind == "hash":
        payload["limitations"] = ["Tampered after signing."]
        expected = "portability evidence hash mismatch"
    else:
        payload["unexpected"] = True
        payload["sha256"] = hashlib.sha256(
            canonical_json({key: value for key, value in payload.items() if key != "sha256"})
        ).hexdigest()
        expected = "invalid top-level schema"
    evidence_path.write_bytes(canonical_json(payload) + b"\n")

    with pytest.raises(ValueError, match=expected):
        build_study_report(
            _STUDY,
            _RECEIPT,
            _COLLECTED,
            portability_evidence_path=evidence_path,
        )


def test_report_bundle_includes_original_portability_evidence_bytes(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "portability-evidence.json"
    _, content = _write_evidence(evidence_path)
    report = build_study_report(
        _STUDY,
        _RECEIPT,
        _COLLECTED,
        portability_evidence_path=evidence_path,
    )

    bundle = write_study_report(report, tmp_path / "bundles")

    assert (bundle / "inputs" / "portability-evidence.json").read_bytes() == content


def test_omitted_portability_evidence_preserves_legacy_report_shape() -> None:
    report = build_study_report(_STUDY, _RECEIPT, _COLLECTED)

    inputs = report["inputs"]
    bundle_inputs = report["_bundle_inputs"]
    assert isinstance(inputs, dict)
    assert isinstance(bundle_inputs, dict)
    assert "portability_evidence" not in inputs
    assert "portability_evidence" not in bundle_inputs
    assert json.loads(_COLLECTED.read_text())["report_sha256"] == inputs[
        "collected_report_sha256"
    ]
