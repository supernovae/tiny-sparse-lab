"""Offline neutral metadata acceptance through the public report command."""

import json
import subprocess
from pathlib import Path

import pytest
from report_fixtures import write_report_inputs

from sparselab.cli.main import build_parser

MATRIX = "e1e4b9eae75254117f0d1c2b7ec7a8deaf075f1d196d467d1eeb2fcc31359bfc"
STUDY = "ec13286d47d1d4f770940f415f7caa90f01bb560b6a781d12d7f578c799b51dc"
CONFIGS = [
    "0a2c6cff8756684c12f1ce2743eb392e40a161ce64caa14dbd039aeaa56a4787",
    "7299f564f23a6493afe3bcc2ec187d731028ea7aa1fdc45ff5a6331dfe140df8",
]


@pytest.fixture
def acceptance(tmp_path):
    root = write_report_inputs(tmp_path / "inputs", historical=True)
    expected = {
        "format": "sparselab-study-identity-expectations-v1",
        "matrix_sha256": MATRIX,
        "study_sha256": STUDY,
        "config_identity_mode": "legacy_default_fields",
        "config_sha256": CONFIGS.copy(),
    }
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(expected))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()
    args = [
        "study",
        "report",
        str(root / "study.yaml"),
        str(root / "receipt.json"),
        "--evidence",
        str(root / "collected-report.json"),
        "--output",
        str(tmp_path / "out"),
        "--identity-expectations",
        str(path),
        "--candidate-commit",
        commit,
    ]
    return tmp_path, root, path, expected, args, commit


def invoke(args):
    parsed = build_parser().parse_args(args)
    parsed.handler(parsed)


def test_public_identity_acceptance_records_candidate_without_payloads(
    acceptance, monkeypatch, capsys
):
    tmp, _root, _path, expected, args, commit = acceptance
    unrelated = tmp / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    invoke(args)
    bundle = Path(capsys.readouterr().out.strip())
    report = json.loads((bundle / "report.json").read_text())
    checked = report["identity_acceptance"]
    assert checked["verified"] is True
    assert checked["expected"] == expected
    assert checked["candidate_commit"] == commit
    assert len(checked["candidate_package_sha256"]) == 64
    assert not (tmp / "runs").exists()
    assert not (tmp / "workers").exists()


@pytest.mark.parametrize(
    "mutation",
    ["matrix", "study", "mode", "order", "missing", "invalid", "extra", "candidate"],
)
def test_public_identity_acceptance_refuses_drift_before_output(acceptance, mutation):
    tmp, _root, path, expected, args, _commit = acceptance
    if mutation == "matrix":
        expected["matrix_sha256"] = "0" * 64
    elif mutation == "study":
        expected["study_sha256"] = "0" * 64
    elif mutation == "mode":
        expected["config_identity_mode"] = "current"
    elif mutation == "order":
        expected["config_sha256"].reverse()
    elif mutation == "missing":
        del expected["study_sha256"]
    elif mutation == "invalid":
        expected["config_sha256"] = ["invalid"]
    elif mutation == "extra":
        expected["approve"] = True
    elif mutation == "candidate":
        args[-1] = "0" * 40
    path.write_text(json.dumps(expected))
    with pytest.raises(ValueError):
        invoke(args)
    assert not (tmp / "out").exists()


@pytest.mark.parametrize("flag", ["--candidate-commit", "--identity-expectations"])
def test_public_identity_acceptance_requires_paired_flags(acceptance, flag):
    tmp, _root, _path, _expected, args, _commit = acceptance
    position = args.index(flag)
    del args[position : position + 2]
    with pytest.raises(ValueError, match="required together"):
        invoke(args)
    assert not (tmp / "out").exists()
