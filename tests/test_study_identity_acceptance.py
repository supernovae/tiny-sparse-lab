"""Offline neutral metadata acceptance through the public report command."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from report_fixtures import write_report_inputs
from study_identity_vectors import CONFIGS, MATRIX, STUDY

from sparselab.cli.main import build_parser
from sparselab.experiments import identity_acceptance
from sparselab.training.manifest import source_identity


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
    commit = "a" * 40
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


@pytest.fixture
def committed_package(tmp_path):
    repo = tmp_path / "package-repo"
    package = repo / "src" / "sparselab"
    shutil.copytree(
        Path(identity_acceptance.__file__).resolve().parents[1],
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Commit package fixture",
        ],
        cwd=repo,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    return repo, package, commit


def invoke_copy(args, committed_package, cwd):
    repo, _package, commit = committed_package
    args = list(args)
    args[args.index("--candidate-commit") + 1] = commit
    env = dict(os.environ, PYTHONPATH=str(repo / "src"))
    return subprocess.run(
        [sys.executable, "-m", "sparselab", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("mode", ["legacy_default_fields", "current"])
def test_public_identity_acceptance_records_candidate_without_payloads(
    acceptance, committed_package, mode
):
    tmp, _root, path, expected, args, _commit = acceptance
    _repo, _package, commit = committed_package
    if mode == "current":
        root = write_report_inputs(tmp / "current", historical=False)
        receipt = json.loads((root / "receipt.json").read_text())
        expected.update(
            matrix_sha256=receipt["matrix_sha256"],
            study_sha256=receipt["study_sha256"],
            config_identity_mode="current",
            config_sha256=[row["config_sha256"] for row in receipt["runs"]],
        )
        path.write_text(json.dumps(expected))
        args[2:4] = [str(root / "study.yaml"), str(root / "receipt.json")]
        args[args.index("--evidence") + 1] = str(root / "collected-report.json")
    package = source_identity()
    unrelated = tmp / "unrelated"
    unrelated.mkdir()
    result = invoke_copy(args, committed_package, unrelated)
    assert result.returncode == 0, result.stderr
    bundle = Path(result.stdout.strip())
    report = json.loads((bundle / "report.json").read_text())
    checked = report["identity_acceptance"]
    assert checked["verified"] is True
    assert checked["expected"] == expected
    assert checked["candidate_commit"] == commit
    assert checked["candidate_package_sha256"] == package["sha256"]
    assert checked["candidate_package_algorithm"] == "package-path-sha256-v2"
    assert (
        checked["expectations_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert checked["candidate_package_status"] == "CLEAN_AND_COMMITTED"
    assert not (tmp / "runs").exists()
    assert not (tmp / "workers").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "matrix",
        "study",
        "mode",
        "order",
        "missing",
        "invalid",
        "invalid-matrix",
        "invalid-study",
        "extra",
        "candidate",
        "format",
    ],
)
def test_public_identity_acceptance_refuses_drift_before_output(
    acceptance, mutation, monkeypatch
):
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
    elif mutation == "invalid-matrix":
        expected["matrix_sha256"] = "invalid"
    elif mutation == "invalid-study":
        expected["study_sha256"] = "invalid"
    elif mutation == "extra":
        expected["approve"] = True
    elif mutation == "format":
        expected["format"] = "sparselab-study-identity-expectations-v2"
    elif mutation == "candidate":
        args[-1] = "0" * 40
    monkeypatch.setattr(
        identity_acceptance,
        "git_provenance",
        lambda paths: {
            "status": "CLEAN_AND_COMMITTED",
            "source_commit": _commit,
            "declarations": [
                {"sha256": digest} for _, digest in sorted(source_identity()["files"])
            ],
        },
    )
    path.write_text(json.dumps(expected))
    schema_cases = {
        "missing",
        "invalid",
        "invalid-matrix",
        "invalid-study",
        "extra",
        "format",
    }
    if mutation in schema_cases:
        monkeypatch.setattr(
            "sparselab.experiments.reporting.build_study_report",
            lambda *a, **k: pytest.fail("report built before expectation validation"),
        )
        with pytest.raises(ValidationError) as error:
            invoke(args)
        field = {
            "missing": "study_sha256",
            "invalid": "config_sha256",
            "invalid-matrix": "matrix_sha256",
            "invalid-study": "study_sha256",
            "extra": "approve",
            "format": "format",
        }[mutation]
        assert error.value.errors()[0]["loc"][0] == field
    else:
        message = (
            "candidate commit does not match"
            if mutation == "candidate"
            else "study identity expectations differ from verified report"
        )
        with pytest.raises(ValueError, match=message):
            invoke(args)
    assert not (tmp / "out").exists()


@pytest.mark.parametrize("flag", ["--candidate-commit", "--identity-expectations"])
def test_public_identity_acceptance_requires_paired_flags(
    acceptance, flag, monkeypatch
):
    tmp, _root, _path, _expected, args, _commit = acceptance
    monkeypatch.setattr(
        "sparselab.experiments.reporting.build_study_report",
        lambda *a, **k: pytest.fail("report built before flag validation"),
    )
    position = args.index(flag)
    del args[position : position + 2]
    with pytest.raises(ValueError, match="required together"):
        invoke(args)
    assert not (tmp / "out").exists()


@pytest.mark.parametrize(
    "status", ["DIRTY_DECLARATION", "UNTRACKED_DECLARATION", "UNKNOWN"]
)
def test_candidate_package_provenance_refuses_before_output(
    acceptance, monkeypatch, status
):
    tmp, _root, _path, _expected, args, commit = acceptance
    monkeypatch.setattr(
        identity_acceptance,
        "git_provenance",
        lambda paths: {"source_commit": commit, "status": status, "declarations": []},
    )
    with pytest.raises(ValueError, match="CLEAN_AND_COMMITTED"):
        invoke(args)
    assert not (tmp / "out").exists()


def test_invalid_expectations_refuse_before_report_work(acceptance, monkeypatch):
    tmp, _root, path, _expected, args, _commit = acceptance
    path.write_text('{"format":"future"}')
    monkeypatch.setattr(
        "sparselab.experiments.reporting.build_study_report",
        lambda *a, **k: pytest.fail("report built before expectation validation"),
    )
    with pytest.raises(ValueError):
        invoke(args)
    assert not (tmp / "out").exists()


def test_expectation_digest_uses_the_bytes_parsed(acceptance, monkeypatch):
    _tmp, _root, path, expected, _args, _commit = acceptance
    data = path.read_bytes()
    original = identity_acceptance.parse_document

    def parse(text, *, source):
        path.write_text('{"changed":true}')
        return original(text, source=source)

    monkeypatch.setattr(identity_acceptance, "parse_document", parse)
    parsed, digest = identity_acceptance.read_expectations(path)
    assert parsed.model_dump() == expected
    assert digest == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("kind", ["dirty", "untracked-file", "untracked-package"])
def test_real_package_changes_refuse_without_output(
    acceptance, committed_package, kind
):
    tmp, _root, _path, _expected, args, _commit = acceptance
    repo, package, commit = committed_package
    if kind == "dirty":
        path = package / "experiments" / "reporting.py"
        path.write_text(path.read_text() + "\n# Dirty fixture verifier\n")
    elif kind == "untracked-file":
        (package / "extra.py").write_text("# Untracked fixture module\n")
    else:
        subprocess.run(["git", "rm", "-rq", "--cached", "src"], cwd=repo, check=True)
        (repo / "anchor.txt").write_text("Unrelated repository\n")
        subprocess.run(["git", "add", "anchor.txt"], cwd=repo, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "Keep only unrelated anchor",
            ],
            cwd=repo,
            check=True,
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    result = invoke_copy(args, (repo, package, commit), tmp)
    assert result.returncode != 0
    assert "candidate package must be CLEAN_AND_COMMITTED" in result.stderr
    assert not (tmp / "out").exists()


def test_inventory_provenance_hash_mismatch_refuses_without_output(
    acceptance, monkeypatch
):
    tmp, _root, _path, _expected, args, commit = acceptance
    inventory = sorted(source_identity()["files"])

    def provenance(paths):
        package_root = Path(identity_acceptance.__file__).resolve().parents[1]
        assert tuple(paths) == tuple(package_root / name for name, _ in inventory)
        declarations = [{"sha256": digest} for _, digest in inventory]
        declarations[0]["sha256"] = "0" * 64
        return {
            "status": "CLEAN_AND_COMMITTED",
            "source_commit": commit,
            "declarations": declarations,
        }

    monkeypatch.setattr(identity_acceptance, "git_provenance", provenance)
    with pytest.raises(ValueError, match="changed during source verification"):
        invoke(args)
    assert not (tmp / "out").exists()
