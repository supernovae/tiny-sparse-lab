"""Opt-in fixed identity assertions for an already verified native study report."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StrictStr

from sparselab.config.models import StrictModel
from sparselab.documents import parse_document
from sparselab.recovery.provenance import git_provenance
from sparselab.training.manifest import sha256_file, source_identity


class IdentityExpectations(StrictModel):
    format: Literal["sparselab-study-identity-expectations-v1"]
    matrix_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    study_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    config_identity_mode: Literal["current", "legacy_default_fields"]
    config_sha256: list[Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        min_length=1, max_length=1000
    )


def accept_identity(report: dict, path: Path, candidate_commit: str) -> dict:
    """Refuse drift before report output creation; never run or load a model."""
    expected = IdentityExpectations.model_validate(
        parse_document(path.read_text(encoding="utf-8"), source=path)
    )
    actual = {
        key: report["inputs"][key]
        for key in ("matrix_sha256", "study_sha256", "config_identity_mode")
    }
    actual["config_sha256"] = [row["config_sha256"] for row in report["runs"]]
    if actual != expected.model_dump(exclude={"format"}):
        raise ValueError("study identity expectations differ from verified report")
    provenance = git_provenance((Path(__file__).resolve(),))
    if provenance["source_commit"] != candidate_commit or not candidate_commit:
        raise ValueError("candidate commit does not match verifier source checkout")
    package = source_identity()
    return {
        "verified": True,
        "expectations_sha256": sha256_file(path),
        "expected": expected.model_dump(),
        "candidate_commit": provenance["source_commit"],
        "candidate_package_sha256": package["sha256"],
        "candidate_package_algorithm": package["algorithm"],
        "verifier_file_status": provenance["status"],
        "scope": "Fixed metadata identities only; no payload or scientific qualification. "
        "Package hash identifies actual loaded-source bytes; Git commit alone "
        "does not assert a clean package or installed-runtime equivalence.",
    }
