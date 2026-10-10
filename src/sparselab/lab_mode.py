"""Lab mode: one local command from a YAML delta to a scored comparison.

``sparselab try`` is the fast path for normal, authorized local experiments. It
derives the candidate through native ``config derive``, trains both arms through
the native trainer, scores both on the held-out validation split through native
checkpoint-bound evaluation and writes one compact, self-hashed run record.

Lab mode deliberately skips ExperimentPlan locks, Campaign approvals and
reconciliation, corpus admission reviews and proposal/stop documents. It keeps:

* resource limits: storage preflight, optional ``ResourceEnvelope`` and an
  optional, caller-chosen wall limit (no implicit short timeout);
* data identity: tokenizer, train and validation digests for each arm;
* safe cancellation: SIGINT/SIGTERM or a cancel sentinel stop training at a
  checkpointed step boundary and the record says so;
* held-out checks: both arms are scored on the same validation bytes, with the
  same tokenizer and scored-target count, or the comparison is refused.

A lab record is evidence of one local comparison, not a release, promotion or
scientific conclusion. Release runs keep the full ExperimentPlan/Campaign path.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import secrets
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RECORD_FORMAT = "sparselab-lab-try-v1"
DELTA_VERSION = 1
_DELTA_KEYS = {"lab_delta_version", "question", "set"}
EXIT_NOT_COMPARABLE = 3
EXIT_INTERRUPTED = 130


@dataclass(frozen=True)
class Candidate:
    """A candidate arm: either a delta over the baseline or a full config."""

    kind: str  # "delta" or "config"
    source: Path
    settings: dict[str, Any]
    question: str | None


def _canonical(value: Any) -> bytes:
    from sparselab.training.manifest import canonical_json

    return canonical_json(value)


def lab_root(work_dir: Path, override: Path | None = None) -> Path:
    """Lab-mode state lives under the persistent work root, outside Git."""
    return (override if override is not None else work_dir / "lab").absolute()


def read_candidate(path: Path) -> Candidate:
    """Classify CANDIDATE as a lab delta document or a full RunConfig."""
    from sparselab.derivation import _settings
    from sparselab.experiments.plan import read_document

    path = Path(path).absolute()
    document = read_document(path)
    if "schema_version" in document:
        return Candidate("config", path, {}, None)
    unknown = set(document) - _DELTA_KEYS
    if unknown:
        raise ValueError(
            "lab delta accepts only lab_delta_version, question and set; "
            f"unexpected: {', '.join(sorted(unknown))}"
        )
    version = document.get("lab_delta_version", DELTA_VERSION)
    if version != DELTA_VERSION or type(version) is not int:
        raise ValueError(f"unsupported lab_delta_version: {version!r}")
    settings = document.get("set")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("lab delta requires a nonempty 'set' mapping of dotted fields")
    _settings(settings)
    question = document.get("question")
    if question is not None and not isinstance(question, str):
        raise ValueError("lab delta question must be a string")
    return Candidate("delta", path, dict(settings), question)


def _sha256_file(path: Path) -> str:
    from sparselab.training.manifest import sha256_file

    return sha256_file(path)


def _with_runs_dir(config: Any, runs: Path) -> Any:
    """Operational relocation of the run store; scientific settings unchanged."""
    return config.model_copy(
        update={"logging": config.logging.model_copy(update={"root_dir": runs})}
    )


def _with_backend(config: Any, backend: str | None) -> Any:
    if backend is None:
        return config
    return config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"backend": backend})}
    )


def _config_key(config: Any) -> str:
    from sparselab.training.manifest import config_sha256, source_identity

    return hashlib.sha256(
        _canonical(
            {
                "config_sha256": config_sha256(config.model_dump(mode="json")),
                "source_identity_sha256": source_identity()["sha256"],
            }
        )
    ).hexdigest()


def _progress(run: Path) -> dict[str, Any]:
    path = run / "progress.json"
    if path.is_symlink() or not path.is_file():
        return {"status": "missing"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return {"status": "unreadable"}
    return value if isinstance(value, dict) else {"status": "unreadable"}


def _stop_reason(progress: Mapping[str, Any]) -> str | None:
    stages = progress.get("stages")
    if isinstance(stages, list) and stages and isinstance(stages[-1], dict):
        reason = stages[-1].get("reason")
        if isinstance(reason, str):
            return reason
    return None


def _reusable_baseline(runs: Path, prefix: str) -> tuple[str | None, str]:
    """Find a completed baseline with the same effective config and source."""
    attempts = sorted(
        path.name
        for path in runs.glob(prefix + "*")
        if path.is_dir()
        and not path.is_symlink()
        and (path.name == prefix or path.name[len(prefix) :].startswith("-"))
    )
    for name in attempts:
        if _progress(runs / name).get("status") == "completed":
            return name, prefix
    # Never mutate or reuse an interrupted attempt; start a new one beside it.
    return None, prefix if not attempts else f"{prefix}-{len(attempts) + 1}"


def _data_identity(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "data" / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise TypeError("prepared-data manifest must be an object")
    return {
        "source": manifest.get("source"),
        "prepared_manifest_sha256": manifest.get("manifest_sha256"),
        "source_identity_sha256": manifest.get("source_identity_sha256"),
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "train_sha256": manifest.get("train", {}).get("sha256"),
        "train_tokens": manifest.get("train", {}).get("tokens"),
        "validation_sha256": manifest.get("validation", {}).get("sha256"),
        "validation_tokens": manifest.get("validation", {}).get("tokens"),
    }


def _run_identity(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    return {
        "effective_config_sha256": manifest.get("effective_config_sha256"),
        "git_commit": manifest.get("git_commit"),
        "git_dirty": manifest.get("git_dirty"),
        "package_version": manifest.get("package_version"),
        "source_identity_sha256": (manifest.get("source_identity") or {}).get("sha256")
        if isinstance(manifest.get("source_identity"), dict)
        else manifest.get("source_identity"),
    }


def code_revision() -> dict[str, Any]:
    """Best-effort Git revision of the running checkout (read-only Git calls)."""
    import subprocess

    from sparselab.recovery.provenance import _git, repository_root

    root = repository_root(Path(__file__).parent)
    if root is None:
        return {"git_commit": None, "git_dirty": None}
    try:
        commit = _git(root, "rev-parse", "HEAD").decode().strip()
        dirty = bool(_git(root, "status", "--porcelain=v1", "--untracked-files=no"))
    except OSError, subprocess.CalledProcessError, UnicodeError:
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _peak_rss_bytes() -> int | None:
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except OSError, ValueError:
        return None
    # Linux reports KiB; macOS reports bytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


class LabCancelled(Exception):
    """Training stopped at a safe boundary before the comparison completed."""

    def __init__(self, arm: str, reason: str, row: dict[str, Any]) -> None:
        super().__init__(f"{arm} arm stopped before completion: {reason}")
        self.arm = arm
        self.reason = reason
        self.row = row


def _train_and_score(
    *,
    arm: str,
    config: Any,
    run_id: str,
    reuse: bool,
    cancel_path: Path,
    max_wall_seconds: float | None,
    authorization: Any,
    resource_envelope: Any,
    tokenizer_batch_documents: int,
    tokenizer_batch_source_bytes: int,
    train_fn: Callable[..., str],
) -> dict[str, Any]:
    from sparselab.evaluation.inference import load_run, write_inference_result

    runs = config.logging.root_dir
    started = time.monotonic()
    if not reuse:
        returned = train_fn(
            config,
            run_id=run_id,
            cancel_path=cancel_path,
            max_wall_seconds=max_wall_seconds,
            authorization=authorization,
            resource_envelope=resource_envelope,
            tokenizer_batch_documents=tokenizer_batch_documents,
            tokenizer_batch_source_bytes=tokenizer_batch_source_bytes,
        )
        if returned != run_id:
            raise ValueError(f"trainer returned unexpected run id {returned!r}")
    train_seconds = time.monotonic() - started
    run = runs / run_id
    progress = _progress(run)
    row: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run),
        "reused": reuse,
        "status": progress.get("status"),
        "stop_reason": _stop_reason(progress),
        "train_seconds": None if reuse else round(train_seconds, 3),
    }
    if progress.get("status") != "completed":
        row["status"] = progress.get("status") or "interrupted"
        raise LabCancelled(arm, str(row["stop_reason"] or row["status"]), row)
    started = time.monotonic()
    loaded = load_run(run_id, runs, None, None, authorization=authorization)
    result = loaded.evaluate()
    result.update({"source": "lab_try_eval", "identity": loaded.identity})
    observation = write_inference_result(loaded.run, "eval", result)
    identity = loaded.identity
    row.update(
        {
            "eval_seconds": round(time.monotonic() - started, 3),
            "checkpoint": identity.get("checkpoint_relative_path"),
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "step": identity.get("step"),
            "tokens_seen": identity.get("tokens_seen"),
            "parameters": (identity.get("parameter_inventory") or {}).get("total"),
            "seed": config.seed,
            "heldout": {
                "split": "validation",
                "loss": result["loss"],
                "perplexity": result["perplexity"],
                "valid_targets": result["valid_targets"],
                "batches": result["batches"],
                "observation": str(observation),
            },
            "data": {
                **_data_identity(run),
                "eval_data_sha256": identity.get("data_sha256"),
                "eval_tokenizer_sha256": identity.get("tokenizer_sha256"),
            },
            "code": _run_identity(run),
        }
    )
    return row


def heldout_checks(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict:
    """Refuse comparisons that are not scored on identical held-out inputs."""
    base_data = baseline["data"]
    cand_data = candidate["data"]
    checks = {
        "validation_distinct_from_train": all(
            data["validation_sha256"] is not None
            and data["validation_sha256"] != data["train_sha256"]
            for data in (base_data, cand_data)
        ),
        "same_validation_data": base_data["validation_sha256"]
        == cand_data["validation_sha256"]
        and base_data["validation_sha256"] is not None,
        "same_tokenizer": base_data["eval_tokenizer_sha256"]
        == cand_data["eval_tokenizer_sha256"],
        "same_scored_targets": baseline["heldout"]["valid_targets"]
        == candidate["heldout"]["valid_targets"]
        and baseline["heldout"]["valid_targets"] > 0,
        "finite_losses": all(
            math.isfinite(row["heldout"]["loss"]) for row in (baseline, candidate)
        ),
    }
    return {
        "comparable": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


def compare(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict:
    checks = heldout_checks(baseline, candidate)
    if not checks["comparable"]:
        return {"verdict": "NOT_COMPARABLE", **checks}
    base_loss = float(baseline["heldout"]["loss"])
    cand_loss = float(candidate["heldout"]["loss"])
    delta = cand_loss - base_loss
    verdict = (
        "CANDIDATE_LOWER_LOSS"
        if delta < 0
        else "CANDIDATE_HIGHER_LOSS"
        if delta > 0
        else "TIE"
    )
    return {
        "verdict": verdict,
        "heldout_loss_delta": delta,
        "heldout_loss_relative": delta / base_loss if base_loss else None,
        **checks,
    }


def _seal(record: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return {**body, "record_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def write_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    sealed = _seal(record)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(sealed, indent=2, sort_keys=True, default=str) + "\n")
    return sealed


def read_record(path: Path) -> dict[str, Any]:
    """Read a lab record and reject edited content."""
    path = Path(path)
    if path.is_dir():
        path = path / "try.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("format") != RECORD_FORMAT:
        raise ValueError(f"not a lab try record: {path}")
    if _seal(record)["record_sha256"] != record.get("record_sha256"):
        raise ValueError(f"lab try record digest mismatch: {path}")
    return record


def run_try(
    candidate_path: Path,
    baseline_path: Path,
    *,
    work_dir: Path,
    lab_dir: Path | None = None,
    seed: int | None = None,
    backend: str | None = None,
    max_wall_seconds: float | None = None,
    authorization: Any = None,
    resource_envelope: Any = None,
    tokenizer_batch_documents: int | None = None,
    tokenizer_batch_source_bytes: int | None = None,
    fresh_baseline: bool = False,
    train_fn: Callable[..., str] | None = None,
    now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Train/score baseline and candidate locally and write one lab record."""
    from sparselab.config.loading import load_config
    from sparselab.data.encoding import (
        TOKENIZER_BATCH_DOCUMENTS,
        TOKENIZER_BATCH_SOURCE_BYTES,
    )
    from sparselab.data.legacy import require_current_dataset
    from sparselab.derivation import _config, _delta, derive_config
    from sparselab.experiments.compiler import patchable_config
    from sparselab.resource_envelope import check_envelope, current_process_rss_bytes
    from sparselab.training.manifest import source_identity

    if train_fn is None:
        from sparselab.training.trainer import train as train_fn
    if max_wall_seconds is not None and (
        not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0
    ):
        raise ValueError("--max-wall-seconds must be positive and finite")
    started_at = (now or (lambda: datetime.now(UTC)))()
    started = time.monotonic()
    root = lab_root(Path(work_dir), lab_dir)
    runs = root / "runs"
    baseline_path = Path(baseline_path).absolute()
    candidate = read_candidate(Path(candidate_path))
    baseline_source = load_config(baseline_path)
    require_current_dataset(baseline_source.dataset)

    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    try_id = f"try-{stamp}-{secrets.token_hex(4)}"
    try_dir = root / "tries" / try_id
    try_dir.mkdir(parents=True, exist_ok=False)
    try:
        baseline_file = baseline_path
        if seed is not None:
            baseline_file = try_dir / "baseline.yaml"
            derive_config(baseline_path, baseline_file, {"seed": seed})
        declared: dict[str, Any] | None = None
        if candidate.kind == "delta":
            settings = dict(candidate.settings)
            if seed is not None:
                settings.setdefault("seed", seed)
            candidate_file = try_dir / "candidate.yaml"
            declared = derive_config(baseline_path, candidate_file, settings)[
                "declared_delta"
            ]
        else:
            candidate_file = candidate.source
            if seed is not None:
                candidate_file = try_dir / "candidate.yaml"
                derive_config(candidate.source, candidate_file, {"seed": seed})
        base_config = _with_backend(
            _with_runs_dir(load_config(baseline_file), runs), backend
        )
        cand_config = _with_backend(
            _with_runs_dir(load_config(candidate_file), runs), backend
        )
        require_current_dataset(cand_config.dataset)
        delta = _delta(
            _config(patchable_config(load_config(baseline_file)), baseline_file.parent),
            _config(
                patchable_config(load_config(candidate_file)), candidate_file.parent
            ),
        )
        base_key = _config_key(base_config)
        if not delta or base_key == _config_key(cand_config):
            raise ValueError("candidate does not change any setting from the baseline")
        # Resource limits are checked before any training starts.
        envelope_reading = (
            check_envelope(
                resource_envelope,
                workspace=root,
                rss_bytes=current_process_rss_bytes(),
            )
            if resource_envelope is not None
            else None
        )
    except Exception as error:
        # Keep the failed attempt visible instead of leaving an unexplained folder.
        write_record(
            try_dir / "try.json",
            {
                "format": RECORD_FORMAT,
                "mode": "lab",
                "try_id": try_id,
                "created_at": started_at.isoformat(),
                "inputs": {
                    "baseline": {"path": str(baseline_path)},
                    "candidate": {"path": str(candidate.source)},
                },
                "status": "failed",
                "failure": f"{type(error).__name__}: {error}",
                "exit_status": 2,
            },
        )
        raise
    cancel_path = try_dir / "CANCEL"
    runs.mkdir(parents=True, exist_ok=True)
    print(
        f"lab try {try_id}: touch {cancel_path} (or Ctrl-C) to stop at a safe step",
        file=sys.stderr,
    )
    baseline_prefix = f"lab-base-{base_key[:16]}"
    reuse_id, new_id = (
        (None, f"{baseline_prefix}-{try_id}")
        if fresh_baseline
        else _reusable_baseline(runs, baseline_prefix)
    )
    record: dict[str, Any] = {
        "format": RECORD_FORMAT,
        "mode": "lab",
        "try_id": try_id,
        "created_at": started_at.isoformat(),
        "question": candidate.question,
        "inputs": {
            "baseline": {
                "path": str(baseline_path),
                "sha256": _sha256_file(baseline_path),
            },
            "candidate": {
                "kind": candidate.kind,
                "path": str(candidate.source),
                "sha256": _sha256_file(candidate.source),
                "derived_config": str(candidate_file),
            },
            "seed_override": seed,
            "backend_override": backend,
        },
        "declared_delta": declared,
        "delta": delta,
        "code": {
            "source_identity_sha256": source_identity()["sha256"],
            **code_revision(),
        },
        "resources": {
            "resource_envelope": None
            if resource_envelope is None
            else resource_envelope.model_dump(mode="json"),
            "envelope_reading": envelope_reading,
            "max_wall_seconds_per_arm": max_wall_seconds,
            "cancel_path": str(cancel_path),
        },
        "skipped_release_gates": [
            "experiment_plan_lock",
            "campaign_approval_and_reconciliation",
            "corpus_admission_review",
            "proposal_binding_stop_documents",
        ],
        "limitations": [
            "single local comparison with one seed per arm; not a significance test",
            "lab records are not release, promotion or scientific-conclusion evidence",
        ],
        "arms": {},
        "status": "running",
    }
    common = {
        "cancel_path": cancel_path,
        "max_wall_seconds": max_wall_seconds,
        "authorization": authorization,
        "resource_envelope": resource_envelope,
        "tokenizer_batch_documents": tokenizer_batch_documents
        or TOKENIZER_BATCH_DOCUMENTS,
        "tokenizer_batch_source_bytes": tokenizer_batch_source_bytes
        or TOKENIZER_BATCH_SOURCE_BYTES,
        "train_fn": train_fn,
    }
    exit_status = 0
    try:
        record["arms"]["baseline"] = _train_and_score(
            arm="baseline",
            config=base_config,
            run_id=reuse_id or new_id,
            reuse=reuse_id is not None,
            **common,
        )
        record["arms"]["candidate"] = _train_and_score(
            arm="candidate",
            config=cand_config,
            run_id=f"lab-{try_id}",
            reuse=False,
            **common,
        )
        record["comparison"] = compare(
            record["arms"]["baseline"], record["arms"]["candidate"]
        )
        record["status"] = "completed"
        if record["comparison"]["verdict"] == "NOT_COMPARABLE":
            exit_status = EXIT_NOT_COMPARABLE
    except LabCancelled as error:
        record["arms"][error.arm] = error.row
        record["status"] = "interrupted"
        record["interruption"] = {"arm": error.arm, "reason": error.reason}
        exit_status = EXIT_INTERRUPTED
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["interruption"] = {"arm": None, "reason": "keyboard_interrupt"}
        exit_status = EXIT_INTERRUPTED
    except Exception as error:  # noqa: BLE001 - every failure is retained in the record
        record["status"] = "failed"
        record["failure"] = f"{type(error).__name__}: {error}"
        exit_status = 1
    finally:
        record["resources"]["wall_seconds"] = round(time.monotonic() - started, 3)
        record["resources"]["peak_rss_bytes"] = _peak_rss_bytes()
        record["exit_status"] = exit_status
        sealed = write_record(try_dir / "try.json", record)
    return sealed, try_dir / "try.json"


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return "n/a" if value is None else str(value)


def summarize(record: Mapping[str, Any], path: Path | None = None) -> str:
    """Human-readable comparison; ``--json`` carries the same record."""
    lines = [f"LAB TRY {record['try_id']}  status: {record['status']}  (lab mode)"]
    if record.get("question"):
        lines.append(f"  question: {record['question']}")
    for field, change in (record.get("delta") or {}).items():
        lines.append(f"  delta {field}: {change['base']!r} -> {change['variant']!r}")
    for arm in ("baseline", "candidate"):
        row = record.get("arms", {}).get(arm)
        if row is None:
            lines.append(f"  {arm:<9} not scored")
            continue
        heldout = row.get("heldout", {})
        reused = " (reused)" if row.get("reused") else ""
        lines.append(
            f"  {arm:<9} {row['run_id']}{reused}  held-out loss "
            f"{_fmt(heldout.get('loss'))}  ppl {_fmt(heldout.get('perplexity'), 2)}"
            f"  targets {_fmt(heldout.get('valid_targets'))}"
        )
    comparison = record.get("comparison")
    if comparison:
        if comparison["verdict"] == "NOT_COMPARABLE":
            lines.append(
                "  verdict: NOT_COMPARABLE (failed held-out checks: "
                + ", ".join(comparison["failed"])
                + ")"
            )
        else:
            lines.append(
                f"  verdict: {comparison['verdict']}  Δ held-out loss "
                f"{comparison['heldout_loss_delta']:+.4f}"
            )
    if record.get("interruption"):
        stop = record["interruption"]
        lines.append(f"  stopped safely: arm={stop['arm']} reason={stop['reason']}")
    if record.get("failure"):
        lines.append(f"  failure: {record['failure']}")
    resources = record.get("resources", {})
    lines.append(
        f"  wall {_fmt(resources.get('wall_seconds'), 1)}s  "
        f"peak rss {_fmt(resources.get('peak_rss_bytes'))} bytes"
    )
    if path is not None:
        lines.append(f"  record: {path}")
    return "\n".join(lines)
