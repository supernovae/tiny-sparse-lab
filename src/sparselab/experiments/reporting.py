"""Read-only validation and publication of static architecture-study evidence.

This module deliberately does not load models, run inference, or open the writer-side
experiment store.  A report bundle is an immutable presentation of supplied evidence.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sparselab.engram.packs import _rename_noreplace
from sparselab.evaluation.capabilities import _passes
from sparselab.evaluation.evidence import experiment_evidence
from sparselab.experiments.analysis import build_research_analysis
from sparselab.experiments.charts import render_charts
from sparselab.experiments.study import (
    _comparison_report,
    _reject_duplicate_pairs,
    plan_study,
)
from sparselab.model.inspection import inspection_report, parameter_inventory
from sparselab.research.catalog import (
    DatasetProfile,
    DatasetsFile,
    ProfilesFile,
    ResearchEntry,
    ResearchRecipe,
    ScaleProfile,
)
from sparselab.training.manifest import (
    canonical_json,
    config_sha256,
    read_manifest,
    sha256_file,
)

_REPORT_FORMAT = "sparselab-static-study-report"
_BUNDLE_FORMAT = "sparselab-study-report-bundle"
_MAX_JSON = 32 * 1024 * 1024

_MAX_TELEMETRY_ROWS_PER_RUN = 100_000


def _read_json(path: Path, description: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular nonsymlink file")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {description}: {error}") from error
    if len(data) > _MAX_JSON:
        raise ValueError(f"{description} exceeds 32 MiB")
    try:
        value = json.loads(
            data,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object")
    _reject_nonfinite(value, description)
    return value, data


def _reject_nonfinite(value: object, description: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{description} contains a non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite(child, description)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite(child, description)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_run_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise ValueError("run_id must be a single safe path component")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_research(path: Path, study_path: Path) -> dict[str, object]:
    metadata, raw = _read_json(path, "research metadata")
    if (
        metadata.get("format") != "sparselab-research-scaffold"
        or type(metadata.get("version")) is not int
        or metadata["version"] != 1
        or not _is_sha256(metadata.get("research_sha256"))
    ):
        raise ValueError("unsupported research scaffold metadata")
    body = {key: value for key, value in metadata.items() if key != "research_sha256"}
    if hashlib.sha256(canonical_json(body)).hexdigest() != metadata["research_sha256"]:
        raise ValueError("research metadata hash mismatch")
    inventory = metadata.get("inputs")
    coordinates = metadata.get("coordinates")
    if not isinstance(inventory, list) or not isinstance(coordinates, list):
        raise ValueError("research metadata input inventory is missing")
    entry_data = metadata.get("entry")
    selection = metadata.get("selection")
    if (
        not isinstance(entry_data, dict)
        or not isinstance(selection, dict)
        or set(selection) != {"scale", "data", "backend", "design"}
        or not all(isinstance(value, str) for value in selection.values())
        or not _is_sha256(metadata.get("entry_sha256"))
        or not _is_sha256(metadata.get("recipe_sha256"))
    ):
        raise ValueError("research metadata lacks strict entry or selection data")
    entry = ResearchEntry.model_validate(entry_data)
    if (
        selection["backend"] not in {"cpu", "mps", "cuda", "rocm", "xpu"}
        or not selection["design"]
        or entry.id != metadata["entry"].get("id")
    ):
        raise ValueError("research metadata entry or selection is invalid")
    expected = {
        "base.yaml",
        "tokenizer.yaml",
        "matrix.yaml",
        "study.yaml",
        "research_sources/catalog-entry.json",
        "research_sources/recipe.json",
        "research_sources/profiles.json",
        "research_sources/datasets.json",
    }
    for coordinate in coordinates:
        coordinate_values = (
            coordinate.get("coordinate") if isinstance(coordinate, dict) else None
        )
        if (
            not isinstance(coordinate, dict)
            or not isinstance(coordinate_values, dict)
            or not _is_sha256(coordinate.get("config_sha256"))
            or coordinate.get("path") != f"configs/{coordinate['config_sha256']}.yaml"
            or any(
                not isinstance(axis, str) or not isinstance(label, str)
                for axis, label in coordinate_values.items()
            )
            or coordinate["path"] in expected
        ):
            raise ValueError("invalid research coordinate inventory")
        expected.add(coordinate["path"])

    root = path.parent.resolve()
    names: set[str] = set()
    study_digest: str | None = None
    source_hashes: dict[str, str] = {}
    for item in inventory:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not _is_sha256(item.get("sha256"))
        ):
            raise ValueError("invalid research metadata input inventory")
        name = item["path"]
        relative = Path(name)
        if (
            not name
            or "\\" in name
            or relative.is_absolute()
            or relative.as_posix() != name
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or name in names
        ):
            raise ValueError("invalid or duplicate research input path")
        candidate = root
        for part in relative.parts:
            candidate /= part
            if candidate.is_symlink():
                raise ValueError("research scientific input traverses a symlink")
        resolved = candidate.resolve(strict=True)
        if (
            root not in resolved.parents
            or not resolved.is_file()
            or sha256_file(resolved) != item["sha256"]
        ):
            raise ValueError("research scientific input inventory mismatch")
        names.add(name)
        if name.startswith("research_sources/"):
            source_hashes[name] = item["sha256"]
        if name == "study.yaml":
            study_digest = item["sha256"]
    if (
        names != expected
        or study_digest is None
        or study_path.is_symlink()
        or source_hashes.get("research_sources/catalog-entry.json")
        != metadata["entry_sha256"]
        or source_hashes.get("research_sources/recipe.json")
        != metadata["recipe_sha256"]
    ):
        raise ValueError("research metadata input inventory is incomplete")
    entry_source, _ = _read_json(
        root / "research_sources/catalog-entry.json", "research entry source"
    )
    recipe_source, _ = _read_json(
        root / "research_sources/recipe.json", "research recipe source"
    )
    profiles_source, _ = _read_json(
        root / "research_sources/profiles.json", "research profile source"
    )
    datasets_source, _ = _read_json(
        root / "research_sources/datasets.json", "research dataset source"
    )
    source_entry = ResearchEntry.model_validate(entry_source)
    recipe = ResearchRecipe.model_validate(recipe_source)
    profiles = ProfilesFile.model_validate(profiles_source)
    datasets = DatasetsFile.model_validate(datasets_source)
    recipe_scales = recipe.designs.get(selection["design"])
    if (
        source_entry != entry
        or recipe.id != entry.id
        or recipe_scales is None
        or selection["scale"] not in recipe_scales
        or set(profiles.scales) != {"smoke", "nano", "micro", "tiny"}
        or selection["scale"] not in profiles.scales
        or set(datasets.datasets) != {"offline", "tinystories"}
        or selection["data"] not in datasets.datasets
    ):
        raise ValueError(
            "research source files differ from declared entry or selection"
        )
    scale_recipe = recipe_scales[selection["scale"]]
    factorial_designs = [
        item.model_dump(mode="json") for item in scale_recipe.factorial_designs
    ]
    if metadata.get("factorial_designs", []) != factorial_designs:
        raise ValueError("research factorial designs differ from the bound recipe")
    design_axes = {
        axis: [{"label": option.label, "set": option.set} for option in options]
        for axis, options in scale_recipe.axes.items()
    }
    return {
        "sha256": _sha(raw),
        "research_sha256": metadata["research_sha256"],
        "metadata": metadata,
        "dataset_profile": {
            "name": selection["data"],
            "sha256": source_hashes["research_sources/datasets.json"],
            "profile": datasets.datasets[selection["data"]].model_dump(mode="json"),
        },
        "scale_profile": {
            "name": selection["scale"],
            "sha256": source_hashes["research_sources/profiles.json"],
            "profile": profiles.scales[selection["scale"]].model_dump(mode="json"),
        },
        "factorial_designs": factorial_designs,
        "design_axes": design_axes,
    }


def _read_telemetry(
    root: Path, run_ids: set[str], endpoints: dict[str, int]
) -> tuple[dict[str, list[dict[str, object]]], set[str]]:
    """Read only endpoint-bounded logical metric rows through a read-only URI."""
    database = root / "experiments.sqlite3"
    if not run_ids or not database.is_file() or database.is_symlink():
        return {}, set()
    allowed = (
        "train/loss",
        "validation/loss",
        "performance/step_seconds",
        "performance/tokens_per_second",
        "memory/device_peak_allocated_bytes",
        "memory/device_sampled_peak_bytes",
        "memory/process_peak_rss_bytes",
        "allocation/raw_tokens",
        "allocation/valid_targets",
        "allocation/weighted_neural_supervision_mass",
        "allocation/owner/neural_targets",
        "allocation/owner/lexical_targets",
        "allocation/owner/semantic_targets",
        "allocation/owner/hybrid_targets",
    )
    output: dict[str, list[dict[str, object]]] = {}
    truncated_runs: set[str] = set()
    try:
        with sqlite3.connect(
            database.resolve().as_uri() + "?mode=ro", uri=True
        ) as connection:
            connection.execute("PRAGMA query_only=ON")
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(metrics)")
            }
            required = {"run_id", "step", "tokens_seen", "wall_time", "name", "value"}
            if not required <= columns:
                return {}, set()
            placeholders = ",".join("?" for _ in allowed)
            query = (
                "SELECT run_id,step,tokens_seen,wall_time,name,value FROM metrics "
                f"WHERE run_id=? AND step<=? AND (name IN ({placeholders}) "
                "OR name GLOB 'engram/*' OR name GLOB 'moe/*' "
                "OR name GLOB 'attention/*' OR name GLOB 'allocation/*') "
                "ORDER BY step,name LIMIT ?"
            )
            for run_id in sorted(run_ids):
                endpoint = endpoints.get(run_id)
                if endpoint is None or endpoint < 0:
                    continue
                rows = connection.execute(
                    query,
                    (run_id, endpoint, *allowed, _MAX_TELEMETRY_ROWS_PER_RUN + 1),
                ).fetchall()
                if len(rows) > _MAX_TELEMETRY_ROWS_PER_RUN:
                    truncated_runs.add(run_id)
                    rows = rows[:_MAX_TELEMETRY_ROWS_PER_RUN]
                for (
                    actual_run,
                    step,
                    tokens,
                    wall_time,
                    name,
                    value,
                ) in rows:
                    if (
                        actual_run == run_id
                        and isinstance(step, int)
                        and isinstance(tokens, int)
                        and isinstance(wall_time, (int, float))
                        and math.isfinite(float(wall_time))
                        and isinstance(name, str)
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                    ):
                        output.setdefault(run_id, []).append(
                            {
                                "step": step,
                                "tokens_seen": tokens,
                                "wall_time": wall_time,
                                "name": name,
                                "value": value,
                            }
                        )
    except sqlite3.Error:
        return {}, set()
    return output, truncated_runs


def _cards_by_identity(study: Any, receipt: dict[str, object]) -> dict[str, str]:
    received = receipt.get("cards")
    if not isinstance(received, list) or len(received) != len(study.cards):
        raise ValueError("receipt card inventory does not match study")
    planned = [
        (item.card.name, item.card.digest, item.reference) for item in study.cards
    ]
    seen: set[tuple[str, str]] = set()
    mapping: dict[str, str] = {}
    for item in received:
        if not isinstance(item, dict):
            raise ValueError("invalid receipt card")
        identity = (item.get("name"), item.get("digest"))
        if not all(isinstance(x, str) for x in identity) or identity in seen:
            raise ValueError("duplicate or malformed receipt card")
        seen.add(identity)  # type: ignore[arg-type]
        matches = [
            reference
            for name, digest, reference in planned
            if (name, digest) == identity
        ]
        if len(matches) != 1 or not isinstance(item.get("reference"), str):
            raise ValueError("receipt cards differ from study")
        mapping[item["reference"]] = matches[0]
    if len(mapping) != len(planned):
        raise ValueError("receipt cards differ from study")
    return mapping


def _validate_capability(
    card: Any, value: object, expected_config: str, run_id: str
) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or value.get("name") != card.name
        or value.get("digest") != card.digest
    ):
        raise ValueError("capability card identity mismatch")
    result = value.get("result")
    if not isinstance(result, dict):
        if value.get("valid") is False and isinstance(value.get("error"), str):
            return {"valid": False, "error": value["error"]}
        raise ValueError("capability result missing retained result")
    if result.get("card") != card.name or result.get("card_digest") != card.digest:
        raise ValueError("capability result card identity mismatch")
    identity = result.get("identity")
    if not isinstance(identity, dict) or identity.get("run_id") != run_id:
        raise ValueError("capability run identity mismatch")
    config = identity.get("config")
    if not isinstance(config, dict) or config_sha256(config) != expected_config:
        raise ValueError("capability config identity mismatch")
    if (
        not _is_sha256(identity.get("checkpoint_sha256"))
        or not isinstance(identity.get("step"), int)
        or isinstance(identity.get("step"), bool)
        or not isinstance(identity.get("tokens_seen"), int)
        or isinstance(identity.get("tokens_seen"), bool)
        or not _is_sha256(identity.get("tokenizer_sha256"))
        or not _is_sha256(identity.get("source_identity_sha256"))
    ):
        raise ValueError("capability checkpoint identity is malformed")
    data_identity = identity.get("data_sha256")
    if (
        not isinstance(data_identity, dict)
        or not _is_sha256(data_identity.get("train"))
        or not _is_sha256(data_identity.get("validation"))
    ):
        raise ValueError("capability data identity is malformed")
    if not isinstance(result.get("valid"), bool):
        raise ValueError("capability validity is malformed")
    if value.get("valid") is not result["valid"]:
        raise ValueError("capability wrapper validity differs from retained result")
    cases = {case.identifier: case for case in card.cases}
    if result.get("case_count") != len(cases) or isinstance(
        result.get("case_count"), bool
    ):
        raise ValueError("capability case count mismatch")
    rows = result.get("results")
    if not isinstance(rows, list):
        raise ValueError("capability results must be a list")
    seen: set[str] = set()
    passed = 0
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("malformed capability case")
        case = cases.get(row["id"])
        if case is None or row["id"] in seen:
            raise ValueError("unknown or duplicate capability case")
        seen.add(row["id"])
        if (
            row.get("prompt") != case.prompt
            or row.get("expected") != case.expected
            or not isinstance(row.get("response"), str)
            or not isinstance(row.get("passed"), bool)
        ):
            raise ValueError("capability case evidence mismatch")
        actual = _passes(card, row["response"], case.expected)
        if actual != row["passed"]:
            raise ValueError("capability scorer result was tampered")
        passed += int(actual)
    if result["valid"]:
        score = result.get("score")
        reported_passed = result.get("passed")
        if (
            seen != set(cases)
            or not isinstance(reported_passed, int)
            or isinstance(reported_passed, bool)
            or reported_passed != passed
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0 <= float(score) <= 1
            or float(score) != passed / len(cases)
            or value.get("passed") != passed
            or value.get("case_count") != len(cases)
            or value.get("score") != score
        ):
            raise ValueError("capability score or case inventory mismatch")
    elif (
        rows
        or result.get("score") is not None
        or not isinstance(result.get("failures"), list)
        or value.get("score") is not None
    ):
        raise ValueError("invalid capability failure record")
    return result


def _endpoint(run: dict[str, object], config: Any) -> dict[str, object]:
    identity = run.get("identity")
    if not isinstance(identity, dict):
        return {"status": "unavailable", "reason": "run identity unavailable"}
    step, tokens = identity.get("step"), identity.get("tokens_seen")
    if (
        not isinstance(step, int)
        or isinstance(step, bool)
        or step < 0
        or not isinstance(tokens, int)
        or isinstance(tokens, bool)
        or tokens < 0
    ):
        return {"status": "unavailable", "reason": "run counters unavailable"}
    max_steps, max_tokens = config.training.max_steps, config.training.max_tokens
    complete = (
        step <= max_steps
        and tokens <= max_tokens
        and (step == max_steps or tokens == max_tokens)
    )
    return {
        "status": "complete" if complete else "partial",
        "step": step,
        "tokens_seen": tokens,
        "configured_max_steps": max_steps,
        "configured_max_tokens": max_tokens,
    }


def _validate_local_run(
    run_path: Path,
    run_id: str,
    expected_config: str,
    record: dict[str, object],
) -> dict[str, object]:
    try:
        if run_path.is_symlink() or not run_path.is_dir():
            raise ValueError("run directory is missing or unsafe")
        manifest_path = run_path / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("run manifest is a symlink")
        manifest = read_manifest(manifest_path)
        identity = record.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("collected run identity is unavailable")
        effective = manifest.get("effective_config")
        source = manifest.get("source_identity")
        artifacts = manifest.get("artifacts")
        if (
            manifest.get("run_id") != run_id
            or not isinstance(effective, dict)
            or config_sha256(effective) != expected_config
            or not isinstance(source, dict)
            or identity.get("run_id") != run_id
            or not isinstance(identity.get("config"), dict)
            or config_sha256(identity["config"]) != expected_config
            or identity.get("source_identity_sha256") != source.get("sha256")
            or not isinstance(artifacts, list)
        ):
            raise ValueError("local run manifest does not match collected identity")
        artifact_hashes = {
            item["relative_path"]: item["sha256"]
            for item in artifacts
            if isinstance(item, dict)
            and isinstance(item.get("relative_path"), str)
            and isinstance(item.get("sha256"), str)
        }
        data_identity = identity.get("data_sha256")
        if (
            identity.get("tokenizer_sha256") != artifact_hashes.get("tokenizer.json")
            or not isinstance(data_identity, dict)
            or data_identity
            != {
                "train": artifact_hashes.get("data/train.npy"),
                "validation": artifact_hashes.get("data/validation.npy"),
            }
        ):
            raise ValueError("local run artifacts differ from collected identity")
        local = experiment_evidence(run_path)
        if (
            local.get("run_id") != run_id
            or local.get("source_identity_sha256") != source.get("sha256")
            or local.get("verified_checkpoints") is not True
        ):
            raise ValueError("local run artifacts or checkpoints did not verify")
        checkpoint_digest = identity.get("checkpoint_sha256")
        step, tokens = identity.get("step"), identity.get("tokens_seen")
        matches = [
            item
            for item in local.get("checkpoints", [])
            if isinstance(item, dict)
            and item.get("verified") is True
            and item.get("digest") == checkpoint_digest
            and item.get("step") == step
            and item.get("tokens_seen") == tokens
        ]
        if (
            len(matches) != 1
            or identity.get("checkpoint_relative_path")
            != f"checkpoints/{matches[0]['path']}"
        ):
            raise ValueError("selected checkpoint does not match local run evidence")
        training_runtime = identity.get("training_runtime")
        runtime = manifest.get("runtime")
        if not isinstance(training_runtime, dict) or not isinstance(runtime, dict):
            raise ValueError("local runtime identity is unavailable")
        runtime_fields = (
            "engine",
            "backend",
            "device_index",
            "device_name",
            "framework_version",
            "os",
        )
        if any(training_runtime.get(key) != runtime.get(key) for key in runtime_fields):
            raise ValueError("local runtime differs from collected identity")
        return {
            "status": "validated",
            "evidence_level": local.get("evidence_level"),
            "manifest_sha256": hashlib.sha256(canonical_json(manifest)).hexdigest(),
            "selected_checkpoint": matches[0],
            "experiment_evidence": local,
        }
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        return {"status": "rejected", "reason": str(error)}


def _cache_layout(config: Any) -> dict[str, object]:
    """Describe configured cache tensors without constructing a model."""
    kind = config.attention.kind
    if config.runtime.engine != "pytorch":
        return {
            "status": "runtime_specific",
            "reason": "static PyTorch cache layout is not applied to the native MLX engine",
        }
    if kind == "block_sparse":
        return {
            "status": "unsupported",
            "reason": "block-sparse reference attention does not expose incremental KV caching",
        }
    heads = config.model.num_heads
    head_dim = config.model.hidden_dim // heads
    value_dim = config.attention.latent_dim // heads if kind == "mla" else head_dim
    mode = "sliding" if kind == "sliding_window" else kind
    dtype = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }.get(config.runtime.precision, "runtime_dependent")
    output: dict[str, object] = {
        "status": "configured",
        "kind": "architectural_shape",
        "mode": mode,
        "capacity_upper_bound": config.model.max_seq_len,
        "request_input_ids_shape": ["batch", "capacity"],
        "layers": config.model.num_layers,
        "key_shape_per_layer": ["batch", heads, "capacity", head_dim],
        "value_shape_per_layer": ["batch", heads, "capacity", value_dim],
        "dtype": dtype,
        "source": "configured architecture; actual request capacity and runtime dtype vary",
    }
    if kind == "sliding_window":
        output["attention_window"] = config.attention.window_size
    return output


def _cost_observations(
    telemetry: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    per_run: dict[str, dict[str, object]] = {}
    for run_id, rows in sorted(telemetry.items()):
        update_seconds = [
            float(row["value"])
            for row in rows
            if row.get("name") == "performance/step_seconds"
            and isinstance(row.get("value"), (int, float))
            and not isinstance(row.get("value"), bool)
        ]
        throughput = [
            float(row["value"])
            for row in rows
            if row.get("name") == "performance/tokens_per_second"
            and isinstance(row.get("value"), (int, float))
            and not isinstance(row.get("value"), bool)
        ]
        if update_seconds or throughput:
            per_run[run_id] = {
                "captured_update_samples": len(update_seconds),
                "captured_optimizer_update_seconds": sum(update_seconds)
                if update_seconds
                else None,
                "observed_tokens_per_second": throughput,
                "interpretation": "captured update telemetry only; evaluation, checkpoint, setup, and idle time are excluded",
            }
    return {
        "kind": "measured_update_telemetry" if per_run else "unavailable",
        "per_run": per_run,
        "monetary_cost": {
            "value": None,
            "unit": None,
            "reason": "no cloud billing or energy meter is present in the supplied evidence",
        },
        "missing_reason": None
        if per_run
        else "no endpoint-bounded optimizer duration or throughput samples were available",
    }


def _card_bundle_files(study: Any) -> list[dict[str, object]]:
    output = []
    for item in study.cards:
        content = (
            canonical_json(
                {
                    "reference": item.reference,
                    "card": asdict(item.card),
                }
            )
            + b"\n"
        )
        output.append(
            {
                "path": f"cards/{item.card.digest}.json",
                "content": content,
            }
        )
    return output


def build_study_report(
    study_path: Path,
    receipt_path: Path,
    evidence_path: Path,
    *,
    research_path: Path | None = None,
    runs_dir: Path | None = None,
) -> dict[str, object]:
    """Validate supplied collected evidence and return a presentation-only report."""
    study = plan_study(study_path)
    receipt, receipt_bytes = _read_json(receipt_path, "study receipt")
    if (
        receipt.get("format") != "sparselab-architecture-study-receipt"
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
    ):
        raise ValueError("unsupported study receipt")
    if (
        receipt.get("study_sha256") != study.study_sha256
        or receipt.get("matrix_sha256") != study.matrix_sha256
    ):
        raise ValueError("study specification differs from receipt")
    remap = _cards_by_identity(study, receipt)
    evidence, evidence_bytes = _read_json(evidence_path, "collected report")
    if (
        evidence.get("format") != "sparselab-architecture-study-report"
        or type(evidence.get("report_version")) is not int
        or evidence["report_version"] != 1
    ):
        raise ValueError("unsupported collected report")
    claimed = evidence.get("report_sha256")
    material = {key: value for key, value in evidence.items() if key != "report_sha256"}
    if (
        not _is_sha256(claimed)
        or hashlib.sha256(canonical_json(material)).hexdigest() != claimed
    ):
        raise ValueError("collected report hash mismatch")
    if (
        evidence.get("study_sha256") != study.study_sha256
        or evidence.get("matrix_sha256") != study.matrix_sha256
        or evidence.get("receipt_sha256") != _sha(receipt_bytes)
    ):
        raise ValueError("collected report identity differs from supplied inputs")
    receipt_runs = receipt.get("runs")
    records = evidence.get("runs")
    if (
        not isinstance(receipt_runs, list)
        or not isinstance(records, list)
        or len(records) != len(study.expanded)
        or len(receipt_runs) != len(study.expanded)
    ):
        raise ValueError("run inventory differs from study")

    evaluated: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for planned, submitted, raw in zip(
        study.expanded, receipt_runs, records, strict=True
    ):
        if not isinstance(submitted, dict) or not isinstance(raw, dict):
            raise ValueError("invalid run inventory")
        digest = config_sha256(planned.config.model_dump(mode="json"))
        run_id = _safe_run_id(submitted.get("run_id"))
        if (
            run_id in seen_ids
            or submitted.get("coordinate") != planned.coordinate
            or submitted.get("config_sha256") != digest
            or raw.get("run_id") != run_id
            or raw.get("coordinate") != planned.coordinate
            or raw.get("config_sha256") != digest
        ):
            raise ValueError("receipt/evidence coordinate identity mismatch")
        seen_ids.add(run_id)
        raw_identity = raw.get("identity")
        if (
            not isinstance(raw_identity, dict)
            or raw_identity.get("run_id") != run_id
            or not isinstance(raw_identity.get("config"), dict)
            or config_sha256(raw_identity["config"]) != digest
            or not _is_sha256(raw_identity.get("checkpoint_sha256"))
            or not isinstance(raw_identity.get("step"), int)
            or isinstance(raw_identity.get("step"), bool)
            or not isinstance(raw_identity.get("tokens_seen"), int)
            or isinstance(raw_identity.get("tokens_seen"), bool)
            or not _is_sha256(raw_identity.get("tokenizer_sha256"))
            or not _is_sha256(raw_identity.get("source_identity_sha256"))
        ):
            raise ValueError("collected run identity is malformed")
        data_identity = raw_identity.get("data_sha256")
        if (
            not isinstance(data_identity, dict)
            or not _is_sha256(data_identity.get("train"))
            or not _is_sha256(data_identity.get("validation"))
        ):
            raise ValueError("collected dataset identity is malformed")
        capabilities = raw.get("capabilities")
        if not isinstance(capabilities, dict):
            raise ValueError("run capabilities unavailable")
        normalized: dict[str, object] = {}
        for receipt_ref, study_ref in remap.items():
            source = capabilities.get(receipt_ref)
            if source is None and study_ref != receipt_ref:
                source = capabilities.get(study_ref)
            card = next(
                item.card for item in study.cards if item.reference == study_ref
            )
            checked = _validate_capability(card, source, digest, run_id)
            result_identity = checked.get("identity")
            if isinstance(result_identity, dict) and any(
                raw_identity.get(key) != result_identity.get(key)
                for key in (
                    "run_id",
                    "checkpoint_sha256",
                    "checkpoint_relative_path",
                    "step",
                    "tokens_seen",
                    "config",
                    "tokenizer_sha256",
                    "data_sha256",
                    "source_identity_sha256",
                    "training_runtime",
                )
            ):
                raise ValueError("run and capability identities differ")
            normalized[study_ref] = (
                {**source, "result": checked}
                if isinstance(source, dict) and "result" in source
                else checked
            )
        record = dict(raw)
        record["capabilities"] = normalized
        checkpoint_observations = raw.get("checkpoint_observations")
        if checkpoint_observations is not None:
            if not isinstance(checkpoint_observations, list):
                raise ValueError("checkpoint observations must be a list")
            # The collector records each immutable checkpoint's own identity and
            # card output. Preserve these raw rows verbatim; endpoint validation
            # above remains the comparison input and no curve point is selected.
            for observation in checkpoint_observations:
                if not isinstance(observation, dict):
                    raise ValueError("checkpoint observation must be an object")
                observed_identity = observation.get("identity")
                if not isinstance(observed_identity, dict) or (
                    observed_identity.get("run_id") != run_id
                    or config_sha256(observed_identity.get("config", {})) != digest
                    or not _is_sha256(observed_identity.get("checkpoint_sha256"))
                    or not isinstance(observed_identity.get("step"), int)
                    or isinstance(observed_identity.get("step"), bool)
                    or not isinstance(observed_identity.get("tokens_seen"), int)
                    or isinstance(observed_identity.get("tokens_seen"), bool)
                    or observed_identity.get("source_identity_sha256")
                    != raw_identity.get("source_identity_sha256")
                ):
                    raise ValueError(
                        "checkpoint observation identity does not bind this run"
                    )
            record["checkpoint_observations"] = checkpoint_observations
        record["endpoint_status"] = _endpoint(record, planned.config)
        evaluated.append(record)

    comparisons: list[dict[str, object]] = []
    for pair in study.pairs:
        baseline = evaluated[pair.baseline_index]
        variant = evaluated[pair.variant_index]
        complete = (
            baseline["endpoint_status"].get("status") == "complete"
            and variant["endpoint_status"].get("status") == "complete"
        )
        if complete:
            comparison = _comparison_report(study, pair, evaluated)
            comparison["endpoint_status"] = "complete"
        else:
            unavailable = [dict(item) for item in evaluated]
            unavailable[pair.baseline_index]["capabilities"] = {}
            unavailable[pair.variant_index]["capabilities"] = {}
            unavailable[pair.baseline_index]["validation"] = {
                "error": "configured endpoint was not reached"
            }
            unavailable[pair.variant_index]["validation"] = {
                "error": "configured endpoint was not reached"
            }
            comparison = _comparison_report(study, pair, unavailable)
            comparison["endpoint_status"] = "inconclusive"
            comparison["endpoint_reason"] = (
                "one or both configured endpoints are partial or unavailable"
            )
        comparisons.append(comparison)

    supplied = evidence.get("comparisons")
    if not isinstance(supplied, list) or len(supplied) != len(comparisons):
        raise ValueError("collected comparison inventory differs from study")
    reverse_remap = {study_ref: receipt_ref for receipt_ref, study_ref in remap.items()}
    for expected, actual in zip(comparisons, supplied, strict=True):
        if not isinstance(actual, dict):
            raise ValueError("collected comparison identity mismatch")
        for field in (
            "id",
            "baseline",
            "variant",
            "context",
            "configuration_differences",
        ):
            if actual.get(field) != expected.get(field):
                raise ValueError("collected comparison identity mismatch")
        if expected["endpoint_status"] != "complete":
            continue
        if actual.get("validation_loss") != expected.get("validation_loss"):
            raise ValueError("collected validation comparison was tampered")
        old_caps = actual.get("capabilities")
        if not isinstance(old_caps, dict):
            raise ValueError("collected comparison capabilities are missing")
        for study_ref, cap in expected["capabilities"].items():
            receipt_ref = reverse_remap.get(study_ref, study_ref)
            if old_caps.get(receipt_ref) != cap:
                raise ValueError("collected pair capability comparison was tampered")

    research: dict[str, object] | None = None
    if research_path is not None:
        research = _validate_research(research_path, study_path)
        metadata = research["metadata"]
        selection = metadata["selection"]
        if metadata["entry"]["id"] != study.name:
            raise ValueError("research entry id does not match study name")
        declared_coordinates = metadata["coordinates"]
        if len(declared_coordinates) != len(study.expanded):
            raise ValueError("research coordinate inventory differs from the study")
        dataset_info = research["dataset_profile"]
        scale_info = research["scale_profile"]
        dataset = DatasetProfile.model_validate(dataset_info["profile"])
        scale = ScaleProfile.model_validate(scale_info["profile"])
        for planned, declared in zip(study.expanded, declared_coordinates, strict=True):
            digest = config_sha256(planned.config.model_dump(mode="json"))
            config = planned.config
            if (
                declared["coordinate"] != planned.coordinate
                or declared["config_sha256"] != digest
                or config.runtime.backend != selection["backend"]
                or config.model.vocab_size != dataset.vocab_size
                or config.model.hidden_dim != scale.hidden_dim
                or config.model.num_layers != scale.num_layers
                or config.model.num_heads != scale.num_heads
                or config.model.max_seq_len != scale.max_seq_len
                or config.training.seq_len != scale.seq_len
                or config.training.max_steps != scale.max_steps
                or config.training.max_tokens != scale.max_tokens
                or config.training.micro_batch_size != scale.micro_batch_size
                or config.training.gradient_accumulation != scale.gradient_accumulation
                or config.dataset.source != dataset.source
                or config.dataset.revision != dataset.revision
                or config.dataset.train_max_documents != dataset.train_max_documents
                or config.dataset.validation_max_documents
                != dataset.validation_max_documents
                or config.dataset.train_max_tokens != dataset.train_max_tokens
                or config.dataset.validation_max_tokens != dataset.validation_max_tokens
                or config.dataset.synthetic_seed != dataset.dataset_seed
            ):
                raise ValueError("research selection differs from planned run configs")
    endpoints = {
        str(item["run_id"]): int(item["endpoint_status"]["step"])
        for item in evaluated
        if isinstance(item.get("endpoint_status"), dict)
        and isinstance(item["endpoint_status"].get("step"), int)
        and not isinstance(item["endpoint_status"].get("step"), bool)
    }
    if runs_dir is None:
        telemetry, truncated_telemetry = {}, set()
    else:
        telemetry, truncated_telemetry = _read_telemetry(
            runs_dir.resolve(),
            {str(item["run_id"]) for item in evaluated},
            endpoints,
        )
    for run in evaluated:
        allocation_metrics = [
            row
            for row in telemetry.get(str(run["run_id"]), [])
            if isinstance(row.get("name"), str)
            and str(row["name"]).startswith("allocation/")
        ]
        if allocation_metrics:
            # Read-only measured rows remain distinct from manifest identity and
            # from any estimate-labelled FLOP comparison.
            run["allocation_metrics"] = allocation_metrics

    local_evidence: dict[str, dict[str, object]] = {}
    if runs_dir is not None:
        root = runs_dir.resolve()
        for run in evaluated:
            run_id = str(run["run_id"])
            run_path = root / run_id
            if run_path.is_symlink() or run_path.parent.resolve() != root:
                local_evidence[run_id] = {
                    "status": "rejected",
                    "reason": "unsafe run path",
                }
                continue
            local_evidence[run_id] = _validate_local_run(
                run_path,
                run_id,
                str(run["config_sha256"]),
                run,
            )

    quantities = []
    for planned in study.expanded:
        inventory = parameter_inventory(planned.config)
        quantities.append(
            {
                "config_sha256": config_sha256(planned.config.model_dump(mode="json")),
                "inventory": asdict(inventory),
                "inspection": inspection_report(planned.config),
                "dataset": {
                    "source": planned.config.dataset.source,
                    "revision": planned.config.dataset.revision,
                    "config": planned.config.dataset.dataset_config,
                    "train_max_documents": planned.config.dataset.train_max_documents,
                    "validation_max_documents": planned.config.dataset.validation_max_documents,
                    "train_max_tokens": planned.config.dataset.train_max_tokens,
                    "validation_max_tokens": planned.config.dataset.validation_max_tokens,
                    "seed": planned.config.dataset.synthetic_seed,
                },
                "memory": planned.config.model.model_dump(
                    mode="json",
                    include={
                        "memory",
                        "memory_table_size",
                        "memory_ngram_size",
                        "memory_dim",
                        "memory_injection",
                        "memory_hash_heads",
                        "memory_ngram_orders",
                    },
                ),
                "cache_state": _cache_layout(planned.config),
            }
        )
    missing_or_rejected = [
        {"run_id": run_id, **value}
        for run_id, value in local_evidence.items()
        if value.get("status") == "rejected"
    ]
    if runs_dir is None:
        missing_or_rejected = [
            {
                "run_id": str(run["run_id"]),
                "status": "not_requested",
                "reason": "report-only mode did not receive a local runs directory",
            }
            for run in evaluated
        ]
    cards = [
        {
            "reference": item.reference,
            "name": item.card.name,
            "digest": item.card.digest,
            "hypothesis": item.card.hypothesis,
            "limitations": item.card.limitations,
            "scorer": item.card.scorer,
            "controls": dict(item.card.controls),
            "case_count": len(item.card.cases),
        }
        for item in study.cards
    ]
    research_analysis: dict[str, object] | None = None
    if research is not None:
        research_metadata = research.get("metadata")
        analysis_entry = (
            research_metadata.get("entry")
            if isinstance(research_metadata, dict)
            else None
        )
        analysis_selection = (
            research_metadata.get("selection")
            if isinstance(research_metadata, dict)
            else None
        )
        if not isinstance(analysis_entry, dict) or not isinstance(
            analysis_selection, dict
        ):
            raise ValueError("validated research analysis inputs are unavailable")
        research_analysis = build_research_analysis(
            evaluated,
            entry=analysis_entry,
            selection=analysis_selection,
            factorial_designs=research.get("factorial_designs"),
            design_axes=research.get("design_axes"),
            quantities=quantities,
            cards=cards,
        )
    report_inputs: dict[str, object] = {
        "study_sha256": study.study_sha256,
        "matrix_sha256": study.matrix_sha256,
        "receipt_sha256": _sha(receipt_bytes),
        "collected_report_sha256": claimed,
        "verification_scope": (
            "checksummed_report_only"
            if runs_dir is None
            else "report_plus_local_checkpoint_validation"
        ),
    }
    if research is not None:
        report_inputs["research_sha256"] = research["research_sha256"]
        dataset_profile = research["dataset_profile"]
        scale_profile = research["scale_profile"]
        report_inputs["dataset_profile_sha256"] = dataset_profile["sha256"]
        report_inputs["scale_profile_sha256"] = scale_profile["sha256"]
    return {
        "format": _REPORT_FORMAT,
        "version": 1,
        "inputs": report_inputs,
        "research": research,
        "cards": cards,
        "runs": evaluated,
        "research_analysis": research_analysis,
        "comparisons": comparisons,
        "architectural_quantities": quantities,
        "implementation_quantities": {
            "kind": "measured" if telemetry else "unavailable",
            "telemetry": telemetry,
            "truncated_run_ids": sorted(truncated_telemetry),
            "source": "read_only_experiment_store" if telemetry else "unavailable",
            "missing_reason": None
            if telemetry
            else "local endpoint-bounded telemetry was unavailable",
        },
        "costs": _cost_observations(telemetry),
        "learning_observations": local_evidence,
        "local_validation": local_evidence,
        "missing_or_rejected": missing_or_rejected,
        "limitations": [
            "Static report rendering does not train, load a model, or evaluate cards.",
            "Collected report values are checksummed evidence, not a new experiment attestation.",
            "Cache shapes are configuration-derived templates, not live request allocations.",
            "Factorial effects require four matched endpoint cells; missing or identity-mismatched cells remain inconclusive.",
            "Nondominance uses declared directions and exact observations; it is not a significance or winner claim.",
            "Allocation heatmaps show configuration-derived parameter inventories, not measured runtime cost.",
            "Boundary sweeps show configured levels only; no failure threshold is inferred.",
            "Monetary cost is unavailable without billing or energy-meter evidence.",
        ],
        "reproducibility_commands": [
            (
                f"sparselab study report {study_path} {receipt_path} "
                f"--evidence {evidence_path} --output REPORTS_DIR"
            )
        ],
        "_bundle_inputs": {
            "receipt": receipt_bytes,
            "collected": evidence_bytes,
            "study_path": study_path,
            "study_file_sha256": sha256_file(study_path),
            "matrix_path": study.matrix_path,
            "matrix_file_sha256": sha256_file(study.matrix_path),
            "receipt_path": receipt_path,
            "evidence_path": evidence_path,
            "research_path": research_path,
            "cards": _card_bundle_files(study),
        },
    }


def _html(report: dict[str, object], charts: dict[str, str]) -> str:
    research = report.get("research")
    metadata = research.get("metadata") if isinstance(research, dict) else None
    entry = metadata.get("entry") if isinstance(metadata, dict) else None
    entry = entry if isinstance(entry, dict) else {}
    title = html.escape(str(entry.get("title") or "Static study report"))
    question = html.escape(str(entry.get("question", "")))
    hypothesis = html.escape(str(entry.get("hypothesis", "")))
    run_rows: list[str] = []
    for run in report.get("runs", []):
        if not isinstance(run, dict):
            continue
        endpoint = run.get("endpoint_status")
        endpoint = endpoint if isinstance(endpoint, dict) else {}
        validation = run.get("validation")
        validation = validation if isinstance(validation, dict) else {}
        scores = []
        capabilities = run.get("capabilities")
        if isinstance(capabilities, dict):
            for reference, value in sorted(capabilities.items()):
                if not isinstance(value, dict):
                    continue
                result = value.get("result", value)
                score = result.get("score") if isinstance(result, dict) else None
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    scores.append(f"{reference}: {float(score):.6g}")
        columns = (
            run.get("run_id", ""),
            endpoint.get("status", "unavailable"),
            endpoint.get("step", ""),
            endpoint.get("tokens_seen", ""),
            validation.get("loss", "unavailable"),
            "; ".join(scores) if scores else "unavailable",
        )
        run_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in columns)
            + "</tr>"
        )
    runs_table = (
        "<table><thead><tr><th>Run</th><th>Endpoint</th><th>Step</th>"
        "<th>Tokens</th><th>Validation loss</th><th>Card scores</th></tr></thead>"
        f"<tbody>{''.join(run_rows)}</tbody></table>"
    )
    raw_cases = "".join(
        "<details><summary>"
        + html.escape(str(run.get("run_id", "run")))
        + " retained cases</summary><pre>"
        + html.escape(
            json.dumps(
                run.get("capabilities", {}),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
        + "</pre></details>"
        for run in report.get("runs", [])
        if isinstance(run, dict)
    )
    links = "".join(
        f'<li><a href="charts/{html.escape(name, quote=True)}">{html.escape(name)}</a></li>'
        for name in sorted(charts)
    )

    def block(heading: str, value: object) -> str:
        content = json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        return (
            f"<section><h2>{html.escape(heading)}</h2>"
            f"<pre>{html.escape(content)}</pre></section>"
        )

    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{title}</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse}th,td{border:1px solid #aaa;padding:.4rem;text-align:left}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head><body>"
        f"<h1>{title}</h1><p>Observed evidence only; no winner or efficiency claim is inferred.</p>"
        f"<h2>Question</h2><p>{question}</p><h2>Hypothesis</h2><p>{hypothesis}</p>"
        "<h2>Runs and endpoints</h2>"
        f"{runs_table}<h2>Observed comparisons</h2>"
        f"{block('Comparisons', report.get('comparisons', []))}"
        f"{block('Research analyses', report.get('research_analysis', {}))}"
        f"{block('Architectural quantities and cache layouts', report.get('architectural_quantities', []))}"
        f"{block('Implementation telemetry', report.get('implementation_quantities', {}))}"
        f"{block('Measured costs and missing values', report.get('costs', {}))}"
        f"{block('Cards and limitations', report.get('cards', []))}"
        f"{block('Local evidence validation', report.get('local_validation', {}))}"
        f"<h2>Retained raw cases</h2>{raw_cases}"
        f"<h2>Charts</h2><ul>{links}</ul>"
        f"{block('Report limitations', report.get('limitations', []))}"
        "</body></html>\n"
    )


def write_study_report(report: dict[str, object], output: Path) -> Path:
    """Publish an immutable content-addressed bundle, or verify an identical replay."""
    if (
        report.get("format") != _REPORT_FORMAT
        or type(report.get("version")) is not int
        or report["version"] != 1
    ):
        raise ValueError("unsupported static study report")
    report = dict(report)
    inputs = report.pop("_bundle_inputs", None)
    if not isinstance(inputs, dict):
        raise ValueError("report was not built by build_study_report")
    report_bytes = canonical_json(report) + b"\n"
    charts = render_charts(report)
    staging_parent = output.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=staging_parent))
    try:
        (temp / "charts").mkdir()
        (temp / "inputs").mkdir()
        report_inputs = report.get("inputs")
        if not isinstance(report_inputs, dict):
            raise ValueError("report input identities are unavailable")

        def add_input(relative: str, content: bytes) -> None:
            path = Path(relative)
            if (
                not relative
                or "\\" in relative
                or path.is_absolute()
                or path.as_posix() != relative
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError("unsafe report bundle input path")
            destination = temp / "inputs" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(content)

        research_path = inputs.get("research_path")
        research = report.get("research")
        if isinstance(research_path, Path):
            if research_path.is_symlink() or not isinstance(research, dict):
                raise ValueError("research source changed before report publication")
            metadata = research.get("metadata")
            inventory = metadata.get("inputs") if isinstance(metadata, dict) else None
            if not isinstance(inventory, list):
                raise ValueError("validated research input inventory is unavailable")
            research_bytes = research_path.read_bytes()
            if _sha(research_bytes) != research.get("sha256"):
                raise ValueError("research metadata changed before report publication")
            add_input("research.json", research_bytes)
            root = research_path.parent.resolve()
            for item in inventory:
                if not isinstance(item, dict):
                    raise ValueError("validated research inventory changed")
                relative = item.get("path")
                expected_digest = item.get("sha256")
                if not isinstance(relative, str) or not _is_sha256(expected_digest):
                    raise ValueError("validated research inventory changed")
                source = root / relative
                if source.is_symlink() or not source.is_file():
                    raise ValueError("research scientific input disappeared")
                content = source.read_bytes()
                if _sha(content) != expected_digest:
                    raise ValueError(
                        "research scientific input changed before report publication"
                    )
                add_input(f"research-scaffold/{relative}", content)

        for key, name, digest_key in (
            ("study_path", "study.yaml", "study_file_sha256"),
            ("matrix_path", "matrix.yaml", "matrix_file_sha256"),
        ):
            source = inputs.get(key)
            expected = inputs.get(digest_key)
            if isinstance(source, Path):
                if source.is_symlink() or not source.is_file():
                    raise ValueError(f"{name} changed before report publication")
                content = source.read_bytes()
                if not _is_sha256(expected) or _sha(content) != expected:
                    raise ValueError(f"{name} changed before report publication")
                add_input(name, content)
        for key, name in (
            ("receipt", "receipt.json"),
            ("collected", "collected-report.json"),
        ):
            content = inputs.get(key)
            if not isinstance(content, bytes):
                raise ValueError("report input bytes are unavailable")
            add_input(name, content)
        cards = inputs.get("cards")
        if not isinstance(cards, list):
            raise ValueError("report card inventory is unavailable")
        for item in cards:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not isinstance(item.get("content"), bytes)
            ):
                raise ValueError("report card inventory is invalid")
            add_input(item["path"], item["content"])

        (temp / "report.json").write_bytes(report_bytes)
        for name, content in charts.items():
            (temp / "charts" / name).write_text(content, encoding="utf-8")
        (temp / "index.html").write_text(_html(report, charts), encoding="utf-8")
        report_inputs = report["inputs"]
        summary = [
            "# Static study report",
            "",
            "Observed evidence only; no winner, cost, or efficiency claim is inferred.",
            f"- Study SHA-256: `{report_inputs['study_sha256']}`",
            f"- Matrix SHA-256: `{report_inputs['matrix_sha256']}`",
            f"- Verification scope: `{report_inputs['verification_scope']}`",
            f"- Runs: {len(report.get('runs', []))}",
            "- Raw input copies: `inputs/`",
            "- Structured report: `report.json`",
            "- Static presentation: `index.html`",
            "",
        ]
        (temp / "summary.md").write_text("\n".join(summary), encoding="utf-8")
        files = [
            {
                "path": str(path.relative_to(temp)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(temp.rglob("*"))
            if path.is_file()
        ]
        manifest: dict[str, object] = {
            "format": _BUNDLE_FORMAT,
            "version": 1,
            "files": files,
            "study_sha256": report_inputs["study_sha256"],
            "receipt_sha256": report_inputs["receipt_sha256"],
            "report_sha256": report_inputs["collected_report_sha256"],
        }
        manifest["bundle_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
        (temp / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        destination = output / str(manifest["bundle_sha256"])
        if os.path.lexists(destination):
            load_report_bundle(destination)
            for path in temp.rglob("*"):
                if path.is_file():
                    existing = destination / path.relative_to(temp)
                    if (
                        not existing.is_file()
                        or existing.read_bytes() != path.read_bytes()
                    ):
                        raise ValueError(
                            f"conflicting existing report bundle: {destination}"
                        )
            return destination
        output.mkdir(parents=True, exist_ok=True)
        _rename_noreplace(temp, destination)
        return destination
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def load_report_bundle(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("report bundle must be a real directory")
    root = path.resolve(strict=True)
    children = list(root.rglob("*"))
    if len(children) > 10_000:
        raise ValueError("report bundle contains too many paths")
    if any(child.is_symlink() for child in children):
        raise ValueError("report bundle contains symlinked content")
    manifest, _ = _read_json(root / "manifest.json", "bundle manifest")
    bundle_digest = manifest.get("bundle_sha256")
    if (
        manifest.get("format") != _BUNDLE_FORMAT
        or type(manifest.get("version")) is not int
        or manifest["version"] != 1
        or not _is_sha256(bundle_digest)
    ):
        raise ValueError("unsupported report bundle manifest")
    body = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    if (
        hashlib.sha256(canonical_json(body)).hexdigest() != bundle_digest
        or root.name != bundle_digest
    ):
        raise ValueError("report bundle manifest hash mismatch")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("report bundle file inventory missing")
    declared: set[str] = set()
    for item in files:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "size"}
            or not isinstance(item.get("path"), str)
            or not _is_sha256(item.get("sha256"))
            or not isinstance(item.get("size"), int)
            or isinstance(item.get("size"), bool)
            or item["size"] < 0
        ):
            raise ValueError("invalid report bundle file inventory")
        name = item["path"]
        relative = Path(name)
        if (
            not name
            or "\\" in name
            or relative.is_absolute()
            or relative.as_posix() != name
            or any(part in {"", ".", ".."} for part in relative.parts)
            or name in declared
        ):
            raise ValueError("unsafe or duplicate report bundle file path")
        child = root
        for part in relative.parts:
            child /= part
            if child.is_symlink():
                raise ValueError("report bundle child is a symlink")
        resolved = child.resolve(strict=True)
        if (
            root not in resolved.parents
            or not resolved.is_file()
            or resolved.stat().st_size != item["size"]
            or sha256_file(resolved) != item["sha256"]
        ):
            raise ValueError("report bundle child integrity failure")
        declared.add(name)
    report, _ = _read_json(root / "report.json", "bundle report")
    actual = {
        str(child.relative_to(root))
        for child in children
        if child.is_file() and child.relative_to(root).as_posix() != "manifest.json"
    }
    if actual != declared:
        raise ValueError("report bundle contains undeclared or missing content")
    directories = {
        str(parent)
        for name in declared
        for parent in Path(name).parents
        if str(parent) != "."
    }
    actual_directories = {
        str(child.relative_to(root)) for child in children if child.is_dir()
    }
    if actual_directories != directories:
        raise ValueError("report bundle contains unexpected directories")
    if (
        report.get("format") != _REPORT_FORMAT
        or type(report.get("version")) is not int
        or report["version"] != 1
    ):
        raise ValueError("invalid bundle report")
    report_inputs = report.get("inputs")
    if (
        not isinstance(report_inputs, dict)
        or manifest.get("study_sha256") != report_inputs.get("study_sha256")
        or manifest.get("receipt_sha256") != report_inputs.get("receipt_sha256")
        or manifest.get("report_sha256") != report_inputs.get("collected_report_sha256")
    ):
        raise ValueError("report bundle input identities differ from manifest")
    return {"manifest": manifest, "report": report, "path": str(root)}
