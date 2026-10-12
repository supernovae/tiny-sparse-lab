"""Read-only follow-up plans; rendering never prepares a runtime or declaration."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

from sparselab.dashboard.probe_data import ProbeEntry

# Missing differs from explicitly saved null (no override / no selected policy).
REQUEST_FIELDS = (
    "tier",
    "final",
    "fast_fail",
    "lm_eval_tasks",
    "lm_eval_limit",
    "lab_dir",
    "runs_dir",
    "runtime_profile_path",
    "resource_envelope_path",
    "backend_override",
    "runtime_id",
    "runtime_profile",
    "resource_envelope",
)


def followup_plan(
    entry: ProbeEntry, owner: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Keep sealed evidence intact; state the sole intended policy change.

    Embedded try batteries get runtime/resource policy from their sealed owner,
    but training settings are not presented as a ProbeConfig declaration.
    Paths remain recorded locations, never resolved against the reader's cwd.
    """
    result = entry.result
    record = owner if owner is not None else result
    request = record.get("request")
    request = copy.deepcopy(request) if isinstance(request, dict) else {}
    verdict = result.get("verdict") or {}
    action = verdict.get("action")
    tier_field = "tier" if entry.source == "probe" else "probe_tier"
    tier = request.get(tier_field)
    if entry.source == "try" and tier is None:
        tier = result.get("tier")
    change = {}
    if action == "escalate":
        change = {"tier": {"from": tier, "to": verdict.get("next_tier")}}
    missing = [name for name in REQUEST_FIELDS if name not in request]
    if entry.source == "try":
        # These are native embedded-battery settings, not training selectors.
        missing = [name for name in missing if name != "tier"]
        if "probe_tier" not in request:
            missing.append("probe_tier")
    bindings = {
        role: copy.deepcopy(result.get(role)) for role in ("target", "baseline")
    }
    for role, value in bindings.items():
        if role == "baseline" and value is None:
            continue
        for field in ("checkpoint", "checkpoint_sha256"):
            if not isinstance(value, dict) or not value.get(field):
                missing.append(f"{role}.{field}")
    if "baseline_provenance" not in record:
        missing.append("baseline_provenance (historical selector context unknown)")
    if not (result.get("suite") or {}).get("sha256"):
        missing.append("suite.sha256")
    if "record_sha256" not in record:
        missing.append("record_sha256")
    if action == "escalate" and not verdict.get("next_tier"):
        missing.append("next_tier")
    return {
        "action": action,
        "source_record": str(entry.path.absolute()),
        "record_sha256": record.get("record_sha256"),
        "record_kind": entry.source,
        "sealed_request": request,
        "recorded_tier": tier,
        "intended_changes": change,
        "checkpoint_bindings": bindings,
        "baseline_provenance": copy.deepcopy(record.get("baseline_provenance")),
        "suite": copy.deepcopy(result.get("suite")),
        "benchmark_groups": [
            {
                key: copy.deepcopy((row.get("details") or {}).get(key))
                for key in ("benchmark_group", "tasks", "limit")
            }
            for row in result.get("probes") or []
            if row.get("id") == "lm_eval"
        ],
        "missing_fields": missing,
        "preparation": (
            "Separate explicit action required: author a ProbeConfig YAML from the "
            "sealed probe policy (for a try, review embedded battery settings separately). "
            "Copy saved runtime_profile/resource_envelope values into new native YAML "
            "files and reference those files; do not reuse possibly changed source files "
            "or resolve historical relative selectors in the reader directory. Preserve "
            "benchmark tasks/limit, final, fast_fail and backend policy (a saved null "
            "backend override inherits the verified checkpoint config). Null benchmark "
            "selectors do not freeze current default tasks: verify the recorded suite "
            "and measured benchmark group. Escalation adds the next tier checks; "
            "if no benchmark was measured yet, review the new benchmark explicitly. Apply only "
            "intended_changes; rerun changes nothing. Use the source record for the "
            "candidate and verify its recorded digest/checkpoint; select and verify the "
            "baseline's recorded checkpoint and provenance separately, never a newer "
            "run latest. Review relocated paths, runtime availability and suite/benchmark "
            "identity before executing sparselab probe with that declaration and saved "
            "backend override. Missing fields require explicit reconstruction; current "
            "defaults are not historical evidence. This plan does not promise reproduction "
            "and creates no files or execution authority."
        ),
    }


def render_followup(plan: Mapping[str, Any]) -> str:
    """One text representation for dashboard and other readers."""
    return json.dumps(dict(plan), indent=2, sort_keys=True)
