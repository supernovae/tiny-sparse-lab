"""Join pinned selection rationale, file-level rights and retained source yield.

The structured fields are review aids, not a synthetic quality grade or an
assertion that a repository's root license grants rights to every file.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from sparselab.corpus.project import load_project


def inventory(project_path: Path, release: Path, pool_report: Path) -> dict:
    project = load_project(project_path)
    metrics = json.loads(pool_report.read_text(encoding="utf-8"))
    rights_report = json.loads(
        (release / "license-report.json").read_text(encoding="utf-8")
    )
    summaries = defaultdict(
        lambda: {
            "bytes": 0,
            "files": 0,
            "roles": Counter(),
            "training_eligibility": Counter(),
            "boundary_flags": Counter(),
            "file_spdx": Counter(),
        }
    )
    for file in rights_report["files"]:
        summary = summaries[file["source_id"]]
        summary["files"] += 1
        summary["bytes"] += file["size"]
        summary["roles"][file["role"]] += 1
        policy = file.get("rights")
        if policy is not None:
            summary["training_eligibility"][policy["training_eligibility"]] += 1
            summary["file_spdx"][
                policy.get("detected_spdx_expression") or "repository_default"
            ] += 1
            summary["boundary_flags"].update(policy["boundary_flags"])
    rows = []
    for source in sorted(project.sources, key=lambda item: item.id):
        selected = summaries[source.id]
        counts = metrics["source_counts"].get(source.id, {})
        acquisition = source.acquisition.model_dump(mode="json")
        rows.append(
            {
                "source_id": source.id,
                "family": source.source_family,
                "split": project.splits.assignments[source.source_family],
                "first_party_origin": source.canonical_uri,
                "revision": source.revision,
                "selection_rationale": source.notes,
                "claimed_domains": source.domains,
                "declared_document_kinds": source.document_kinds,
                "include_patterns": acquisition.get("include"),
                "exclude_patterns": acquisition.get("exclude"),
                "selected_file_count": selected["files"],
                "selected_snapshot_bytes": selected["bytes"],
                "selected_file_roles": dict(selected["roles"]),
                "file_rights_training_decisions": dict(
                    selected["training_eligibility"]
                ),
                "file_specific_spdx_evidence": dict(selected["file_spdx"]),
                "path_boundary_flags": dict(selected["boundary_flags"]),
                "retained_normalized": counts,
                "explicit_training_restriction": source.explicit_training_restriction,
                "license_url": source.license_url,
                "review_limits": "Source selection and pinned rights evidence are not proof of pedagogical quality, semantic isolation, or permission for model-weight publication.",
            }
        )
    return {
        "release_id": metrics["release_id"],
        "sources": rows,
        "source_count": len(rows),
        "quality_score": None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("pool_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            inventory(args.project, args.release, args.pool_report),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
