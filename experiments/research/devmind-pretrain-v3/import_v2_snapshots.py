"""Register byte-verified immutable v2 snapshots in the separate v3 workspace.

Copy snapshot directories with `cp -a --reflink=auto` first. This importer
refuses changed declarations, missing copies and an existing target lock.
It never writes to the v2 workspace or relaxes snapshot hash verification.
"""

import argparse
import os
import tempfile
from pathlib import Path

from sparselab.corpus.acquisition import (
    _project_sha,
    declaration_sha256,
    verify_acquisition,
    verify_snapshot,
)
from sparselab.corpus.project import load_project, source_declaration_payload
from sparselab.training.manifest import canonical_json


def import_snapshots(
    v2_recipe: Path, v2_work: Path, v3_recipe: Path, v3_work: Path
) -> int:
    old = load_project(v2_recipe)
    new = load_project(v3_recipe)
    if old.config.id == new.config.id:
        raise ValueError("source and target projects must have different IDs")
    original = verify_acquisition(old, v2_work)
    previous = {source.id: source for source in old.sources}
    base = v3_work / "corpora" / new.config.id
    destination = base / "acquisition.json"
    if destination.exists():
        raise ValueError("refusing to replace an existing target acquisition lock")
    if set(previous) != {source.id for source in new.sources}:
        raise ValueError("import only the unchanged v2 foundation before extending v3")
    entries = {}
    for source in new.sources:
        past = previous[source.id]
        if source_declaration_payload(source) != source_declaration_payload(past):
            raise ValueError(f"modified v2 source declaration: {source.id}")
        receipt = original["sources"][source.id]
        snapshot = base / "snapshots" / source.id / receipt["snapshot_sha256"]
        manifest = verify_snapshot(snapshot)
        if (
            manifest["declaration_sha256"] != declaration_sha256(source)
            or manifest["declaration"] != source_declaration_payload(source)
            or manifest["snapshot_sha256"] != receipt["snapshot_sha256"]
        ):
            raise ValueError(f"copied v2 snapshot differs: {source.id}")
        entries[source.id] = {**receipt, "snapshot_path": str(snapshot.resolve())}
    lock = {
        "schema_version": 1,
        "project_id": new.config.id,
        "project_sha256": _project_sha(new),
        "sources": entries,
    }
    handle, temporary = tempfile.mkstemp(dir=base, prefix=".import-acquisition-")
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(canonical_json(lock) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, destination)
        os.unlink(temporary)
        verify_acquisition(new, v3_work)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return len(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v2_recipe", type=Path)
    parser.add_argument("v2_work", type=Path)
    parser.add_argument("v3_recipe", type=Path)
    parser.add_argument("v3_work", type=Path)
    args = parser.parse_args()
    print(import_snapshots(args.v2_recipe, args.v2_work, args.v3_recipe, args.v3_work))
