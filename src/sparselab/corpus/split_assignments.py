"""Authenticated frozen document assignments with private operation-owned lookup."""

import shutil
from collections.abc import Mapping
from pathlib import Path

from sparselab.corpus.bounded_metadata import current_metadata_store
from sparselab.corpus.jsonl_records import records_from_path
from sparselab.corpus.row_index import RowIndex
from sparselab.training.manifest import sha256_file


def assignment_path(directory, reference):
    from sparselab.corpus.project import SplitAssignmentsReference

    reference = SplitAssignmentsReference.model_validate(reference).model_dump(
        mode="json"
    )
    path = Path(directory) / ("assignments-" + reference["sha256"] + ".jsonl")
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ValueError("missing or symlinked split assignments")
    if sha256_file(path) != reference["sha256"]:
        raise ValueError("split assignment digest mismatch")
    return path


class Assignments(Mapping):
    def __init__(self, path, reference):
        def rows():
            previous = None
            count = 0
            for record in records_from_path(path, max_record_bytes=8 * 1024**2):
                row = record.value
                if (
                    not isinstance(row, dict)
                    or set(row) != {"document_id", "split"}
                    or not isinstance(row["document_id"], str)
                    or not row["document_id"]
                    or row["split"] not in ("train", "validation", "test")
                    or previous is not None
                    and row["document_id"] <= previous
                ):
                    raise ValueError("invalid or unordered split assignment")
                previous = row["document_id"]
                count += 1
                yield row
            if count != reference["rows"]:
                raise ValueError("split assignment count mismatch")
            if sha256_file(path) != reference["sha256"]:
                raise ValueError("split assignments changed during lookup")

        self.index = RowIndex(
            current_metadata_store().connection, rows(), lambda row: row["document_id"]
        )

    def __getitem__(self, key):
        return self.index[key]["split"]

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return len(self.index)


def split_policy(project):
    policy = project.splits.model_dump(mode="json")
    reference = policy.get("assignment_file")
    if reference is not None:
        from sparselab.corpus.project import project_path

        directory = project_path(project.root, project.config.splits).parent
        policy["assignments"] = Assignments(
            assignment_path(directory, reference), reference
        )
    return policy


def copy_assignments(project, destination):
    reference = project.splits.model_dump(mode="json").get("assignment_file")
    if reference is None:
        return
    from sparselab.corpus.project import project_path

    source = assignment_path(
        project_path(project.root, project.config.splits).parent, reference
    )
    target = Path(destination) / source.name
    with target.open("xb") as output, source.open("rb") as original:
        shutil.copyfileobj(original, output)
    assignment_path(destination, reference)


def verify_document_assignments(root, identity, documents):
    reference = identity["split"].get("assignment_file")
    if reference is None:
        return
    assignments = Assignments(assignment_path(root, reference), reference)
    for doc in documents:
        if assignments.get(doc["document_id"]) != doc["split"]:
            raise ValueError("document differs from frozen split assignment")
