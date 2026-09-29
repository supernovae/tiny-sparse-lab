"""Behavioral checks for v4 diagnostic-only header-normalized overlap scoring."""

import hashlib
import json
import runpy
from pathlib import Path

from sparselab.corpus.near_duplicates import _shingles

_SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "research"
    / "devmind-pretrain-v4"
)


def _audit(monkeypatch):
    monkeypatch.syspath_prepend(str(_SCRIPT_DIR))
    return runpy.run_path(str(_SCRIPT_DIR / "audit_overlap.py"))


def test_shared_apache_header_is_not_shared_implementation(monkeypatch) -> None:
    without_header = _audit(monkeypatch)["without_repeated_header"]
    license_text = (
        "// Copyright 2024 The Example Authors\n"
        '// Licensed under the Apache License, Version 2.0 (the "License");\n'
        "// you may not use this file except in compliance with the License.\n"
        "// You may obtain a copy of the License at\n"
        "// http://www.apache.org/licenses/LICENSE-2.0\n"
        "// Unless required by applicable law or agreed to in writing, software\n"
        "// distributed under the License is distributed on an AS IS BASIS,\n"
        "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.\n"
        "// See the License for the specific language governing permissions and\n"
        "// limitations under the License.\n\n"
    )
    left = license_text + "package buildkite\n"
    right = license_text + "package fileutil\n"
    left_body, left_kind = without_header(left)
    right_body, right_kind = without_header(right)
    assert (left_kind, right_kind) == ("apache", "apache")
    assert (left_body, right_body) == ("package buildkite\n", "package fileutil\n")
    assert "Copyright" in left and "Copyright" in right
    raw_a, raw_b = _shingles(left, 100), _shingles(right, 100)
    body_a, body_b = _shingles(left_body, 100), _shingles(right_body, 100)
    assert len(raw_a & raw_b) / len(raw_a | raw_b) > 0.65
    assert not (body_a & body_b)


def test_diagnostic_header_normalization_preserves_unrecognized_and_shebang(
    monkeypatch,
) -> None:
    without_header = _audit(monkeypatch)["without_repeated_header"]
    script = "#!/bin/bash\n# Copyright 2024 Example\necho status\n"
    comment = (
        "// Copyright 2024 Example\n// Parse user-provided config.\npackage config\n"
    )
    assert without_header(script) == (script, None)
    assert without_header(comment) == (comment, None)
    assert without_header(
        "# SPDX-License-Identifier: MIT\n# Copyright 2024 Example\nrun()\n"
    ) == (
        "run()\n",
        "spdx_copyright",
    )
    authored = (
        "// Explains the operator failure mode.\n" * 5
        + "// Licensed under the Apache License, Version 2.0\n"
        + "// limitations under the License.\n"
        + "package operator\n"
    )
    assert without_header(authored) == (authored, None)


def test_normalized_mode_proposes_content_pair_hidden_by_different_headers(
    monkeypatch, tmp_path: Path
) -> None:
    audit = _audit(monkeypatch)
    audit["RATES"]["developer_train"] = 1
    audit["RATES"]["engineering_heldout"] = 1
    work = tmp_path / "release"
    work.mkdir()
    (work / "manifest.json").write_text('{"release_id":"fixture"}')
    body = "configure transaction timeout and verify the write succeeded\n"
    texts = [
        (
            "# SPDX-License-Identifier: MIT\n# Copyright 2024 Example\n" + body,
            "train",
        ),
        (
            "# SPDX-License-Identifier: Apache-2.0\n# Copyright 2025 Different\n"
            + body,
            "validation",
        ),
    ]
    rows = [
        {
            "document_id": hashlib.sha256(f"{split}:{index}".encode()).hexdigest(),
            "source_id": f"v4_fixture_{index}",
            "source_location": f"source/{index}.txt",
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "split": split,
            "drop_reason": None,
            "text": text,
        }
        for index, (text, split) in enumerate(texts)
    ]
    (work / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    report = audit["compare_modes"](work)
    assert report["release_id"] == "fixture"
    assert report["raw_candidates"] == 0
    assert report["normalized_candidates"] == 1
    assert report["coverage"]["removed_header"]["spdx_copyright:documents"] == 2
    assert len(report["candidate_union"]) == 1
    pair = report["candidate_union"][0]
    assert pair["boilerplate_normalized_similarity"] == 1.0
    assert pair["raw_similarity"] < 0.65
    assert pair["proposed_by"] == ["boilerplate_normalized"]
    assert [json.loads(line)["text"] for line in (work / "documents.jsonl").open()] == [
        text for text, _ in texts
    ]
