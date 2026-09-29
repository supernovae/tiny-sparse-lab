"""Review cross-split lexical candidates in both original and diagnostic-only header-normalized text.

The frozen source/normalized document bytes are never changed. Deterministic
sampling, full-text-size exclusions and exact caps are reported, not mistaken
for an exhaustive semantic independence proof.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from measure_pool import DEV_V2

from sparselab.corpus.near_duplicates import _shingles, candidates

RATES = {
    "developer_train": 30,
    "other_train": 350,
    "wikibooks_heldout": 12,
    "engineering_heldout": 1,
}
MAX_DOCUMENTS = 100_000
MAX_DOCUMENT_CHARS = 48_000
MAX_TEXT_BYTES = 256 * 1024 * 1024
MAX_SHINGLES = 18_000_000
MAX_REVIEW_PAIRS = 20_000
MAX_REVIEW_TEXT_BYTES = 256 * 1024 * 1024
_APACHE_START = re.compile(
    r"licensed under the apache license|apache[- ]2\.0", re.IGNORECASE
)
_APACHE_END = re.compile(r"limitations under the license", re.IGNORECASE)
_SPDX = re.compile(r"SPDX-License-Identifier\s*:", re.IGNORECASE)
_COPYRIGHT = re.compile(r"copyright|\(c\)\s*\d{4}", re.IGNORECASE)
_COMMENT = re.compile(r"^\s*(?://|#|/\*|\*|;|--|<!--|\*/|-->)")


def without_repeated_header(text: str) -> tuple[str, str | None]:
    """Remove only a recognizable leading Apache block or SPDX/copyright lines.

    This is for similarity computation, not a source normalization or release
    transform. Ordinary comments, code, unknown licenses and embedded notices
    remain untouched. No header beyond 64 physical LF-delimited lines is removed.
    """
    if text.startswith("#!"):
        return text, None
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    if not lines:
        return text, None
    header = []
    for index, line in enumerate(lines[:64]):
        if not line.strip() or _COMMENT.match(line):
            header.append((index, line))
            continue
        break
    if not header or not _COMMENT.match(header[0][1]):
        return text, None
    apache_start = next(
        (index for index, line in header if _APACHE_START.search(line)), None
    )
    apache_end = next(
        (index for index, line in header if _APACHE_END.search(line)), None
    )
    if (
        apache_start is not None
        and apache_start < 5
        and apache_end is not None
        and apache_end > apache_start
    ):
        end = apache_end + 1
        while end < len(header) and (
            not header[end][1].strip() or header[end][1].strip() in {"*/", "-->"}
        ):
            end += 1
        return "".join(lines[end:]), "apache"
    end = 0
    seen_spdx = False
    while end < len(header):
        line = header[end][1]
        if _SPDX.search(line):
            seen_spdx = True
        elif _COPYRIGHT.search(line) or not line.strip():
            pass
        else:
            break
        end += 1
    if seen_spdx:
        return "".join(lines[end:]), "spdx_copyright"
    return text, None


def _stratum(row: dict) -> str:
    if row["split"] == "train":
        return (
            "developer_train"
            if row["source_id"].startswith(("v3_", "v4_")) or row["source_id"] in DEV_V2
            else "other_train"
        )
    return (
        "wikibooks_heldout"
        if row["source_id"] == "wikibooks_20260901"
        else "engineering_heldout"
    )


def _selected(
    release: Path,
    counts: dict | None,
    normalized: bool,
    header_counts: Counter | None = None,
):
    with (release / "documents.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["drop_reason"] is not None:
                continue
            stratum = _stratum(row)
            if counts is not None:
                counts["population"][stratum + ":documents"] += 1
                counts["population"][stratum + ":utf8_bytes"] += len(
                    row["text"].encode("utf-8")
                )
            digest = hashlib.sha256(row["document_id"].encode()).digest()
            if int.from_bytes(digest[:8], "big") % RATES[stratum]:
                continue
            if len(row["text"]) > MAX_DOCUMENT_CHARS:
                if counts is not None:
                    counts["excluded_oversize"][stratum + ":documents"] += 1
                    counts["excluded_oversize"][stratum + ":utf8_bytes"] += len(
                        row["text"].encode("utf-8")
                    )
                continue
            if counts is not None:
                counts["sampled"][stratum + ":documents"] += 1
                counts["sampled"][stratum + ":utf8_bytes"] += len(
                    row["text"].encode("utf-8")
                )
            text, header = (
                without_repeated_header(row["text"])
                if normalized
                else (row["text"], None)
            )
            if header_counts is not None and header:
                header_counts[header + ":documents"] += 1
            if normalized and not text.strip():
                continue
            yield {
                k: row[k]
                for k in (
                    "document_id",
                    "split",
                    "source_id",
                    "source_location",
                    "content_sha256",
                )
            } | {
                "text": text,
                "content_sha256": (
                    hashlib.sha256(text.encode()).hexdigest()
                    if normalized
                    else row["content_sha256"]
                ),
            }


def _score(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compare_modes(release: Path) -> dict:
    counts = {
        "population": Counter(),
        "sampled": Counter(),
        "excluded_oversize": Counter(),
        "removed_header": Counter(),
    }
    kwargs = {
        "max_documents": MAX_DOCUMENTS,
        "max_input_text_bytes": MAX_TEXT_BYTES,
        "max_total_shingles": MAX_SHINGLES,
        "max_shingles_per_document": MAX_DOCUMENT_CHARS,
        "max_bucket_documents": 1000,
        "max_comparisons": 200_000,
        "max_candidates": MAX_REVIEW_PAIRS,
    }
    raw = candidates(_selected(release, counts, False), **kwargs)
    normalized = candidates(
        _selected(release, None, True, counts["removed_header"]), **kwargs
    )
    pair_modes: dict[tuple[str, str], set[str]] = {}
    metadata: dict[str, dict] = {}
    for name, audit in (("raw", raw), ("boilerplate_normalized", normalized)):
        for pair in audit["candidates"]:
            ids = (pair["left_document_id"], pair["right_document_id"])
            pair_modes.setdefault(tuple(sorted(ids)), set()).add(name)
            for side in ("left", "right"):
                identity = pair[side + "_document_id"]
                metadata[identity] = {
                    "source_id": pair[side + "_source_id"],
                    "location": pair[side + "_source_location"],
                    "split": pair[side + "_split"],
                }
    if len(pair_modes) > MAX_REVIEW_PAIRS:
        raise ValueError("cross-mode candidate pair cap exceeded")
    bodies = {}
    bytes_loaded = 0
    needed = set(metadata)
    with (release / "documents.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = row["document_id"]
            if identity not in needed:
                continue
            text = row["text"]
            metadata[identity]["raw_content_sha256"] = row["content_sha256"]
            bytes_loaded += len(text.encode("utf-8"))
            if bytes_loaded > MAX_REVIEW_TEXT_BYTES:
                raise ValueError("candidate review text byte cap exceeded")
            bodies[identity] = text
            needed.remove(identity)
            if not needed:
                break
    if needed:
        raise ValueError("candidate review document missing from frozen release")
    shingles = {}
    inspected = {}
    for identity, text in bodies.items():
        normalized_text, kind = without_repeated_header(text)
        shingles[identity] = (
            _shingles(text, MAX_DOCUMENT_CHARS),
            _shingles(normalized_text, MAX_DOCUMENT_CHARS)
            if normalized_text.strip()
            else frozenset(),
        )
        inspected[identity] = kind
    pairs = []
    for (left, right), modes in sorted(pair_modes.items()):
        original = _score(shingles[left][0], shingles[right][0])
        body = _score(shingles[left][1], shingles[right][1])
        pairs.append(
            {
                "left_document_id": left,
                "right_document_id": right,
                "left": metadata[left],
                "right": metadata[right],
                "raw_similarity": original,
                "boilerplate_normalized_similarity": body,
                "left_recognized_header": inspected[left],
                "right_recognized_header": inspected[right],
                "proposed_by": sorted(modes),
            }
        )
    return {
        "release_id": json.loads((release / "manifest.json").read_text())["release_id"],
        "sampling_moduli": RATES,
        "max_complete_document_characters": MAX_DOCUMENT_CHARS,
        "coverage": {k: dict(sorted(v.items())) for k, v in counts.items()},
        "raw_cross_split_comparisons": raw["cross_split_comparisons"],
        "normalized_cross_split_comparisons": normalized["cross_split_comparisons"],
        "raw_candidates": len(raw["candidates"]),
        "normalized_candidates": len(normalized["candidates"]),
        "raw_screened_documents": raw["document_count"],
        "normalized_screened_documents": normalized["document_count"],
        "candidate_union": pairs,
        "limitations": "Only deterministic sampled documents no longer than 48000 characters are screened in full. Both MinHash/LSH modes can miss lexical similarities and cannot establish semantic independence, paraphrase absence or coverage of excluded large files. Recognized headers are removed only in memory for diagnostic similarity; original source and release remain unchanged.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(compare_modes(args.release), indent=2, sort_keys=True) + "\n"
    )
