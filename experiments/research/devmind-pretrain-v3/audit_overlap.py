"""Bounded, coverage-labelled lexical candidate review for a frozen v3 release.

Include all non-Wikibooks heldout documents, sample the large Wikibooks and
train pools by immutable document ID, then run the existing shingle/MinHash
candidate scorer. Prefix-only screening and sampling cannot prove isolation.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from measure_pool import SOURCE_FAMILIES

from sparselab.corpus.near_duplicates import candidates

RATES = {
    "developer_train": 20,
    "other_train": 250,
    "wikibooks_heldout": 10,
    "engineering_heldout": 1,
}
MAX_DOCUMENTS = 100_000
MAX_TEXT_BYTES = 128 * 1024 * 1024
PREFIX_CHARS = 16_384
MAX_SHINGLES = 12_000_000


def review(release: Path) -> dict:
    # The frozen release is verified separately by corpus freeze/publication.
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    population = Counter()
    selected = Counter()

    def source() -> object:
        with (release / "documents.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["drop_reason"] is not None:
                    continue
                if row["split"] == "train":
                    stratum = (
                        "developer_train"
                        if (
                            row["source_id"].startswith("v3_")
                            or row["source_id"] in SOURCE_FAMILIES
                        )
                        else "other_train"
                    )
                elif row["source_id"] == "wikibooks_20260901":
                    stratum = "wikibooks_heldout"
                else:
                    stratum = "engineering_heldout"
                population[stratum + ":documents"] += 1
                population[stratum + ":utf8_bytes"] += len(row["text"].encode("utf-8"))
                modulus = RATES[stratum]
                digest = hashlib.sha256(row["document_id"].encode()).digest()
                if int.from_bytes(digest[:8], "big") % modulus:
                    continue
                selected[stratum + ":documents"] += 1
                prefix = row["text"][:PREFIX_CHARS]
                selected[stratum + ":prefix_utf8_bytes"] += len(prefix.encode("utf-8"))
                # No publication of source text in the candidate report.
                yield {
                    k: row[k]
                    for k in (
                        "document_id",
                        "split",
                        "source_id",
                        "source_location",
                        "content_sha256",
                    )
                } | {"text": prefix}

    result = candidates(
        source(),
        max_documents=MAX_DOCUMENTS,
        max_input_text_bytes=MAX_TEXT_BYTES,
        max_total_shingles=MAX_SHINGLES,
        max_shingles_per_document=PREFIX_CHARS,
        max_bucket_documents=1000,
        max_comparisons=200_000,
        max_candidates=20_000,
    )
    return {
        "release_id": manifest["release_id"],
        "sampling_moduli": RATES,
        "prefix_characters": PREFIX_CHARS,
        "population": dict(sorted(population.items())),
        "sampled": dict(sorted(selected.items())),
        "candidate_audit": result,
        "limitations": "Heldout families except Wikibooks fully sampled, but only first 16384 characters per document. Other strata sampled by SHA-256/document ID. LSH can miss lexical overlaps; middle/end passages, unsampled mirrors and paraphrases remain unproved.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(review(args.release), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
