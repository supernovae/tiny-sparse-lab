"""Emit gold-blinded Card 03 review questions from authored draft items."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sparselab.evaluation.kml_card03_items import _chunks, _families, _rows
from sparselab.training.manifest import canonical_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--new-chunks", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    authored = json.loads(args.items.read_text(encoding="utf-8"))
    if isinstance(authored, dict):
        authored = authored["items"]
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    added = json.loads(args.new_chunks.read_text(encoding="utf-8"))
    if isinstance(added, dict):
        added = added["chunks"]
    docs = {
        row["document_id"]: row
        for row in _rows(args.release / "documents.jsonl")
        if row["drop_reason"] is None
    }
    families = _families(args.families, docs)
    chunk_bank = {chunk["id"]: chunk for chunk in candidate["chunks"]}
    for chunk in added:
        if chunk["id"] in chunk_bank and chunk_bank[chunk["id"]] != chunk:
            raise ValueError("authored chunk conflicts with candidate identity")
        chunk_bank[chunk["id"]] = chunk
    chunks = _chunks(list(chunk_bank.values()), docs, families)
    if len(authored) != 200 or len({item["id"] for item in authored}) != 200:
        raise ValueError("blind author partition needs 200 distinct items")
    rows: list[dict] = []
    for item in authored:
        if item["review_status"] != "draft":
            raise ValueError("blind worksheet requires unreviewed author draft")
        visible = []
        for index, chunk_id in enumerate(item["support_chunk_ids"], 1):
            chunk = chunks[chunk_id]
            doc = docs[chunk["document_id"]]
            visible.append(
                {
                    "ordinal": index,
                    "chunk_id": chunk_id,
                    "source_document_id": chunk["document_id"],
                    "text": doc["text"][chunk["start"] : chunk["end"]],
                }
            )
        rows.append(
            {
                "id": item["id"],
                "author_item_sha256": item["content_sha256"],
                "suite": item["suite"],
                "question": item["question"],
                "visible_gold_context": visible,
                "parent_document_ids": item["parent_document_ids"],
            }
        )
    with args.output.open("xb") as stream:
        for row in rows:
            stream.write(canonical_json(row) + b"\n")
    manifest = {
        "schema_version": 1,
        "status": "BLIND_REVIEW_WORKSHEET_NOT_ITEM_APPROVAL",
        "items_sha256": sha256_file(args.items),
        "candidate_sha256": sha256_file(args.candidate),
        "new_chunks_sha256": sha256_file(args.new_chunks),
        "family_inventory_sha256": sha256_file(args.families),
        "worksheet_sha256": sha256_file(args.output),
        "item_count": len(rows),
        "ids_sha256": hashlib.sha256(canonical_json([row["id"] for row in rows])).hexdigest(),
    }
    with args.output.with_suffix(args.output.suffix + ".manifest.json").open("xb") as stream:
        stream.write(canonical_json(manifest) + b"\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
