# LAB86 private preparation policy v1

Authority provenance: owner approvals recorded on canonical LAB86,
https://app.notion.com/p/3fb3b68cca2281a5a68fc549ca0c62d6 (2026-10-17 04:09 UTC;
configuration assembly reaffirmed 04:55 UTC). This document records that scope;
it does not grant authority, release a gate, accept terms, or clear LAB39/LAB70.

Use only the pinned candidate shards in shards.json. The first three are the
approved feasibility pilot. Remaining compressed byte allocations are operational
bounds, not token ratios or a claim that the pilot represents the entire corpus.
Stop at the 4 GiB cumulative body gate and retain failed acquisition receipts.
Continue only on the native report-bound reviewed decision using the same ledger.
All metadata and retry response bodies count toward 100 GiB. No automatic retry,
budget reset, reweighting, source substitution or repeated target exposure.

Keep everything private and quarantined pending actual admission review. SmolLM's
ODC-By card label does not license every underlying FineWeb page. Retain URL,
dump/date, source ID, original row/shard hashes and notices; inspect source rights
and exclusions. Cosmopedia uses generated text only, not prompt extracts; preserve
seed_data, prompt, audience and format provenance. Group common seed families
before splitting. A dataset-level seed label is a coarse family, not permission
to invent document-level seeds. Reject absent provenance.

Code needs repository/path/license attribution. Initial allowed labels are MIT,
Apache-2.0, BSD-2-Clause, BSD-3-Clause and ISC. Inspect file notices; exclude
conflicts, absent licenses, vendored/generated content and secrets. Allowed
languages: Python, JavaScript, TypeScript, Go, Rust, Shell, SQL and Markdown.
The loader's Apache-2.0 notice is not a blanket license for repository content.
No external loader execution or unreviewed language derivation is authorized here.

Use exact/canonical dedup and reviewed cross-source near-duplicate clusters while
preserving code indentation. Group repositories, seed families and duplicates
before a frozen 98/1/1 train/validation/test split. Report benchmark contamination
coverage and gaps; absence of checks is not decontamination. Fewer than 100
eligible families per stratum blocks the native split; do not weaken grouping.

Fit one fresh 32768 byte-level BPE, reserved native special tokens, from TRAIN
only: <=512 MiB, 60/20/20 bytes web/synthetic/code. Preserve whitespace, case and
byte coverage. Audit >=1000 unseen TRAIN documents per stratum where available;
report shortages, roundtrip/UNK, bytes per token, identifiers, numbers,
multilingual text and code whitespace. Never borrow validation/test for fitting.
Measure unique usable TRAIN targets after admission/dedup/splitting/tokenization.
7.2B/2.4B/2.4B are conditional floors, not available-supply claims. Shuffle native
mixtures deterministically without replacement (max_exposures=1). Pack native
int32 tokens at sequence length 2048. No model steps or inference are authorized.

48 hours includes operator waits and cleanup. At most eight CPU threads; native
library thread settings are one. 16 GiB RSS is sampled observation with stop,
not a hard cap. Require measured RAM and disk headroom. 300 GiB additional disk
includes retained inputs/intermediates/caches/receipts. Parquet expanded byte
measurement is unsupported; do not claim it is bounded by compressed bytes.
A 5M-row prefix and 8 GiB emitted JSONL ceiling per shard are conservative maximums,
not observed counts. Record any cutoff/shortfall; do not claim whole-shard coverage.
No package installation, model/corpus download, registry action or runner job was
performed to assemble this package.
