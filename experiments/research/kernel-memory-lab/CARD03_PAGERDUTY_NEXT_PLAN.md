# Card 03 next domain-source gate — pinned PagerDuty documents

Status: **PROPOSAL ONLY; NO CONTENT ACQUISITION OR FILE ADMISSION APPROVED**. The completed [Common Pile pilot](results/2026-10-07-card03-quarantined-pilot.md) cannot supply the incident-response evidence domain, and its [record screen](CARD03_PILOT_ROW_SCREEN.md) admits no language rows. The metadata-only [file inventory](CARD03_PAGERDUTY_FILE_INVENTORY.md) pins the smallest complete PagerDuty text selection: 36 Markdown files plus `LICENSE` and `README.md` at repository commit `464fc9d3e47e19e9d8da17cec1a41dc09624e95a` and tree `0a6d6a7e05784260ddd649513af6d42f003c5927`. No PDF, image or `docs/assets/**` path is selected.

First make a **focused native acquisition-path extension**, separately reviewable and testable without remote content. The current Git adapter calls `git fetch` before a selected-blob byte cap can run, so it cannot satisfy a strict transport allowance. Add an optional explicit pinned-Git-blob HTTP mode under existing `sparselab corpus acquire`/project declarations, reusing its immutable snapshots and durable response-body ledger. Bind each declared path, Git blob SHA-1, raw byte length, commit/tree ID and rights metadata; require an exact `Content-Length` and authenticated HTTPS at the pinned raw path, hash the Git `blob <size>\0` object as well as SHA-256 of retained bytes, and reject an unexpected path, size, Git object, missing length or changed selection before admitting output. Stream only the declared body size with no one-byte overread. Count metadata, redirects, errors, body bytes, interrupted attempts, retries and resumes in one persistent ledger created **before** the acquisition's own metadata requests. Enforce byte and deadline caps before reservation and during reads; preserve existing declaration/receipt digests when the optional mode is absent. Mock exact files, redirects/errors, truncated/extra bodies, checksum failures, interrupted transfers and exhausted/resumed ledgers. Keep this code fix separate from the subsequent content attempt.

Only after that native path is tested and the owner separately approves acquisition, propose **one rights-quarantined content attempt** with these exact limits:

| Limit | Ceiling |
| --- | ---: |
| Declared raw source bytes, first complete pass | 317,526 (302,785 Markdown + 11,391 LICENSE + 3,350 README) |
| Source response bodies including at most one retry per file | **635,052 bytes** |
| Metadata, redirect and error response bodies | **131,072 bytes** |
| Combined response bodies | **766,124 bytes** |
| Transfers | **76** maximum (38 files, at most one retry each) |
| Wall time from one ledger creation | **600 seconds** |
| Added disk and inodes | **128 MiB**, **250 inodes** |
| Retained decoded Markdown/rights context | **1 MiB**, only the 38 selected files |
| Optimizer updates, GPU, cloud use and spend | **zero** |

The metadata-only tree observed 2026-10-07 used 77,724 bytes, leaving 53,348 bytes in the proposed metadata cap for a new attempt's required metadata/error/redirect bodies. Recheck file sizes and Git identities under that new ledger; stop if they differ or metadata cannot fit. Require `Accept-Encoding: identity`; if the server cannot provide declared raw `Content-Length`, stop rather than switching to an unbounded Git fetch. The source and metadata caps are separate; neither is a training-token budget. The sample's physical staging high-water, retained output and free-space/inode margin must be checked at actual paths before launch and monitored through publication.

After an authorized, verified snapshot, review **each of the 36 Markdown files** for third-party text, notices, attribution, personal data and secrets, then bind file families/versions and mark admitted, excluded or review-required with a reviewer and rationale. Keep uncertain files quarantined. Only rights-admitted language/evidence material can feed the main tokenizer, family-separated release and frozen 200/400-item suites; those later steps need their own measured preparation and evaluation bounds. This plan does not change the 341,885,952-parameter Card 04 architecture or authorize Card 05 training.
