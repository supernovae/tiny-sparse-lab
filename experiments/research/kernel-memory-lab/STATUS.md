# Current project status

Record revision: 2026-10-07.11
Snapshot date: 2026-10-07
Source audit revision: 06efc4db82ecf3da97b50cff518cba605ad27b33
Actual WSL checkout revision at Card 01 start: cf29aff79ec7259f7ec93988cd6a87065f4350e5 (`main`, clean before edits)
Repository documentation path: `experiments/research/kernel-memory-lab` (present in local `main`)
Current selected card: 03 — complete-shard pilot proposal READY FOR REVIEW; full acceptance blocked on source rights, transport fix and acquisition approval
Next proposed action: owner review of the exact Card 03 pilot proposal and limits before any source-content transfer
Working branch: `codex/kernel-memory-lab` (rolling local branch; remote push deferred by owner)
Runtime authority: [KML-D02](DECISIONS.md#kml-d02--bounded-card-02-confirmation-approved-2026-10-07) exercised and exhausted (89/89); no source acquisition or further training authority
Last reviewed project result: [KML-20261007-C02-A2](results/2026-10-07-card02-confirmation.md), accepted for tiny-fixture scope by [KML-D03](DECISIONS.md#kml-d03--card-02-accepted-with-failed-attempt-note-2026-10-07)
Latest project result: [KML-20261007-C03-S3](results/2026-10-07-card03-pilot-proposal.md), bounded complete-shard proposal READY FOR REVIEW; full gate blocked
Latest research decision: [KML-D05](DECISIONS.md#kml-d05--retain-two-common-pile-candidates-for-pilot-planning-2026-10-07), both Common Pile components retained for planning only
Latest project attempt: [KML-20261007-C02-A2](results/2026-10-07-card02-confirmation.md), reviewed in KML-D03; [A1](results/2026-10-07-card02-tiny-fixtures.md) remains FAILED
Budget code fix: [KML-20261007-C02-P1](results/2026-10-07-card02-budget-repair.md), committed in `77ec6e7` and confirmed by A2

## Delivery facts

The source-release research plan and workbook were synchronized at revision 2026-10-07.1. This repository edition contains documented adoption/privacy edits and a separate future expert track; the Library DOCX snapshots do not include that new track. Repository infrastructure was inspected at the source revision above; see `SOURCE_AUDIT.md`. No project-specific fixture, main config, dataset/tokenizer, trained checkpoint, reader, router, insertion result or offload result was produced by this document task.

The local read-only reconciliation found no Kernel Memory Lab Campaign, run or
artifact in the configured work root `/srv/sparselab/state/experiments/` or
in the inspected legacy/default roots. The documentation-only Card 01 result
was pending review at that point; it did not establish Gate 0.

The owner subsequently accepted the Card 01 documentation contract. Card 02
created fresh CPU fixtures and passed functional checks, but repeated test runs
used 257 cumulative optimizer updates against a 200-step cap. Its retained
result records the deviation and failed gate. Gate 0 remains unverified.

A proposed persistent budget ledger and Card 02 clarification now charge every
selected test, retry and resumed run within one attempt, and stop at the shared
deadline. The [89-update, 120-second confirmation](CARD02_CONFIRMATION.md)
was approved for one new CPU attempt by KML-D02. [A2](results/2026-10-07-card02-confirmation.md)
used exactly 89 charged updates and completed the selected tests and native
checks within about 36.4 seconds. The owner accepted A2 for Card 02's tiny
fixture scope in [KML-D03](DECISIONS.md#kml-d03--card-02-accepted-with-failed-attempt-note-2026-10-07).
The original A1 attempt remains FAILED; its missing aggregate budget accounting
was repaired before A2. Gate 0 remains unverified because its provisional
data/tokenizer decision and hardware inventory are still missing.

Card 03 now has a [reviewable specification](CARD03_DATA_CONTRACT.md) for an
incident-response evidence domain and rights-audited Common Pile language base.
[KML-D04](DECISIONS.md#kml-d04--card-03-source-candidates-selected-for-specification-2026-10-07)
records the owner's candidate choice, not source admission. The proposed mixture,
train-only tokenizer, family splits, overlap checks and 200/400-item rubrics have
no realized artifacts or scores. Exact source rights/revisions and a bounded
preparation approval are missing; [S1](results/2026-10-07-card03-specification.md)
is pending review. Gate 0 still lacks a provisional data/tokenizer decision and
hardware inventory.

[S2](results/2026-10-07-card03-metadata-audit.md) pins candidate PagerDuty and
Common Pile component revisions from remote metadata. It found that the smallest
candidate Common Pile shards exceed S1's 64 MiB download proposal, while the
native Git path fetches before checking selected-file bytes. The [metadata
audit](CARD03_SOURCE_METADATA_AUDIT.md) supersedes that pilot request without
changing S1 or admitting any source. No content, tokenizer or evaluation items
were acquired. A file-level rights and bounded-transport plan is still needed.

The owner retained both Common Pile components as candidates in
[KML-D05](DECISIONS.md#kml-d05--retain-two-common-pile-candidates-for-pilot-planning-2026-10-07)
without approving acquisition. [S3](results/2026-10-07-card03-pilot-proposal.md)
links the [revised proposal](CARD03_PILOT_PROPOSAL.md): one complete smallest
shard from each component, an attempt-wide byte allocation including one retry,
rights quarantine and a focused native transport prerequisite. PagerDuty remains
the domain candidate outside this pilot. No source bytes or training tokens were
used. The 341,885,952-parameter architecture is unchanged; Gate 0 is unverified.

Future expert/sensemaking track: PLANNED ONLY; see `EXPERT_TRACK.md`. No additional runtime card is selected. Source candidates remain unapproved for acquisition or ingestion.

## Single current checklist

| Card | Deliverable | Status | Evidence or blocker |
| --- | --- | --- | --- |
| 01 | Foundation protocol and minimal missing scaffolding | EVIDENCE VERIFIED (documentation scope) | Owner accepted [foundation contract](FOUNDATION.md) in [KML-D01](DECISIONS.md#kml-d01--foundation-contract-accepted-2026-10-07); Gate 0 remains unverified |
| 02 | Fresh tiny correctness fixtures using native controls | EVIDENCE VERIFIED (done with notes) | [KML-D03](DECISIONS.md#kml-d03--card-02-accepted-with-failed-attempt-note-2026-10-07) accepted [A2](results/2026-10-07-card02-confirmation.md); [A1](results/2026-10-07-card02-tiny-fixtures.md) remains FAILED for its 57-update cap breach |
| 03 | Data, tokenizer and frozen evaluation contracts | READY FOR REVIEW (pilot proposal); full gate BLOCKED | [Contract](CARD03_DATA_CONTRACT.md), [revised pilot](CARD03_PILOT_PROPOSAL.md) and [S3](results/2026-10-07-card03-pilot-proposal.md); acquisition approval, native transport fix, item rights and realized artifacts missing |
| 04 | Main shape and measured fit | NOT STARTED | Fixtures, inputs and exact profile approval |
| 05 | One bounded language training tranche | NOT STARTED | Data, fit receipts and run approval |
| 06 | Raw-text oracle evidence baseline | NOT STARTED | Qualified fresh checkpoint and frozen suite |
| 07 | Lexical retrieval baseline | NOT STARTED | Card 06 and approved versioned store |
| 08 | Explicit dense-to-new-reader transfer | NOT STARTED | Tiny contracts and selected Card 06 origin |
| 09 | One integrated reader | NOT STARTED | Cards 06, 07, 08 and bounded approval |
| 10 | Optional learned router or index experiment | NOT STARTED | Measured bottleneck; initially skip recommended |
| 11 | Frozen-weight insertion and correction | NOT STARTED | Useful reader and approved new records |
| 12 | Same-store RAM/NVMe comparison | NOT STARTED | Useful reader, fixed store and approved caps |
| 13 | Review and next decision | NOT STARTED | Actual relevant receipts |

These are prerequisite notes, not fabricated failed runs. Use BLOCKED when a real missing input prevents the selected task. After adoption, change this snapshot only from actual results, receipts and review. Append the result first and record the reviewer; a status change never grants approval.
