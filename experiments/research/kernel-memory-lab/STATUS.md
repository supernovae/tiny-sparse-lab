# Current project status

Record revision: 2026-10-07.8
Snapshot date: 2026-10-07
Source audit revision: 06efc4db82ecf3da97b50cff518cba605ad27b33
Actual WSL checkout revision at Card 01 start: cf29aff79ec7259f7ec93988cd6a87065f4350e5 (`main`, clean before edits)
Repository documentation path: `experiments/research/kernel-memory-lab` (present in local `main`)
Current selected card: NONE; Card 02 EVIDENCE VERIFIED for tiny-fixture scope with original A1 FAILED
Next proposed action: scope Card 03's data/tokenizer/evaluation specification after a domain and source/license decision
Runtime authority: [KML-D02](DECISIONS.md#kml-d02--bounded-card-02-confirmation-approved-2026-10-07) exercised and exhausted (89/89); no further training authority
Last reviewed project result: [KML-20261007-C02-A2](results/2026-10-07-card02-confirmation.md), accepted for tiny-fixture scope by [KML-D03](DECISIONS.md#kml-d03--card-02-accepted-with-failed-attempt-note-2026-10-07)
Latest research decision: [KML-D03](DECISIONS.md#kml-d03--card-02-accepted-with-failed-attempt-note-2026-10-07)
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

Future expert/sensemaking track: PLANNED ONLY; see `EXPERT_TRACK.md`. No additional runtime card is selected. Source candidates remain unapproved for acquisition or ingestion.

## Single current checklist

| Card | Deliverable | Status | Evidence or blocker |
| --- | --- | --- | --- |
| 01 | Foundation protocol and minimal missing scaffolding | EVIDENCE VERIFIED (documentation scope) | Owner accepted [foundation contract](FOUNDATION.md) in [KML-D01](DECISIONS.md#kml-d01--foundation-contract-accepted-2026-10-07); Gate 0 remains unverified |
| 02 | Fresh tiny correctness fixtures using native controls | EVIDENCE VERIFIED (done with notes) | [KML-D03](DECISIONS.md#kml-d03--card-02-accepted-with-failed-attempt-note-2026-10-07) accepted [A2](results/2026-10-07-card02-confirmation.md); [A1](results/2026-10-07-card02-tiny-fixtures.md) remains FAILED for its 57-update cap breach |
| 03 | Data, tokenizer and frozen evaluation contracts | NOT STARTED | Card 01 and source/license decisions |
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
