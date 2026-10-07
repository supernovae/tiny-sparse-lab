# Current project status

Record revision: 2026-10-07.21
Snapshot date: 2026-10-07
Source audit revision: 06efc4db82ecf3da97b50cff518cba605ad27b33
Actual WSL checkout revision at Card 01 start: cf29aff79ec7259f7ec93988cd6a87065f4350e5 (`main`, clean before edits)
Repository documentation path: `experiments/research/kernel-memory-lab` (present in local `main`)
Current selected work: Card 03 quarantined pilot and Card 04 CPU synthetic tokenizer fit completed; Card 04 GPU attempts [P3](results/2026-10-07-card04-gpu-profile-interrupted.md) and [P4](results/2026-10-07-card04-gpu-retry-preflight-failed.md) failed on unavailable `amd-smi` CLI telemetry. [P5](results/2026-10-07-card04-tracked-monitor-tools.md) tracks the replacement monitor tools for review. Both full-card gates remain BLOCKED.
Next proposed actions: review the [new Card 04 monitor-repair allocation](CARD04_RETRY2_PROPOSAL.md); neither KML-D10 nor KML-D11 grants another GPU attempt. For Card 03, separately decide whether to approve the [bounded PagerDuty rights-quarantined content attempt](CARD03_PAGERDUTY_NEXT_PLAN.md) using the tested native pinned-Git mode.
Working branch: `codex/kernel-memory-lab` (rolling branch; P4's fixed GPU source checkout was `04f6e478d6b50b5396e64f68bd633ad5acb24aad`)
Runtime authority: Card 02 KML-D02 exhausted (89/89); Card 03 KML-D08 pilot and Card 04 KML-D09 CPU fit exercised; KML-D10 and KML-D11 each reserved 7/120 updates in separate failed GPU attempts, with no completed stage or profile and no retry or train allocation
Last reviewed project result: [KML-20261007-C04-R1](results/2026-10-07-card04-preparation-acceptance.md), P1 preparation accepted by [KML-D07](DECISIONS.md#kml-d07--card-04-preparation-accepted-for-reuse-2026-10-07)
Latest project results: [C03-P1](results/2026-10-07-card03-quarantined-pilot.md) verifies quarantined acquisition; [C04-P2](results/2026-10-07-card04-synthetic-tokenizer.md) verifies synthetic tokenizer; [C04-P3](results/2026-10-07-card04-gpu-profile-interrupted.md) and [C04-P4](results/2026-10-07-card04-gpu-retry-preflight-failed.md) preserve failed GPU attempts; [C04-P5](results/2026-10-07-card04-tracked-monitor-tools.md) provides reviewable monitor tools
Latest research decision: [KML-D11](DECISIONS.md#kml-d11--one-card-04-replacement-gpu-attempt-approved-2026-10-07) approved the one failed replacement attempt; KML-D06/D07 accepted earlier Card 03/04 steps
Latest project attempt: [KML-20261007-C02-A2](results/2026-10-07-card02-confirmation.md), reviewed in KML-D03; [A1](results/2026-10-07-card02-tiny-fixtures.md) remains FAILED
Budget code fix: [KML-20261007-C02-P1](results/2026-10-07-card02-budget-repair.md), committed in `77ec6e7` and confirmed by A2

## Delivery facts

Sample-evidence correction [C04-S3](results/2026-10-07-card04-sample-log-review.md)
is READY FOR REVIEW in a follow-up to merged PR #51. It makes failed sample-log
writes stop supervision and updates Retry2's launcher hash. Runtime approval
and local idle/no-training verification remain separate pending gates.

Tracked monitor review [C04-S2](results/2026-10-07-card04-tracked-monitor-review.md)
supersedes S1's missing-source blocker: base `75f9a8d` supplies the scripts.
Focused timeout, watchdog, cleanup and launch-input corrections are READY FOR
REVIEW in draft PR #51. Retry2 references the corrected tracked tool hashes;
local no-training validation and a separate runtime decision remain pending.

Cloud code review [C04-S1](results/2026-10-07-card04-cloud-safety-review.md)
is READY FOR REVIEW on a separate branch. It fixes budget-runner exception
cleanup using mocked tests; it does not validate the external AMD SMI wrapper,
change historical attempt accounting, or approve Retry2. Measured fit remains
BLOCKED.

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
was repaired before A2. Gate 0 remains unverified. Later Card 03 and Card 04
reviews supply candidate data/tokenizer specifications and read-only hardware
inventory; Gate 0 has not had its own criterion-by-criterion promotion review.

Card 03 now has a [reviewable specification](CARD03_DATA_CONTRACT.md) for an
incident-response evidence domain and rights-audited Common Pile language base.
[KML-D04](DECISIONS.md#kml-d04--card-03-source-candidates-selected-for-specification-2026-10-07)
records the owner's candidate choice, not source admission. The proposed mixture,
train-only tokenizer, family splits, overlap checks and 200/400-item rubrics have
no realized artifacts or scores. Exact source rights and a bounded
preparation approval are missing; [S1](results/2026-10-07-card03-specification.md)
was pending review when recorded and is now accepted for specification scope in
KML-D06. The accepted candidate specification is not an admission of main
training data or tokenizer artifacts.

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

[S4](results/2026-10-07-card03-transport-fix.md) records the focused native
bounded-Hub transport fix in local code commits `23db56c` and `c77074e`.
Local/mock tests and the broader corpus suite passed; no real candidate
metadata or shard was requested by this code task. The code prerequisite is
accepted for reuse by [KML-D06](DECISIONS.md#kml-d06--card-03-specification-and-transport-work-accepted-for-reuse-2026-10-07),
but the pilot still needs source-level rights evidence, explicit acquisition
approval and a fresh storage/inode preflight. Card 03
and Gate 0 have no realized data/tokenizer/evaluation evidence yet.

The owner asked to push the rolling branch and proceed to the next card. The
branch was pushed; [Card 04 P1](results/2026-10-07-card04-shape-preparation.md)
records a proposed synthetic-only profile config, a fresh tokenizer declaration,
native inspection of exactly 341,885,952 parameters, and read-only WSL2/ROCm
inventory. The tokenizer output does not exist and no GPU profile was launched.
[The bounded Card 04 proposal](CARD04_PROFILE_PROPOSAL.md) requests a CPU-only
synthetic tokenizer preparation first, then a separate GPU profile decision.
The owner accepted P1's preparation evidence in [KML-D07](DECISIONS.md#kml-d07--card-04-preparation-accepted-for-reuse-2026-10-07),
and accepted Card 03's S1-S4 design/code steps in KML-D06. These reviewed steps
can be incorporated into subsequent work. Card 03's rights/acquisition gate,
Card 04's measured-fit gate and Gate 0 remain unverified.

The owner subsequently approved one exact Card 03 rights-quarantined pilot in
[KML-D08](DECISIONS.md#kml-d08--card-03-rights-quarantined-two-shard-pilot-approved-2026-10-07)
and one Card 04 CPU synthetic tokenizer fit in
[KML-D09](DECISIONS.md#kml-d09--card-04-cpu-synthetic-tokenizer-fit-approved-2026-10-07).
At that approval snapshot, the [source-level rights preflight](CARD03_SOURCE_RIGHTS_PREFLIGHT.md)
and [native pilot declarations](corpus-pilot/corpus.yaml) were ready for final
local validation; neither bounded operation had run yet.

The approved [Card 03 pilot](results/2026-10-07-card03-quarantined-pilot.md)
transferred and verified both pinned shards, retaining 8 Gutenberg and 21
Wikimedia rows; the [record screen](CARD03_PILOT_ROW_SCREEN.md) excludes nine
non-content rows and leaves 20 under rights review. The [Card 04 P2 fit](results/2026-10-07-card04-synthetic-tokenizer.md)
produced and verified a synthetic 32,768-entry tokenizer. These executed steps
are evidence for their bounded scope; no source row is admitted. The owner then
approved one bounded local GPU profile in KML-D10. [P3](results/2026-10-07-card04-gpu-profile-interrupted.md)
was interrupted after a device-memory sampler failure; seven stage updates were
reserved, and no stage bundle or profile receipt completed. A replacement
attempt requires a new explicit allocation. Card 03's full data/evaluation
gate, Card 04's measured-fit gate and Gate 0 remain unverified.

The subsequent metadata-only [PagerDuty inventory](CARD03_PAGERDUTY_FILE_INVENTORY.md)
confirmed 36 pinned Markdown blobs. The optional native bounded-Git transport
was implemented and locally tested in [S5](results/2026-10-07-card03-bounded-git-transport.md),
and the exact [38-file domain declaration](corpus-domain-pilot/corpus.yaml) parses.
No domain content ledger has been initialized or file acquired; the owner's
separate content-acquisition decision is pending. S5 does not admit a file.

The owner approved one replacement Card 04 GPU attempt under KML-D11.
[P4](results/2026-10-07-card04-gpu-retry-preflight-failed.md) passed the
readiness, shape, tokenizer and storage gates, then stopped before native
staging when the `amd-smi` CLI failed its first device-memory preflight. The
new ledger charged seven reserved stage updates but no optimizer update was
observed. A directly called AMD SMI library reader and fail-closed candidate
wrapper passed idle and mocked tests after P4; [a fresh attempt](CARD04_RETRY2_PROPOSAL.md)
requires separate approval. Neither P3 nor P4 supplies measured fit.

[P5](results/2026-10-07-card04-tracked-monitor-tools.md) places portable copies
of the direct AMD SMI reader and fail-closed phase launcher under the root-level `tools/kernel-memory-lab/` with
CPU/mock tests. This preserves the external originals and makes the proposed
monitor repair reviewable in Git. No new GPU attempt ran; the Card 04 measured-fit
gate is still blocked.

Future expert/sensemaking track: PLANNED ONLY; see `EXPERT_TRACK.md`. No additional runtime card is selected. Only the two-shard rights-quarantined pilot is approved for acquisition; no source row is approved for training or evaluation ingestion.

## Single current checklist

| Card | Deliverable | Status | Evidence or blocker |
| --- | --- | --- | --- |
| 01 | Foundation protocol and minimal missing scaffolding | EVIDENCE VERIFIED (documentation scope) | Owner accepted [foundation contract](FOUNDATION.md) in [KML-D01](DECISIONS.md#kml-d01--foundation-contract-accepted-2026-10-07); Gate 0 remains unverified |
| 02 | Fresh tiny correctness fixtures using native controls | EVIDENCE VERIFIED (done with notes) | [KML-D03](DECISIONS.md#kml-d03--card-02-accepted-with-failed-attempt-note-2026-10-07) accepted [A2](results/2026-10-07-card02-confirmation.md); [A1](results/2026-10-07-card02-tiny-fixtures.md) remains FAILED for its 57-update cap breach |
| 03 | Data, tokenizer and frozen evaluation contracts | EVIDENCE VERIFIED (S1-S4 and quarantined pilot); S5 code ready for review; full gate BLOCKED | [C03-P1](results/2026-10-07-card03-quarantined-pilot.md) verifies two snapshots; zero rows admitted. [S5](results/2026-10-07-card03-bounded-git-transport.md) tested bounded Git transport and [the domain declaration](CARD03_PAGERDUTY_NEXT_PLAN.md) is proposed; content approval, family splits, main tokenizer and frozen evaluation artifacts are missing |
| 04 | Main shape and measured fit | EVIDENCE VERIFIED (P1/P2 preparation); measured-fit gate BLOCKED | [C04-P3](results/2026-10-07-card04-gpu-profile-interrupted.md) and [C04-P4](results/2026-10-07-card04-gpu-retry-preflight-failed.md) preserve failed GPU attempts; [P5](results/2026-10-07-card04-tracked-monitor-tools.md) supplies reviewable tools, but no measured fit, and [new monitor-repair allocation](CARD04_RETRY2_PROPOSAL.md) needs owner approval |
| 05 | One bounded language training tranche | NOT STARTED | Data, fit receipts and run approval |
| 06 | Raw-text oracle evidence baseline | NOT STARTED | Qualified fresh checkpoint and frozen suite |
| 07 | Lexical retrieval baseline | NOT STARTED | Card 06 and approved versioned store |
| 08 | Explicit dense-to-new-reader transfer | NOT STARTED | Tiny contracts and selected Card 06 origin |
| 09 | One integrated reader | NOT STARTED | Cards 06, 07, 08 and bounded approval |
| 10 | Optional learned router or index experiment | NOT STARTED | Measured bottleneck; initially skip recommended |
| 11 | Frozen-weight insertion and correction | NOT STARTED | Useful reader and approved new records |
| 12 | Same-store RAM/NVMe comparison | NOT STARTED | Useful reader, fixed store and approved caps |
| 13 | Review and next decision | NOT STARTED | Actual relevant receipts |

These are prerequisite notes, not fabricated failed runs. Use BLOCKED when a real missing input prevents the selected task. After adoption, change this snapshot only from actual results, receipts and review. The owner's "proceed" accepts a completed evidenced step for reuse when recorded in a decision/result; it does not complete an unmet full-card gate or grant separately withheld runtime approval.
