# Current project status

Record revision: 2026-10-08.27
Snapshot date: 2026-10-08
Source audit revision: 06efc4db82ecf3da97b50cff518cba605ad27b33
Actual WSL checkout revision at Card 01 start: cf29aff79ec7259f7ec93988cd6a87065f4350e5 (`main`, clean before edits)
Repository documentation path: `experiments/research/kernel-memory-lab` (present in local `main`)
Current selected work: Full Card 03 remains accepted and Gate 0 EVIDENCE VERIFIED. [C05-N1](results/2026-10-08-card05-fresh-tranche-negative-language.md) completed the one fresh 5-million-target Card 05 tranche and 200-item evaluation under KML-D24, but scored 0/200 and is **NOT ELIGIBLE** under the adopted reader thresholds; its result is READY FOR OWNER REVIEW. [C05-F1](results/2026-10-08-card05-full-tranche-stop.md) remains FAILED, fully charged and unchanged. C03-S4/S2, Card 02 A1 and Card 04 P3/P4 remain failed with evidence preserved.
Next action: review [C05-I11's](results/2026-10-08-card05-hosted-owned-root-stop.md) failed hosted inode-cap and macOS cleanup evidence, then decide on a separate offline footprint/cleanup correction or revised allocation. C05-I6, C05-I8 and C05-I11 remain failed and spent; no dispatch authority exists. PR #54 stays draft. Q2 is owner accepted only as narrow CPU/fp32 evidence; Q1 remains FAILED. Model-path hosted qualification, live ROCm, real-panel/script parity and reviewer independence remain unqualified; no research runtime, acquisition, GPU work or Card 06 progression is authorized.
Working branch: `codex/kernel-memory-lab` (rolling branch; P6's fixed GPU source checkout was `fc75c548c88fb08973e334b805acdf497e3ccfc4`)
Runtime authority: Earlier Card 02/03/04 allocations remain exhausted or completed as recorded. KML-D20's 32-update measurement remains consumed. KML-D22's failed full-tranche attempt and KML-D24's fresh attempt each charged 4,883/4,883 updates. KML-D24's stage/train/evaluation authority is spent. KML-D25's one 600-second, zero-update, 18-generation diagnostic completed in C05-X1. KML-D28's C05-Q1 attempt is spent administratively without a ledger or model update. KML-D29's separate C05-Q2 attempt is spent with one v2 ledger charged 3/3 updates and 77/77 actual targets. KML-D32, KML-D34 and KML-D37 hosted dispatches are spent administratively before model tests, with no observed model updates or generation. No retry or further runtime remains authorized.
Last reviewed project result: [C05-Q2](results/2026-10-08-card05-native-cpu-qualification-q2.md) was accepted by the owner at `4b5976b` as narrow CPU/fp32 integration evidence in KML-D30. [C05-C1](results/2026-10-08-card05-native-monitor-slice.md) and [C05-C2](results/2026-10-08-card05-native-attempt-contract-slice.md) remain accepted offline; C05-C3's original coverage/binding guarantee remains unaccepted pending review of C05-C4. [C05-I1](results/2026-10-08-card05-v2-offline-diagnosis.md) remains accepted as a working explanation; C05-R1/C05-M1 and Card 03/Gate 0 retain their accepted scopes.
Latest project results: [C05-I9](results/2026-10-08-card05-hosted-guard-exit-repair.md) records the scoped offline guard correction, READY FOR OWNER REVIEW; [C05-I8](results/2026-10-08-card05-hosted-retry1-stop.md) preserves the FAILED hosted retry; [C05-I7](results/2026-10-08-card05-hosted-guard-sampler-repair.md) records the offline hosted guard repair, READY FOR OWNER REVIEW; [C05-I6](results/2026-10-08-card05-hosted-ci-qualification-stop.md) preserves the FAILED hosted preledger stop; [C05-I5](results/2026-10-08-card05-hosted-ci-guard-preflight.md) records the original guard and passing offline checks; [C05-I4](results/2026-10-08-card05-pr-format-repair.md) repairs the 13 documented Ruff-format failures; [C05-I3](results/2026-10-08-card05-draft-pr-safety-lane.md) is the limited draft integration lane; [C05-I2](results/2026-10-08-card05-integration-pr-preflight.md) preserves the earlier PR stop; [C05-Q2](results/2026-10-08-card05-native-cpu-qualification-q2.md) is owner accepted for the tiny CPU path, and [C05-Q1](results/2026-10-08-card05-native-cpu-qualification-stop.md) remains FAILED. [C05-C4](results/2026-10-08-card05-native-binding-corrections.md) remains corrected offline panel/native wiring evidence, READY FOR OWNER REVIEW. [C05-C3](results/2026-10-08-card05-native-panel-readiness-slice.md) retains its historical scoped evidence but its original coverage/binding claim is unaccepted. C05-C1/C2 are accepted offline; C05-P4 is the accepted plan. [C05-X1](results/2026-10-08-card05-matched-diagnostic.md) remains exploratory; [C05-N1](results/2026-10-08-card05-fresh-tranche-negative-language.md) remains the 0/200 negative result and [C05-F1](results/2026-10-08-card05-full-tranche-stop.md) the failed first attempt. Card 03/Gate 0 acceptance and earlier evidence remain intact.
Latest research decision: [KML-D35](DECISIONS.md#kml-d35--hosted-retries-paused-and-offline-guard-correction-authorized-2026-10-08) pauses hosted retries and authorizes the offline correction; [KML-D34](DECISIONS.md#kml-d34--one-fresh-hosted-serving-only-retry-authorized-and-stopped-2026-10-08) authorized and spent one fresh hosted dispatch; [KML-D33](DECISIONS.md#kml-d33--focused-offline-hosted-guard-startup-repair-authorized-2026-10-08) authorized the offline repair only; [KML-D32](DECISIONS.md#kml-d32--one-guarded-hosted-serving-only-validation-authorized-2026-10-08) was spent at preflight. KML-D31 authorized the narrow draft lane. KML-D30 accepts Q2 narrowly and records the earlier PR CI block. KML-D29 records the spent Q2 allocation, KML-D28 preserves Q1's stop, KML-D27 accepts C05-C1/C2 offline, and KML-D24's full-run authority is spent.
Latest project attempt: [C05-I11](results/2026-10-08-card05-hosted-owned-root-stop.md) failed during hosted dependency setup at the 50,000-inode cap before model tests; [C05-I8](results/2026-10-08-card05-hosted-retry1-stop.md) and [C05-I6](results/2026-10-08-card05-hosted-ci-qualification-stop.md) remain separate earlier failed dispatches. [C05-Q2](results/2026-10-08-card05-native-cpu-qualification-q2.md) remains the latest successful model-path qualification, with 3 CPU updates, 77 nonmasked targets and zero generation. None changes C05-N1's failed reader eligibility at 0/200 or C05-F1's earlier FAILED monitoring stop. Earlier accepted timing, preparation and synthetic-fit results remain intact.
Latest offline implementation: [C05-I10](results/2026-10-08-card05-hosted-owned-root-implementation.md) implements the owned-root five-job phase split under KML-D36; its [audit](CARD05_HOSTED_OWNED_ROOT_AUDIT.md) and local tests do not qualify hosted behavior or authorize a dispatch.
Owned-root policy decision: [KML-D36](DECISIONS.md#kml-d36--owned-root-policy-accepted-for-offline-implementation-2026-10-08) accepted the narrower policy for offline implementation; KML-D37 later approved and spent one separate dispatch.
Latest runtime decision: [KML-D37](DECISIONS.md#kml-d37--one-owned-root-hosted-dispatch-approved-and-stopped-2026-10-08) approved and spent the single owned-root dispatch; no subsequent hosted allocation is authorized.
Budget code fix: [KML-20261007-C02-P1](results/2026-10-07-card02-budget-repair.md), committed in `77ec6e7` and confirmed by A2

## Delivery facts

[C05-Q2](results/2026-10-08-card05-native-cpu-qualification-q2.md) passed 22/22
preledger checks, including the exact config round-trip, and one training node.
The native CPU run and v2 ledger agree on 3 updates and 77 targets, with a
32/32/13 final mask and zero generations. Both owned supervisors reported zero
surviving descendants. This qualifies only the bounded synthetic CPU path;
[C05-Q1](results/2026-10-08-card05-native-cpu-qualification-stop.md) preserves
its earlier 21/21 precheck and failed preledger trace unchanged.

[C05-I2](results/2026-10-08-card05-integration-pr-preflight.md) adds an explicit
opt-in guard to the optimizer-bearing Q2 fixture. Three selected zero-update
checks passed. Automatic PR CI still selects real training and generation,
so the prepared integration PR is not opened without a new bounded CI decision.

[C05-I3](results/2026-10-08-card05-draft-pr-safety-lane.md) implements the
owner-approved exact-digest historical lint exceptions and separate limited
same-repository draft CI lane. It preserves ordinary checks outside this draft
case and restores them on `ready_for_review`. The normal full-tree Ruff format
check still reports 13 untouched files; no broad suite or hardware proof is
claimed by the draft checks.

[C05-I4](results/2026-10-08-card05-pr-format-repair.md) formats only those 13
reported Python files. Whole-tree Ruff format and rule checks now pass; all 13
ASTs and the three pinned historical legacy artifacts remain unchanged. This
does not authorize the normal PR jobs or full CPU suites.

[C05-I5](results/2026-10-08-card05-hosted-ci-guard-preflight.md) adds the
owner-approved opt-in, exact-SHA hosted serving-only path under KML-D32. Its
preflight and resource watcher fail before model work on a missing or expired
workflow identity, setup/storage/process fault, collection drift or ledger
violation. Offline doubles verify persistent per-job charging and owned child
shutdown. The clean guard at `ed7b837` and its safe draft checks passed, but
the one [hosted dispatch](results/2026-10-08-card05-hosted-ci-qualification-stop.md)
failed before its ledger or model work when the ambient storage baseline scan
exhausted its three-second sampling deadline. The allocation is spent and the
full CPU suites remain unqualified.

[C05-I7](results/2026-10-08-card05-hosted-guard-sampler-repair.md) repairs the
startup receipt and ambient sampler offline, with a 15-second whole-sweep
deadline, retained offending-root/path context and incomplete-init
finalization. Thirty-two selected zero-model-work checks passed. Local fixture
traversal costs do not establish hosted Linux or macOS behavior; the fresh
[proposal](CARD05_HOSTED_RETRY1_PROPOSAL.md) received a separate approval and
was attempted once as [C05-I8](results/2026-10-08-card05-hosted-retry1-stop.md).
Linux baselines still exhausted the 15-second whole-sweep deadline; macOS
baselines passed but guarded dependency setup failed before model work. The
fresh allocation is spent, and hosted model-path qualification remains open.

[C05-I9](results/2026-10-08-card05-hosted-guard-exit-repair.md) fixes
stale-process exit detection and cleanup-failure receipt ordering offline.
[The owned-root policy](CARD05_HOSTED_OWNED_ROOT_POLICY_PROPOSAL.md) was accepted
for one offline implementation pass in KML-D36. [C05-I10](results/2026-10-08-card05-hosted-owned-root-implementation.md)
records the changed workflow/guard and fixture audit. Provider setup remains
time-bounded and observational for storage; the checkout and job root are
measured during lab execution. A fresh hosted allocation is still required.

[C05-C4](results/2026-10-08-card05-native-binding-corrections.md) requires
unique exact frozen-suite coverage and a pinned renderer reproducing each
verified panel prompt; it adds the v2 attempt CLI, selected Campaign phases and
cold native receipt-derived actual counters. The separate Q2 optimizer-bearing
CPU integration completed under KML-D29; live ROCm and real-panel parity
remain unqualified. It did not reassess historical scores.


[C05-C3](results/2026-10-08-card05-native-panel-readiness-slice.md) adds
absent-field-compatible panel decoder controls, an exact token-ID comparison
for two authenticated panels, and independent review-ledger score checks
bound to the accepted frozen item content and readiness policy. It preserves
missing and failed score states without promotion; no real generation or
historical score reassessment occurred.

[C05-C2](results/2026-10-08-card05-native-attempt-contract-slice.md) adds a
content-pinned optional plan reference and v2 allocation in the existing
attempt ledger. It reserves cumulative optimizer, target, generation-call and
generated-token maxima before monitored phases, including zero-update work,
under one immutable deadline and baseline. Selected offline checks preserve
legacy absent-field lock identities and v1 ledger behavior. Authenticated
model-run counters and update-bearing integration are qualified only for the
tiny C05-Q2 CPU fixture; no new research runtime or reader eligibility follows.

[C05-C1](results/2026-10-08-card05-native-monitor-slice.md) implements and
offline-tests the first native monitor/readiness slice under KML-D26. Legacy
optional-absent serialization is preserved; the full common-root storage cap
includes final monitor writes. The tracked Card 05 scripts remain until parity,
and no live ROCm or research runtime claim follows from these CPU checks.

[C05-P4](results/2026-10-08-card05-consolidation-planning.md) proposes
[prospective base, instruction, evidence-reader and agentic stages](CARD05_CONSOLIDATION_PLAN.md)
without changing any accepted result or reader gate. It maps reusable Card 05
monitor, attempt and panel behavior to native APIs and orders safety/readiness
correctness before the next separately approved experiment. The TODO update is
code work only; no runtime or main integration is authorized by this record.

[C05-X1](results/2026-10-08-card05-matched-diagnostic.md) froze eight distinct
held-out passage/question pairs before one bounded inference run. Eighteen of
eighteen generations completed with 1,048 retained completion IDs against 1,152
reserved positions; the two preselected uncached reruns matched cached token IDs
exactly. All eight short questions were unanswered despite answer-bearing
contexts. The repaired native monitor and subreaper reported no violations and
zero live descendants. This exploratory result does not isolate a cause or
establish reader eligibility. Full raw prompts and completions remain in the
private external task root; the frozen 200-item suite is unchanged.

[C05-I1](results/2026-10-08-card05-v2-offline-diagnosis.md) inspects the v2
loss and raw-panel trajectories without model work. One-batch held-out loss fell
from 10.668 at initialization to 6.461 at step 4,883, but all 200 language
items remained incorrect and format-noncompliant. Retained packing and token-ID
checks found no shift, mask, context-fit or display-decoding defect. Raw prose
fluency versus instruction following is not isolated by this panel; a new
zero-update matched-prompt diagnostic was separately approved and has completed
as C05-X1.

[C05-N1](results/2026-10-08-card05-fresh-tranche-negative-language.md) ran once from
clean pushed `88095de` after 113 passing offline tests and cold preflight. Its
independent ledger charged 4,883 updates; the native run completed exactly
5,000,000 nonmasked targets and retained all eleven checkpoints. The frozen
selected terminal checkpoint passed full verification at finite one-batch loss
6.461394. The one selected-checkpoint suite and all 200 frozen language prompts
completed under the time, VRAM, RSS, disk, inode and output caps. Independent
item-level reviews agreed on 0/200 correct; one formatting disagreement was
adjudicated, leaving 0/200 format compliant and 0/20 on each axis. Reader
eligibility is NOT MET; C05-N1 awaits owner review and grants no Card 06
progression. The earlier C05-F1 failed ledger and evidence remain unchanged.

[C05-R1](results/2026-10-08-card05-offline-monitor-repair.md), accepted by the owner in KML-D24, repairs only the
monitoring path after C05-F1. Bounded fresh-pass byte/inode traversal includes
live SQLite WAL/SHM files while genuine errors and exhausted deadlines fail
closed. An owner-tracked subreaper now stops timeout, monitor and worker
descendants across process groups before reporting completion. Code commit
`16cfa8b` is pushed; 112 focused CPU/mock tests passed. The failed ledger,
baseline and partial run were not modified. C05-R1 itself grants no runtime
authority; KML-D24 separately authorizes one fresh conditional attempt.

[C05-F1](results/2026-10-08-card05-full-tranche-stop.md) used the owner-approved
KML-D22 allocation from clean pushed `428e9b1`. Offline adaptation passed 101
focused tests; cold release, tokenizer, mixture and packed-cache checks, inspect,
storage preflight and three idle UUID-matched VRAM reads passed. The new-config
zero-update validation stage completed, but the fresh training attempt failed
closed when `du` raced ephemeral SQLite WAL files. The ledger charged all 4,883
reserved updates; the shared metrics store contains 500 completed updates and
512,000 targets. No terminal checkpoint or 200-item evaluation was produced.
The child `timeout` group survived the launcher's initial group kill and was
then explicitly terminated; GPU process inventory was empty afterward. Card 05
eligibility is unestablished. No retry or resume is authorized.

[C05-P3](results/2026-10-08-card05-full-tranche-planning.md) proposes a fresh
seed-17 4,883-update/5,000,000-target config with a 100-update warmup and
full-horizon cosine decay, eleven preserved one-batch validation/checkpoint
events, and a bounded native-panel evaluation of all 200 frozen language items.
Its [single approval request](CARD05_FULL_TRANCHE_PROPOSAL.md) fixes resource caps,
the selected-checkpoint and scoring rules, a preledger offline launcher
adaptation, and the no-retry/stop policy. This planning step performed no model
initialization, staging, GPU or optimizer work and grants no runtime authority.

[C05-M1](results/2026-10-08-card05-real-data-timing.md) used the pushed clean
`ec1651f` checkout and one persistent ledger to validate-stage the accepted
prepared bundle with zero updates, then run one fresh 32-update/32,768-target
measurement. The terminal checkpoint verified fully and stage/train monitors
reported no cap violation. Its measured 20-update steady window and separated
overhead yield a 1,353-second raw forecast plus a stated 20% planning
allowance for 5 million positions under an assumed every-500 checkpoint and
validation cadence. Full-suite evaluation cost remains unmeasured; the owner
accepted this result as timing evidence in KML-D21, without granting
full-tranche runtime authority.

[C05-S1](results/2026-10-08-card05-monitor-adaptation.md) implements and
CPU/mock-tests the exact Card 05 stage/train launcher and validator under the
owner's KML-D20 approval. It was committed and pushed as `ec1651f` before the
runtime ledger or model activity; C05-M1 then used that fixed checkout.

[C03-R1/G00-R1/C05-P2](results/2026-10-08-card03-acceptance-gate0-card05-review.md)
records the owner's full Card 03 acceptance for the exact frozen local-research
release and read-only cold verification of release/prepared identities. It maps
all original Gate 0 criteria to accepted evidence and marks that starting-point
gate EVIDENCE VERIFIED. It supersedes C05-P1's dated acquisition/preparation
blockers. C05-M1 supplies owner-accepted real-data timing; KML-D22 subsequently
approved one full-run allocation, which stopped and is exhausted in C05-F1.

[C03-C2](results/2026-10-07-card03-evaluation-continuation.md) used the verified C03-C1 release, tokenizer and prepared bundle with no new source request or model update. Three author partitions recorded decisions for all 600 original candidate IDs. Different reviewers blind-solved and checked every exact final item version, retaining negative earlier decisions. The final cold screen found zero hard errors, and native `require_complete=True` freeze published a 200/400-item manifest with all 20 category denominators and 20 clarification plus 20 unanswerable evidence items. The three language boilerplate-overlap flags were individually adjudicated as different held-out facts. The owner accepted C03-C2 together with C03-C1 under KML-D19; C03-C2 alone is not a model result.

[C03-C1](results/2026-10-07-card03-offline-continuation.md) used the retained
four-source snapshots without network activity. A revised family grouping and
full lexical recheck preceded a cold-verified 5,994-document release. The
train-only 32,768-entry tokenizer and measured unique positions clear all three
accepted source and family floors. The native mixture and sealed bundle verify
exactly 5,000,000 supervised targets at 65/25/10, at most two exposures and
zero optimizer updates. At C03-C1's stop, its unreviewed 600-item candidate
bank had sampled semantic errors and blocked the frozen evaluation requirement.
KML-D18 later accepted these preparation artifacts for reuse; C03-C2 separately
corrected and froze the evaluation suite. KML-D19 accepted them together as the
complete Card 03 release. C03-C1 alone was not a complete Card 03 admission.

[C03-S4](results/2026-10-07-card03-scale-retry1-stop.md) records the one KML-D16
retry. The durable ledger shows 473 complete transfers and 845,577,448 source
bytes; PagerDuty commit/tree metadata passed first, and all four pinned snapshots
were verified. Native admission, spot audits and family splits produced a latest
offline build with 5,994 kept documents. A full lexical screen found 181
cross-split candidates in an earlier build; revised admission excluded 5,183
short Wikimedia stubs and grouped observed families. The repeat screen was
interrupted when copied CPU test receipts exposed 10 post-ledger optimizer
updates in a broad regression suite against the approved zero-update cap.
At C03-S4's stop, no Card 03 release, tokenizer, denominator, mixture or frozen
evaluation existed. C03-C1 later prepared the first four within separate
authorization, and C03-C2 prepared the frozen evaluation accepted in KML-D19. All
earlier raw evidence and failed ledgers remain retained.

[C03-A1](results/2026-10-07-card03-source-admission-v1.md) applies the owner's
accepted source-policy v1 to the two already verified pilot snapshots. Native
offline build and cold release verification retain 19 qualifying documents,
exclude nine non-content rows and quarantine one biography for privacy review.
The original source declarations, acquisition receipt and historical
zero-admitted [screen](CARD03_PILOT_ROW_SCREEN.md) are unchanged. The new release
is metadata-only and has **no selected LM/chat records**. It is too small for the
main language mix and does not complete Card 03 or authorize Card 05. The
[consolidated preparation plan](CARD03_TO_CARD05_PREPARATION_PLAN.md) remains a
proposal.

[C03-S2](results/2026-10-07-card03-scale-metadata-stop.md) stopped the first
scaled preparation attempt during bounded Git metadata reading. The new native
deadline fallback has offline tests, but there are zero newly acquired source
bytes or snapshots. The stopped ledger and monitor traces are retained; the
[retry-1 proposal](CARD03_SCALE_RETRY1_PROPOSAL.md) needs separate approval.

[C04-P6](results/2026-10-07-card04-gpu-retry2-measured-fit.md) records the one
owner-approved Retry2 synthetic GPU attempt. Native staging completed seven
updates and fresh profiling completed 113 more under one 1,800-second ledger,
with finite losses/gradients, a full verified checkpoint, and no sampled
resource-cap or monitor failure. [KML-D13](DECISIONS.md#kml-d13--card-04-p6-synthetic-fit-evidence-accepted-2026-10-07)
accepts this synthetic-fit scope; older P3/P4 failures remain unchanged.
Synthetic throughput is not a real-data forecast. Gate 0 was subsequently
reviewed and verified for its separate starting-point criteria in C03-R1/G00-R1.

[C05-P1](results/2026-10-07-card05-prerequisite-audit.md) was the historical
preparation audit. C03-C1 and C03-C2 now supply the owner-accepted full Card 03
release; its prior acquisition/preparation blockers are superseded. C03-R1/G00-R1
maps every Gate 0 criterion. Its formerly missing real-data timing evidence is
now supplied by owner-accepted C05-M1. The later C05-F1 full-run attempt failed;
that one-time authority is exhausted without a Card 05 result.

Inherited-output correction [C04-S4](results/2026-10-07-card04-inherited-output-review.md)
addresses a reproduced sampler descendant/pipe-EOF hang after the owner reported
local hung-sensor failures. Earlier cloud CI did not clear those local failures.
The corrected tracked tools passed local verification at `fc75c54` before the
separate KML-D12 runtime decision; the code review alone granted no GPU authority.

Sample-evidence correction [C04-S3](results/2026-10-07-card04-sample-log-review.md)
was READY FOR REVIEW in a follow-up to merged PR #51. It makes failed sample-log
writes stop supervision and updates Retry2's launcher hash. Runtime approval
and local idle/no-training verification were handled separately before P6.

Tracked monitor review [C04-S2](results/2026-10-07-card04-tracked-monitor-review.md)
supersedes S1's missing-source blocker: base `75f9a8d` supplies the scripts.
Focused timeout, watchdog, cleanup and launch-input corrections are READY FOR
REVIEW in draft PR #51. Retry2 references the corrected tracked tool hashes;
local no-training validation and a separate runtime decision preceded P6.

Cloud code review [C04-S1](results/2026-10-07-card04-cloud-safety-review.md)
is READY FOR REVIEW on a separate branch. It fixes budget-runner exception
cleanup using mocked tests; it does not validate the external AMD SMI wrapper,
change historical attempt accounting, or approve Retry2. P6 later supplied
synthetic measured-fit evidence for review.

## Historical progression (each statement describes its own date)

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
later received separate KML-D12 approval. Neither P3 nor P4 supplies measured fit.

[P5](results/2026-10-07-card04-tracked-monitor-tools.md) places portable copies
of the direct AMD SMI reader and fail-closed phase launcher under the root-level `tools/kernel-memory-lab/` with
CPU/mock tests. This preserves the external originals and makes the proposed
monitor repair reviewable in Git. P5 itself ran no GPU attempt; P6 later used
these tracked tools under KML-D12.

Future expert/sensemaking track: PLANNED ONLY; see `EXPERT_TRACK.md`. No additional runtime card is selected. KML-D16's later scaled acquisition completed four pinned source snapshots, C03-C1 prepared a local-use release accepted for reuse, and C03-C2 froze evaluation subsequently accepted with C03-C1 in KML-D19. The earlier two-shard pilot and all stopped attempts retain their own identities. KML-D12's single Card 04 GPU authority is spent.

## Single current checklist

| Card | Deliverable | Status | Evidence or blocker |
| --- | --- | --- | --- |
| 01 | Foundation protocol and minimal missing scaffolding | EVIDENCE VERIFIED (documentation scope) | Owner accepted [foundation contract](FOUNDATION.md) in [KML-D01](DECISIONS.md#kml-d01--foundation-contract-accepted-2026-10-07); separate Gate 0 review completed in [G00-R1](results/2026-10-08-card03-acceptance-gate0-card05-review.md) |
| 02 | Fresh tiny correctness fixtures using native controls | EVIDENCE VERIFIED (done with notes) | [KML-D03](DECISIONS.md#kml-d03--card-02-accepted-with-failed-attempt-note-2026-10-07) accepted [A2](results/2026-10-07-card02-confirmation.md); [A1](results/2026-10-07-card02-tiny-fixtures.md) remains FAILED for its 57-update cap breach |
| 03 | Data, tokenizer and frozen evaluation contracts | EVIDENCE VERIFIED (exact local-research release) | [KML-D19](DECISIONS.md#kml-d19--full-card-03-accepted-gate-0-reviewed-card-05-measurement-proposed-2026-10-08) accepts [C03-C1](results/2026-10-07-card03-offline-continuation.md) and [C03-C2](results/2026-10-07-card03-evaluation-continuation.md) together; [C03-S4](results/2026-10-07-card03-scale-retry1-stop.md) and [C03-S2](results/2026-10-07-card03-scale-metadata-stop.md) remain FAILED |
| 04 | Main shape and measured fit | EVIDENCE VERIFIED (synthetic fit) | [C04-R2](results/2026-10-07-card04-synthetic-fit-acceptance.md) records owner acceptance of [P6](results/2026-10-07-card04-gpu-retry2-measured-fit.md); P3/P4 remain failed; separate real-data timing is now accepted in C05-M1 |
| 05 | One bounded language training tranche | TRANCHE COMPLETED; READER ELIGIBILITY NOT MET (owner review pending) | [C05-N1](results/2026-10-08-card05-fresh-tranche-negative-language.md) completed 4,883 updates / exactly 5,000,000 targets and 200/200 frozen items, scoring 0/200; [C05-I1](results/2026-10-08-card05-v2-offline-diagnosis.md) is owner accepted as a working explanation, and exploratory [C05-X1](results/2026-10-08-card05-matched-diagnostic.md) found 0/8 answered matched questions with two exact cache-parity checks. [C05-F1](results/2026-10-08-card05-full-tranche-stop.md) remains a separate FAILED first attempt with its full reservation charged. Native CPU qualification [C05-Q1](results/2026-10-08-card05-native-cpu-qualification-stop.md) failed before its ledger/model; the separate [C05-Q2](results/2026-10-08-card05-native-cpu-qualification-q2.md) completed its tiny CPU integration and is owner accepted narrowly. No runtime authority remains; [C05-P1](results/2026-10-07-card05-prerequisite-audit.md) is historical |
| 06 | Raw-text oracle evidence baseline | BLOCKED (not started) | The accepted Card 03 suite exists, but C05-N1's fresh checkpoint failed the adopted Card 05 reader-eligibility gate; no qualified checkpoint or Card 06 runtime authority exists |
| 07 | Lexical retrieval baseline | NOT STARTED | Card 06 and approved versioned store |
| 08 | Explicit dense-to-new-reader transfer | NOT STARTED | Tiny contracts and selected Card 06 origin |
| 09 | One integrated reader | NOT STARTED | Cards 06, 07, 08 and bounded approval |
| 10 | Optional learned router or index experiment | NOT STARTED | Measured bottleneck; initially skip recommended |
| 11 | Frozen-weight insertion and correction | NOT STARTED | Useful reader and approved new records |
| 12 | Same-store RAM/NVMe comparison | NOT STARTED | Useful reader, fixed store and approved caps |
| 13 | Review and next decision | NOT STARTED | Actual relevant receipts |

These are prerequisite notes, not fabricated failed runs. Use BLOCKED when a real missing input prevents the selected task. After adoption, change this snapshot only from actual results, receipts and review. The owner's "proceed" accepts a completed evidenced step for reuse when recorded in a decision/result; it does not complete an unmet full-card gate or grant separately withheld runtime approval.
