# Revision and synchronization log

## 2026-10-08.4 — C05-M1 accepted and one full-tranche/evaluation request, 8 October 2026

- Recorded KML-D21 accepting C05-M1 at `01184e8` for real-data timing evidence only; the measured checkpoint, old failures, Card 03 release and Gate 0 assessment remain unchanged
- Derived a new seed-17 4,883-step/5,000,000-target full-run config with 100-step warmup, full-horizon cosine decay and every-500 one-batch validation/checkpoint retention; authored fixed 200-language-item prompt order, generation panel and selected-checkpoint evaluation suite
- Validated the declarations offline against the frozen 600-item content digest, ten 20-item language axes, native config/parser, exact tokenizer context fit and conservative checkpoint/storage math; [C05-P3](results/2026-10-08-card05-full-tranche-planning.md) records the evidence and [the single proposal](CARD05_FULL_TRANCHE_PROPOSAL.md) states exact limits and thresholds
- No staging, model initialization, GPU execution, optimizer updates, acquisition, evaluation generation, cloud spend or Card 06 progression occurred; full Card 05 runtime still needs owner approval and a focused offline launcher adaptation before ledger creation

## 2026-10-08.3 — bounded Card 05 real-data timing measurement, 8 October 2026

- Pushed passing launcher adaptation as fixed clean `ec1651f`, cold-verified the accepted prepared inputs and stage/run configuration, and completed the one KML-D20 ledger-bounded attempt: zero-update validate stage followed by one fresh 32-update/32,768-target train stop
- Recorded [C05-M1](results/2026-10-08-card05-real-data-timing.md) with full checkpoint verification, stage/train monitor receipts, resource deltas, 20 valid steady timing updates, overhead and a 1,353-second raw five-million-position forecast plus a separately labeled 20% planning allowance
- Preserved the auxiliary observational resource-check failure and correction, all earlier failed evidence, and the unchanged sealed release; no frozen-suite scoring, acquisition, retry, resume or full Card 05 training occurred
- Marked timing evidence READY FOR OWNER REVIEW; the single runtime allocation is consumed and a full Card 05 optimizer run requires a separate decision

## 2026-10-08.2 — Card 05 timing allocation and monitored launcher, 8 October 2026

- Recorded the owner's exact KML-D20 one-attempt real-data timing approval, including zero-update validation stage, 32-update/32,768-position train limit, aggregate deadline, common-root storage baseline, one-batch initial/terminal validation and no frozen-suite run
- Added the focused tracked Card 05 launcher/validator and a kind selector in the existing bounded-measurement helper; 64 focused Card 04/05 CPU/mock tests and 28 adjacent budget/monitor tests passed, with Ruff and shell syntax checks
- Kept model initialization, staging and GPU work outside this code-preparation revision; [C05-S1](results/2026-10-08-card05-monitor-adaptation.md) records the tested tool hashes and remaining clean-checkout preflight before the one authorized runtime ledger

## 2026-10-08.1 — full Card 03 acceptance, Gate 0 review and Card 05 profile proposal, 8 October 2026

- Recorded the owner's KML-D19 acceptance of C03-C2 at `4ced162` together with previously accepted C03-C1 for the exact local-research release, preserving failed C03-S2/S4 and older attempt evidence
- Cold-verified available release and prepared identities, recomputed the frozen evaluation content digest, and mapped every original Gate 0 criterion to accepted evidence in [C03-R1/G00-R1/C05-P2](results/2026-10-08-card03-acceptance-gate0-card05-review.md); Gate 0 is evidence verified for its starting-point scope
- Kept [C05-P1](results/2026-10-07-card05-prerequisite-audit.md) as historical, superseded its dated acquisition/preparation blockers, and proposed one separately approved 32-update real-data measurement from a fresh main initializer with exact monitored limits; no model/GPU work occurred in this review

## 2026-10-07.24 — source-level admission policy v1 and retained-pilot screen, 7 October 2026

- Recorded the owner's acceptance of proportionate pinned-source/license rules in KML-D14 and updated the Card 03 contract for inherited record provenance, automated exceptions, spot audits and nonblocking optional metadata issues; preserved the earlier rights preflight and zero-admitted pilot screen as historical evidence
- Added a versioned 29-row admission manifest and an offline Corpus release variant over the original snapshots, with 19 locally qualifying documents, nine non-content exclusions and one privacy quarantine; no LM/chat view selected
- Extended native v2 corpus build and cold verification to enforce the exact admission inventory and preserve old declaration/release hashes when the optional field is absent; recorded focused/broader offline tests and immutable build/release identities in [C03-A1](results/2026-10-07-card03-source-admission-v1.md)
- Proposed one consolidated preparation path from source sizing through broad/domain data, family splits, train-only main tokenizer, realized mixture, frozen evaluations and real-data Card 05 forecast, without authorizing new acquisition or training

## 2026-10-07.23 — P6 accepted and Card 05 prerequisites audited, 7 October 2026

- Recorded the owner's [R2 acceptance](results/2026-10-07-card04-synthetic-fit-acceptance.md) of P6 for Card 04 synthetic measured fit in KML-D13, preserving P3/P4 failures and the synthetic-only limit
- Audited [Card 05 prerequisites](results/2026-10-07-card05-prerequisite-audit.md): Card 02 is accepted, but Card 03 rights-admitted data/main tokenizer/frozen evaluation and real-data timing or validated forecast are missing; Gate 0 has no promotion review
- Kept Card 05 training unstarted and proposed the existing bounded PagerDuty rights-quarantined content pilot as the smallest next operational task, subject to its own separate approval

## 2026-10-07.22 — bounded Card 04 synthetic GPU profile, 7 October 2026

- Recorded one owner-approved Retry2 attempt in [P6](results/2026-10-07-card04-gpu-retry2-measured-fit.md): seven stage and 113 profile updates completed under the shared deadline/caps, with finite numerical observations, native monitor receipts and a full verified checkpoint
- Preserved P3/P4 failed attempts and kept synthetic fit READY FOR REVIEW before the later KML-D13 owner decision; no real-data throughput, Card 03 admission or Gate 0 promotion followed from P6 alone

## 2026-10-07.21 — Card 04 monitor tools tracked for review, 7 October 2026

- Added portable tracked copies of the direct AMD SMI VRAM reader and fail-closed phase launcher under root-level `tools/kernel-memory-lab/`, removing machine paths and device UUID literals while preserving the external originals and P3/P4 failure evidence
- Added 18 CPU/mock tests for identity, measurement failures, preflight and mid-run caps, storage limits, budget/path guards and native phase command construction; 25 related native monitor/budget tests, Ruff and shell syntax passed
- Linked the tracked tools from the [new attempt proposal](CARD04_RETRY2_PROPOSAL.md) and [P5 code-preparation result](results/2026-10-07-card04-tracked-monitor-tools.md); no GPU attempt, data acquisition or new runtime authority occurred, and Card 04 measured fit remains blocked

## 2026-10-07.20 — Card 04 replacement telemetry preflight failed, 7 October 2026

- Recorded KML-D11 approving one replacement GPU attempt under the exact Card 04 retry proposal; its new ledger reserved seven stage updates
- Preserved [P4](results/2026-10-07-card04-gpu-retry-preflight-failed.md): `amd-smi` CLI failed before native staging, so no optimizer update, stage bundle, profile receipt or fit evidence exists; P3 remains unchanged
- Prepared an installed AMD SMI library reader and fail-closed candidate wrapper; 20 live idle reader invocations, a wrong-UUID rejection, shell syntax and a mocked mid-run reader failure passed without training
- Added [one new bounded proposal](CARD04_RETRY2_PROPOSAL.md); Card 04 measured fit, Card 03 full gate and Gate 0 remain unverified, with no further GPU authority from KML-D11

## 2026-10-07.19 — native bounded PagerDuty declaration ready, 7 October 2026

- Implemented optional pinned-Git-blob HTTPS acquisition in the existing native
  Corpus path, with one durable response-body ledger and exact commit/tree/file
  checks; committed/pushed reusable code as `ec91e9f` before these records
- Passed 46 focused acquisition/identity tests and Ruff; broader corpus suite
  had 297 passes and two intermittent proof-reuse failures that both passed
  individually, so a clean all-corpus pass is not claimed
- Added [S5](results/2026-10-07-card03-bounded-git-transport.md) and a parsed,
  exact [38-file domain pilot declaration](corpus-domain-pilot/corpus.yaml),
  separately from any content attempt; updated the proposal and code TODO
- Kept PagerDuty content acquisition, individual file admission, Card 03 full
  gate, Card 04 replacement GPU attempt and Gate 0 pending their own decisions

## 2026-10-07.18 — Card 03 domain metadata inventory and bounded next gate, 7 October 2026

- Rechecked the immutable PagerDuty Git tree metadata without acquiring any
  Markdown or license content: 36 selected Markdown blobs, 302,785 raw bytes,
  plus the pinned LICENSE and README identities
- Added the exact [file inventory](CARD03_PAGERDUTY_FILE_INVENTORY.md) and a
  [rights-quarantined next plan](CARD03_PAGERDUTY_NEXT_PLAN.md) separating the
  native bounded-Git-blob transport code prerequisite from later acquisition
  approval; neither code mode nor content attempt has started
- Recorded unavailable expanded-stream/peak-staging counters from C03-P1 as
  missing evidence rather than inferred measurements; the 29-row screen still
  excludes nine and admits zero records
- Kept Card 04 P3 interrupted, both full-card gates and Gate 0 unverified

## 2026-10-07.17 — Card 04 GPU stage interruption and replacement proposal, 7 October 2026

- Began one fixed-checkout local ROCm attempt under KML-D10 after tokenizer,
  inspect, workspace, research-lint and dense-readiness checks passed
- Charged all seven stage-pilot updates to its persistent ledger; a transient
  device-memory sampler failure left no completed stage bundle or profile run,
  so the owner stopped the attempt and preserved raw monitor/runner evidence
- Corrected the external sampler wrapper to fail closed on unavailable
  measurements; shell syntax, repeated idle readings and a mocked mid-run
  sampler failure passed, without starting another GPU job
- Added [P3](results/2026-10-07-card04-gpu-profile-interrupted.md) and a
  [new exact allocation proposal](CARD04_RETRY_PROPOSAL.md). KML-D10 grants no
  retry; Card 04 measured fit, Card 03 full gate and Gate 0 remain unverified

## 2026-10-07.16 — quarantined Card 03 pilot and synthetic tokenizer executed, 7 October 2026

- Retained the first failed Card 03 acquisition invocation and every later
  metadata/deadline failure in one durable ledger; native fixes for the live
  urllib response and Hub LFS `oid` metadata were committed separately
- Completed the owner-approved two-shard transport pilot within its original
  byte/time/disk bounds; authenticated 8 Gutenberg and 21 Wikimedia review rows
  in two native snapshots, with zero rows admitted for training or evaluation
- Completed and verified the approved Card 04 CPU synthetic tokenizer fit at
  exactly 32,768 entries; recorded KML-D10's later separate approval of one
  bounded local GPU profile, still pending at this revision
- Preserved the unrelated broader corpus-suite failure, Card 02 A1 failure,
  the full Card 03/04 blockers and unverified Gate 0

## 2026-10-07.15 — bounded Card 03 pilot and Card 04 CPU fit authority, 7 October 2026

- Recorded the owner's explicit approval of one rights-quarantined Card 03
  two-shard pilot and one Card 04 CPU-only synthetic tokenizer fit in KML-D08/D09
- Added the source-level rights preflight and a native, pinned Corpus project
  with shared response-body accounting, distinct expanded/retained/disk caps,
  review-required source rights and no selected training release view
- Kept PagerDuty acquisition, individual row admission, main tokenizer,
  evaluation-item materialization, Card 04 GPU profile and model training
  outside these approvals; Card 03/04 full gates and Gate 0 remain unverified
- This revision records authority and prepared declarations, not execution;
  the historical Card 02 A1 failure is unchanged

## 2026-10-07.14 — Card 03 and Card 04 owner acceptance, 7 October 2026

- Recorded the owner's clarification that "proceed" accepts a completed,
  reviewable step for incorporation into later cards; future handoffs must
  record that acceptance instead of leaving the step pending review
- Appended [KML-D06](DECISIONS.md#kml-d06--card-03-specification-and-transport-work-accepted-for-reuse-2026-10-07)
  and [C03-R1](results/2026-10-07-card03-acceptance.md), accepting S1-S4
  specification, corrected pilot design and native transport prerequisite;
  reconciled the contract, metadata audit and pilot proposal's current wording
- Appended [KML-D07](DECISIONS.md#kml-d07--card-04-preparation-accepted-for-reuse-2026-10-07)
  and [C04-R1](results/2026-10-07-card04-preparation-acceptance.md), accepting
  P1's shape/config/inventory preparation; earlier results remain unchanged
- Kept source-row rights and acquisition, realized Card 03 artifacts, Card 04
  tokenizer/fit, and Gate 0 open; no download, tokenizer fit, training or GPU
  work was performed by this review, and Card 02 A1 remains FAILED

## 2026-10-07.13 — Card 04 synthetic shape preparation, 7 October 2026

- Pushed rolling `codex/kernel-memory-lab` through S4 to GitHub as requested
- Selected Card 04 for synthetic-only preparation while Card 03 source rights
  and acquisition remain blocked; added two proposed declarations and the
  [bounded profile plan](CARD04_PROFILE_PROPOSAL.md)
- Read-only native inspection recounted 341,885,952 parameters, reported a
  4,102,632,296-byte checkpoint estimate and marked memory fit UNKNOWN;
  inventoried the registered RX 7900 XTX/ROCm environment
- Appended [P1](results/2026-10-07-card04-shape-preparation.md). No tokenizer
  fit, staging, optimizer update, GPU profile, acquisition or Gate 0 promotion

## 2026-10-07.12 — Card 03 native transport prerequisite, 7 October 2026

- Implemented exact pinned Hub-shard metadata selection and an explicitly
  initialized, durable attempt-wide transport ledger in code commits `23db56c`
  and `c77074e`, separately from this research-record update
- Passed 26 focused acquisition tests and 293 corpus tests with local/mock
  inputs; retained the legacy declaration hash and snapshot reuse when the new
  optional fields are absent
- Appended [S4](results/2026-10-07-card03-transport-fix.md) and updated the
  proposal implementation note; no live candidate metadata/shard acquisition,
  rights admission, optimizer updates or GPU work
- Kept the S3 pilot limits unapproved, Card 03 full gate blocked, Card 02 A1
  FAILED and Gate 0 unverified

## 2026-10-07.11 — Card 03 complete-shard pilot proposal, 7 October 2026

- Recorded KML-D05 retaining filtered Project Gutenberg and Wikimedia as
  candidates for a revised plan, without source-acquisition approval
- Proposed the smallest listed complete shard from each pinned component:
  843,172,419 first-transfer bytes, 1,323,415,169 source-body bytes including
  one worst-case retry, plus exact time, disk, row and metadata caps
- Identified the native filename/config/split and aggregate transfer-ledger
  prerequisites in `TODO.md`; the plan excludes Git and keeps PagerDuty for a
  separate bounded source proposal
- Kept ingestion separate from training-token/update budgets; no source bytes,
  tokenizer, optimizer updates or GPU work, and no change to the 341M shape,
  Card 02 A1 failure or unverified Gate 0

## 2026-10-07.10 — Card 03 metadata audit and pilot correction, 7 October 2026

- Renamed the clean local working branch to rolling `codex/kernel-memory-lab`;
  deferred remote pushes under the owner's current instruction
- Pinned candidate PagerDuty commit/tree and filtered Project Gutenberg and
  Wikimedia dataset revisions from metadata; admitted no file or source row
- Found that the smallest proposed Common Pile shards exceed S1's 64 MiB
  download cap and native Git selection does not bound its earlier fetch;
  appended S2 and superseded only the infeasible pilot request
- Recorded the conditional native Git transfer-bound gap in `TODO.md`; retained
  S1's original document/hash, Card 02 A1 failure and unverified Gate 0

## 2026-10-07.9 — Card 03 specification draft, 7 October 2026

- Recorded KML-D04: the owner chose incident-response prose and a rights-audited
  Common Pile subset as source **candidates for specification**, not ingestion
- Added a reviewable source/rights manifest, 65/25/10 proposed language mixture,
  family split and leakage rules, train-only 32,768-entry tokenizer contract,
  separate 200/400-item rubrics and a bounded later source/split pilot request
- Recorded the conditional native mixture-materialization gap in `TODO.md` and
  appended S1; Card 03 specification is READY FOR REVIEW, while full acceptance
  awaits exact source rights/files, acquisition approval and realized evidence
- No source bytes, tokenizer fit, evaluation scores, optimizer updates or runtime
  approval; Card 02 A1 remains FAILED and Gate 0 remains unverified

## 2026-10-07.8 — Card 02 owner review and closure, 7 October 2026

- Recorded KML-D03 accepting A2 for Card 02's tiny-fixture scope and marked
  Card 02 EVIDENCE VERIFIED (done with notes)
- Kept A1 and its 257/200-update failure intact; identified the completed
  aggregate-budget code fix and the separate bounded 89-update confirmation
- Kept Gate 0 unverified pending a provisional data/tokenizer decision and
  hardware inventory; did not start Card 03 or grant runtime authority

## 2026-10-07.7 — bounded Card 02 confirmation completed, 7 October 2026

- Appended A2 with the separate 89-update, approximately 36.4-second CPU
  confirmation receipts and native checkpoint identities
- Marked Card 02 READY FOR REVIEW pending acceptance; original A1 remains
  FAILED for its 57-update cap breach, and Gate 0 remains unverified
- Exhausted KML-D02 runtime authority and did not select Card 03

## 2026-10-07.6 — one Card 02 CPU confirmation approved, 7 October 2026

- Recorded KML-D02 from the owner's approval of the exact 89-update,
  120-second confirmation and authorization to commit and push a branch
- Kept the original 200-update ceiling and 257-update A1 failure unchanged;
  the approved confirmation has a new ledger and no training-bearing retries
- Card 02 and Gate 0 remain unverified pending the new result and review

## 2026-10-07.5 — Card 02 budget repair proposal, 7 October 2026

- Clarified that Card 02 update and time caps cover the whole approved attempt,
  including every test, retry and resumed run, and that failed reservations stay
  charged
- Added a persistent per-attempt budget ledger and bounded command wrapper,
  plus a proposed 89-update, 120-second CPU-only confirmation
- Preserved the original 257-update Card 02 attempt as FAILED and Gate 0 as
  unverified; this revision grants no confirmation runtime authority

## 2026-10-07.4 — Card 01 acceptance and Card 02 failed attempt, 7 October 2026

- Recorded the owner's Card 01 contract acceptance in `DECISIONS.md` and marked only its documentation scope EVIDENCE VERIFIED
- Added fresh, bounded CPU fixture tests and a Card 02 result with actual source, tokenizer, run and checkpoint identities
- Preserved the passed functional observations and the cumulative optimizer-step cap breach (257 against 200); Card 02 is FAILED pending review, and Gate 0 remains unverified
- No scientific threshold, source acquisition, GPU run, Campaign transition or later card was authorized or completed

## 2026-10-07.3 — Card 01 foundation documentation, 7 October 2026

- Reconciled local `main` at `cf29aff79ec7259f7ec93988cd6a87065f4350e5` with the pinned source audit; intervening tracked changes are guidance and planning documents
- Added a bounded foundation contract for fresh fixtures, a separately initialized main model, strict same-experiment resume and optional dense-to-reader transfer
- Recorded unresolved inputs and a pending Card 01 result; updated the single checklist without claiming Gate 0, a runtime receipt or reviewer acceptance
- Left the source-release DOCX snapshots historical; no theory threshold, card order or runtime authority changed

## 2026-10-07.2 — repository adoption proposal, 7 October 2026

- Added the complete Markdown research plan and thirteen-card workbook under the existing research namespace, with explicit historical-delivery banners
- Updated the entrypoint/status for repository use; preserved one current checklist, native Campaign authority and external mutable outputs
- Preserved existing root guidance with a minimal scoped bootstrap pointer and registered the planned question in the research roadmap
- Added the future calibrated-expert training track and rights-aware source/generator catalog; no generators, ingestion or training implemented
- Kept DOCX snapshots and the ZIP in Library because repository research lint permits reviewed text records, not those binary payloads
- Removed the private architecture-discussion download link from the public plan; retained its hypothesis-source role
- Converted package checksums to a Markdown provenance record; all source-package checksums were verified before adaptation
- No scientific acceptance gate, foundation architecture or execution-card order changed; no runtime authority granted

## 2026-10-07.1 — 7 October 2026

- Refreshed the TinySparseLab baseline from `c54b17a59cc59a81e51949ff699ab859e707fc28` to `06efc4db82ecf3da97b50cff518cba605ad27b33`
- Converted newly shipped infrastructure work into reuse and verification steps while preserving the unexecuted project cards
- Retained the full fresh-kernel rationale, comprehension/evidence focus, controls, thirteen-card backlog and all resource/approval boundaries
- Added official Codex CLI/WSL2 startup, instruction-discovery and resume guidance
- Delivered faithful Markdown versions of both updated editable DOCX references plus a single operating record and templates
- No repository edits, PR, installation, corpus acquisition, GPU run, research evaluation or cloud spending by this document task

## Future change protocol

1. Name the theory/card/status change and cite the evidence or explicit owner decision.
2. Reconcile edits from either DOCX into the Markdown operating record. Do not keep independently diverging backlogs.
3. Review changes to hypotheses, acceptance thresholds, resource bounds or data scope before using them. A documentation edit does not grant runtime authority.
4. Add a new revision here. Refresh human DOCX references from that same accepted revision when delivering an update.
5. Record source-release checksums in `PACKAGE_PROVENANCE.md`; use Git for the repository edition. Generate a new checksum manifest outside the research tree when packaging a new ZIP. Do not treat source-release checksums as current after editing files.

Privacy adaptation: personal cloud-credit balance was removed; technical hardware targets and explicit resource-approval gates are preserved. The plan/workbook therefore have documented narrow redactions as well as their adoption banners.
