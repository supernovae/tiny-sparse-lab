# Decision log

Append entries; do not rewrite earlier decisions. Correct a decision with a new entry citing it. No experimental go/no-go decision or runtime approval is recorded at delivery.

## KML-D11 — one Card 04 replacement GPU attempt approved, 2026-10-07

- Owner decision: After the first GPU attempt [P3](results/2026-10-07-card04-gpu-profile-interrupted.md) and its exact [replacement proposal](CARD04_RETRY_PROPOSAL.md), Byron asked to **“re-try the GPU attempt”** and check whether it works. This approved one fresh attempt under that proposal's existing bounds; it did not revive KML-D10's spent attempt or authorize a series of retries.
- Fixed scope and allocation: unchanged synthetic Card 04 config SHA-256 `0a7a04740b822d0f4149bc9ae287981a3c3687667ed7e8ba62e3b4e310b5ea49` and tokenizer SHA-256 `f6df298e90e2aca41b4ba66d57e138a96bf864c3d61860972e1bedfbbb778e2e`; one local RX 7900 XTX BF16/reference-dense attempt, 120 aggregate updates (7 stage plus conditionally 113 profile), one 1,800-second deadline, at most 122,880 target positions, 20 GiB VRAM, 24 GiB process-tree RSS, 20 GiB additional disk, 1,000 inodes, zero cloud use/spend, no retry or resume.
- Outcome: [P4](results/2026-10-07-card04-gpu-retry-preflight-failed.md) stopped at its first device-memory preflight before native staging. Its new ledger conservatively charged seven reserved stage updates, with zero observed optimizer updates and no stage/profile receipt. The unallocated 113 updates are not an approved retry. P3 remains interrupted and unchanged; Card 04 measured fit and Gate 0 remain unverified.
- Next decision: A third attempt needs the new exact [monitor-repair allocation](CARD04_RETRY2_PROPOSAL.md) and explicit owner approval after reviewing P4. No Card 03 content acquisition or Card 05 training is authorized here.

## KML-D10 — Card 04 bounded local GPU profile approved, 2026-10-07

- Owner decision: After [P2](results/2026-10-07-card04-synthetic-tokenizer.md) produced and verified the exact 32,768-entry synthetic tokenizer, Byron answered **“Approve this bounded GPU profile”** to a question naming the [Card 04 profile proposal](CARD04_PROFILE_PROPOSAL.md) and its exact limits.
- Fixed input and scope: one local RX 7900 XTX `rocm-7900xtx` BF16/reference-dense profile of [card04-synthetic-profile.yaml](card04-synthetic-profile.yaml) SHA-256 `0a7a04740b822d0f4149bc9ae287981a3c3687667ed7e8ba62e3b4e310b5ea49`, with synthetic tokenizer SHA-256 `f6df298e90e2aca41b4ba66d57e138a96bf864c3d61860972e1bedfbbb778e2e`. Fresh random initialization only; no earlier weights or optimizer state.
- Exact aggregate ceilings: **120 optimizer updates**, comprising two native stage pilots of 2 smoke and 5 warmup updates plus at most 113 fresh profile updates; one **1,800-second** deadline from one ledger; at most **122,880 target positions**, **20 GiB device memory**, **24 GiB process-tree RSS**, **20 GiB added disk** and **1,000 inodes**. Stop on the first limit, nonfinite numerical result or OOM. No retry/reset, no cloud use or spend. Profile timing uses its last 100 updates after the first 13 warmup updates; stage pilots are distinct initializations.
- Prerequisites: fixed clean source checkout, registered ROCm interpreter, native `inspect`/workspace preflight, verified tokenizer and prepared inputs, free-space/inode margin, explicit deadline/update ledger and resource monitor. If these do not pass, do not launch the dependent phase. Preserve failed receipts.
- Boundary: this approves synthetic fit evidence only. It does not admit Card 03 source rows, provide real-data throughput, authorize Card 05 language training or complete Card 04's measured-fit gate before receipts and review.

## KML-D09 — Card 04 CPU synthetic tokenizer fit approved, 2026-10-07

- Owner decision: In response to an explicit question naming this as a separately reserved runtime step, Byron answered **“Yes, approve those two bounded steps.”** This approves one Card 04 native CPU synthetic tokenizer fit under the unchanged [proposal](CARD04_PROFILE_PROPOSAL.md), separately from the Card 03 acquisition in KML-D08.
- Exact limit: `card04-synthetic-tokenizer.yaml` SHA-256 `fbd8497a7a09ec961bddaa76a3f78523cb123bab42757623b48fb940502e3803`; at most **600 seconds** from process launch, **1 GiB additional disk**, **1,000 inodes**, **50,000 train documents** and declared **5,000,000 train-input bytes**; **zero optimizer updates, zero GPU/cloud work and zero spend**. No retry under a reset deadline. Keep its source, output and cache under the declared external task workspace; stop at the first cap or identity failure.
- Prerequisites: fresh storage/inode preflight, unchanged declaration digest, no existing output, deadline and disk monitoring. Verify the resulting tokenizer manifest, exact 32,768 vocabulary, special IDs, source provenance and round-trip behavior; a mismatch blocks Card 04 fit rather than changing its architecture.
- Boundary: this does not approve the later GPU profile, staging pilots or any model optimizer work. Card 04's full measured-fit gate and Gate 0 remain unverified until their own evidence review.

## KML-D08 — Card 03 rights-quarantined two-shard pilot approved, 2026-10-07

- Owner decision: In response to an explicit question naming the earlier separately withheld acquisition, Byron answered **“Yes, approve those two bounded steps.”** This approves one rights-quarantined Card 03 pilot under the accepted [S3 design](CARD03_PILOT_PROPOSAL.md) and [source-level rights preflight](CARD03_SOURCE_RIGHTS_PREFLIGHT.md), subject to fresh native identity/storage checks.
- Exact limit: only the pinned Project Gutenberg `project_gutenberg-dolma-0014.json.gz` and Wikimedia `wikimedia-0027.json.gz` files in [the native declaration](corpus-pilot/corpus.yaml); **843,172,419** bytes for the first two full transfers, **1,323,415,169** charged source-body bytes including at most one interruption-only retry, **1,048,576** metadata-body bytes, **1,324,463,745** combined response-body bytes, **2,700 seconds**, **5 GiB additional disk**, **1,000 inodes**, at most **4,096 scanned rows and 32 retained rows/32 MiB per shard**, plus a stricter **2 GiB expanded-stream cap per shard**. CPU only; **zero model target tokens, optimizer updates, GPU/cloud jobs and spend**. One durable ledger covers failures, redirects, retries and resumes; do not reset it.
- Prerequisites: source-level terms must support the restricted review, exact revision/path/size/hash/config/split metadata and storage margin must verify, and local native transport tests must remain valid. Stop and preserve receipts on the first failed check or cap. Acquired rows remain `review_required`, with no selected release view; record-level rights review is still required before any tokenizer/training/evaluation use.
- Boundary: this pilot does not approve PagerDuty acquisition, a full corpus build, main-tokenizer fit, frozen evaluation materialization or model training. Its sample alone cannot satisfy the full Card 03 gate or Gate 0.

## KML-D07 — Card 04 preparation accepted for reuse, 2026-10-07

- Owner review: Byron clarified that saying "proceed" after a reviewable card step means the completed step is accepted/admitted and may be incorporated into subsequent work. His earlier request to proceed to Card 04 and this clarification accept [P1](results/2026-10-07-card04-shape-preparation.md) for its **synthetic shape and inventory preparation** scope; see the [append-only review](results/2026-10-07-card04-preparation-acceptance.md).
- Accepted evidence: the two pinned synthetic proposal declarations; native read-only count of **341,885,952** parameters; **4,102,632,296-byte** estimated checkpoint; WSL2/RX 7900 XTX/registered ROCm inventory. The profile may use these as reviewed candidate inputs, subject to verifying their identities at execution.
- Gate boundary: this is **EVIDENCE VERIFIED for P1 preparation**, not full Card 04 measured fit. No tokenizer output, initializer receipt, finite-loss/gradient test, VRAM/RSS measurement, throughput or profile run exists. The full fit gate stays BLOCKED.
- Authorization: accepting P1 does not start or approve the separately proposed CPU tokenizer fit, stage pilots, GPU profile, optimizer updates, cloud use or spending. Its 600-second preparation and 120-update/1,800-second profile allocations remain proposed operational ceilings, not spent or executable approval.
- Consequence: carry the fixed 341M architecture and synthetic profile design forward; the smallest next dependent task remains the separately bounded fresh synthetic tokenizer fit. Card 03's rights/acquisition gate and Gate 0 remain open; Card 02 A1 remains FAILED.

## KML-D06 — Card 03 specification and transport work accepted for reuse, 2026-10-07

- Owner review: Byron clarified that "proceed" accepts the completed card step for incorporation into later work. His prior progression through Card 03 and this clarification accept [S1](results/2026-10-07-card03-specification.md), [S2](results/2026-10-07-card03-metadata-audit.md), [S3](results/2026-10-07-card03-pilot-proposal.md) and [S4](results/2026-10-07-card03-transport-fix.md) for their **specification, metadata, bounded pilot design and native code-prerequisite** scopes; see the [append-only review](results/2026-10-07-card03-acceptance.md).
- Accepted decisions: incident-response prose remains the evidence-domain candidate; the pinned Project Gutenberg filtered and Wikimedia filtered complete shards are admitted as **candidates for a rights-quarantined pilot design**. S2 supersedes S1's infeasible 64 MiB pilot request. S3's exact shard identities and limits are the reviewed planning baseline, and S4's tested transport fix is the native prerequisite. These may be incorporated into later card design.
- Gate boundary: no candidate file or record is admitted as training/evaluation material. Source-level and record-level rights checks, live acquisition, content hashes, realized splits, tokenizer and frozen evaluation items are absent. The full Card 03 data gate stays BLOCKED. This decision does not revise S1-S4 historical results or claim their unexecuted steps occurred.
- Authorization: earlier explicit statements withholding **acquisition approval** remain in force. Acceptance of the S3 design does not authorize its 1,324,463,745-byte combined response-body, 2,700-second, 5 GiB pilot, tokenizer fitting, training, GPU/cloud work or spending. A future acquisition decision must name its exact bounds after source-rights and live preflight checks.
- Consequence: use the reviewed contract and native path; retain rights quarantine and the separate ingestion-versus-training budgets. Card 02 A1 remains FAILED and Gate 0 remains unverified.

## KML-D05 — retain two Common Pile candidates for pilot planning, 2026-10-07

- Question: Should Card 03 continue planning around the pinned Project Gutenberg and Wikimedia filtered components after S2 showed the 64 MiB pilot was infeasible?
- Owner decision: Byron recommended keeping **both** components as candidates and requested the smallest practical rights and bounded-acquisition plan using verified shard sizes.
- Evidence reviewed: [S2 metadata audit](results/2026-10-07-card03-metadata-audit.md) and its pinned component revisions, file sizes and native transport blockers.
- Scope: Select these two components for **planning only**. This does not admit any record or authorize a source download, tokenizer fit, model training, GPU/cloud work, spending or a Campaign transition.
- Consequence: Replace S1's infeasible pilot request with a separately reviewable [Card 03 pilot proposal](CARD03_PILOT_PROPOSAL.md) covering complete, pinned shards and an aggregate transfer budget. Keep PagerDuty as the domain candidate but outside this first Common Pile acquisition pilot.
- Unchanged: Card 02's failed A1 and accepted A2, Gate 0's unverified status, and the proposed 341,885,952-parameter architecture. A source-ingestion allowance is separate from any later training-token/update budget.
- Revisit trigger: File/record rights cannot be cleared, a shard identity changes, a bounded native transport cannot be established, or the owner chooses another source.

## KML-D04 — Card 03 source candidates selected for specification, 2026-10-07

- Question: Which first evidence domain and base-language source should the Card 03 specification cover?
- Owner choice: Byron selected “Incident-response prose + Common Pile (recommended)” in response to a question that explicitly limited the choice to a specification candidate.
- Interpretation: Draft Card 03 around incident-response documentation, with a rights-audited Common Pile subset as the base-language candidate. PagerDuty's incident-response documentation is a proposed domain source, not an admitted corpus.
- Evidence and uncertainty: [The source catalog](SOURCE_CATALOG.md) identifies these leads and their rights gates; exact Common Pile components/files, immutable revisions, per-item rights, PagerDuty file exclusions, and content digests are unresolved.
- Authorization: Specification work only. This choice does not approve acquisition, tokenizer fitting, evaluation-item materialization, model training, GPU/cloud work or spending.
- Consequence: Prepare the [Card 03 contract](CARD03_DATA_CONTRACT.md) for review and leave its source-admission and acquisition gate open. Card 02's accepted tiny-fixture result and failed A1 attempt remain unchanged; Gate 0 remains unverified.
- Revisit trigger: The owner chooses a different domain/source or the rights audit disqualifies a candidate.

## KML-D03 — Card 02 accepted with failed-attempt note, 2026-10-07

- Question: Does the separate bounded Card 02 confirmation satisfy the tiny-fixture gate despite the preserved first attempt's resource failure?
- Evidence reviewed: [A1 failed attempt](results/2026-10-07-card02-tiny-fixtures.md) SHA-256 `ab1390669ab673055057e576bbdac1cd1d97bb2cf233bc9b45f4c15e2610b4c1`; [A2 confirmation](results/2026-10-07-card02-confirmation.md) SHA-256 `f81540d63a07c6b6b74da076e7b73291fd1aa6d47ef34a9049bbaf7bd869a05a`; retained A2 ledger JSON SHA-256 `7adebf61338c3459e97579636624a229ad91cd96fa1f95ffe2c93e4e47e5980f`. A2 selected tests passed 3/3 plus 1/1, charged exactly 89/89 updates, ended with 83.605 seconds remaining, and verified the selected native generations. The original A1 result and its artifacts remain unchanged.
- Owner review and decision: Byron stated “card 02 has been approved/reviewed” and requested Card 02 be marked done with notes. Accept **A2** as Card 02 EVIDENCE VERIFIED for its tiny CPU fixture scope. Keep **A1 FAILED**: its 257 cumulative updates exceeded its actual 200-update approval by 57, and later success does not erase that breach.
- Code-fix note: the observed A1 failure was missing aggregate attempt accounting across test reruns, the native full/parent/resumed runs, and the separate sidecar test. Commit `77ec6e73ff84b523001c974aaa37a8dbdacfca6f` added the persistent shared budget ledger, up-front reservations and deadline wrapper; six zero-training unit tests passed. A2 confirmed the fix operationally. No further Card 02 code defect is established by these receipts.
- Boundary: acceptance covers synthetic label, gradient, overfit, semantic-control and strict-resume plumbing. It does not establish language comprehension, factual retrieval quality, main-tokenizer readiness, GPU fit, or the full Gate 0. Gate 0 still needs its provisional data/tokenizer decision and hardware inventory under the original criteria.
- Authorization: this is a review decision, not a new training, acquisition, GPU, cloud, Campaign or later-card runtime approval. KML-D02's 89-update confirmation allocation is exhausted.
- Consequence: mark Card 02 EVIDENCE VERIFIED (done with A1 failure noted); retain both append-only results and all external artifacts. Card 03 can be scoped separately around its domain and source/license prerequisites.
- Revisit trigger: new evidence contradicts the A2 identities or checks, or a specific residual code defect is reproduced.

## KML-D02 — bounded Card 02 confirmation approved, 2026-10-07

- Question: May one fresh CPU confirmation run test the repaired aggregate budget accounting after failed attempt `KML-20261007-C02-A1`?
- Evidence inspected: [failed A1 record](results/2026-10-07-card02-tiny-fixtures.md), [budget repair proposal](results/2026-10-07-card02-budget-repair.md), [exact confirmation plan](CARD02_CONFIRMATION.md), and six passing zero-training budget unit tests. The A1 total of 257 updates exceeded its actual 200-update approval cap; the cap was not a per-command allowance or a mistaken calculation.
- Decision: The owner replied “i approve this update” to the request to approve the exact 89-update, 120-second CPU confirmation, and authorized committing and pushing a branch. Approve that bounded confirmation only. Preserve A1 as FAILED; do not retroactively raise its cap.
- Authorized scope and bounds: one fresh external-root Card 02 CPU confirmation under `CARD02_CONFIRMATION.md`; at most 89 optimizer updates and 120 seconds from ledger creation, at most 1 MiB generated text and 64 selected cases, with no external data, GPU, cloud spend or Campaign apply. One ledger covers all selected tests, retries and resumed runs. No training-bearing retry is allocated.
- Owner and reviewer: Byron, project owner; confirmation result review remains pending.
- Consequences: Commit and push the reviewable branch, then run only this confirmation. A passing run may become READY FOR REVIEW in a new append-only result; Card 02 is not EVIDENCE VERIFIED and Gate 0 stays unverified until review and its own requirements are met. Later cards need separate approval.
- Revisit trigger: Any failed check, exhausted budget, expired deadline, source mismatch or proposed scope change stops further runtime and requires a new decision.

## KML-D01 — foundation contract accepted, 2026-10-07

- Question: Does the documentation-only Card 01 foundation contract satisfy its reviewed scope?
- Evidence inspected: [foundation contract](FOUNDATION.md) SHA-256 `df57d600f99a8d16fb20dcc50caef6814718b4ba39132712ac3b764a769e178d`; [Card 01 result](results/2026-10-07-card01-foundation.md) SHA-256 `253a46ded454176da494af7cffda63213c7c06659084849b526198461b7270fa`; checkout `cf29aff79ec7259f7ec93988cd6a87065f4350e5` with the recorded uncommitted documentation diff.
- Options considered: accept the bounded contract or request a correction.
- Decision: The project owner stated “i accept contract, please proceed” in this conversation. Accept Card 01's documentation contract as reviewed evidence. This does not complete Gate 0 or verify any runtime artifact.
- Reason and uncertainty: The contract separates fixture, main, strict resume and optional transfer identities and names missing dependent inputs. No project fixture or native run existed at acceptance.
- Owner and reviewer: Byron, project owner; acceptance recorded by Codex.
- Authorized scope and bounds: “please proceed” selects Card 02 only, under the workbook's tiny CPU limits: at most 1 MiB generated text, 64 cases, 200 optimizer steps and 20 CPU minutes. No external acquisition, GPU/cloud operation or later card is authorized.
- Consequences for dependent cards: Card 02 may begin; Cards 03–13 retain their own prerequisites and approvals.
- Revisit trigger: A specific reviewer correction to Card 01 or an artifact contradicting the starting-state record.

## Document release 2026-10-07.1

- Date: 2026-10-07
- Decision recorded: synchronized document delivery only
- Scope: refreshed source baseline, faithful Markdown operating pack, WSL2/Codex guidance
- Research implementation or runtime authorized: NONE
- Evidence: `SOURCE_AUDIT.md`, `REVISION_LOG.md`, complete plan and workbook
- Next step: read-only reconciliation in the actual project checkout

## Template for a future reviewed decision

- Decision ID and UTC timestamp: [actual values]
- Question: [one bounded question]
- Evidence/results/receipts inspected: [real IDs and paths or links]
- Options: [bounded alternatives]
- Decision: [continue / revise / stop / defer]
- Reason, limitations and uncertainty: [supported explanation]
- Owner and reviewer: [actual roles/names]
- Exact authorization, if any: [current approval reference and bounds; otherwise NONE]
- Effects on dependent cards: [specific changes]
- Revisit trigger: [new evidence or condition]
- Correction to earlier decision: [ID and reason, if any]
