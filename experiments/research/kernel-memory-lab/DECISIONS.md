# Decision log

Append entries; do not rewrite earlier decisions. Correct a decision with a new entry citing it. No experimental go/no-go decision or runtime approval is recorded at delivery.

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
