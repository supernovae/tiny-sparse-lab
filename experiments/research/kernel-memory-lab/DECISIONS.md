# Decision log

Append entries; do not rewrite earlier decisions. Correct a decision with a new entry citing it. No experimental go/no-go decision or runtime approval is recorded at delivery.

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
