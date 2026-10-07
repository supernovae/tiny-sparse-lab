# Kernel Memory Lab Execution Workbook

> Repository adoption note (2026-10-07.2): the full foundation guide below is retained from release 2026-10-07.1. Statements about extracting the pack or no PR refer to that historical delivery. Use [README](README.md), [BOOTSTRAP](BOOTSTRAP.md) and [STATUS](STATUS.md) for current repository context. The later [expert track](EXPERT_TRACK.md) is an additional proposal; foundation gates and all thirteen unstarted cards remain unchanged.

Fresh language kernel checklist and copy ready GPT prompts
Working title • Prepared for Byron • 7 October 2026
Revision 2026-10-07.1

### Start here

Use this workbook to turn the research plan into one reviewable task at a time. The goal is a fresh kernel that learns comprehension, nuance and evidence use, with new facts supplied by external memory. Research and operating documents are authored. Every project-specific implementation and runtime card remains NOT STARTED; the audited repository already supplies useful infrastructure.

1. For Codex CLI on the WSL2 GPU PC, extract the supplementary Markdown pack into the reviewed experiment namespace. Read the new CLI section below. The DOCX files are editable human references; CLI context comes from explicit file reads.

2. Start from the repository root and read its existing AGENTS.md plus the pack’s BOOTSTRAP.md. Supply the actual path and current revision. Use the master context and first read-only assessment; no prior checkpoint or corpus is a prerequisite.

3. The first response should be an evidence-backed gap assessment and one proposed foundation change. Paste that response, its verified revision and artifact links back to the reviewing GPT or dot. Do not paste a claim that tests passed without the receipt.

4. After review, select exactly one numbered task card. Paste master context, the current handoff envelope, that card’s prompt and the required artifacts. An unverified or missing dependency blocks only that dependent task.

5. After each task, append its result and decision to the record. Advance only after the acceptance gate is independently checked. A recommendation, campaign next result or completed prompt is not permission to run the next stage.

### First prompt to paste into Luna

Use the fresh-kernel plan and workbook, or their synchronized Markdown files. Perform one read-only assessment of the actual TinySparseLab checkout. Compare it with the 7 October audit 06efc4db82ecf3da97b50cff518cba605ad27b33. Read existing AGENTS.md and verify the experiments/research/<campaign> convention plus external runtime root. Identify shipped configuration derivation, sealed staging, state-digest API, semantic probes/sidecar batches, overlap analysis, preparation benchmark, attention and campaign contracts. Separate reuse from genuinely missing reader/transfer work. Return the real revision, evidence-backed gaps, unresolved inputs and one smallest foundation change with proposed tests. Do not edit, install, acquire data, run GPU work or invoke campaign apply/resume. If access is missing, name it. Stop after assessment.

What to paste back: the complete assessment, repository revision, verified file/command references, proposed change and unresolved inputs. The review should decide whether Card 01 is ready or needs a smaller scope. No run results are expected yet.

## Use the operating pack in Codex CLI

The practical handoff is a directory Codex can read, rather than reattaching Word files to each conversation. This delivery synchronizes both editable DOCX references and their full Markdown counterparts at revision 2026-10-07.1. Download and extract Kernel_Memory_Lab_agent_pack.zip. The instructions below are for you to apply; this task made no repository, installation or system changes.

### Choose the working location

Use the WSL2 distribution containing the project’s Linux Python/GPU environment. Prefer a Linux path such as ~/code/tiny-sparse-lab. After reviewing the repository’s existing guidance, place the pack contents at experiments/research/kernel-memory-lab. That path is proposed, not detected on your PC. Keep configs, fixtures and results in the corresponding experiment namespace shown in the artifact layout. Do not copy an unrelated experiment or its weights to get started.

In the WSL shell, echo "$WSL_DISTRO_NAME", pwd and command -v codex show the distribution, current location and selected executable. They do not verify GPU readiness. If Codex is absent, follow the official WSL setup page separately; the documents do not authorize an installation.

### Launch a read only first session

After substituting your real repository path, use:
codex --cd "$HOME/code/tiny-sparse-lab" --sandbox read-only --ask-for-approval never "Read AGENTS.md and experiments/research/kernel-memory-lab/BOOTSTRAP.md. Reconcile the project state without edits, installs, downloads, GPU work or campaign apply. Report missing evidence and one next task for approval."

These are current official CLI flags. The initial shell sandbox is read-only. The never approval setting disables escalation requests. The explicit task also forbids mutating tools; a sandbox flag is not a scientific resource budget. If a read is blocked, report the missing access. For later approved work, choose an appropriate scoped write/approval mode in a separate launch; do not weaken the initial mode to make an unapproved action succeed.

### Read the complete context once and the live record every time

BOOTSTRAP.md names the required files explicitly. Read the complete research plan on the first pass, its concise invariants on each later pass, current STATUS.md, recent decisions/results and exactly one selected card. CONTEXT.md does not replace the complete theory. SOURCE_AUDIT.md separates shipped infrastructure from missing research mechanisms. Keep the brief short enough to inspect and require targeted reads of large logs.

Preserve root AGENTS.md. AGENTS_POINTER_PROPOSAL.md is only a reviewed snippet for a later authorized edit. Codex discovers instruction files from the project root to its launch directory; a nested AGENTS file is not automatically active from a repository-root launch. Explicit bootstrap reads work without replacing instructions. Restart after accepted instruction-file changes. The documented default combined instruction limit is 32 KiB.

Official references: the plan’s Sources section links the CLI command reference, WSL guidance, AGENTS discovery and Windows desktop environment setting. The desktop app’s Agent environment → Windows Subsystem for Linux setting requires restart; its terminal-shell preference is a separate setting.

## Keep one record across agents and sessions

### Authority and synchronization

After you adopt the pack, RESEARCH_PLAN.md and EXECUTION_WORKBOOK.md define the accepted theory and cards; STATUS.md is the one current checklist. Native Campaign manifests, run IDs, receipts and immutable artifacts establish execution facts. DECISIONS.md records reviewed changes. Each results entry preserves the actual command, hashes, tests and blockers. No new task tracker or native campaign schema is being invented.

The editable DOCX files are synchronized reference snapshots. If you change either in Word, reconcile that change into the Markdown files with review and a REVISION_LOG.md entry before continuing. Refresh the DOCX references from the same revision. Never let an agent silently prefer stale prose over actual receipts, or promote a proposal to a completed result.

### Resume safely

From the same repository, codex resume opens saved sessions; codex resume --last selects the most recent matching session. Prefer a known session ID when several projects or sessions could be confused. Repeat the desired sandbox and approval flags when resuming. Begin by reconciling file state even if the old conversation is available. A fresh session uses BOOTSTRAP.md and can reach the same evidence-backed state.

### Close a bounded card

Return the result first. Include repository revision and diff, exact input/output hashes, native campaign/run/receipt IDs, executed tests and metrics with denominators, failed attempts, resource usage, limitations, gate verdict and one next decision. Within the task’s authorized write scope, append a result entry and update STATUS.md to point to it. Leave reviewer approval pending until a real review occurs. A read-only session reports a proposed record update instead of writing it.

### Choose the right review

Use Sol for implementation correctness: initialization/loading, tensor mapping, optimizer and gradient membership, numerics, device or memory defects. Bring a minimal reproducible case, revision, config, logs and expected behavior. Use Astra for an unresolved research decision after plumbing is checked: oracle/retrieval/fusion gaps, ambiguous ablations, objective design or competing mechanisms. Bring matched controls, error categories, uncertainty and cost. Neither agent can approve runtime or spend.

### Stop instead of filling an evidence gap

Missing checkpoint, wrong tokenizer, stale store manifest or unsealed prepared inputs means stop the dependent task and name the gap. Do not retrain, fetch replacement data, invent hashes, open a PR, dispatch to Colab or advance campaign apply. Shipped config and experiment derivation, staging, semantic probes and prep benchmarks can reduce engineering work, but they cannot manufacture this fresh project’s artifacts or prove the proposed reader works.

## Master context for each new session

Copy the following context unchanged unless the project owner approves a documented revision. The working title is a convenience, not a name the user has committed to.

Project: Kernel Memory Lab, a new experiment using TinySparseLab tooling. Goal: a compact kernel that learns language comprehension, nuance, contextual interpretation, instruction following and evidence use. External memory should supply useful new and versioned facts without requiring those facts to be trained into the weights. Script writing is not the primary objective.

Start fresh. Do not depend on MODEL0 or MODEL-0, its checkpoint, tokenizer, corpus, optimizer, runtime approval or reproduction. Use newly generated tiny synthetic fixtures for plumbing and a separately randomly initialized main core. Proposed native main shape: vocabulary 32,768, width 1,024, 24 layers, 16 attention heads, feed forward width 2,816, tied embeddings, 341,885,952 parameters under the pinned native implementation. Recount if the configuration changes. Tiny fixtures are disposable; their weights never initialize the main core.

Use the 7 October audit 06efc4db82ecf3da97b50cff518cba605ad27b33 as a pinned source and inspect the actual current revision. Reference dense attention materializes scores and uses float32 softmax; opt-in native SDPA now exists, but actual backend and fit are unverified. Local first in WSL2 on the user-reported 7900 XTX with 24 GB VRAM. Host RAM, storage, ROCm compatibility and throughput are unverified. Optional cloud availability, usable hours and provider burn rate require verification. There is no automatic cloud spending.

Keep fresh initialization, strict same-experiment resume and explicit same-project dense-to-augmented transfer distinct. Language learning, reader/fusion learning and router learning are separate. Establish raw-text oracle and lexical controls before integrated fusion, learned routing or virtual PKM. Demonstrate useful reading in RAM before comparing the same store on NVMe. Do not freeze a random core and call it a capable reader.

Do exactly the supplied task card within its approved scope. No unapproved large data download, GPU training, paid job, deployment or unrelated repository rewrite. Treat campaign apply as mutating even without execute-runs; it can prepare or pilot work. Campaign status, next and explain are observations, not approvals. Never fabricate lock hashes, invent receipts, auto-approve a gate, silently substitute artifacts or retrain a missing checkpoint.

Read the canonical Markdown record under the adopted experiments/research/kernel-memory-lab path, then return a short result, exact changed artifacts, revision and hashes, native run/receipt IDs, tests, gate assessment and one next decision. Append the result and update STATUS.md only within the authorized write scope. Mark unsupported claims UNVERIFIED. Stop when the bounded task is complete, blocked or reaches its stop condition; do not chain into another card.

### Roles and review

Luna operates a well-specified task and assembles evidence. Sol reviews code, loading, gradients, optimizer groups, numerics and device behavior. Astra reviews competing explanations or research design after correctness is established. Byron approves scope, data, costs and runtime bounds. Model names designate roles; a role does not confer approval authority.

## Stage checklist and current record

These statuses describe this project at revision 2026-10-07.1, not the whole repository. The cards remain NOT STARTED even where native infrastructure is already shipped. After adoption, STATUS.md is the single current checklist; this table is its delivery snapshot. EVIDENCE VERIFIED requires reviewed receipts and acceptance criteria. Technical status and approval remain separate.

| Card | Deliverable | Depends on | Initial status |
| --- | --- | --- | --- |
| 01 | Foundation change and experiment contract | First assessment | NOT STARTED |
| 02 | New tiny correctness fixtures | 01 | NOT STARTED |
| 03 | Data tokenizer and evaluation splits | 01 | NOT STARTED |
| 04 | Native configuration and fit calibration | 02 and 03 for real data | NOT STARTED |
| 05 | One budgeted language training tranche | 03 and measured 04 | NOT STARTED |
| 06 | Raw text oracle evidence test | 05 capable checkpoint | NOT STARTED |
| 07 | Lexical retrieval baseline | 06 | NOT STARTED |
| 08 | Dense to augmented transfer contract | 02 and selected 06 checkpoint | NOT STARTED |
| 09 | Integrated reader fusion test | 06 07 and 08 | NOT STARTED |
| 10 | Optional learned router or PKM test | 07 09 and measured bottleneck | NOT STARTED |
| 11 | Frozen weight knowledge update test | Useful 07 or 09 reader | NOT STARTED |
| 12 | Same store RAM and NVMe comparison | Useful 07 or 09 and same store | NOT STARTED |
| 13 | Review and next go or no go | Relevant completed evidence | NOT STARTED |

### What is actually complete

Research documents and the operating pack are authored at this synchronized revision. A fresh source-only audit pins main 06efc4db82ecf3da97b50cff518cba605ad27b33, including PR49. Typed derivation, staged-input reuse, digest comparison, semantic probes, overlap analysis, preparation benchmarking, native SDPA and hosted relay contracts are shipped infrastructure. They do not complete this project’s cards. No repository test execution, fresh configuration, fixture, data/tokenizer, trained checkpoint or memory result was produced by this document task.

### Status vocabulary

NOT STARTED → IN PROGRESS → READY FOR REVIEW → EVIDENCE VERIFIED. BLOCKED records the missing input and owner; FAILED records evidence and the bounded next diagnosis. SKIPPED requires a reason and reviewer decision. User approval is recorded separately from technical status. Editing a status never manufactures approval.

## Handoff envelope and artifact discipline

Attach or link the actual files required for the chosen card. Replace bracketed fields only with observed information. These are template fields, not artifacts that already exist.

Task card and exact subtask: [one card and one bounded action]
Requested mode: [read-only assessment / code change / approved runtime]
Repository location and observed revision: [verified value]
Current branch or diff: [verified value or none]
Inputs with purpose, location and hash: [actual artifact list]
Relevant previous receipt IDs: [verified receipts]
Approved data sources and maximum bytes: [approval or NOT APPROVED]
Runtime authority: [exact user approval and date or NOT APPROVED]
Bounds: [device, steps, target tokens, wall time, memory, spend/credits; first cap stops]
Expected outputs and acceptance gate: [from selected card]
Unresolved inputs and blockers: [explicit list]
Return destination: [artifact directory or review conversation]
Do not infer missing permissions from previous task completion.

### Proposed artifact layout

The fresh audit confirms the repository’s experiment namespace convention. The locations below are proposals for this new project, not files already installed. Card 01 must confirm the exact name, existing instructions and accepted placement. Use one operating record under experiments/research/kernel-memory-lab; preserve the root AGENTS.md and all unrelated experiment folders.

| Proposed location | Purpose |
| --- | --- |
| experiments/research/kernel-memory-lab/ | Proposed canonical Markdown protocol, cards, status pointer and small reviewed decisions/results |
| experiments/research/kernel-memory-lab/configs/ | Proposed native declarations; confirm exact local naming and resolution before authoring |
| Campaign namespace or scoped test fixtures | Fresh tiny generators/tests only; follow the existing test and fixture conventions |
| <work-root>/experiments/kernel-memory-lab/ | Native mutable/large outputs, manifests, logs, checkpoints and staged bundles; resolve real paths |
| Campaign results/ and DECISIONS.md | Small reviewed summaries and references only; native Campaign receipts remain execution evidence |

Artifact manifest fields: role, actual path, creator/run ID, byte size, content hash and hashing algorithm, repository revision, model/config identity, tokenizer and dataset/split IDs, creation time, and status. Record null plus a reason for unresolved values; never insert invented hashes. Keep secrets and signed URLs out of records.

### Missing artifact rule

If a required file is missing, first verify the referenced path and receipt. Report the exact gap and its effect. Do not fetch a replacement corpus, use an older checkpoint, change tokenizer identity or retrain automatically. Ask for the missing artifact or a separately approved recovery task.

## Card 01 Prepare the foundation change

Owner: Luna implementation with Sol review
Depends on: First read-only assessment and selected repository access
Current status: NOT STARTED

### Bound and expected artifacts

Code and documentation only; no downloads, GPU work or campaign apply. Run only relevant existing CPU validations; propose any unavailable dependency.

Reviewable diff or PR-ready branch; fresh project protocol; mode/initialization contract; test plan; unresolved-input manifest.

### Acceptance gate

A reviewer confirms no prior experiment artifact is a prerequisite; fixtures, main training and optional transfer have separate identities; required missing inputs block execution rather than being fabricated. Schema-valid design is not the same as run-ready inputs.

### Prompt to paste with context and envelope

Implement only the foundation change explicitly approved after the assessment. Reuse current typed config/experiment derivation, dataset/campaign schemas, sealed staging, semantic probes, sidecar construction and digest APIs instead of duplicating them. Add only the missing isolated protocol and minimal scaffolding under experiments/research/kernel-memory-lab, following existing AGENTS.md. Put mutable/large outputs in the external campaign work root. Distinguish fresh initialization, strict resume and proposed dense-to-new-reader transfer. Record unresolved data/tokenizer/hardware/budget inputs and evaluation categories. A derivation receipt proves authoring, not runtime readiness. Run only the approved relevant CPU validations and return a reviewable diff, receipts and blockers. Do not install, acquire data, push/open a PR, create fixtures or run GPU/campaign work under this foundation card.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Card 02 Build new tiny correctness fixtures

Owner: Luna with Sol for loader and gradient defects
Depends on: 01 protocol and accessible CPU test environment
Current status: NOT STARTED

### Bound and expected artifacts

New synthetic fixtures only, at most 1 MiB generated text and 64 test cases; maximum 200 optimizer steps and 20 CPU minutes, first cap stops. No external corpus.

The update and time ceilings apply to the **entire approved Card 02 attempt**,
including every selected test, native full run, interrupted parent, resumed
child and retry. They are not per pytest invocation or per run ID. Before any
optimizer work, create one fresh ledger under the external project work root
with the approved total update and wall-time limits. Run all training-bearing
commands through `sparselab.training.attempt_budget`; Card 02 tests require
its ledger environment variable and reserve their full update maxima before
work starts. A separately selected legacy sidecar test needs its own up-front
reservation. Reservations remain charged after a failure or interruption;
neither rerunning pytest nor using a new run ID resets the allowance. The
deadline starts when the ledger is created and covers setup, preparation,
training, validation and confirmation checks across commands. The wrapper
terminates its process group at the deadline. Missing ledger, exhausted
allowance or expired deadline stops the task before more work. Preserve the
ledger, failed outputs and native counters, then request a new bounded decision
if a complete confirmation no longer fits. Do not initialize a replacement
ledger under the same approval.

Deterministic generator; tiny model and fixture tokenizer specs; expected labels/masks; overfit, gradients, disabled-memory and resume tests; receipts.

### Acceptance gate

Deterministic fixture bytes match real hashes; target masks are hand-checked; loss decreases by at least 80% on the deliberately overfit fixture within its cap or is diagnosed as failed; intended parameters receive finite gradients; frozen ones do not; resume and disabled-memory outputs meet a predeclared tolerance.

### Prompt to paste with context and envelope

Build only fresh tiny correctness fixtures within the approved protocol and CPU bounds. Use an independently initialized tiny model and deterministic fixture tokenizer; never use its weights for the main core. Reuse native semantic probe declarations for supplied-vector no-memory, oracle and wrong-vector inference controls. Trainer code already builds SemanticQueryBatch from verified FP32 query and boolean-mask sidecars; test that path rather than reimplementing it. Cover label shift, masks, batching, gradients/optimizer membership, repeated-batch overfit and strict save/resume with RNG/scheduler/counters. Use the training-state digest API only for its exact supported comparison; retain artifact hashes and separate numerical-equivalence checks. These are vector/plumbing tests, not natural-language chunk reading. Stop at the first approved CPU/time/step cap; retain failures and return receipts.

For any confirmation after `KML-20261007-C02-A1`, keep that original attempt
FAILED and append a separate result. Use a new approval and one new persistent
ledger; preallocate the sum of every planned optimizer update, including any
legacy regression, before execution. Check the ledger before each active
command and record charged and observed updates separately. A passing rerun
cannot erase the original limit breach or complete Gate 0 automatically.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Card 03 Approve data tokenizer and splits

Owner: Luna specification with Sol pipeline review
Depends on: 01; domain choice and source/license decision from Byron
Current status: NOT STARTED

### Bound and expected artifacts

Specification first; no bulk downloads or tokenizer training by this prompt. Any later acquisition must name source, license, maximum bytes and purpose.

Source/license manifest; proposed language mixture; main tokenizer contract; deterministic split and dedup rules; frozen comprehension/evidence rubrics; bounded acquisition request.

### Acceptance gate

Broad language foundation is distinct from the factual store; train-only tokenizer fitting and held-out coverage tests are specified; source families split before chunking; synthetic splits hold out templates/compositions and entities; no evaluation answers enter training or retrieval queries. Owner approves sources and acquisition bound.

### Prompt to paste with context and envelope

Prepare only the data, main-tokenizer and evaluation specification. Prioritize comprehension, contextual nuance and evidence use; code is optional. Propose a 32,768-entry tokenizer fitted on approved training-only material, with special-token, Unicode, round-trip and coverage tests. Recount if vocabulary changes. Define a separate 200-item language diagnostic and 300–500-item evidence suite with held-out entities/document families, including negation, scope, reference, conditions, ambiguity, constraints, contradiction and missing evidence. Accept multiple valid answers and needed clarification. Specify mixture order/token accounting. Reuse memorization analyze for bounded descriptive source-overlap checks; it is not a leakage or quality guarantee. Retain exact/near-duplicate split checks and answer exclusion. Return one bounded preparation request. Do not download or train.

### After approval paste this separate preparation prompt

Execute only the separately approved data-preparation subtask. Verify sources/licenses, acquisition bytes, CPU/time/disk caps and tokenizer specification. Reuse native authoring/preparation paths, then split/deduplicate before fitting the tokenizer on training-only data. Preserve provenance, mixture ordering and token accounting. Produce real snapshots, tokenizer, frozen rubrics/items, stable gold chunk IDs and withheld insertion/correction records. Record exact hashes and leakage/coverage receipts. Reuse a prepared-inputs bundle only after its seal and identities verify; an array cache is insufficient. The shipped benchmark-preparation creates synthetic inputs and runs subprocess preparation, so it needs its own bounded task and cannot substitute for real-corpus evidence. No model training or corpus expansion; stop at the first cap.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Card 04 Validate the main shape and measured fit

Owner: Luna operator with Sol numerical and device review
Depends on: 02; approved realized 03 artifacts for real-data timing, or clearly labeled fresh synthetic profile inputs
Current status: NOT STARTED

### Bound and expected artifacts

Prepare first. Runtime only with a recorded approval: one local context 1,024/microbatch 1 profile, at most 20 warm-up plus 100 measured optimizer steps and 30 wall minutes. No cloud use by default.

Validated native config; parameter count; fresh initializer manifest; hardware/software inventory; profile receipt, peak VRAM/RSS and sustained target tokens/s; failures.

### Acceptance gate

Native inspection matches 341,885,952 parameters or explains a legitimate recount; random init loads no weights or optimizer; loss/gradients remain finite; approved memory/time caps hold. Timing excludes warm-up. Synthetic throughput is labeled and never silently treated as real-data throughput.

### Prompt to paste with context and envelope

Handle only the main shape and fit gate. Validate vocabulary 32,768, width 1,024, 24 layers, 16 heads and FFN 2,816. Reuse typed config derive if an approved authoring task needs it; its receipt is not runtime proof. Inventory the actual WSL2 driver/distribution/kernel, ROCm/framework and device. Reference dense is default; native SDPA now exists for PyTorch dense/equal-head configurations. Propose one bounded profile and state the attention choice. Only with exact configuration/device/time/memory approval, run that profile, report the actual selected backend and stop at the first cap or nonfinite/OOM. Runtime attention itself runs forward/backward and needs scope approval. Do not install, switch hardware, use Colab, enable SDPA or change precision automatically. Preparation benchmarks do not prove GPU fit. Return one decision, not a training tranche.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Card 05 Run one budgeted language tranche

Owner: Luna operator with Sol readiness review
Depends on: 03 realized approved data/tokenizer/splits; 04 real-data throughput or conservative validated forecast; explicit run approval
Current status: NOT STARTED

### Bound and expected artifacts

Proposed first pilot ceiling: 10M target tokens or 2 local GPU hours, whichever occurs first; only the owner-approved bound is operative. Also require step, memory, checkpoint and evaluation caps. No automatic 1B/3B/10B run.

Readiness review; explicit run budget; initial and final checkpoints; target-token/step accounting; held-out loss and comprehension report; immutable receipts.

### Acceptance gate

All input identities match; actual tokens/time are within approval; best checkpoint and failures remain recoverable. Frozen-reader eligibility is a proposed ≥60% aggregate and ≥50% per diagnostic axis, at least 20 scored items per axis, plus finite held-out loss and stable instruction format. Thresholds need pre-run adoption, not retrospective adjustment.

### Prompt to paste with context and envelope

Handle one language-training tranche only. Verify the selected fresh checkpoint or random-initialization manifest, tokenizer, approved data, split, optimizer groups, scheduler and evaluation contract. Use measured sustained throughput to estimate total wall time including setup, checkpoints and evaluation. If exact runtime approval or any artifact is missing, return BLOCKED and a concrete request; do not train a substitute. With valid approval, run only that tranche, stopping at the first token/step/time/memory/spend boundary, nonfinite loss or declared regression. Save evaluation and resume receipts. Report per-axis comprehension errors and whether this checkpoint is eligible for a frozen-reader study. More language or supervised curriculum requires a new tranche decision; do not continue automatically.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Card 06 Establish raw text oracle evidence use

Owner: Luna evaluation; Sol correctness; Astra only for unresolved learning design
Depends on: 05 checkpoint that passes the adopted comprehension eligibility gate; frozen 03 evidence suite
Current status: NOT STARTED

### Bound and expected artifacts

Preparation/evaluation contract first; reader-task training requires its own explicit cap. Proposed pilot training ceiling 1M supervised target tokens or 1 local GPU hour; evaluation at most 500 frozen items and 128 generated answer tokens/item.

No-evidence, gold-evidence, wrong, shuffled and absent-evidence results; matched prompt/answer budgets; category errors; supported-claim and abstention scoring. Selected dense-reader checkpoint ID/hash, explicitly unchanged from Card 05 if no adaptation occurred.

### Acceptance gate

Provisional target: ≥15 percentage-point oracle accuracy gain on at least 200 answerable items, with a paired 95% interval excluding zero; unsupported-answer rate improves; ≥50 missing/conflict cases are inspected. Adopt a baseline-appropriate threshold before the run. A ceiling-limited baseline requires predeclared alternative criteria.

### Prompt to paste with context and envelope

Establish only the raw-text oracle evidence baseline using the selected capable fresh checkpoint. Keep query, decoder and answer budgets fixed across no evidence, gold evidence, plausible wrong evidence, shuffled evidence and absent evidence. Start with four chunks of 128 tokens, counting prompt plus evidence within the actual context limit. Verify answer leakage is absent. If bounded evidence-use training is needed, propose it and stop unless exact approval is supplied; do not call a frozen random model a reader. Score meaning, source support, contradictions, abstention and ambiguity handling separately. Return paired results and error examples. Do not add a retriever, cross-attention reader or learned router. Identify the selected dense-reader checkpoint for Cards 07–09 and account for all evidence-adaptation training.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Card 07 Build the lexical retrieval baseline

Owner: Luna implementation with Sol review
Depends on: 06 selected dense-reader checkpoint and fixed versioned store; approved source/license and indexing bounds
Current status: NOT STARTED

### Bound and expected artifacts

Use only the approved small corpus; start at ≤10,000 chunks and ≤1 GiB total approved input. No bulk acquisition. Evaluation cap ≤500 items; neural training is outside this card.

Immutable chunk manifest with provenance; lexical index; retrieval recall and latency; generated-answer comparison to the same oracle/no-evidence controls.

### Acceptance gate

Provisional gold-evidence recall at 4 ≥85%; actual-retrieval answer gain retains ≥80% of the oracle gain. Fix definitions for multi-chunk questions, denominators and confidence intervals beforehand. Poor recall blocks router conclusions but can lead to a bounded chunk/query diagnosis.

### Prompt to paste with context and envelope

Build only a lexical chunk-retrieval baseline on the approved store. Verify stable chunk/document IDs, versions, source locator, license, offsets and content/tokenizer hashes. Retrieve at most four chunks at the fixed evidence budget with no answer in the query. Measure gold-evidence recall, index bytes, query latency and end-to-end answer quality using the same reader as Card 06. Keep missing, stale and conflicting evidence visible. If native tools lack this index, implement the smallest reviewable path; do not add a neural embedding model, graph, learned router or NVMe subsystem. Return the diff, exact manifest/index identity, receipts and the oracle-to-retrieval gap. Use exactly Card 06’s selected dense-reader checkpoint and record its hash.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Card 08 Prove dense to augmented initialization

Owner: Sol engineering; Luna runs bounded tests
Depends on: 02 tiny contract tests; selected dense-reader checkpoint from 06 for actual transfer verification
Current status: NOT STARTED

### Bound and expected artifacts

Implement the contract with tiny CPU tests first; no model training or external/legacy checkpoint. Actual-main verification is separate: one transfer, at most 16 fixed inputs of 128 tokens and 20 wall minutes, with explicitly approved device and RAM/VRAM cap.

Explicit tensor mapping; reader-module initializer; strict mismatch diagnostics; optimizer/trainable-group manifest; disabled-memory equivalence and gradient tests. Separate tiny-contract and actual-main verification receipts; Card 09 requires both.

### Acceptance gate

Every dense tensor maps once with identical shape/tokenizer semantics; only declared new modules are randomly initialized. Reject unexplained keys/dtypes/dimensions; reset optimizer/scheduler/counters for transfer. At memory disabled, FP32 reference logits match within predeclared absolute/relative tolerance, initially 1e-6/1e-5; device-specific tolerance requires evidence and review.

### Prompt to paste with context and envelope

Implement and test only the explicitly approved transfer from Card 06’s selected same-project dense-reader checkpoint to a model with a disabled new reader interface. Config/experiment derive does not perform this transfer; do not substitute its authoring receipt. Map every compatible dense tensor, initialize only declared new modules, reject unexplained missing/unexpected keys and verify tokenizer/embedding identity. Reset optimizer/scheduler/token counters; this is not strict resume. Check disabled-memory output equivalence, parameter counts, gradient flow and optimizer membership using tiny fixtures and predeclared tolerance. Then request the separately bounded actual-main mapping/equivalence check and run it only with exact device/time/memory approval. Tiny tests do not prove the main artifact. Do not retrain a missing checkpoint, train fusion or continue to Card 09. Return contract, diff and numerical receipts.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Card 09 Test one integrated reader

Owner: Sol integration; Luna bounded evaluation; Astra reviews ambiguous outcomes
Depends on: 06 and 07 controls on the same selected dense-reader origin; 08 tiny and actual-main receipts; approved architecture/runtime budget
Current status: NOT STARTED

### Bound and expected artifacts

One adapter only. Proposed reader: 2 encoder layers, width 256, 4 heads, FFN 1,024; one 4-head cross-attention adapter after block 16; four ×128-token chunks. Training only with explicit cap, initially ≤1M target tokens or 1 GPU hour.

Reader/config and parameter accounting; gated residual; trainable-group/gradient receipt; matched raw-text versus integrated oracle/retrieval results and resource report.

### Acceptance gate

The integrated reader retains ≥90% of raw-text oracle task score at the same chunks/evidence budget, or meets a predeclared worthwhile latency/memory trade-off. Report full encoder/reader cost. Language regression ceiling, initially ≤2 percentage points on the diagnostic, must be adopted before training.

### Prompt to paste with context and envelope

Implement and test one integrated chunk reader according to the approved design, using the verified same-project transfer contract. Freeze the qualified dense checkpoint initially and train only the declared encoder/adapter parameters within explicit approval; otherwise prepare the exact request and stop. Check that a small or zero residual gate does not silently prevent required gradients. Compare raw-text and integrated reading with identical oracle chunks, then the same lexical retrieval. Keep strict-from-scratch components explicit and report all new parameters. If joint training is needed, request a separate bounded control with language replay. Return quality, grounding, per-axis regressions, memory and latency. Do not train a router or introduce paging. Use the same Card 06 origin for all branches; report branch-specific training and total compute rather than claiming an unmatched gain is architectural.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Card 10 Consider a learned router only if needed

Owner: Astra design review then Sol engineering; Luna bounded operation
Depends on: 07 and 09; a specific measured lexical-retrieval limitation; separate approval for the selected experiment
Current status: NOT STARTED

### Bound and expected artifacts

Optional and initially SKIP RECOMMENDED unless a bottleneck is evidenced. One conventional content-router experiment before virtual PKM; approved token/step/time/index caps required.

Bottleneck evidence; one falsifiable router hypothesis; contrastive-data/negative contract; exact candidate/read budget; recall/update/quality/cost comparison.

### Acceptance gate

Predeclare the minimum recall or total-system improvement worth the complexity, with a non-inferiority bound on evidence task score and insertion behavior. Same queries, candidates, read count and evidence tokens across indices. If no bottleneck or gain exists, retain the simpler baseline.

### Prompt to paste with context and envelope

Assess only whether a learned content router is justified by the attached lexical and reader receipts. Separate retrieval miss from reader failure. If justified, propose one bounded contrastive query/chunk training experiment with known positives and hard negatives; explicitly describe gradient flow because native detached CPU top-1 lookup does not train a router. Do not execute training under this assessment prompt. Treat virtual PKM as a later alternative to the same router’s index, with versioned content-to-bucket assignment and measured recall loss. Do not treat a fixed document-ID classifier as an insertion-capable router. Return a go/no-go proposal, exact comparison and approval request; stop.

### After review paste this separate implementation prompt

Implement only the single router or index experiment selected in the attached Card 10 decision, preserving the approved control and candidate/read budgets. Verify exact data, architecture, steps, token, wall-time, memory and spend limits. If runtime approval is absent, prepare code and tests only and return the bounded request. With exact approval, run only that experiment. Report recall, answer score, insertion behavior, index size, total cost and uncertainty against the fixed baseline. Do not automatically try virtual PKM after a failed content router or rebuild the full archive. Stop after the approved comparison.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Card 11 Test knowledge insertion and correction

Owner: Luna evaluation with Sol provenance checks
Depends on: Useful 07 raw-text reader or useful 09 integrated reader; selected path/checkpoint and frozen hashes; approved new documents
Current status: NOT STARTED

### Bound and expected artifacts

No neural training. Proposed ≤100 new approved chunks, ≤200 targeted questions and ≤128 generated answer tokens/item; no unapproved source acquisition.

Before/after store manifests; immutable model hash; new index-entry receipt; insertion, correction, stale-source and conflict results; regression slice.

### Acceptance gate

Weights and tokenizer unchanged; only declared store/index changes occur. On at least 50 answerable inserted-fact items, propose ≥15-point gain over the pre-insertion store with paired uncertainty; correct-source precedence and unsupported answers pass predeclared rubric. Old-fact score falls no more than the adopted 2-point tolerance or is flagged.

### Prompt to paste with context and envelope

Test only knowledge updates with frozen weights. Verify the before-state checkpoint and store/index identities. Add the approved small set of previously withheld facts or corrected versions, preserving source provenance and supersession links. Update only necessary index entries; a changed embedding/codebook requires an explicit migration proposal, not silent reuse. Evaluate the same questions before and after, plus old-fact and conflicting/stale-source controls. Verify hashes prove no neural weight update and that evaluation answers were not used to train the reader. Return update cost, evidence-supported quality and failures. Do not fine-tune away a failed insertion result.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Card 12 Compare the same store in RAM and NVMe

Owner: Sol storage design; Luna bounded benchmark
Depends on: Useful 07 raw-text reader or useful 09 integrated reader; selected path/checkpoint; same store/query set and approved local caps
Current status: NOT STARTED

### Bound and expected artifacts

No larger corpus and no model training. One bounded benchmark with ≤200 fixed queries, warm and controlled-cold passes, declared cache/RAM reserve and wall-time cap. Any privileged cache operation requires separate approval.

Identical manifest hashes; RAM/mapped-NVMe configurations; residency/page-cache evidence; p50/p95 retrieval/TTFT/decode; RSS/VRAM/cache/transfer measurements.

### Acceptance gate

Results establish the same quality and evidence payload in both placements. Cold-state validity is measured, not assumed from memory mapping. Meet owner-approved latency and memory limits; otherwise report the frontier. A small store resident in page cache cannot demonstrate cold-NVMe scalability.

### Prompt to paste with context and envelope

Prepare one same-store RAM-versus-NVMe comparison. Verify store/index, checkpoint, queries, precision, batch size and chunk budget are identical. Measure retrieval, page faults, cache hits, host-to-device transfer, reader compute, time to first token and decode separately for warm and controlled-cold conditions. Use a safe scoped method to bound cache; do not drop system-wide caches or change security/system settings without explicit approval. If current data cannot exceed the cache budget, state that limitation and request a separately bounded scale test rather than downloading more. Run only with the exact benchmark approval; otherwise return the protocol. Stop after the comparison.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Card 13 Review evidence and choose the next step

Owner: Sol correctness review or Astra architecture review; Byron approves commitment
Depends on: Actual manifests, diffs and receipts for the question being decided; incomplete stages remain unverified
Current status: NOT STARTED

### Bound and expected artifacts

Read-only review. No repairs, retraining, reruns, cloud dispatch or new mechanisms under this prompt.

Gate-by-gate evidence verdict; unresolved risks; one continue/revise/stop recommendation; exact next bounded task and any approval request.

### Acceptance gate

Every completion claim resolves to real artifacts and receipts. Distinguish comprehension, oracle-reader, retrieval, fusion, update and offload results. A recommendation names the decisive evidence and cost; missing evidence remains BLOCKED/UNVERIFIED. Never promote a status based on chat recollection.

### Prompt to paste with context and envelope

Review the attached evidence for this one decision: [state the specific gate or next tranche]. Verify artifact identities, counts, splits, authorization bounds, failed runs and test receipts. Check that fresh initialization, comprehension eligibility, oracle controls, lexical comparison, transfer/fusion and any update/offload claims were actually established where relevant. Do not infer later gates from earlier toy tests. Return EVIDENCE VERIFIED, FAILED or BLOCKED for each relevant criterion, with precise artifact references and one bounded next action. If the simple baseline wins, recommend retaining it. If approval or a critical artifact is missing, identify it; do not repair or retrain automatically. Stop after the review.

### Close the card

Return the evidence bundle, append one run record and request the named review. Mark EVIDENCE VERIFIED only after review. Keep failures and blockers visible. Do not start the next card automatically.

## Approve a run and use native commands safely

Every runtime request must state what will happen and where its authority stops. The budgets in the cards are proposed ceilings, not approvals. Increasing data, switching device/provider, changing architecture or raising spend requires a fresh decision. The first reached cap ends the run; a stopped run may be informative without passing its research gate.

### Run approval request template

Proposed run ID and purpose: [value]
Configuration and exact input artifact hashes: [verified values]
Device and provider: [local hardware or named cloud runtime]
Data sources and maximum bytes: [approved values]
Maximum optimizer steps and target tokens: [values]
Maximum wall time including setup/evaluation: [value]
VRAM, host-memory reserve and disk limits: [values]
For cloud: verified available GPU, credit burn/price, total credit/currency cap, runtime constraints and termination method: [values]
Evaluation/checkpoint cadence and required outputs: [values]
Stop on: first cap, nonfinite loss, OOM, identity mismatch or predeclared regression
Exact approval requested: May I run only this bounded operation?
User response and timestamp: [actual response, never inferred]

### Native command templates and side effects

Verify these against the actual installed CLI before use. CONFIG and CAMPAIGN are shell variables pointing to real validated files; they are not proposed filenames to invent. Use an already available locked environment; do not interpret a missing dependency as permission to install.

uv run --locked --no-sync sparselab inspect "$CONFIG" --json

Preparation only: uv run --locked --no-sync sparselab workspace preflight "$CONFIG"

uv run --locked --no-sync sparselab campaign validate "$CAMPAIGN" --json

uv run --locked --no-sync sparselab campaign status "$CAMPAIGN" --json

uv run --locked --no-sync sparselab campaign next "$CAMPAIGN" --json

uv run --locked --no-sync sparselab campaign explain "$CAMPAIGN" --json

Inspect and campaign validate/status/next/explain are observational templates. Workspace preflight checks capacity but may initialize the work-root/scratch, so keep it outside a strict read-only audit. Campaign apply/resume can prepare, pilot, evaluate or generate without execute-runs. Derive writes fresh declarations/receipts; stage may warm up. Prep benchmarks/readiness smoke are active work. Preflight and stage have no json flag. Prepared-inputs needs a sealed staging bundle, not an array cache. Each active operation needs its own authorized scope. [2]

## Append only run record

After adoption, append one result under the canonical Markdown record for every attempted operation, including failed and preparation-only tasks. This page supplies the template; do not maintain a second live log in this DOCX. Correct an earlier entry with a new linked entry. Native run receipts, logs and artifact hashes are the evidence; the project record is their human-readable index.

Record ID: [unique observed/project-assigned ID]
UTC timestamp and operator/reviewer: [values]
Card and exact action: [value]
Starting status and relevant decision ID: [values]
Repository revision and diff reference: [verified values]
Input artifacts and hashes: [actual values or UNVERIFIED]
Approval reference and exact bounds: [actual authorization or NOT APPLICABLE]
Device/software and numerical precision: [observed values]
Command or action actually performed: [exact command/action]
Measured steps, target tokens, time, peak VRAM/RSS, disk and spend/credits: [values or NOT MEASURED]
Outputs and hashes, including failed logs/checkpoints: [actual values]
Tests and metrics with denominators: [receipts]
Gate outcome: [READY FOR REVIEW / EVIDENCE VERIFIED / FAILED / BLOCKED]
Reviewer and evidence checked: [actual reviewer or pending]
Known limitations or unexplained differences: [values]
Next bounded decision needed: [one decision]
Correction to earlier record, if any: [record ID and reason]

### Initial entries

| Record | Completed state | Evidence boundary |
| --- | --- | --- |
| PLAN 002 | Research plan updated 7 October | Current pinned source audit; no experiment execution |
| WORKBOOK 002 | Workbook and Markdown pack updated | Synced revision 2026-10-07.1; no card executed |
| IMPLEMENTATION | NOT STARTED | No repository changes or PR by this task |
| RUNTIME | NOT STARTED | No calibration, training, evaluation or cloud spend by this task |

Keep native campaign IDs, run IDs and evidence-bundle paths in the result entry. Add a JSONL mirror only if a reviewed repository need justifies it; do not create a competing status system. STATUS.md points to the newest reviewed result and decision under experiments/research/kernel-memory-lab. Store approval references without secrets or signed URLs.

## Decision log review and resume prompts

### Decision log template

Decision ID and date: [values]
Question being decided: [one question]
Evidence records and artifacts inspected: [IDs and verified links/paths]
Options considered: [bounded alternatives]
Decision: [continue / revise / stop / defer]
Reason and uncertainty: [evidence-based explanation]
Owner and reviewer: [actual names/roles]
Authorized scope and bounds, if any: [exact user approval; otherwise NONE]
Consequences for dependent cards: [list]
Revisit trigger: [specific new evidence or condition]

### Sol review prompt

Review only Card [number] using the attached diff, manifests and receipts. Check implementation correctness, input identities, initialization/loading semantics, numerical tolerances, gradient/optimizer membership, resource/authorization caps and failed-test handling. Confirm cited commands and artifacts exist. Return criterion-by-criterion VERIFIED, FAILED or UNVERIFIED with exact evidence. Name one smallest correction if needed. Do not edit, rerun, retrain, approve spending or advance the campaign. Stop after the review.

### Astra research review prompt

Review only this experimental decision: [one question]. Use the attached results and controls to separate base comprehension, reader learning, retrieval miss, fusion loss and physical storage limits. Check leakage, matched budgets, uncertainty, category errors and whether a simpler control explains the gain. Recommend continue, revise or stop, with one falsifiable bounded next test. Missing evidence is a blocker, not permission to reconstruct results or run work. Do not add mechanisms or authorize a training tranche. Stop after the recommendation.

### Resume prompt for a new conversation

Resume Kernel Memory Lab from the adopted experiments/research/kernel-memory-lab Markdown record. Read BOOTSTRAP.md, CONTEXT.md, RESEARCH_PLAN.md, STATUS.md, the latest decision and result entries, the selected EXECUTION_WORKBOOK.md card, and actual native campaign manifests/receipts. Do not rely on previous chat memory. Verify the repository revision and each required artifact identity/hash. Reconstruct the state as EVIDENCE VERIFIED, READY FOR REVIEW, FAILED, BLOCKED or NOT STARTED with source receipts. Report missing or contradictory evidence without substitution or retraining. Identify the single next authorized action and exact bounds. Do not execute a new task until that scope is approved. A saved Codex session does not replace this reconciliation.

Source boundary: this workbook and its Markdown counterpart share revision 2026-10-07.1. The repository audit was refreshed on 7 October at 06efc4db82ecf3da97b50cff518cba605ad27b33. Infrastructure existence is separated from this project’s unexecuted cards. Paths, thresholds, budgets and prompts remain proposed until adopted; no GPU, training, evaluation, cloud-spend or capability result is claimed. Keep the plan’s primary bibliography and SOURCE_AUDIT.md with the record.
