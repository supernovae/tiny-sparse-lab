# Semantic kernel MVP v1: hypothesis register and staged protocol

Authored 2026-10-02. Stages 1–5 are proposals; only Stage 0's compiler and replay
pilot are implemented. Before each behavioral stage, commit a separate sealed
declaration with exact cases, partitions, model/tokenizer revisions, budgets,
scorer and acceptance criteria. This document is not a sealed model campaign.

## The question we are preserving

Can a small learned language controller use stable typed interfaces to external
facts and executable capabilities, reducing the need to acquire domain instances
through gradient descent while preserving correct composition and explanation?

Three objects remain distinct: **facts** (external records), **operations**
(ordinary executable software), and **the policy** (learned selection, argument
binding, ambiguity handling and explanation). Later learned adapters are a
fourth object with their own cost and compatibility claims.

| ID | Falsifiable hypothesis | Disconfirming outcome | Necessary control |
|---|---|---|---|
| H1 grounding | Same frozen model answers source questions more accurately with the bank than without it | No improvement, or fabricated unsupported assertions | Closed-book; same facts as text; ordinary JSON tools |
| H2 composition | Policy succeeds on held-out combinations of known operations | Only seen workflow templates succeed | Single-operation cases; matched-length seen compositions |
| H3 replacement | Replacing source facts changes answers without weight updates | Stale answers persist despite correct retrieval | A→B→A source snapshots; unchanged checkpoint; no-change repeat |
| H4 interface learning | Policy fine-tuning improves held-out use of a fixed ABI | Only wording/task memorization improves | Same model before tuning; equal-budget ordinary tool-trace tuning |
| H5 neural efficiency | Smaller controller reaches a fixed capability threshold at lower measured total cost | Savings vanish after compilation, retrieval, adaptation and retries | Larger model on same bank and tasks; cost-matched alternatives |
| H6 representation | Learned structure adapter improves on typed serialized records | No benefit over JSON, or transfer requires task-answer calibration | Serialized records; random/permuted adapters; two independently trained recipients |

H1–H4 do not establish H5 or H6. A reliable deterministic runtime does not make
the model's tool selection reliable. Stable interfaces do not make independently
trained latent spaces compatible. New domains may need expertise to interpret
their abstractions; installing a solver does not teach when its assumptions hold.

## MVP domain and reality boundary

Use real versioned lab Python source first, with exact source evidence. Tasks:
locate definitions, inspect declared arguments, identify syntactic call sites,
compare signatures across snapshots, and distinguish unknown from unsupported.
Do not call AST call sites resolved callers. Reflection, imports, aliases,
inheritance and dynamic dispatch require additional analysis or explicit unknowns.

Later add one physical demonstration: a two-body orbital simulator with units,
assumptions, energy/angular-momentum diagnostics and timestep/solver interventions.
Use cited physical constants and transparent initial conditions. Deliberately
injected unit, frame and solver faults are **controlled faults**, not naturally
occurring field data. Numerical agreement validates computation under the stated
model; it does not prove the model correctly describes a real orbit.

## Stages and decisions

| Stage | Deliverable | Gate to advance | What a pass means |
|---|---|---|---|
| 0 substrate | Source compiler, typed read-only runtime, replay traces | Independent source spot-checks; integrity/error tests; bounded operation/output behavior | Software wiring only |
| 1 frozen controller | Real-task bundle, deterministic baseline, local-model adapter and replayable action/result loop | Seal scorer/splits/limits; report every baseline/model case, including failures | Measures whether a controller uses the interface |
| 2 structure SFT | Executed train-only v2 tool transcripts, matched tuning controls | H4 contrast under fixed endpoint; held-out source families, wording and compositions | Interface-policy learning in this domain |
| 3 expand/replace | New source snapshots and a second capability package | H2/H3 cases with checkpoint frozen; missing/conflicting/stale data handled | Bounded composition/replacement |
| 4 instrumented diagnosis | Orbit simulator, explicit hypotheses, discriminating experiments | Correct diagnosis with verified interventions under held-out fault combinations | Bounded active diagnosis |
| 5 optional latent adapter | Structure encoder + recipient adapter; Engram comparator | H6 matched controls, costs and two-recipient transfer | A separately defined representation result |

### Stage 1 preregistration requirements

Target 90 independently authored, human-checkable cases: 30 development, 60 test.
These are design targets, not already collected cases. Partition **source module
families and workflow composition patterns** before authoring policy training
traces. Do not randomly split adjacent functions or paraphrases from one task.
All test facts may be in the inference bank: this tests retrieval/use, not unseen
facts inaccessible to the system. They must be absent from SFT targets. Keep
scorer answers and hidden tests separate from model-visible bank and prompts.

Include missing symbols, duplicate short names, misleading call spelling,
incomplete observations, and questions the AST cannot answer. Require evidence
IDs and source lines in final claims. Report exact IDs/argument correctness,
unsupported-claim rate, abstention precision/recall, invalid actions, steps,
retries, timeout rate, input/output tokens, wall time and bank bytes by task
family. A correct final answer is not sufficient if its cited evidence disagrees.

Baseline arms: deterministic exact lookup (reports its applicability/coverage),
frozen model closed-book, frozen model with equal facts serialized as text,
frozen model with ordinary typed JSON tools. Use identical model and decoding
settings. This pilot's ABI is itself JSON; only later nontext/learned variants
can support an architectural claim beyond ordinary tool calling. Match source
coverage and output/interaction budgets; report prompt cost rather than silently
matching by truncating one arm's evidence. The deterministic baseline can win.

Proposed fixed budgets: 8 tool steps, 4,096 generated tokens including reasoning
where available, 120 seconds per task; one attempt per case. Freeze values after
resource smoke, before opening test outcomes. Record hidden/unavailable thinking
costs as unavailable. The generic replay pilot has separate wiring-only limits.

Stage 1 success proposal: grounded exact success ≥80% overall, ≥60% in every
supported task family, unsupported factual claim rate ≤5%, and a positive paired
accuracy difference versus closed-book. Report paired confidence intervals and
all arm contrasts; if the interval includes zero, call the advantage unresolved.
These thresholds must be accepted/sealed before test execution. A ten-case demo
does not close them. For Stage 2, seal a fresh protocol and an explicit ≥10
percentage-point gain over both untuned and matched tool-SFT controls on supported
held-out cases without increasing unsupported claims; retain fixed endpoints.

## Training and data ownership

Start with a 1–2B **pretrained** candidate if it fits the chosen hardware. That is
a sensible trial size, not a measured minimum. Compare smaller/larger models
later. The existing lab's native DenseLM scratch training and an imported
pretrained model/LoRA runner are different execution paths; support for model
imports, chat templates and adapter training must be verified or built explicitly.
Do not imply the current `local_chat` trainer imports arbitrary 1–2B checkpoints.

Compile source facts without SGD. Train the policy, if needed, on successful
executed operations and truthful unsupported/clarification responses. Preserve
source provenance and ownership. Never auto-convert all scraped text into facts:
AST declarations are extractable; explanatory claims need reviewed semantics.
Export train-only `format_version: 2`, `loss_mode: assistant_only` conversations
through the existing local-chat schema. Observation fields are context-only,
while assistant tool calls/final text receive loss. This masking does **not**
stop the policy from learning factual associations from supervised final answers.
Report instance overlap and separate a policy-only operation-target arm if testing
whether domain answers need to be learned at all.

Compiler input excludes benchmark annotations, hidden fault labels and expected
answers. Training trace generation uses only its assigned source/task partition.
Retain rejected traces; teacher-generated actions require execution and scoring,
not acceptance because they look plausible. Teacher compute is a separate cost.
Do not claim trajectories reveal or reproduce a model's private reasoning.

## Instrumentation, boundaries and publication

Keep per-step actions, typed errors, observations and source IDs. Distinguish
observed/derived/inferred/unknown claims; provenance establishes where a value
came from, not that it is true. Deterministic calculations can implement the wrong
formula. Hypothesis elimination is conditional on correct likelihoods, measurement
error and hypothesis coverage; include `other/unknown` and avoid invented priors.

Record source commit/dirty status, source/compiler/ABI hashes, model revision,
tokenizer/chat-template hashes, budgets, trainable ownership, hardware, wall time,
compiler cost, bank bytes, inference and tuning tokens. Keep stage declarations
and small evidence references in Git; datasets, banks, traces and checkpoints in
the external persistent root. Freeze identities before behavioral runs. Never
overwrite failed attempts or adjust acceptance after viewing outcomes.

Publish incrementally: (1) reproducible substrate software and limitations;
(2) frozen-controller results, including zero/negative outcomes; (3) training
contrast; (4) composition/diagnosis; (5) any adapter result separately. Release
model weights only after source/base/adapter licensing and privacy review. A
publishable artifact can be a well-instrumented negative result. None of these
stages retroactively proves that lexical N-grams represent semantic knowledge.
