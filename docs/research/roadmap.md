# Capability status and evidence roadmap

This roadmap describes what SparseLab can execute, what has been smoke-tested, what outcomes have been observed, and what still blocks stronger claims. Progress is organized by capability and evidence—not release phases. **Implemented** means the path exists; **smoke-tested** means it ran; neither means a model is useful or an architecture is better. Missing code belongs in [TODO.md](../../TODO.md); scientific questions and next experiments remain here and in the versioned lifecycle.

Durable experiment definitions use the [`experiments/`](../../experiments/)
layout: copyable teaching material under `samples/`, actual campaigns under
`research/`, and mutable execution output under the ignored
`sparselab-work/experiments/` tree. GitHub's Code task and Research experiment
issue templates preserve the same boundary.

## Decision coordination

The lifecycle sidecar now records bounded Findings, declared next tests, reviewed
promotion decisions, and known-good baseline availability without treating a
metric as an automatic scientific decision. The first dense synthetic alias
reference remains an unpromoted candidate: its one-run capture reached the budget
and acquisition gates, but held-out validation loss rose from 6.176402 to 7.056087.
The [known-good baseline guide](known-good-baselines.md) records the evidence and
the failed gate alongside the bounded CPU/offline route and independent branches.
It does not close useful-model, semantic text-query/compiler, full allocation,
independent lexical generalization, representation portability, CUDA/XPU, or
cross-host evidence.

## Capabilities available now

| Capability | Current boundary |
|---|---|
| Model workflow | Train, inspect, evaluate, checkpoint, resume/promote, generate, and chat on documented CPU, MPS, and MLX paths. Worker-based independent runs are supported; no distributed backward or expert sharding. |
| Learn and research | Packaged mechanism lessons, offline/TinyStories/FineWeb-Edu profiles, controlled recipes, explicit scaffolds, capability cards, paired/factorial comparisons, static reports, and read-only research browsing. Preparation, training, and collection remain explicit commands. |
| Architecture | PyTorch dense, sliding-window, block-sparse and MLA attention; local Top-K MoE; token/byte/portable Engram. MLX supports dense and native block-sparse attention, not every PyTorch mechanism. |
| Lexical memory | Token- and byte-addressed trainable tables, placement options, diagnostics, and checkpoint integration. Addressing/reuse/collisions are observable mechanisms, not proof of useful retrieval. |
| Portable memory | Verified byte-table export/load and a frozen table with a trainable target adapter. This is distinct from a semantic EngramPack and from a text encoder. |
| Semantic EngramPack | Verified exact retrieval from caller-supplied vectors, structured toy-world controls, and direct PyTorch `DenseLM` adapters. The standard text chat/training path does not construct semantic queries. |
| Useful model behavior | No independently validated useful domain model yet. Synthetic association and instruction runs provide learning/failure examples, not general assistant capability. |

## Smoke and experiment evidence

- The [CPU/offline FFN smoke and nano reports](sample-report.md) completed 18 runs and 21 comparisons per campaign. All three alias-card scores were zero for every run; lexical memory showed no consistent held-out-loss benefit. The fixed-seed rerun reproduced all 18 losses, but adds no independent seed evidence.
- The [FineWeb-Edu micro study](../model-scaling.md#completed-fineweb-edu-micro-study) completed 18 MPS runs at 1,024 updates / 262,144 targets. Narrower FFNs had higher held-out loss; lexical deltas changed by seed/width, and all three alias cards were zero. This is bounded, descriptive evidence, not a scaling law or a general quality result.
- The [MLA analysis smoke](sample-report.md#research-analysis-pipeline-smoke) completed 12 CPU/offline runs and exercised factorial, nondominance, allocation, and boundary-sweep reporting. Its 36 card outcomes were measured zeros; the run verifies analysis paths, not an MLA/memory benefit.
- The [byte-Engram smoke](../byte-addressing.md#verified-smoke-execution) verified nontrivial UTF-8 addresses and two training updates. It does not establish learned byte-memory value.
- The [portable Engram comparison](../portable-engram.md) tested two held-out cases; baseline, random-table, and trained-adapter systems all had zero exact matches, and the trained adapter did not outperform the random control. General transfer remains unverified.
- The [ownership-allocation smoke](../path-domain-corpus.md#partial-cpu-allocation-smoke-observation) executed one of 75 configured coordinates. It produced zero card passes at every recorded checkpoint; no allocation optimum or full curve is established.
- The [instruction starter](../from-toy-to-useful.md#measured-starter-example-what-improved-what-did-not) lowered synthetic validation loss, but its actual replies still failed ordinary arithmetic, color, and unrelated questions. This is not a useful assistant.

## Open research questions

These are evidence gaps and proposed investigations, not implementation TODOs.
If an experiment exposes missing or defective software, open a separate Code task
and add that implementation gap to [TODO.md](../../TODO.md). The main open
questions are:

- **Useful tasks:** establish one low-risk real task against an independent test population and a simple baseline, with declared error handling and human review.
- **Corpus shape and synthetic fraction:** preregister paired releases from the same licensed source snapshots with declared shape/fraction filters, held-out source/world/template units, fixed tokenizer/target token budget and controlled model/seed/evaluation suites. Compare recall, paraphrase, troubleshooting, tool use and abstention separately; report negative and mixed effects rather than a universal synthetic percentage or shape ranking. The [Corpus Forge ledger](../corpus-provenance.md) provides measurement inputs, not an observed outcome.
- **DevMind dense-control candidate:** [v0 A](../../experiments/research/devmind-pretrain-v0/source-report.md) froze source evidence and eleven underpowered cards; [v0 B](../../experiments/research/devmind-pretrain-v0/tokenizer-report.md) failed the nine-kind gate before fitting. The [v1 pilot evidence](../../experiments/research/devmind-pretrain-v1/evidence.md) seals a separate private metadata-only source candidate and underpowered diagnostic cards, resolves bounded lexical near-duplicates and preregisters six paired tokenizer kinds, but source origin/privacy/obligation and weight-publication review is still open; no tokenizer was fit, no distinct source-token supply measured and no model trained. Independent JSON/TOML/log kinds, broad-language coverage, ≥200 genuinely independent test parents per card and the original 150M-token DATA_RICH gate remain unresolved. Next: review selected file rights/content and publication policy; only then fit/measure the frozen bakeoff, choose supply-calibrated dense MODEL-0, and test ROCm/BF16 runtime on the provisioned worker before a fresh seed-42 plan. No corpus build, card or smoke substitutes for model quality or promotion.
- **DevMind v3 engineering-source supply:** [v2's frozen candidate](../../experiments/research/devmind-pretrain-v2/protocol.md) has 49,120,567 distinct engineering train bytes; the separate [v3 immutable release and evidence](../../experiments/research/devmind-pretrain-v3/protocol.md#frozen-release-and-measured-source-supply) retain 510,860,750 developer bytes from first-party code/manual/spec sources, with 3,012 independent engineering validation/test documents and unchanged academic/web sources. The pinned peS2o title screen flags only 744 of 280,730 papers for topical review without relabeling them. At one other-source pass, 20/30/40% byte-proxy developer exposure requires 1.597/2.738/4.260 developer passes; near-overlap sampling finds Apache-license boilerplate pairs and leaves unsampled semantic independence unproved. **Decision `EXPAND_MORE`:** next independently versioned source/split protocol should add rights-reviewed SQL, JavaScript, shell, configs, operations, build/example/schema material and independent config/operator heldouts, then rerun exact/near leakage screens. No tokenizer or model was fit; no evidence of model quality or a model-weight license follows.
- **DevMind v4 source decision:** the separate [v4 frozen release and measured evidence](../../experiments/research/devmind-pretrain-v4/protocol.md#frozen-release-and-retained-source-supply) retain 735,281,099 distinct developer train bytes and independent SQL, JS/TS, shell, build, operations and schema heldouts without changing v2/v3 or generic pools. At one pass through nondeveloper data, 25/30% descriptive byte-proxy share takes 1.480/1.903 developer passes. The aspirational 1.0–1.2 GB developer target was missed; small IaC/build/operator/example pools and bounded lexical overlap coverage remain explicit risks. The preregistered corpus-source decision is **READY_FOR_TOKENIZER**, not approval for model training or weights publication. A subsequent separately specified tokenizer/model study must measure quality, rights and exposure rather than collecting without a concrete new gap.
- **DevMind v5 reproducible successor:** the separately committed
  [v5 protocol](../../experiments/research/devmind-pretrain-v5/protocol.md)
  preserves [v2/v3/v4 historical identities](../../experiments/research/devmind-pretrain-v5/historical-releases.json)
  as nonreconstructable with the available authenticated closure. It copies all
  123 reviewed final-v4 source declarations and heldouts without historical
  snapshot imports, pins one fresh producer, and requires exact independent
  build/release recovery before tokenizer selection and dense MODEL-0. Corpus,
  tokenizer, ROCm training and evaluation outcomes are not yet observed.
- **Lexical Engram:** test transfer/generalization across independent task data, held-out wording/facts, seeds, and collision/capacity controls; keep token and byte results distinct.
- **Portable byte Engram:** extend the negative two-case result to multiple recipient configurations and held-out cases; prove adapter updates leave source table bytes unchanged and retain disabled/random/frozen-only controls.
- **Semantic EngramPack:** distinguish exact retrieval of supplied vectors from natural-language understanding. Any text-query capability first needs a reproducible encoder/space contract, an exercised query path, leakage-audited tasks, pack controls, and measured encoder/retrieval cost.
- **Other mechanisms and allocation:** run selected full comparisons beyond wiring smokes, preserve every seed/outcome, and separate task quality from parameter/cache estimates and synchronized device timing.
- **Learning and cost evidence:** add a versioned observation protocol for held-out outcomes, actual token/checkpoint boundaries, threshold censoring, wall/device time, and memory. Update duration is not end-to-end experiment time.
- **Hardware:** CPU/Apple evidence cannot close CUDA/HIP/ROCm/XPU or real cross-host execution gates.

## Dense-lm-v1 descendants

The canonical lifecycle now promotes dense-lm-v1 as a bounded three-seed learning reference: seeds 42, 17, and 73 each completed 4,096 steps / 4,194,304 tokens; all fixed terminal gates passed; and endpoints were reached while learning, with no plateau observed. Terminal losses were 2.4824737093453306, 2.484718531778414, and 2.470433681211826, respectively. These results do not establish chat quality; OOD capability cards remain descriptive. Candidate validation passes independently; the unrelated Engram global validation error remains visible.

**Finding — dense-lm-token-budget-v1:** Increasing supervised target exposures from 4.19M to 16.78M consistently reduced held-out loss across all three dense reference seeds, but fixed-panel generation behavior did not improve monotonically. No plateau was observed at the terminal budget. The [scale-study control audit](../../experiments/research/dense-lm-scale-v1/protocol.md#prior-control-audit) verifies the unchanged pre-parent decay horizon and full-state lineage. This is not an automatic promotion; `dense-lm-v1` remains the promoted parent/reference.

The completed [dense-lm-scale-v1 comparison](../../experiments/research/dense-lm-scale-v1/results.md) held the same 16.78M supervised targets, fixed dataset/optimizer/decay, ROCm backend and six greedy prompts across the mature ~30M dense reference and the ~50M dense candidate. At all three seeds the larger model had lower terminal held-out loss by 0.0496–0.0607 nats/target and higher runtime/memory cost; fixed stories remained mixed, including regressions in bread/object and causal continuity. [Frozen protocol](../../experiments/research/dense-lm-scale-v1/protocol.md), [input registration](../../experiments/research/dense-lm-scale-v1/preregistration.md), and [raw matched observations](../../experiments/research/dense-lm-scale-v1/evidence.json) preserve the decision boundary. No further baseline is promoted; this is neither an architecture novelty claim nor evidence of generally better prose.

The executed [dense-lm-decoding-v1 study](../../experiments/research/dense-lm-decoding-v1/results.md) reused only the three paired mature checkpoints: 22 development prompts selected a supported non-greedy policy by a frozen mechanical rule, then 55 separate prompts were generated once under greedy and the selected policy. Sampling reduced n-gram loops in both widths but also exposed entity drift and an explicit color contradiction; matched 50M-versus-30M mechanical results on the independent prompts are mixed. An unopened human-review verdict is **unavailable**, not inferred from repetition or LM loss. The original six-prompt deterministic regression decoder and both preceding studies remain unchanged; [raw cells](../../experiments/research/dense-lm-decoding-v1/evidence/summary.json), [prompt hashes](../../experiments/research/dense-lm-decoding-v1/preregistration.json), and a [blind-review protocol](../../experiments/research/dense-lm-decoding-v1/review.md) are retained.

**Finding — tinystories-dense-30m-data-rich-v1 (unpromoted):** The separately preregistered data-rich dense 30M seed-42 run completed 100,663,296 supervised target exposures and a verified full-state endpoint. Its full final 8,000-story held-out loss was 1.610636 nats/native target. On the identical frozen 256 raw stories, the new/old-30M/old-50M bits per UTF-8 byte were 0.668729/0.975226/0.955759; the opened test prompts still contained color contradictions. Source, tokenizer, context, schedule, and precision changed together, so no causal factor or subjective-quality ranking is established. [Results and evidence](../../experiments/research/tinystories-dense-30m-data-rich-v1/results.md) preserve the endpoint and raw outputs; an independent blinded behavior protocol is the next test, not automatic retraining or baseline promotion.

**Open evidence question — independent prose preference:** a sealed
[Surface Review v1](surface-review-v1.md) import can collect one self-blind
reviewer's descriptive votes on existing outputs, but no population preference
or inter-rater agreement follows. Establish whether study prompts are truly
train-disjoint; then preregister a broader independently reviewed prompt
sample and multiple readers before interpreting a subjective ranking. Neither
those votes nor Tier-2 triage diagnostics isolate the effects of data variety,
depth, tokenizer, budget, context and precision changed together in the
data-rich comparison. This is a proposed scientific evidence gate, not a
software implementation task or a promotion decision.

The following descendants remain proposals, not executed work:

- **dense-lm-generation-degeneration-v1:** a still-proposed *cross-milestone* study of seed, repetition, entity continuity and object/color continuity beyond the mature-endpoint decoding comparison. Preserve the established six-prompt regression panel and do not reuse the opened decoding test set for selection.
- **MLA, Engram, FFN thinning, MoE, and sparse attention:** evaluate each as a separate controlled descendant, preserving baseline conditions and binding protocol/evidence before execution.

Every further descendant must retain the baseline conditions for factors not explicitly varied, declare and bind its protocol and evidence before execution, and remain a proposal until actually run and reviewed.

## Deferred capability: teacher-derived semantic representations

**Status: not implemented.** Verified semantic retrieval consumes already encoded canonical vectors; it does not encode prompt text. Current toy worlds use structured keys, not natural-language embeddings. Standard training and generation do not construct semantic query batches. These interfaces and tests establish bounded retrieval mechanics only.

The existing `sparselab engram pack compile` command packages caller-supplied vectors. It does not load a teacher, extract hidden states, choose layers, pool source text, or account for teacher compute. Do not describe manual vectors, record compilation, or the structured toy encoder as teacher knowledge distillation.

A teacher-derived pack remains gated until all of the following are exercised:

1. A frozen, locally available teacher with immutable revision/checkpoint identity; no implicit download or remote code execution.
2. Reproducible query and value representations with declared encoder identities and one explicit compatible feature-space contract.
3. One verified pack used by at least two recipient configurations, with unchanged pack bytes and separate recipient adapters.
4. Held-out evaluation against correct, disabled, random, conflicting, and incomplete pack controls; compiler inputs exclude validation, test, and capability-card content.
5. Source/license/provenance plus measured teacher and compiler costs before any transfer or amortization claim.

Until those conditions hold, do not add an implicit `teacher compile` path, call the synthetic key encoder a text encoder, or describe current retrieval as natural-language understanding.

## Required contract for a future teacher-pack compiler

When the gate is met, compilation is an explicit offline command over already-authorized local assets. Its output is a versioned, verified semantic EngramPack; pack creation never runs implicitly during scaffold, training, evaluation, or dashboard browsing. Any new manifest fields require a versioned schema and must preserve compatibility with existing records-only, lexical, and externally supplied semantic packs.

A compiled pack and its sidecar provenance must bind at least:

- teacher model identifier, immutable revision, checkpoint/config digest, tokenizer digest, architecture, license, and local source identity;
- extraction layer(s), hidden-state selection, pooling/window policy, text template, truncation limits, input-token accounting, dtype, and normalization;
- independent key and value encoder identities, widths, representation-space identity, row/record mapping, and compiler algorithm/version/config/source digest;
- source-record identities, split ownership, license/attribution, skipped/invalid counts, and exact content hashes;
- teacher forward tokens, compiler wall/device time, peak memory, output bytes, and any cached intermediates.

The pack stores representations and provenance, not teacher weights or a promise that the teacher's knowledge is true. A stable hash proves artifact identity, not encoder quality, factual correctness, license sufficiency, retrieval quality, or recipient compatibility. Never infer compatibility from equal vector widths alone.

## Comparisons and cost accounting

A later controlled comparison should hold source split, tokenizer, teacher checkpoint, student initialization, optimizer, student target-token budget, and evaluator fixed. Keep these arms distinct:

1. **Student baseline:** student trained without teacher targets or a semantic pack.
2. **Traditional distillation:** frozen teacher supplies the declared targets; student has no pack.
3. **Teacher + pack / student + same pack:** the same frozen teacher and train-only source produce a verified pack; the student receives that exact pack under the declared training/evaluation protocol.
4. **Frozen-recipient transfer:** freeze each recipient and its adapter, then attach the same verified pack without updating recipient weights.

Report task-level language-model, factual/semantic, reasoning, and unseen-pack results separately. Preserve negative, unknown, conflicting, incomplete, and unavailable outcomes; do not replace them with a pooled quality score. A recipient using a new adapter is a separate trained artifact, not zero-cost transfer.

Account separately for teacher source/input tokens, teacher output/distillation tokens, pack-compiler tokens and forwards, compiler wall time and peak memory, pack storage, student training tokens and targets, student training time, query-encoder and retrieval latency, and recipient adaptation. Record hardware/backend and measurement method. Any amortized figure names the number of recipients and includes pack creation and storage; compiler cost is never treated as free or hidden. Token counts are not interchangeable with FLOPs, energy, wall time, or quality.

## Teaching shapes and boundaries

The intended walkthrough makes the feature path explicit:

`source text → frozen teacher hidden states [B,T,D_T] → declared pooling/window rule → keys [M,K] and values [M,V] → encoded query [B,K] or [B,T,K] → retrieved values [B,T,V] → recipient projection/gate → residual [B,T,D_S]`.

`D_T`, key width `K`, value width `V`, and recipient width `D_S` are independent. The teacher is an offline producer; the pack is frozen; the recipient adapter is trainable only where the experiment says so. Explain which components run at pack-build time, student-training time, and inference time. Do not imply the teacher is absent at query time unless the query path actually avoids loading it.

Current source boundaries: `SemanticQueryBatch` in `src/sparselab/engram/semantic.py` accepts pre-encoded vectors; `SemanticRetriever` verifies and retrieves from pack assets; `SemanticMemoryAdapter` applies the retrieved values. `src/sparselab/data/toy_worlds.py` encodes canonical structured keys for deterministic synthetic controls. These are useful interfaces and test fixtures, not a macro-model compiler. See [the current runtime contract](semantic-memory.md) for exact limits and testable behavior.
