# Research roadmap: teacher representations and semantic memory

This roadmap separates implemented mechanisms from gated research questions. A phase title is not a claim that every item in that phase is runnable. For current semantic-retrieval behavior, see [Verified semantic memory](semantic-memory.md) and [Portable Engram](../portable-engram.md).

## Program sequence

| Phase | Question and intended evidence |
|---|---|
| A — Learn, try, compare, share | Standalone mechanism lessons, controlled study scaffolds, readable static evidence, and independent learning routes. |
| B — Interactions and failure boundaries | Matched factorial cells, per-seed main effects and interaction deltas, explicit missing-cell handling, and failure-boundary sweeps. |
| C — Learning efficiency and cost | Preregistered milestones, censored thresholds, common-support curves, and separately measured training, compilation, retrieval, and hardware costs. |
| D — Executable memory and canonical worlds | Verified frozen packs, exact retrieval, ownership/site controls, changing synthetic worlds, and held-out pack/recipient checks. |
| E — Useful tasks and human evidence | Leakage-controlled task suites, licensed source builders, and separately collected blinded human judgments. |
| F — Allocation and narrower-network training | Provenance-bound ownership, neural-loss allocation, and distinct neural/total/active/token/FLOP regimes. |
| G — References and scaling | Pinned observational model references and carefully bounded scaling recommendations. |
| H — Compiler and teaching substrate | Compile frozen teacher representations into portable packs, test transfer across recipients, and account for every teacher/compiler/student cost. |

## Phase H status and gate

**Status: the design and teaching boundary are documented; the teacher-representation compiler is not implemented.** The existing runtime verifies semantic EngramPack assets and executes bounded exact retrieval. A `SemanticQueryBatch` contains already-encoded vectors; it does not encode prompt text. Current toy worlds use canonical structured keys, not macro-model representations or natural-language embeddings. The standard trainer and generation CLI do not supply a teacher encoder or semantic query batches. Consequently, current exact-retrieval and structured-world tests demonstrate runtime mechanics, not a useful teacher-derived semantic-transfer result.

The existing `sparselab engram pack compile` command packages caller-supplied vectors. It is not a representation compiler: it does not load a teacher, extract hidden states, choose layers, pool source text, or account for teacher compute. Do not describe manual vectors, record compilation, or the structured toy encoder as teacher knowledge distillation.

Phase H implementation must remain gated until a train-only, reproducible semantic representation path has been exercised through the verified pack runtime and shown to transfer across independently configured recipient models. The gate requires:

1. A frozen, locally available teacher and an immutable teacher revision/checkpoint identity; no implicit download or remote code execution.
2. Reproducible query and value representations with declared encoder identities and one explicit compatible feature-space contract.
3. A verified portable pack used by at least two recipient configurations, with unchanged pack bytes and separate recipient adapters.
4. Held-out evaluation against correct, disabled, random, conflicting, and incomplete pack controls. Compiler inputs must exclude validation, test, and capability-card content.
5. Source, license, provenance, and measured teacher/compiler/storage costs recorded before any transfer or amortization claim.

Until these conditions are met, do not add a `teacher compile` command, promote the synthetic key encoder into a text encoder, or label the current retrieval interface as natural-language understanding.

## Future compiler contract

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
