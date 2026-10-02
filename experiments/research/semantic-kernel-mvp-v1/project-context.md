# Tiny Sparse Lab project context: semantic kernel track

Use this file as a source in the existing ChatGPT Project. Git is the source of
truth for declarations; run artifacts establish evidence. Chat summaries are
context, not results. Refresh these files from the checked-out commit when work
changes. This file was created from the 2026-10-02 kernel/interface discussion.

## Our hypothesis

A small language model may become useful as a controller of external structured
facts and executable capabilities. We want to learn whether stable interfaces
reduce domain-instance training and improve grounded composition, not assume
they do. N-gram portability remains a separate mechanism experiment.

## Current position

Stage 0 implements only a Python AST compiler and typed read-only replay runtime.
The domain is real Tiny Sparse Lab source, not invented fact worlds. No model
was trained or tested. No benefit over normal tools/RAG or large models is known.
The bank is not an EngramPack. Subsequent stages and gates are in `protocol.md`;
code dependencies are in `implementation-plan.md`.
`substrate-observation.md` records the real-source wiring run: 199 files,
2,765 definitions, 37,809 unresolved call sites and four scripted probes.
Twelve focused software tests pass. These counts are not model accuracy.

## Project instruction to paste

When helping with this track, first read the latest semantic-kernel protocol,
implementation plan and stage evidence from Git. State the current hypothesis,
stage and smallest missing gate. Keep observed facts, inference and proposals
distinct. Do not modify active DevMind or Engram campaigns. Do not begin large
training to compensate for an untested interface. Preserve negative outcomes,
source/split ownership and complete costs. Suggest new ideas as future tests;
implement only the current bounded stage. Explain each result using an actual
human-readable source/task example. End a completed stage with evidence, what
remains unproved, and the next decision.

## Next coding-agent task

Read `AGENTS.md` and `experiments/research/semantic-kernel-mvp-v1/`.
Implement Stage 1's sealed real-source task bundle, deterministic baseline,
explicit local-model adapter and evaluator in a normal branch. First propose
and seal module-family/composition partitions, independent expected answers,
unknown/ambiguous cases, exact model/tokenizer/chat-template identity and bounded
generation settings. Test the ABI with one local model that is already available
and authorized; if none exists, complete model-independent software and record
the model gate as blocked. Do not download or train a model implicitly. Include
closed-book, same-facts-as-text and ordinary JSON-tool comparisons. Retain every
case outcome and scorer provenance. Do not emit training traces from dev/test
cases. Publish the measured report before proposing structure fine-tuning.

## Parking list

Learned graph encoders; direct hidden-state record injection; portable semantic
adapters; engrams as interface hints; hypothesis-test selection by information
gain; orbital diagnosis; cross-model capability packages. These are future
experiments, not explanations of current results.
