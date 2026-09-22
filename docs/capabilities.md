# Capability experiments

Capability cards turn a narrow architectural hypothesis into a repeatable local experiment. A card is versioned, immutable in code, and contains fixed held-out prompts, an explicit scorer, and a scoped hypothesis. It is neither a general benchmark nor a proxy for overall model quality.

## First card: `engram-recall-v1`

`engram-recall-v1` measures exact recovery of a fixed code-to-value association from held-out record wording. The paired `engram_recall` training source teaches the association through varied records; the card uses separate deterministic record IDs and a `first_normalized_word_exact_v1` scorer.

This is an **associative-recall experiment**, not a test of general reasoning, retrieval, factual knowledge, or chat ability. A dense model can solve it through ordinary weights and attention; an Engram result is useful only relative to a matched dense baseline.

## Run a matched experiment

```sh
uv run sparselab tokenizer train configs/tokenizer_smoke.yaml
uv run sparselab data prepare configs/capability_recall_dense_cpu.yaml
uv run sparselab train configs/capability_recall_dense_cpu.yaml --run-id recall-dense
uv run sparselab train configs/capability_recall_ngram_cpu.yaml --run-id recall-ngram
uv run sparselab capability evaluate recall-dense engram-recall-v1
uv run sparselab capability evaluate recall-ngram engram-recall-v1
uv run sparselab capability compare recall-dense recall-ngram engram-recall-v1
```

`capability compare` requires matched controls: backbone dimensions, attention, dataset, tokenizer, training budget, optimizer, and runtime must be equal. It intentionally permits only the Engram memory settings to differ. A positive score delta supports the named card hypothesis only; a zero or negative delta is equally valuable evidence.

## Iterate without discarding earlier work

Use the same card and scorer at progressively larger configurations. Retain each run’s configuration, tokenizer, prepared data, checkpoint evidence, and card result. This creates a scale series instead of a one-off demo:

```text
recall dense / ngram at small smoke scale
→ recall dense / ngram at 10M
→ recall dense / ngram at 25M
→ a new card with a different hypothesis
```

Do not compare scores across changed cards, tokenizers, or data contracts. For a parameter-efficiency claim, construct a separate matched-total-parameter pair and state that matching policy in the card documentation. For a data-efficiency claim, keep the architecture fixed and compare score curves at declared token budgets.

## Add a card deliberately

A new card needs all of the following before it becomes a claim surface:

1. A versioned name and a one-sentence falsifiable hypothesis.
2. Fixed prompts/cases outside the train document range.
3. A deterministic scorer and explicit normalization policy.
4. A declared dense baseline and allowed variant differences.
5. A run artifact containing card digest, responses, score, checkpoint identity, and conditions.

Good next Engram cards could test distance-sensitive recall, collision sensitivity, multi-order addressing, or parameter-matched recall. Each remains a separate experiment; success on one card never silently broadens the claim.
