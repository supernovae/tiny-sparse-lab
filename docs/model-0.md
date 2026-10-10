# Run and inspect DevMind MODEL-0

**MODEL-0 completed base pretraining; it is not a polished assistant.** The
[canonical result](../experiments/research/history/devmind-pretrain-v5/model0-result.json)
records one completed, evaluated, unpromoted run. It has no developer SFT or
chat-tuning stage. The retained greedy samples are repetitive and language-mixed:
a Python prefix produces repeated C/JavaScript-like blocks, SQL repeats
`--echo #`, and a shell prefix repeats comment markers. These negative outputs
remain part of the result; a successful API response does not change them.

| Recorded property | Value |
| --- | --- |
| Family / node | `devmind-v5-model0` / `model-0` |
| Parameters | 69,317,760 |
| Objective | Next-token base pretraining |
| Tokenizer vocabulary | 32,768 |
| Context | 1,024 tokens |
| Completed updates / supervised targets | 5,525 / 45,260,800 |
| Training runtime | ROCm, BF16, seed 42 |
| Run ID | `762c4104-3a3b-4226-afa6-402cc350356e` |
| Terminal generation | `step_00005525_gen_000004` |
| Checkpoint digest | `52aba43c269204ada5ce4101414f71aa7a8bbe116d5d1d58780eb2e2717d2709` |

The [Family declaration](../experiments/research/history/devmind-pretrain-v5/model0-family.yaml)
binds corpus, tokenizer, plan, checkpoint and evaluation identities. The
[execution record](../experiments/research/history/devmind-pretrain-v5/model0-execution.md)
preserves earlier censored attempts and their later same-attempt reconciliation.
`READY_FOR_NEXT_STAGE` means the declared evidence policy passed; it does not
mean instruction following, usefulness, promotion, publication or child resume
was demonstrated. [MODEL-1's protocol](../experiments/research/history/devmind-pretrain-v5/model1-protocol.md)
is a future comparison declaration, not evidence that a later model exists.

## Artifact lifecycle and availability

The v5 lifecycle is reviewed source declarations → frozen corpus release → LM
export → selected tokenizer → prepared arrays → locked ExperimentPlan and bound
Campaign → worker training → controller ingestion → immutable checkpoint →
checkpoint-bound evaluation/readiness. Source hashes, corpus identity, tokenizer
identity and model checkpoint digest identify different objects; do not replace
one with another or infer provenance from a human-facing model alias.

This is MODEL-0's recorded Forge-backed lifecycle. The current generic
[dataset lock/snapshot workflow](datasets.md) does not rewrite its historical
release, export, tokenizer or prepared-input identities; Forge contracts retain
their release/export gates. Inference verifies the retained run and its original
assets. New acquisition or legacy-input migration creates separate artifacts
for new execution, not a replacement history for this checkpoint. The array
cache returned by `data prepare` and a sealed staging `prepared` bundle are
different objects; neither alone is the complete run needed to load MODEL-0.

The repository contains small declarations and evidence references, **not the
large MODEL-0 weights or prepared arrays**. The canonical result records this
controller location on the original host. It is usable only when the validated
run and its assets actually exist there:

```sh
RUNS=/srv/sparselab/state/campaigns/devmind-v5-model0/417d5245f753873012f2d29b0c31a366bbe586d7ed9f12ef067456f3ad192fda/experiments/plan-devmind-v5-model0/controller
RUN_ID=762c4104-3a3b-4226-afa6-402cc350356e
CHECKPOINT=step_00005525_gen_000004
GENERATION="$RUNS/$RUN_ID/checkpoints/$CHECKPOINT"
uv run --locked --extra cpu sparselab checkpoint inspect "$GENERATION" --json
uv run --locked --extra cpu sparselab checkpoint verify "$GENERATION" --json
uv run --locked --extra cpu sparselab generate "$RUN_ID" \
  --runs-dir "$RUNS" --checkpoint "$CHECKPOINT" --backend cpu \
  --prompt 'def add(a, b):' --max-new-tokens 64 \
  --temperature 0 --top-k 0 --seed 42 --strict-context --show-prompt --json
```

These examples select CPU inference. For an already provisioned vendor runtime,
use `uv run --locked --no-sync` and its authorized runtime selection instead;
see [runtime setup](runtime.md#machine-local-runtime-environments). Do not sync
CPU wheels into that environment. A new CPU sample is a new exploratory
observation, not a reproduction claim for the original ROCm panel.

If that directory is missing, stop at the evidence records and use the
[recovery declaration](../experiments/research/history/devmind-pretrain-v5/model0-recovery.yaml)
to establish what can actually be restored. An archive/reference is not proof
that external payloads are available. No public model download or weight
publication is implied. Do not bypass validation with bare weights, replace the
tokenizer, alter original receipts, or silently retrain.

Retained steps 0, 2,048, 4,096 and 5,525 are recorded in the execution evidence.
Discover their actual manifests and verify each before using it; do not guess
generation suffixes or choose an endpoint by filename ordering. Use the
[TinyText model exploration guide](tinytext-model-guide.md) to compare two
verified paths and save exact prompts/settings. Use the [local API](local-api.md)
for raw `/v1/completions` first; chat is useful for investigating transcript
behavior but cannot supply the missing SFT training. Future checkpoints use the
same interfaces after their own provenance and runtime validation.
