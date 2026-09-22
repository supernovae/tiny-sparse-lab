# Capability experiments

SparseLab is a small-model workbench: train a narrow capability, exercise it through the same chat interface a person uses, and measure whether an architectural change helped. A trained checkpoint is reusable experimental material, not a disposable demo. Keep positive, negative, and inconclusive results.

The aliases below are invented flashcards, not useful world knowledge. For their plain-language meaning and the progression to ordinary chat and domain tasks, read [From flashcards to a useful local assistant](from-toy-to-useful.md).

## A runnable chat-native experiment

```sh
uv run sparselab tokenizer train configs/tokenizer_chat_recall.yaml
uv run sparselab inspect configs/chat_recall_dense_cpu.yaml --json
uv run sparselab train configs/chat_recall_dense_cpu.yaml --run-id chat-dense
uv run sparselab train configs/chat_recall_engram_cpu.yaml --run-id chat-engram
uv run sparselab capability list
uv run sparselab capability describe chat-alias-recall-v1
uv run sparselab capability compare chat-dense chat-engram chat-alias-retention-v1
uv run sparselab capability evaluate chat-dense chat-alias-recall-v1
uv run sparselab capability compare chat-dense chat-engram chat-alias-recall-v1 --vary memory
uv run sparselab capability compare chat-dense chat-engram chat-context-override-v1 --vary memory
uv run sparselab chat chat-engram --system "Answer the requested alias with only its value." --max-new-tokens 12 --transcript /tmp/chat-engram.json
```

These configurations deliberately train small models on a bounded task rather than promise general conversation. The dataset and evaluator share the ordinary `User:` / `Assistant:` formatter. There is no lookup-tool shortcut at inference: responses come from trained model logits. The tokenizer is trained on the training split only.

### What the cards measure

- **`chat-alias-retention-v1`:** training-seen query prompts for every learned alias. This is an acquisition sanity check, explicitly not generalization. Compare step zero to the trained checkpoint before interpreting the harder controls.
- **`chat-alias-recall-v1`:** learned fixed aliases under held-out query wording. The associations are intentionally in training; test wording is not. This measures access to learned information, not discovery of unseen facts.
- **`chat-context-override-v1`:** a conversation-local assignment overrides a learned alias. This control distinguishes static memorization from using supplied conversational context. Failure matters even when static recall succeeds.
- **`engram-recall-v1`:** historical wiring diagnostic. Its prompt contains the answer, and separate record IDs do not establish independent held-out wording. Its first-word scorer accepts extra text. Do not cite it as proof of recall generalization, chat ability, or Engram benefit. Use the newer full-answer chat cards for those narrow questions.

New cards contain frozen prompts, exact normalization/scoring, generation settings, limitations, and a digest. New chat cards use full normalized-answer equality: a correct first word followed by junk does not pass. Prompt plus completion budget must fit; evaluation refuses overflow instead of silently deleting evidence from the context. Balanced fixed-answer controls are reported beside model scores.

The bundled pair trains for up to 1,536 steps/393,216 targets. Epoch-end short batches can make the actual token count smaller when the step limit is reached; comparisons use checkpoint counters. The [recorded review experiment](project-review.md) reached 24/24 retained associations for both models but found no Engram advantage on the held-out or override cards. This is a useful negative architecture result, not a failed framework.

## Evidence and controlled comparisons

Each result retains every prompt, expected answer, generated response, score, and generation settings. Identity includes the exact resolved checkpoint digest, actual step and trained target tokens, run configuration, run-owned tokenizer and data hashes, code identity, runtime, and parameter inventory. Content-addressed filenames preserve earlier checkpoints and reruns instead of overwriting the latest score. The comparison itself is saved too.

```sh
uv run sparselab capability evaluate chat-engram chat-alias-recall-v1 --checkpoint best.json
uv run sparselab evidence chat-engram --json
```

`best.json` means best **validation loss**, not best capability score. Select a checkpoint on validation, not by searching test scores. For a learning curve, evaluate explicit generation directories at preregistered token budgets; preserve the step-zero result as the untrained control. Compare architectures at the same actual budget, not just the same configured maximum.

`capability compare` rejects mismatched card/protocol, seed, actual step/token budget, tokenizer contents, train/validation contents, source-code identity, and runtime. Paths and logging/cadence differences are not scientific differences. Select one permitted experimental axis:

| `--vary` | Permitted change | Interpretation |
|---|---|---|
| `memory` (default) | Engram configuration | Added memory may add total parameters; not a parameter-efficiency result. |
| `attention` | Attention configuration | Attention ablation, not a kernel speedup claim. |
| `ffn` | Dense/MoE feed-forward settings | Report total and active parameters; routing is not free. |
| `scale` | Backbone size | A declared scale comparison, not a mechanism-isolated result. |
| `none` | No scientific configuration changes | Repeatability control. |

Outputs show exact changed fields, paired gains/losses, score delta, and scope-limited interpretation. One seed, a few synthetic cases, or a positive delta is **not** statistical confirmation. Repeat a preregistered seed set, compare each seed to its matched counterpart, retain failures, and report the distribution of deltas. No automatic significance or universal superiority label is issued.

## Bring a new task without changing Python

Export a built-in card as a declarative starting point:

```sh
uv run sparselab capability describe chat-alias-recall-v1
```

Save that JSON, remove its `digest` when editing, and choose a new versioned `name`, hypothesis, limitations, cases, expected responses, and response-token budget. Keep the declared normalization and greedy protocol; update the categorical-answer controls to reflect the cases. Pass the JSON path wherever a card name is accepted. The strict schema rejects unknown fields; custom executable scorers are not loaded. Changing prompts, scorer, or decoding creates a different digest. Never compare it as if it were the old card.

A useful task needs a training source and a separately held-out card. Use [local conversation JSONL](instruction-training.md#local-conversation-corpora) for your own data. Ensure the task's answers are not exposed in test prompts unless testing in-context use. Keep semantic/template overlap audits: distinct row IDs or file hashes alone do not prove generalization.

## Scale without losing the experiment

1. Start with a CPU pair and verify that at least one trained model improves over its step-zero score. If neither learns, fix corpus, context, or optimization before adding mechanisms.
2. Freeze the task/card, source split, tokenizer, optimizer, seed set, and target-token budget. Create a matched dense/Engram pair at the next backbone size; `inspect` the actual parameter counts.
3. Repeat each pair at declared budgets. Log both absolute task scores and paired deltas, held-out loss, memory, and synchronized training throughput. Do not tune on the held-out test set.
4. A parameter-efficiency question requires a separate matched-total-parameter design. A data-efficiency question requires fixed-model score-versus-token curves. Neither follows automatically from adding a table.
5. Add capabilities one at a time: collision stress, longer-distance context, compositional lookup, and realistic domain conversations need their own frozen cards and controls.

The same checkpoint/chat/card path works for larger PyTorch configurations that fit one host. It does not promise arbitrary scale: FP32 memory, training data quality, reference attention costs, no KV cache, and single-device execution are real limits. MLX native training currently does not participate in this chat/capability evidence path. See [model scaling](model-scaling.md), [runtime](runtime.md), and [experiment evidence](evidence.md).
