# Chat-oriented training

Start with [From flashcards to a useful local assistant](from-toy-to-useful.md) for the beginner explanation, the runnable 3.3M `instruction_starter.yaml` recipe, and the distinction between a familiar chat prompt and trained conversational ability.

The optional `instruction_reference` source is a deterministic, offline synthetic curriculum for exercising the local chat transcript path. Each document uses the same plain-text form consumed by `sparselab chat`:

```text
System: You are a concise local assistant.

User: Explain causal attention in one sentence.

Assistant: Causal attention lets each position use earlier positions but not future ones.
```

The curriculum contains bounded arithmetic, word reversal, attribute lookup, concise definitions, sentence completion, and greetings. Its train and validation index ranges differ, but finite task instances/templates can repeat; this is a format/curriculum diagnostic, not a generalization benchmark. For a smaller, measured starting point, use the [chat recall capability pair](capabilities.md) with genuinely held-out query combinations.

## Reference run

```sh
uv run sparselab tokenizer train configs/tokenizer_instruction_8k.yaml
uv run sparselab data prepare configs/instruction_100m.yaml
uv run sparselab inspect configs/instruction_100m.yaml --json
uv run sparselab train configs/instruction_100m.yaml --run-id instruction-100m-pilot --stop-after-step 2
uv run sparselab checkpoint verify runs/instruction-100m-pilot/checkpoints/latest.json --json
uv run sparselab train configs/instruction_100m.yaml --run-id instruction-100m
uv run sparselab chat instruction-100m --system "You are a concise local assistant." --max-new-tokens 32
```

`instruction_100m.yaml` is the initial larger reference: 104,843,648 dense parameters, a 20-million-target-token curriculum budget, effective batch size 16, and block activation checkpointing. It is intentionally a 100M—not 200M—configuration: establish that the corpus, transcript, checkpoint, and held-out behavior are useful before paying the substantially higher memory and training cost of a 200M experiment.

The pilot is a real two-update execution/checkpoint check, not a quality result or an implicit warmup for the fresh run. Current `stage --through warmup` does not execute a measured training pilot. The 100M preset is not a pretrained assistant: its corpus contains repetitive `Reference N:` tasks, and the full training/evaluation must still be performed before claiming useful chat. Its 256-token context includes the system prompt, history, current question, role labels and reserved answer.

## Evaluation boundary

Use chat to inspect behavior, but do not cite a fluent-looking reply as an instruction-following result. Compare the trained run with a matched dense base run where feasible and retain prompts, exact transcript formatting, generation settings, tokenizer, model configuration, token budget, and checkpoint identity. A future conversational corpus must have explicit license/provenance and a held-out multi-turn evaluation before it replaces or extends this synthetic reference.

## Local conversation corpora

Use `dataset.source: local_chat` for a corpus you own or are licensed to use. Supply distinct UTF-8 JSONL files, an explicit license declaration, and acquisition budgets. Paths are relative to the configuration file. Use the same dataset section in your tokenizer-training and model-training configs:

```yaml
dataset:
  source: local_chat
  train_path: ../data/domain-train.jsonl
  validation_path: ../data/domain-validation.jsonl
  license: CC0-1.0
  cache_dir: ../artifacts/data
  train_max_documents: 10000
  validation_max_documents: 1000
  train_max_tokens: 1000000
  validation_max_tokens: 100000
```

Each line is a complete conversation, for example:

```json
{"messages":[{"role":"system","content":"Answer from the workshop handbook."},{"role":"user","content":"Which drawer holds the torque wrench?"},{"role":"assistant","content":"The labeled blue drawer."}]}
```

The optional system message must be first. User and assistant messages alternate in complete pairs; extra fields, unknown roles, blank content, and malformed JSON fail with a file/line error. Multi-turn examples use the exact same transcript formatter as `chat`. The reader streams documents; packing appends EOS. Declare the **actual** license or permission, not the example license above.

Prepared-data identity includes file contents, not just filenames. Changed data cannot reuse stale token arrays. Exact duplicate conversations across train and validation are rejected; this does not detect paraphrases, overlapping facts, or near-duplicates. Split by the unit relevant to your hypothesis (document, customer, task family, or template) before ingestion. Keep separate test cases in a capability card; never put validation examples into the test card.

Tokenizers are immutable artifacts. Training into an existing output directory succeeds only if source-content/training provenance and the tokenizer digest match. Otherwise choose a new tokenizer output directory. Referencing an already-trained tokenizer in a model config remains valid, including a pretrained tokenizer from another corpus.

Training is currently ordinary next-token loss over the **whole transcript**, not assistant-only supervised fine-tuning. There are no tool/function-call roles, preference optimization, or hidden commercial chat templates. Start with short, verifiable responses; train and evaluate the same system prefix. For a new domain, copy the small chat configuration, change dataset/tokenizer paths and budgets, inspect resource usage, and evaluate a separately versioned task card before increasing model size.
