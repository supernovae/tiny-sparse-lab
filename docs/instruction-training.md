# Instruction reference training

The optional `instruction_reference` source is a deterministic, offline synthetic curriculum for exercising the local chat transcript path. Each document uses the same plain-text form consumed by `sparselab chat`:

```text
System: You are a concise local assistant.

User: Explain causal attention in one sentence.

Assistant: Causal attention lets each position use earlier positions but not future ones.
```

The curriculum contains bounded arithmetic, word reversal, attribute lookup, concise definitions, sentence completion, and greetings. Train and validation streams use disjoint deterministic index ranges. It is not a general instruction corpus, a factual knowledge source, or evidence of broad chat ability.

## Reference run

```sh
uv run sparselab tokenizer train configs/tokenizer_instruction_8k.yaml
uv run sparselab data prepare configs/instruction_100m.yaml
uv run sparselab inspect configs/instruction_100m.yaml --json
uv run sparselab stage configs/instruction_100m.yaml --through warmup --output /tmp/sparselab-instruction-stage
uv run sparselab train configs/instruction_100m.yaml --run-id instruction-100m
uv run sparselab chat instruction-100m --system "You are a concise local assistant."
```

`instruction_100m.yaml` is the initial larger reference: 104,843,648 dense parameters, a 20-million-target-token curriculum budget, effective batch size 16, and block activation checkpointing. It is intentionally a 100M—not 200M—configuration: establish that the corpus, transcript, checkpoint, and held-out behavior are useful before paying the substantially higher memory and training cost of a 200M experiment.

## Evaluation boundary

Use chat to inspect behavior, but do not cite a fluent-looking reply as an instruction-following result. Compare the trained run with a matched dense base run where feasible and retain prompts, exact transcript formatting, generation settings, tokenizer, model configuration, token budget, and checkpoint identity. A future conversational corpus must have explicit license/provenance and a held-out multi-turn evaluation before it replaces or extends this synthetic reference.
