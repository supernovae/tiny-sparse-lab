# ADR 0018: Chat-native, artifact-bound small-model experiments

## Status

Accepted; refines [ADR 0017](0017-versioned-capability-cards.md).

## Context

The repository could train and generate, but its first card confused different record IDs with independent held-out wording. Chat silently cropped arbitrary context. Capability outputs overwrote earlier checkpoints, comparisons trusted configured rather than actual budgets, and inference used mutable external tokenizer paths. Those gaps prevented credible progress claims.

## Decision

- Make chat a canonical PyTorch checkpoint interface. Train local conversation JSONL and the synthetic curriculum with the same explicit transcript format used by chat and cards.
- Freeze a verified checkpoint generation once. Use verified run-owned tokenizers, prepared validation data, and portable packages. Preserve exact prompts, responses, generation settings, runtime and checkpoint identities in immutable evidence files.
- Separate training-seen **retention**, held-out wording **recall**, and conversation-local **override**. Do not hide a failed control behind a positive score on another card.
- Compare matched source/tokenizer/data identities, seed, actual trained tokens and steps, and runtime. Allow one explicit architectural axis; publish parameter counts and every changed field. More total parameters are not evidence of efficiency.
- Let users add declarative full-answer cards and licensed JSONL data without writing Python. Split auditing remains the experimenter's responsibility beyond exact conversation-overlap detection.
- Preserve negative and exploratory observations. A single seed or synthetic task is not a general Engram, chat-quality, or scaling result.

## Consequences

A tiny trained model can be useful for a well-defined learned task even while failing new phrasing or context changes. Progress is incremental: prove acquisition, test generalization, test competing explanations, then increase task breadth or scale. Defaults remain simple CPU reference runs; larger configurations reuse the same evidence format within single-host resource limits.

The PyTorch path is canonical. MLX native training, mixed precision, KV-cached serving, broad conversational evaluation, and distributed execution are not implied by this decision. Unsupported paths must fail explicitly or be documented as experimental, not silently represented as equivalent evidence.
