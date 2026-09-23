# ADR 0005: Greedy byte-memory generation reuses the raw prompt stream

## Status

Accepted for Milestone 0.5.

## Decision

Greedy generation for `model.memory: byte` maintains an explicit raw UTF-8 byte stream alongside model input IDs. Prompt token offsets produce the initial causal addresses using the same raw suffix hashing contract as prepared data. After each generated non-special token, its ByteLevel-decoded UTF-8 text extends that stream and produces the address used on the next forward call. Token IDs and addresses are cropped together to model context length.

Empty prompts start from structural BOS with byte address zero. EOS terminates before appending an address. The original prompt string is preserved and generated text is decoded separately for the returned result.

Projected KV decoding is request-local and bounded to the active prompt-plus-
generation budget, capped by model context. Token IDs use the same bounded
request storage so n-gram suffix lookup remains exact without repeated
full-prefix copies. At a context rollover generation rebuilds that storage from
the cropped IDs and addresses, resetting RoPE positions exactly as the
uncached reference path does. Cache allocations are inference-only and never
belong to model weights or checkpoints.

## Consequences

This supports the local tokenizer's ByteLevel decoder and deterministic greedy generation. It does not prove that generated token decoding is a general tokenizer-agnostic byte reconstruction interface, and it makes no retrieval-quality or cross-tokenizer-transfer claim.

## References

- [ByteLevel tokenizers component](https://huggingface.co/docs/tokenizers/components)
- [Tokenizer-Agnostic Engram](https://arxiv.org/html/2607.29065)
