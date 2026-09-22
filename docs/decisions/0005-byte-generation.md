# ADR 0005: Greedy byte-memory generation reuses the raw prompt stream

## Status

Accepted for Milestone 0.5.

## Decision

Greedy generation for `model.memory: byte` maintains an explicit raw UTF-8 byte stream alongside model input IDs. Prompt token offsets produce the initial causal addresses using the same raw suffix hashing contract as prepared data. After each generated non-special token, its ByteLevel-decoded UTF-8 text extends that stream and produces the address used on the next forward call. Token IDs and addresses are cropped together to model context length.

Empty prompts start from structural BOS with byte address zero. EOS terminates before appending an address. The original prompt string is preserved and generated text is decoded separately for the returned result.

## Consequences

This supports the local tokenizer's ByteLevel decoder and deterministic greedy generation. It does not prove that generated token decoding is a general tokenizer-agnostic byte reconstruction interface, and it makes no retrieval-quality or cross-tokenizer-transfer claim.

## References

- [ByteLevel tokenizers component](https://huggingface.co/docs/tokenizers/components)
- [Tokenizer-Agnostic Engram](https://arxiv.org/html/2607.29065)
