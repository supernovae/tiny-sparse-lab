# ADR 0004: Raw UTF-8 byte addressing is a preprocessing contract

## Status

Accepted for Milestone 0.4.

## Context

The 0.3 model accepts only token IDs, so it cannot infer byte-equivalent spans from IDs alone. A ByteLevel BPE token ID is tokenizer-local; decoding an individual ID is not a safe general reconstruction boundary. Byte-equivalent addressing therefore belongs at tokenization/preparation, not as an implicit decoder-side conversion.

## Decision

Define a standalone byte-address utility over explicitly supplied UTF-8 byte spans. The canonical representation is raw UTF-8 with **no Unicode normalization**, case folding, whitespace collapsing, or escaping. A polynomial hash consumes byte values with an explicit length separator. Equal raw byte spans produce equal hashes independent of tokenizer segmentation; distinct spans may collide only through bounded table reduction.

The 0.4 deliverable is a tested address contract and prepared span interface. It does not attach byte addresses to the language-model training path yet: doing that safely requires a versioned prepared-artifact extension that aligns each prediction position with its causal source-byte suffix. That integration follows only after the artifact contract is independently testable.

## Consequences

Raw text that is visually equivalent but has different UTF-8 bytes intentionally receives different addresses. Special tokenizer tokens are structural model IDs, not source-text bytes. This milestone makes no knowledge-transfer, retrieval-quality, or tokenizer-agnostic training claim.

## References

- [Tokenizer-Agnostic Engram](https://arxiv.org/html/2607.29065)
- [polyhash-engram prototype](https://github.com/jararap/polyhash-engram)
