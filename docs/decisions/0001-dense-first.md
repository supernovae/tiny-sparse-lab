# ADR 0001: Dense decoder before sparse mechanisms

## Status

Accepted for the baseline; current execution scope is defined by [ADR 0015](0015-single-host-extension-boundaries.md).

## Decision

Ship a small, explicit dense causal decoder as the first runnable laboratory. Keep input IDs at the model boundary, use ordinary causal attention and a dense SwiGLU feed-forward network, and account for all dense parameters honestly. MoE routing, n-gram memory, hashing, latent-KV attention, distributed execution, and transfer experiments are out of scope until their interfaces and diagnostics can be evaluated independently.

The implementation uses PyTorch float32 device, RNG, synchronization, memory, serialization, and AdamW APIs. It prefers MPS when available, then CUDA, then CPU. Reproducibility is scoped to a fixed environment; it does not claim cross-device or cross-version bitwise identity.

TinyStories remains the baseline remote corpus, pinned to `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, config `default`, with `text`, `train`, and `validation` splits. Prepared artifacts retain source and CDLA-Sharing-1.0 attribution metadata without redistributing corpus text. Later bounded-streaming support adds opt-in FineWeb-Edu and Cosmopedia train streams: validation deterministically skips the configured train-document prefix. Network smoke exercised FineWeb-Edu `default` at `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` and Cosmopedia `khanacademy` at `0ae6ec63f91742bd2d1eaef4f02232c55d719385`.

Use Hugging Face Tokenizers BPE with ByteLevel pre-tokenization and its matching decoder. Training and packing provenance are implemented locally. The dashboard uses independently authored Streamlit and Plotly pages; it borrows no MiMo assets, branding, JavaScript, or CSS.

Original project code is MIT licensed. Dataset and third-party reference licenses remain their own.

## Consequences

The baseline is deliberately not a copy of Qwen4-Exp’s hybrid PLE/QSA/MoE architecture, DeepSeek Engram’s surrounding training code, or any router implementation. Later work separates MoE selection, normalized weights, dispatch, and diagnostics; separates n-gram addressing, table values, and backbone adapter; and treats byte-equivalent hashing as an addressing property rather than guaranteed knowledge transfer.

The baseline teaches RMSNorm, RoPE, causal masking, SwiGLU, residual connections, tied embeddings, next-token prediction, and explicit parameter accounting before adding sparse claims.

## References

- [PyTorch MPS notes](https://docs.pytorch.org/docs/2.14/notes/mps.html), [MPS APIs](https://docs.pytorch.org/docs/2.14/mps.html), [reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html), and [serialization](https://docs.pytorch.org/docs/2.14/notes/serialization.html)
- [Qwen4-Exp documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen4_exp) and [modular source](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen4_exp/modular_qwen4_exp.py)
- [DeepSeek Engram](https://arxiv.org/abs/2601.07372), [demo](https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py), and [license](https://github.com/deepseek-ai/Engram/blob/main/LICENSE)
- [Tokenizer-Agnostic Engram](https://arxiv.org/html/2607.29065) and [prototype](https://github.com/jararap/polyhash-engram)
- [Swiss AI MoE](https://github.com/swiss-ai/MoE/blob/main/moe.py), [Flaxformer routing](https://github.com/google/flaxformer/blob/main/flaxformer/architectures/moe/routing.py), and [Megatron router](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/router.py)
- [TinyStories card](https://huggingface.co/datasets/roneneldan/TinyStories/raw/main/README.md) and [metadata](https://huggingface.co/api/datasets/roneneldan/TinyStories)
- [Datasets loading](https://huggingface.co/docs/datasets/package_reference/loading_methods), [Tokenizers components](https://huggingface.co/docs/tokenizers/components), and [Tokenizer API](https://huggingface.co/docs/tokenizers/v0.22.0/api/tokenizer)
- [MiMo dashboard](https://mimo.xiaomi.com/rl/)
- [LLaMA](https://arxiv.org/abs/2302.13971), [RoFormer](https://arxiv.org/abs/2104.09864), [RMSNorm](https://arxiv.org/abs/1910.07467), [SwiGLU](https://arxiv.org/abs/2002.05202), and [AdamW](https://arxiv.org/abs/1711.05101)
- [SQLite WAL](https://sqlite.org/wal.html) and [Streamlit navigation](https://docs.streamlit.io/develop/api-reference/navigation/st.navigation)
