# Model scaling and accounting

`total` counts every unique parameter storage once. `trainable` is the trainable subset. `active_per_token` is a direct-use convention, not FLOPs or a context-wide unique-weight measurement.

For a tied dense decoder, all parameter storage is active per token. For an untied dense decoder, the convention excludes the full input-embedding table and includes one input row. Tied output-head storage is attributed to `embedding`; `output_head` is zero.

For local Top-K MoE, `total` includes router, shared expert when configured, and every routed expert. `expert` reports all expert SwiGLU storage; `active_per_token` includes router storage, the selected experts, and a shared expert where enabled. This is an accounting convention, not a claim that router work, indexing, or dispatch has zero cost. `ffn` remains the dense SwiGLU category and is zero for a fully MoE model. When an Engram mode is enabled, its table, projection, and gate are reported as memory storage; they are ordinary local model state, not an external-memory exemption.

Model-weight bytes sum unique parameter tensors once. Estimated checkpoint tensor bytes describe persisted tensor payloads for the configured optimizer after its first update; they exclude gradients, RNG, metadata, and serialization overhead. Inspect the configuration rather than extrapolating a fixed ratio across optimizer or architecture changes.

The supplied 8192-vocabulary tied dense presets are 3,344,064 (`micro_dense.yaml`), 6,917,376 (`dense_7m.yaml`), 10,244,160 (`dense_10m.yaml`), 29,893,120 (`dense_25m.yaml`), and 50,274,752 (`dense_50m.yaml`) parameters. `sparselab inspect` instantiates and counts configurations rather than returning fixed constants. All scale presets hold TinyStories revision, tokenizer identity, seed, sequence length, token budget, and optimizer settings constant. `configs/smoke_moe_cpu.yaml` is a small local-router plumbing configuration; it is not a quality or scaling benchmark.
