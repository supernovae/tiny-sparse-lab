# Model scaling and accounting

`total` counts every unique parameter storage once. `trainable` is the trainable subset. `active_per_token` is a direct-use convention, not FLOPs or a context-wide unique-weight measurement.

For a tied dense decoder, all parameter storage is active per token. For an untied dense decoder, the convention excludes the full input-embedding table and includes one input row. Tied output-head storage is attributed to `embedding`; `output_head` is zero.

For local top-1 MoE, `total` includes router and every expert. `expert` reports all expert SwiGLU storage; `active_per_token` includes router storage and one selected expert per decoder block, rather than every expert. This is an accounting convention, not a claim that router work, indexing, or dispatch has zero cost. `ffn` remains the dense SwiGLU category and is zero for a fully MoE model. `engram` remains zero because no external memory exists.

Model-weight bytes sum unique parameter tensors once. Estimated AdamW state bytes are two float32 moments per trainable scalar. `estimated_checkpoint_bytes` is their post-first-update tensor-payload sum; it excludes gradients, RNG, metadata, and serialization overhead.

The supplied 8192-vocabulary tied dense presets are 3,344,064 parameters (`micro_dense.yaml`) and 6,917,376 (`dense_7m.yaml`). `sparselab inspect` instantiates and counts configurations rather than returning fixed constants. `configs/smoke_moe_cpu.yaml` is the small local-router plumbing configuration; it is not a quality or scaling benchmark.
