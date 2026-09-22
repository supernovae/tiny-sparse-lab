# Model scaling and accounting

`total` counts every unique parameter storage once. `trainable` is the trainable subset. `active_per_token` uses a direct-use convention: all output-vocabulary logits count; tied dense models use all parameters; untied models exclude the full input embedding table and include the one selected input row. It is not FLOPs or a context-wide unique-weight measure.

Tied output-head storage is attributed to `embedding` and `output_head` is zero. `expert` and `engram` are zero and explicitly mean “not present in dense model.”

Model-weight bytes sum unique parameter tensors once. Estimated AdamW state bytes are two float32 moments per trainable scalar. `estimated_checkpoint_bytes` is their sum after the first update; it excludes gradients, RNG, metadata, and serialization overhead.

The supplied 8192-vocabulary tied presets are 3,344,064 parameters (`micro_dense.yaml`) and 6,917,376 (`dense_7m.yaml`). `sparselab inspect` instantiates and counts them rather than returning these values as constants.
