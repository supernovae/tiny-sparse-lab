# Metrics

| Metric | Definition | Interpretation |
|---|---|---|
| `train/loss` | Mean next-token negative log-probability over valid supervised targets, in nats. | Lower only under comparable data/tokenizer/objective/budget conditions; excludes auxiliary routing loss. |
| `validation/loss` | Held-out target-weighted mean negative log-probability. | Compare alongside train loss for overfit signals; assistant-only masks also apply to evaluation. |
| `validation/perplexity` | `exp(validation/loss)`. | Not fairly comparable across tokenizers. |
| `optimizer/learning_rate` | Scheduled optimizer learning-rate value. | AdamW uses an update learning rate; Adafactor uses a relative step-size cap, not a numerically equivalent AdamW rate. |
| `optimizer/grad_norm` | Global L2 gradient norm before clipping. | Spikes can signal instability. |
| `performance/tokens_per_second` | Valid targets divided by synchronized optimizer-update duration, including forward/backward and transfers. | Excludes validation/checkpoint/logging; not end-to-end job throughput. |
| `moe/router_auxiliary_loss` | Valid-target-weighted auxiliary loss summed over layers, including its configured coefficient. | Diagnose optimization contribution; it is separate from language-model loss. |
| `moe/layer_*/router_entropy` | Mean entropy of pre-selection softmax probabilities. | A routing diagnostic, not a balance-loss objective. |
| `moe/layer_*/maximum_expert_fraction` | Largest per-expert assignment share for a forward. | High concentration can indicate collapse; no automatic remediation exists. |
| `moe/layer_*/mean_topk_probability` | Mean total routing probability assigned to selected experts. | Shows how much probability mass survives Top-K selection. |
| `engram/*` | Lookups, distinct buckets, collisions, reuse/utilization, and gate mean. | Addressing and mixing diagnostics; not retrieval accuracy. |
| `attention/layer_*/available_tokens` | Causal key positions available across the forward. | Sequence-length dependent baseline for sparse selection. |
| `attention/layer_*/selected_tokens` | Sparse key positions admitted after block selection. | Compare with available positions for the same run. |
| `attention/layer_*/selection_ratio` | `selected_tokens / available_tokens`. | Lower means more aggressive key filtering; it does not establish quality. |
| `attention/layer_*/estimated_flops` | Attention score-product estimate for selected sparse keys. | Relative implementation estimate, not a device benchmark. |
| `attention/layer_*/dense_teacher_mass` | Dense-attention probability mass over the sparse layer's selected keys; requires full diagnostics. | Higher means the selected set retains more of the frozen forward's dense attention distribution. |
| `attention/layer_*/dense_teacher_topk_recall` | Recall of dense score Top-K keys using the sparse selected-key count; requires full diagnostics. | Retrieval-overlap diagnostic, not output equivalence or language-model quality. |

## Runtime, memory, and update measurements

| Metric or family | Interpretation |
|---|---|
| `performance/step_seconds` | Synchronized successful-update duration; includes forward/backward, recomputation and transfers, but excludes staging, validation and checkpointing. |
| `batch/micro_batch_size`, `batch/accumulation_steps`, `batch/effective_batch_size` | Configured microbatch size and actual microbatches/examples in the committed update. Epoch boundaries and final budgets can shorten a window. |
| `batch/tokens_per_micro_batch`, `batch/effective_tokens_per_update` | Valid supervised prediction targets, not raw prompt length or a nominal batch-size product. |
| `memory/device_allocated_bytes`, `memory/device_reserved_bytes` | Native allocation/reservation where exposed; MLX active allocation is distinct from `memory/device_cache_bytes`. |
| `memory/device_peak_allocated_bytes`, `memory/device_peak_reserved_bytes` | Native per-update high-water counters where available. |
| `memory/device_sampled_peak_bytes`, `memory/driver_allocated_bytes` | Sampled allocation maximum and driver reading. MPS sampled peaks are lower bounds, not allocator high-water guarantees. |
| `memory/process_rss_bytes`, `memory/process_peak_rss_bytes`, `memory/system_available_bytes` | Host resident memory, sampled update maximum, and available system RAM. CPU does not invent GPU measurements. |
| `recompute/block_call_ratio` | Recomputed block calls divided by original calls; one means every block was recomputed. It is not a measured time/memory saving. |
| `offload/bytes_to_cpu`, `offload/bytes_to_device`, `offload/peak_host_bytes` | Actual saved-activation transfers and peak live host offload storage. |
| `offload/transfer_seconds`, `offload/transfer_fraction` | Synchronized transfer cost, included in total update time rather than subtracted from throughput. |

Unavailable readings are omitted with explanatory events, not written as zero. Engine/backend/worker/precision/optimizer names are manifest metadata rather than numeric metrics. Apple unified memory is one physical pool: do not add device recommendations and system RAM, or claim extra capacity from MPS host offload. See [memory accounting](memory.md), [runtime policy](runtime.md), and [offload](offload.md).

## Diagnostics and durable aggregation

Architecture diagnostics (`engram/*`, per-layer MoE and attention values) are persisted from the **last nonempty training microbatch before validation**, not averaged over an accumulation window. They expose routing/address use; they do not prove successful retrieval. Scalar diagnostics do not archive full router tensors.

Parameter counts use the direct per-token convention documented in [model scaling](model-scaling.md). Routed experts, shared experts and routers have disjoint inventory categories; any readable expert subtotal must not be added again. Engram totals include every table and adapter, while active accounting includes one retrieved row per stream plus the adapter. These are parameter-use conventions, not measured FLOPs or sparse optimizer storage savings.

Every worker's SQLite database/WAL remains local. The controller imports contiguous, checksum-bound `(origin_id, sequence)` records idempotently, without re-emitting them into its own outbox. A controller disconnect can leave its view stale while the worker continues; terminal execution and verified artifact ingestion are separate states.

The read-only dashboard uses this local projection and retains actual step/target coordinates. Runtime, Memory, Checkpoints, Stages and Learn expose execution conditions and their limits alongside the loss curves. Different backends, optimizers, precision, tokenizers, objectives or observed budgets prevent an unqualified controlled comparison. See [worker acceptance](workers.md#observed-acceptance--2026-09-23).
