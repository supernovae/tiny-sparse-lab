# Metrics

| Metric | Definition | Interpretation |
|---|---|---|
| `train/loss` | Mean next-token negative log-probability in nats. | Lower only under comparable data/tokenizer/budget conditions. |
| `validation/loss` | Held-out mean negative log-probability. | Compare alongside train loss for overfit signals. |
| `validation/perplexity` | `exp(validation/loss)`. | Not fairly comparable across tokenizers. |
| `optimizer/learning_rate` | Scheduled optimizer learning-rate value. | Verify warmup/cosine schedule progression; interpret it with the recorded optimizer choice. |
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
| `attention/layer_*/dense_teacher_mass` | Dense-attention probability mass over the sparse layer's selected keys. | Higher means the selected set retains more of the frozen forward's dense attention distribution. |
| `attention/layer_*/dense_teacher_topk_recall` | Recall of the dense score Top-K keys using the sparse selected-key count. | Retrieval-overlap diagnostic, not output equivalence or language-model quality. |

Architecture diagnostics (`engram/*`, per-layer MoE and attention values) are persisted from the **last nonempty training microbatch before validation**, not averaged over an accumulation window. They expose routing/address use; they do not prove successful retrieval. Scalar diagnostics do not archive full router tensors.

Parameter counts use the direct per-token convention documented in [model scaling](model-scaling.md). Local MoE `expert` includes all expert storage; `active_per_token` includes configured Top-K experts and router storage. Engram totals include every table and adapter, while active accounting includes one retrieved row per table plus the adapter. These are parameter-use conventions, not FLOPs. Memory estimates and checkpoint estimates exclude serialization and metadata overhead where stated.
