# Metrics

| Metric | Definition | Interpretation |
|---|---|---|
| `train/loss` | Mean next-token negative log-probability in nats. | Lower only under comparable data/tokenizer/budget conditions. |
| `validation/loss` | Held-out mean negative log-probability. | Compare alongside train loss for overfit signals. |
| `validation/perplexity` | `exp(validation/loss)`. | Not fairly comparable across tokenizers. |
| `optimizer/learning_rate` | AdamW update learning rate. | Verify warmup/cosine schedule progression. |
| `optimizer/grad_norm` | Global L2 gradient norm before clipping. | Spikes can signal instability. |
| `performance/tokens_per_second` | Valid targets per timed update second. | Device and sequence length affect it. |
| `moe/router_auxiliary_loss` | Mean auxiliary load-balance loss, before its configured coefficient is applied. | Diagnose routing balance; it is not language-model loss. |
| `moe/layer_*/router_entropy` | Mean entropy of pre-selection softmax probabilities. | A routing diagnostic, not a balance-loss objective. |
| `moe/layer_*/maximum_expert_fraction` | Largest per-expert assignment share for a forward. | High concentration can indicate collapse; no automatic remediation exists. |
| `moe/layer_*/mean_topk_probability` | Mean total routing probability assigned to selected experts. | Shows how much probability mass survives Top-K selection. |
| `engram/*` | Lookups, distinct buckets, collisions, reuse/utilization, and gate mean. | Addressing and mixing diagnostics; not retrieval accuracy. |
| `attention/layer_*/available_tokens` | Causal key positions available across the forward. | Sequence-length dependent baseline for sparse selection. |
| `attention/layer_*/selected_tokens` | Sparse key positions admitted after block selection. | Compare with available positions for the same run. |
| `attention/layer_*/selection_ratio` | `selected_tokens / available_tokens`. | Lower means more aggressive key filtering; it does not establish quality. |
| `attention/layer_*/estimated_flops` | Attention score-product estimate for selected sparse keys. | Relative implementation estimate, not a device benchmark. |

Parameter counts use the direct per-token convention documented in [model scaling](model-scaling.md). Local MoE `expert` is all expert storage; `active_per_token` includes one selected expert per block and router storage. Memory and checkpoint estimates exclude serialization and metadata overhead where stated.
