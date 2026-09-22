# Metrics

| Metric | Definition | Interpretation |
|---|---|---|
| `train/loss` | Mean next-token negative log-probability in nats. | Lower only under comparable data/tokenizer/budget conditions. |
| `validation/loss` | Held-out mean negative log-probability. | Compare alongside train loss for overfit signals. |
| `validation/perplexity` | `exp(validation/loss)`. | Not fairly comparable across tokenizers. |
| `optimizer/learning_rate` | AdamW update learning rate. | Verify warmup/cosine schedule progression. |
| `optimizer/grad_norm` | Global L2 gradient norm before clipping. | Spikes can signal instability. |
| `performance/tokens_per_second` | Valid targets per timed update second. | Device and sequence length affect it. |
| Router counts/fractions | Detached per-forward expert assignment totals and token shares. | Exposed on `Top1MoE` blocks; not persisted dashboard metric series yet. |
| Router entropy | Mean entropy of pre-selection softmax probabilities. | A routing diagnostic, not a balance-loss objective. |
| Router maximum fraction | Largest per-expert assignment share for a forward. | High concentration can indicate collapse; no automatic remediation exists in 0.2. |

Parameter counts use the direct per-token convention documented in [model scaling](model-scaling.md). Local MoE `expert` is all expert storage; `active_per_token` includes one selected expert per block and router storage. Memory and checkpoint estimates exclude serialization and metadata overhead where stated.
