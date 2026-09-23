# Latent Attention

The MLA path compresses the joint key/value source into a smaller latent vector, then projects keys and values for attention. Its request-local decoding cache retains latent K/V state and explicit positional state. This is a readable reference implementation, not a fused bandwidth-optimized kernel or a claim of DeepSeek-equivalent inference performance.

Use `configs/smoke_mla_cpu.yaml` for a bounded CPU smoke. Compare dense, MLA, sparse selection, and combined variants at matching data, tokenizer, and token budget.
