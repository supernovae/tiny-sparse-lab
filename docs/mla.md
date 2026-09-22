# Latent Attention

SparseLab includes a small educational latent-KV attention reference under `attention.kind: mla`. It compresses K/V state into a lower-dimensional latent before attention reconstruction. This is a mathematical teaching implementation, not a production MLA kernel; it has no decoding cache or optimized fused projections.

Use `configs/smoke_mla_cpu.yaml` for a bounded CPU smoke. Compare dense, MLA, sparse selection, and combined variants at matching data, tokenizer, and token budget.
