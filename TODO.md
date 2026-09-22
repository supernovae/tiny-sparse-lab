# Remaining experiments and scale limits

- [ ] Add a native NVIDIA CUDA sparse-attention kernel: build and benchmark it on a CUDA GPU.
- [ ] Add a native ROCm/HIP sparse-attention kernel: build and benchmark it on an AMD GPU.
- [ ] Add a native MLX sparse-attention kernel: build and benchmark it on supported Apple hardware.

The supported runtime is one local process using the explicitly requested or correctly detected CPU, MPS, CUDA/ROCm, XPU, or optional MLX Metal engine. Sparse-attention paths currently use verified reference implementations.

## Next capability experiments

The [review and measured example](docs/project-review.md) records completed work and negative results. These are further research tasks, not hidden prerequisites for the current local workbench.

- [ ] Add a context-override curriculum and a new independently held-out assignment/paraphrase card; retain the current failed override control.
- [ ] Run a preregistered multi-seed, multi-budget dense/Engram series; add statistical aggregation only with explicit treatment of paired cases, seed variance, and multiple comparisons.
- [ ] Test Engram collision pressure, address orders/hash heads, and matched-total-parameter alternatives. Report quality and resource costs, not only table use.
- [ ] Curate a redistributable licensed domain conversation corpus and semantic leakage audit. Local user-supplied licensed JSONL is already supported.
- [ ] Add assistant-only training loss and a versioned tool-call conversation contract if domain experiments require them.
- [ ] Implement and test KV-cached generation and actual mixed precision before advertising inference efficiency or larger fit limits.
- [ ] Bring MLX native checkpoint integrity, chat/generation, held-out evidence, and capability comparison to PyTorch parity; current MLX support is experimental dense training/resume.
- [ ] Complete shape-only CLI inspection and actually executed isolated smoke/warmup staging; current stage labels are not measured model-pilot evidence.
- [ ] Implement verified legacy checkpoint import/promotion into current run artifacts and explicit pretrained-model architecture/tokenizer mappings; never reinterpret old configs or arbitrary weight files as compatible chat runs.

## Future distributed and multi-host work

- [ ] Add capacity-aware distributed MoE dispatch: expert sharding, all-to-all token exchange, capacity handling, and multi-rank GPU tests.
- [ ] Add remote worker/controller orchestration: SSH transport, device leases, durable attempt receipts, artifact ingestion, and heterogeneous matrix scheduling.
- [ ] Add homogeneous distributed data-parallel training only after the single-host checkpoint, RNG, and accelerator contracts are stable.
- [ ] Run real ROCm and XPU backend acceptance on their target machines once available.

The supported core is one process on one host using one correctly detected CPU or accelerator. No distributed process groups, cross-host queues, or remote-worker services are part of the current runtime.
