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
- [x] Complete shape-only CLI inspection, backend-auto selection, and conservative accounting without opening model assets.
- [ ] Execute isolated smoke/warmup model pilots and verify their stage bundles before training; current stage labels are not measured model-pilot evidence.
- [ ] Implement verified legacy checkpoint import/promotion into current run artifacts and explicit pretrained-model architecture/tokenizer mappings; never reinterpret old configs or arbitrary weight files as compatible chat runs.

## Single-host runtime foundation

- [x] Bind canonical requested/effective/architecture identities and immutable packed arrays, without rewriting historical manifests or caches.
- [x] Add native checkpoint semantic verification, writer exclusion, finalized-generation recovery, and protected retention.
- [x] Gate full continuation on source/runtime compatibility and retain exact parent-checkpoint identity and selected-device RNG state.
- [ ] Complete transactional metrics migration, checkpoint/attempt ledgers, and crash-recovery accounting before remote ingestion.
- [ ] Add Runtime/Checkpoints dashboard views, live refresh, and stale-run handling.
- [ ] Add isolated precision probes, measured native peak-memory evidence, complete memory-policy search, and recomputation-safe diagnostic accounting.

Verification of this foundation: CPU and byte-addressed split/resume equivalence; local MPS resume; actual CLI AdamW/Adafactor split/resume with identical weights, optimizer state, and RNG; and shape-only inspection of a 16.98B-parameter configuration with missing tokenizer/package assets. The large shape was not trained. These checks do not close the staging, dashboard, metrics-ledger, or worker gates above.

## Future distributed and multi-host work

- [ ] Add capacity-aware distributed MoE dispatch: expert sharding, all-to-all token exchange, capacity handling, and multi-rank GPU tests.
- [ ] Add remote worker/controller orchestration: SSH transport, device leases, durable attempt receipts, artifact ingestion, and heterogeneous matrix scheduling.
- [ ] Add homogeneous distributed data-parallel training only after the single-host checkpoint, RNG, and accelerator contracts are stable.
- [ ] Run real ROCm and XPU backend acceptance on their target machines once available.

The supported core is one process on one host using one correctly detected CPU or accelerator. No distributed process groups, cross-host queues, or remote-worker services are part of the current runtime.
