# Deferred work

- [ ] Add a native NVIDIA CUDA sparse-attention kernel: build and benchmark it on a CUDA GPU.
- [ ] Add a native ROCm/HIP sparse-attention kernel: build and benchmark it on an AMD GPU.
- [ ] Add a native MLX sparse-attention kernel: build and benchmark it on supported Apple hardware.

The supported runtime is one local process using the explicitly requested or correctly detected CPU, MPS, CUDA/ROCm, XPU, or optional MLX Metal engine. Sparse-attention paths currently use verified reference implementations.

## Future distributed and multi-host work

- [ ] Add capacity-aware distributed MoE dispatch: expert sharding, all-to-all token exchange, capacity handling, and multi-rank GPU tests.
- [ ] Add remote worker/controller orchestration: SSH transport, device leases, durable attempt receipts, artifact ingestion, and heterogeneous matrix scheduling.
- [ ] Add homogeneous distributed data-parallel training only after the single-host checkpoint, RNG, and accelerator contracts are stable.
- [ ] Run real ROCm and XPU backend acceptance on their target machines once available.

The supported core is one process on one host using one correctly detected CPU or accelerator. No distributed process groups, cross-host queues, or remote-worker services are part of the current runtime.
