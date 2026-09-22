# Deferred work

- [ ] Add capacity-aware distributed MoE dispatch: implement expert sharding, all-to-all token exchange, capacity handling, and multi-rank GPU tests when a multi-GPU or multi-machine environment is available.
- [ ] Add a native NVIDIA CUDA sparse-attention kernel: build and benchmark it on a CUDA GPU.
- [ ] Add a native ROCm/HIP sparse-attention kernel: build and benchmark it on an AMD GPU.
- [ ] Add a native MLX sparse-attention kernel: build and benchmark it on supported Apple hardware.

The current CPU/MPS dispatch selects the verified PyTorch reference path.

## Hardware acceptance blockers

- [ ] Run the independent ROCm worker acceptance on an actual supported AMD ROCm Linux or WSL2 host: validate runtime discovery, dense update, generation checkpoint verification, same-backend interruption/resume, and CPU-weight promotion.
- [ ] Run the independent XPU worker acceptance on an actual Intel XPU host: validate runtime discovery, dense update, generation checkpoint verification, same-backend interruption/resume, and CPU-weight promotion.
- [ ] Run the heterogeneous controller proof with real Mac MPS, AMD ROCm, and Intel XPU workers concurrently. Verify independent processes, overlapping runs, controller-disconnect progress, and local artifact/outbox ingestion.

Local CPU, MPS, and MLX checks do not substitute for ROCm/XPU evidence.
