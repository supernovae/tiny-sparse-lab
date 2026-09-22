# Deferred work

- [ ] Add capacity-aware distributed MoE dispatch: implement expert sharding, all-to-all token exchange, capacity handling, and multi-rank GPU tests when a multi-GPU or multi-machine environment is available.
- [ ] Add native CUDA, ROCm/HIP, and MLX sparse-attention kernels: build and benchmark each backend on its target runtime. The current CPU/MPS dispatch selects the verified PyTorch reference path.
