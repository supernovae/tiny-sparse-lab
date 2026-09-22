# Deferred work

- [ ] Add capacity-aware distributed MoE dispatch: implement expert sharding, all-to-all token exchange, capacity handling, and multi-rank GPU tests when a multi-GPU or multi-machine environment is available.
- [ ] Add a native NVIDIA CUDA sparse-attention kernel: build and benchmark it on a CUDA GPU.
- [ ] Add a native ROCm/HIP sparse-attention kernel: build and benchmark it on an AMD GPU.
- [ ] Add a native MLX sparse-attention kernel: build and benchmark it on supported Apple hardware.

The current CPU/MPS dispatch selects the verified PyTorch reference path.
