# Runtime policy

SparseLab records an explicit engine, backend, and precision choice in every run manifest. The PyTorch engine supports local `cpu`, `mps`, `cuda`, `rocm`, `xpu`, or `auto`; `auto` resolves only among discovered local runtimes, and an explicitly unavailable backend fails rather than falling back to CPU. The optional MLX engine is a separate local Metal path and requires `engine: mlx` with `backend: metal`.

Memory policy produces an explicit proposed configuration. It never mutates a scientific run configuration. Device allocation, RSS, and estimates are distinct measurements: unified-memory MPS and MLX values are not added to host RAM.

Disk checkpoints preserve durable state. Activation checkpointing recomputes forward blocks during backward. Gradient accumulation combines several microbatches into one optimizer update. These mechanisms solve different resource constraints. None turns a local run into distributed training.
