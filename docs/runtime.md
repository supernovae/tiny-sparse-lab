# Runtime policy

SparseLab records an explicit engine, backend, and precision choice in every run manifest. The PyTorch engine supports local `cpu`, `mps`, `cuda`, `rocm`, `xpu`, or `auto`; `auto` resolves only among discovered local runtimes, and an explicitly unavailable backend fails rather than falling back to CPU. The optional MLX engine is a separate local Metal path and requires `engine: mlx` with `backend: metal`.

PyTorch training currently executes FP32 only (`precision: auto` also resolves to FP32). Explicit BF16/FP16 requests are rejected rather than silently labeled as mixed precision. Hardware supporting a dtype is not evidence that the trainer implements it. Device identity and per-update throughput use the actual selected backend/index.

PyTorch runs support the full train → checkpoint → chat/eval/capability path. Native MLX is experimental dense training/resume only: its current checkpoint inspection checks structure/presence, not a persisted cryptographic inventory. It is excluded from controlled capability comparisons until inference and evidence parity are implemented and tested. CUDA/ROCm/XPU availability is runtime-detected; target-hardware correctness/performance claims still require those machines.

Memory policy produces an explicit proposed configuration. It never mutates a scientific run configuration. Device allocation, RSS, and estimates are distinct measurements: unified-memory MPS and MLX values are not added to host RAM.

Disk checkpoints preserve durable state. Activation checkpointing recomputes forward blocks during backward. Gradient accumulation combines several microbatches into one optimizer update. These mechanisms solve different resource constraints. None turns a local run into distributed training.

Current scaling checks have important limits: CLI `inspect` still instantiates a model; with `backend: auto`, its estimate describes CPU rather than necessarily the backend training will choose. `stage --through smoke|warmup` records those stages without executing an isolated model pilot. Use an explicit target backend and a short real training run to exercise forward/backward/checkpointing; neither an estimate nor a stage label proves measured fit or throughput. The [beginner guide](from-toy-to-useful.md#5-a-smaller-instruction-learner-before-the-100m-run) supplies that workflow.
