# Runtime policy

SparseLab records an explicit PyTorch engine/backend/precision choice in every run manifest. `auto` resolves only among discovered local runtimes; an explicitly unavailable backend fails rather than falling back to CPU.

Memory policy produces an explicit proposed configuration. It never mutates a scientific run configuration. Device allocation, RSS, and estimates are distinct measurements: unified-memory MPS values are not added to host RAM.

Disk checkpoints preserve durable state. Activation checkpointing recomputes forward blocks during backward. Gradient accumulation combines several microbatches into one optimizer update. These mechanisms solve different resource constraints.
