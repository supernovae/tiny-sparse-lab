# Tiny Sparse Lab

Tiny Sparse Lab is an installable PyTorch **architecture-learning laboratory**. It implements dense, sliding-window, and multi-head latent causal attention; local MoE; token and byte-address memory; a combined MLA/MoE/byte-memory reference configuration; byte-aware greedy generation; and verified withheld-fact diagnostics. It remains an educational reference: distributed expert exchange, capacity management, external retrieval, and general language-model benchmarking are not implemented.

## Local workflow

```sh
uv sync --locked --dev
uv run sparselab tokenizer train configs/tokenizer_smoke.yaml
uv run sparselab data prepare configs/smoke_cpu.yaml
uv run sparselab inspect configs/micro_dense.yaml --json
uv run sparselab train configs/smoke_cpu.yaml --run-id dense-smoke
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-smoke
uv run sparselab train configs/smoke_memory_cpu.yaml --run-id memory-smoke
uv run sparselab train configs/smoke_byte_memory_cpu.yaml --run-id byte-memory-smoke
uv run sparselab train configs/smoke_sliding_cpu.yaml --run-id sliding-smoke
uv run sparselab train configs/smoke_mla_cpu.yaml --run-id mla-smoke
uv run sparselab train configs/smoke_combined_cpu.yaml --run-id combined-smoke
uv run sparselab eval moe-smoke
uv run sparselab generate moe-smoke --prompt "Once upon a time" --max-new-tokens 24
uv run sparselab dashboard --runs-dir runs
uv run sparselab facts manifest --seed 0 --output artifacts/withheld-facts-seed-0.json
uv run sparselab facts verify artifacts/withheld-facts-seed-0.json
uv run sparselab facts audit artifacts/withheld-facts-seed-0.json
uv run sparselab facts transfer-evaluate withheld-bytes withheld-bpe artifacts/withheld-facts-seed-0.json
```

Use `--stop-after-step N` to make a durable interrupted checkpoint, then resume into a new child run:

```sh
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-part --stop-after-step 20
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-resumed --resume runs/moe-part/checkpoints/step_00000020.pt
```

A MoE configuration sets `model.ffn: moe` and `model.num_experts: N` where `N >= 2`. Each normalized token is selected by one local expert. The router is deterministic top-1 (`argmax(softmax(logits))`); no token drops, capacity limit, load-balancing loss, or all-to-all dispatch exists in 0.2. Dense and MoE checkpoints are intentionally incompatible because resume rejects a changed model configuration.

Memory configuration sets `model.memory: ngram` with a table size, n-gram size, and independent latent width. Addresses are causal token-ID suffix hashes; they are tokenizer-specific and do not support byte-equivalent or cross-tokenizer matching. The learnable table is checkpointed with the model, so changing memory settings is rejected on resume.

Byte memory uses prepared causal raw-UTF-8 suffix addresses, not tokenizer IDs. Its data artifacts are versioned and hashed; standalone evaluation and greedy generation prepare aligned prompt addresses. This does not establish byte-hash retrieval quality or cross-tokenizer transfer.

The dashboard is read-only and binds to `127.0.0.1`. It shows stored metric observations, events, ancestry, and metric explanations; it never launches training. Loss is not an architecture benchmark: compare runs only when data, tokenizer, budget, device, and architecture conditions are known.

## Data and licensing

Synthetic data is an offline fixture. TinyStories artifacts retain pinned source/revision and license metadata locally; do not commit downloaded corpus text, prepared arrays, run directories, or checkpoints. Project source is MIT licensed; downloaded datasets retain their own terms.

See [using SparseLab](docs/using-sparselab.md), [architecture](docs/architecture.md), [model scaling](docs/model-scaling.md), [experiments](docs/experiments.md), [metrics](docs/metrics.md), [training](docs/training.md), [byte addressing](docs/byte-addressing.md), [withheld facts](docs/withheld-facts.md), and [architecture decisions](docs/decisions/).
