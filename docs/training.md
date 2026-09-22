# Training, checkpoints, and local continuation

SparseLab packs plain token IDs with one EOS per document. A sequence may cross an EOS boundary: EOS separates records but does not create an attention mask. Source-acquisition budgets include EOS; training budgets count valid next-token targets.

## Local runtime contract

A run uses one local process on one host and one explicitly configured or automatically detected device. Use `runtime.backend: cpu|mps|cuda|rocm|xpu|auto` with the PyTorch engine. `auto` selects only an available local backend; an explicitly unavailable backend fails rather than falling back to CPU. The optional MLX engine requires `runtime.engine: mlx` and `runtime.backend: metal`.

CPU reproducibility uses `runtime.backend: cpu` and `training.deterministic: true`. MPS and other accelerators can exercise the same local lifecycle, but are not claimed to be bitwise-identical to CPU.

## Smoke runs

```sh
uv run sparselab train configs/smoke_cpu.yaml --run-id dense-smoke
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-smoke
uv run sparselab train configs/smoke_sparse_cpu.yaml --run-id sparse-smoke
uv run sparselab train configs/smoke_combined_cpu.yaml --run-id combined-smoke
```

A smoke run validates a path through training, checkpointing, evaluation, and metrics. It is not a quality benchmark. Compare runs only when data, tokenizer, device, sequence length, token budget, optimizer, and seed are recorded and intentionally matched.

## Checkpoint and resume workflow

PyTorch checkpoints are immutable local generations under `runs/<run-id>/checkpoints/`; `latest.json` points to the newest validated generation. Verify a checkpoint before continuing it:

```sh
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-part --stop-after-step 20
uv run sparselab checkpoint inspect runs/moe-part/checkpoints/latest.json --json
uv run sparselab checkpoint verify runs/moe-part/checkpoints/latest.json --json
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-resumed \
  --resume runs/moe-part/checkpoints/latest.json
```

Resume creates a child run and retains parent telemetry. It rejects changed model, optimizer, data, tokenizer, budget, engine, or backend configuration; `--allow-runtime-drift` never bypasses scientific compatibility. Recovery applies the same checks. Completed budgets cannot be resumed. Run-owned data, tokenizer, and memory-package artifacts are verified before continuation.

Promotion reuses compatible weights with a fresh optimizer/cursor, using the **destination** dataset and budgets. Source architecture semantics and tokenizer contents must match. It is useful for continuing a model on a new domain/chat corpus, not for automatically resizing a backbone or converting dense weights into MoE.

MLX checkpoints are native local checkpoint directories under `mlx_checkpoints/`. They support the same inspect/verify commands, while resume names the directory directly.

PyTorch validation runs at step zero, each configured cadence, and the terminal update. Exact target counts, finite loss, protocol settings, and artifact identities accompany each checkpoint-bound report. Non-multiple token budgets mask only the remaining valid targets, including partially or fully masked microbatches. Training and validation modes are restored correctly; diagnostics are captured before validation can replace them.

## Mechanism-specific diagnostics

The MoE smoke configuration has local Top-K experts with configurable selection. It has no capacity clipping or distributed expert dispatch. Routing metrics include the auxiliary balance loss, entropy, maximum expert fraction, and selected-probability mass; none demonstrates expert specialization by itself.

Block-sparse attention reports selected and available causal key positions, selection ratio, and a relative selected-key score-product estimate. These are resource diagnostics, not a quality conclusion or a custom-kernel speed claim. Use a matched dense run for a controlled ablation.

See [runtime policy](runtime.md), [metrics](metrics.md), and [architecture](architecture.md) for the terms used by run reports.
