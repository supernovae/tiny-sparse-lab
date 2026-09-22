# Training and resume

SparseLab packs plain token IDs with one EOS per document. A sequence may cross an EOS boundary; EOS is a token separator, not an attention mask. Source acquisition budgets include EOS and are distinct from training target budgets.

CPU reproducibility uses `device: cpu` and `deterministic: true`. Resume creates a child run, retains parent telemetry, and rejects any model, optimizer, data, tokenizer, or training configuration change. Device and logging-root changes are allowed. A dense checkpoint cannot resume a MoE run, nor can an MoE checkpoint resume a dense run. Checkpoints have a sibling SHA-256 manifest; never resume a `.pt` file without it.

## Dense, MoE, and sparse-attention smoke runs

```sh
uv run sparselab train configs/smoke_cpu.yaml --run-id dense-smoke
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-smoke
uv run sparselab train configs/smoke_sparse_cpu.yaml --run-id sparse-smoke
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-part --stop-after-step 20
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-resumed --resume runs/moe-part/checkpoints/step_00000020.pt
```

The MoE smoke config has three local experts with configurable Top-K selection. It performs no capacity clipping or distributed expert dispatch. Logged routing series include the auxiliary balance loss, entropy, maximum expert fraction, and selected-probability mass. Do not interpret a small synthetic run as expert specialization or architecture-quality evidence.

Block sparse attention logs selected and available causal key positions, their ratio, and a relative selected-key score-product estimate. These are resource diagnostics, not a quality conclusion: compare a sparse run with a matched dense run at observed matching token budgets.
