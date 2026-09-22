# Training and resume

SparseLab packs plain token IDs with one EOS per document. A sequence may cross an EOS boundary; EOS is a token separator, not an attention mask. Source acquisition budgets include EOS and are distinct from training target budgets.

CPU reproducibility uses `device: cpu` and `deterministic: true`. Resume creates a child run, retains parent telemetry, and rejects any model, optimizer, data, tokenizer, or training configuration change. Device and logging-root changes are allowed. A dense checkpoint cannot resume a MoE run, nor can an MoE checkpoint resume a dense run. Checkpoints have a sibling SHA-256 manifest; never resume a `.pt` file without it.

## Dense and MoE smoke runs

```sh
uv run sparselab train configs/smoke_cpu.yaml --run-id dense-smoke
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-smoke
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-part --stop-after-step 20
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-resumed --resume runs/moe-part/checkpoints/step_00000020.pt
```

The MoE smoke config has three local experts and top-1 routing. It performs no capacity clipping, balancing loss, or distributed expert dispatch. Router diagnostics are available on each `Top1MoE` block after a forward pass; they are detached and do not yet become SQLite dashboard scalar series. Do not interpret a small synthetic run as an expert-specialization or architecture-quality result.
