# Training and resume

SparseLab packs plain token IDs with one EOS per document. A sequence may cross an EOS boundary; EOS is a token separator, not an attention mask. Source acquisition budgets include EOS and are distinct from training target budgets.

CPU reproducibility uses `device: cpu` and `deterministic: true`. Resume creates a child run, retains parent telemetry, and rejects changed model, optimizer, data, tokenizer, or training configuration. Device and logging-root changes are allowed. Checkpoints have a sibling SHA-256 manifest; never resume a `.pt` file without it.

Use the smoke configuration before bounded TinyStories runs:

```sh
uv run sparselab train configs/smoke_cpu.yaml --run-id smoke
uv run sparselab train configs/smoke_cpu.yaml --run-id part --stop-after-step 20
uv run sparselab train configs/smoke_cpu.yaml --run-id resumed --resume runs/part/checkpoints/step_00000020.pt
```
