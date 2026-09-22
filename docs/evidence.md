# Experiment evidence

SparseLab is intended to make a small experiment inspectable, not to convert a successful demo into a broad capability claim. The lightweight evidence path connects three local facts: an immutable run configuration and artifacts, a verified checkpoint, and a held-out validation observation recorded against that checkpoint.

## Surface workflow

```sh
uv run sparselab train configs/smoke_cpu.yaml --run-id evidence-smoke --stop-after-step 4
uv run sparselab checkpoint verify runs/evidence-smoke/checkpoints/latest.json --json
uv run sparselab evidence evidence-smoke --json
```

Training evaluates at step 0, every `evaluation.every_steps`, and the terminal boundary. Each such evaluation forces a validated checkpoint and writes an immutable report under `runs/<id>/evaluations/`. `sparselab evidence` verifies every checkpoint generation and summarizes the held-out loss/perplexity observations attached to them.

The report level `checkpointed_held_out` means the configured run completed this local evidence path. It does **not** mean the model is useful for arbitrary language, instruction following, safety, or production traffic.

## What each level validates

| Level | Evidence | Does not establish |
|---|---|---|
| Behavior | Focused tests and a real smoke path | Generalization or hardware performance. |
| Artifact | Checkpoint hashes, tensor inventory, and run-owned configuration/data artifacts | Model quality. |
| Local experiment | Verified checkpoint paired with held-out loss/perplexity at recorded budgets | Superiority over an unmatched run or task-level ability. |
| Controlled comparison | Matched source, tokenizer, architecture boundary, optimizer, sequence length, token budget, device, and seed | A universal scaling law or production performance. |
| Hardware implementation | Defined numerical tolerance against reference vectors, one update or state transition where applicable, and measured end-to-end behavior | Correctness on untested operations, shapes, or numerical formats. |

## Hardware and numerical-format experiments

A future RISC, ASIC, or MXFP4 accelerator experiment should use SparseLab as a **reference harness**, not as proof from an isolated kernel demo. Freeze a small input/token batch, model weights, expected outputs, and tolerance policy; run the same operation through the PyTorch reference and the candidate path; then retain both artifact identities, output error statistics, and measured throughput/energy conditions. For training-capable hardware, also compare one optimizer update or an explicitly bounded state transition.

MXFP4 will need its own explicit quantization, rounding, accumulation, overflow, and dequantization contract before such a claim is meaningful. A successful dense FP32 run is useful baseline evidence, but it is not MXFP4 validation. Keep the candidate implementation behind an explicit configuration/backend selection; never silently substitute a reference result for hardware output.

## Experiment discipline

State the hypothesis before running: for example, “block selection retains comparable held-out loss at a fixed selection budget,” or “candidate MXFP4 matmul remains within the declared tolerance on this frozen corpus.” Preserve negative and inconclusive outcomes. Change one boundary per comparison where possible, and report the conditions that would falsify the hypothesis.
