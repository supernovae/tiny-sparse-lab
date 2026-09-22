# Tiny Sparse Lab

**A small, inspectable laboratory for learning how modern decoder mechanisms change a language model.**

Tiny Sparse Lab is for builders who want to move beyond diagrams: change one architectural boundary, train a controlled local run, inspect the resulting artifacts, and explain what the evidence does—and does not—show. It is deliberately a reference implementation rather than a production training stack.

The supported runtime is one process on one host using one correctly detected CPU or accelerator. PyTorch is the canonical engine; the optional MLX engine provides a dense local Metal path on Apple Silicon. Distributed process groups, remote workers, and multi-host scheduling are intentionally deferred.

## Why try it?

- **Learn mechanisms in context.** Follow RMSNorm, RoPE, causal attention, SwiGLU, local MoE routing, latent attention, and addressed memory through runnable code rather than isolated snippets.
- **Run meaningful ablations.** Each smoke configuration makes a narrow behavior testable; the larger presets preserve explicit budgets and provenance.
- **Keep evidence with the run.** Configurations, prepared data, tokenizer artifacts, checkpoints, evaluation reports, metrics, and lineage remain local and inspectable.
- **Make honest claims.** Dense reference paths and Python-level selectors are not advertised as custom-kernel speedups. Small synthetic runs are wiring evidence, not benchmark results.
- **Extend carefully.** New Engram variants, attention paths, and scale experiments can reuse the existing causal data, checkpoint, metric, and comparison boundaries instead of introducing hidden semantics.

## What is implemented

| Area | Explore | Boundary |
|---|---|---|
| Decoder baseline | RMSNorm, RoPE causal attention, SwiGLU, tied or untied output embeddings | Small dense decoder; no hosted model or chat API. |
| Attention | Dense, sliding-window, block-sparse selection, and multi-head latent attention (MLA) | Reference implementations; no native sparse kernels or KV-cache performance claim. |
| Feed-forward sparsity | Local Top-K MoE with an optional shared expert and routing diagnostics | No expert sharding, capacity clipping, token dropping, or all-to-all exchange. |
| Engram memory | Causal token n-gram, raw UTF-8 byte-addressed, and portable frozen-table adapters | Addressing diagnostics are not retrieval-quality or transfer proof. |
| Training evidence | Deterministic local training, child-run resume, validated checkpoints, evaluation, and a read-only dashboard | Cross-backend continuation is promotion, not a distributed resume path. |
| Accelerators | Local CPU, MPS, CUDA/ROCm, and XPU discovery for PyTorch; optional MLX Metal training | Hardware-specific acceptance remains pending where target hardware is unavailable. |

## Start here

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked --dev
uv run sparselab tokenizer train configs/tokenizer_smoke.yaml
uv run sparselab data prepare configs/smoke_cpu.yaml
uv run sparselab inspect configs/smoke_cpu.yaml --json
uv run sparselab train configs/smoke_cpu.yaml --run-id dense-smoke
uv run sparselab eval dense-smoke
uv run sparselab generate dense-smoke --prompt "Once upon a time" --max-new-tokens 24
uv run sparselab chat dense-smoke
uv run sparselab dashboard --runs-dir runs
```

The smoke run is intentionally small. It proves the local tokenizer → prepared data → training → checkpoint → evaluation → generation path; it does not promise fluent generation or comparative model quality.

### Chat with a trained local run

```sh
uv run sparselab chat dense-smoke
uv run sparselab chat dense-smoke --message "Explain causal attention in one sentence."
```

`chat` keeps an in-process plain-text `User:`/`Assistant:` transcript and applies the selected run's ordinary greedy decoder. It works with any saved PyTorch architecture run, including the scale presets. It does **not** make a base model instruction-tuned: the response quality is limited by its training corpus and budget. The optional [instruction reference curriculum](docs/instruction-training.md) trains the transcript format and a small set of deterministic tasks; it is not a general chat dataset or a broad instruction-following claim.

### Resume a local run

Use `--stop-after-step` to create a durable interruption boundary, then resume into a new child run from the latest validated generation:

```sh
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-part --stop-after-step 20
uv run sparselab checkpoint verify runs/moe-part/checkpoints/latest.json --json
uv run sparselab train configs/smoke_moe_cpu.yaml --run-id moe-resumed \
  --resume runs/moe-part/checkpoints/latest.json
```

Resume requires compatible local model, optimizer, data, tokenizer, and training contracts. Use promotion—not resume—when deliberately changing an incompatible architecture or backend.

### Apple Silicon: optional MLX engine

```sh
uv sync --locked --dev --extra mlx
uv run sparselab train configs/smoke_mlx.yaml --run-id mlx-smoke
uv run sparselab checkpoint verify runs/mlx-smoke/mlx_checkpoints/step_00000040 --json
```

MLX checkpoints retain local native Metal state. `checkpoint inspect` and `checkpoint verify` validate their metadata and required state files; MLX resume takes the checkpoint directory.

## Suggested learning path

1. **Establish the baseline.** Run `smoke_cpu`, inspect its parameter inventory, and read [the decoder architecture](docs/architecture.md).
2. **Change one boundary.** Compare dense, [sliding/block-sparse attention](docs/sparse-attention.md), [MLA](docs/mla.md), or [local MoE](docs/moe.md) under matched settings.
3. **Study memory as an architectural mechanism.** Run token and [byte-addressed Engram](docs/engram.md), then read the [portable Engram](docs/portable-engram.md) contract before making transfer claims.
4. **Treat diagnostics as evidence.** Use the [metrics guide](docs/metrics.md), [training and resume guide](docs/training.md), and [withheld-fact diagnostic](docs/withheld-facts.md) to distinguish a measured observation from a conclusion.
5. **Scale deliberately.** Inspect explicit presets first; use the [experiment protocol](docs/experiments.md) to preserve comparison conditions.

## Architecture at a glance

```mermaid
flowchart LR
  ids[Token IDs] --> embed[Embedding]
  embed --> blocks[Decoder blocks\nRMSNorm · RoPE attention · FFN]
  blocks --> memory{Optional Engram}
  memory --> norm[Final RMSNorm]
  norm --> logits[Next-token logits]
  blocks -->|dense or local MoE| ffn[SwiGLU boundary]
```

The architectural options are explicit configuration choices, not automatic optimization. `smoke_combined_cpu.yaml` composes MLA, local MoE, and byte memory as a compatibility check; it is not evidence that the combination improves quality or throughput.

## Documentation map

- [Using SparseLab](docs/using-sparselab.md) — commands, local artifacts, and dashboard.
- [Architecture](docs/architecture.md) — decoder and mechanism boundaries.
- [Training and resume](docs/training.md) — local run lifecycle and checkpoint semantics.
- [Runtime policy](docs/runtime.md) — backend selection and resource terminology.
- [Engram](docs/engram.md), [byte addressing](docs/byte-addressing.md), and [portable Engram](docs/portable-engram.md) — memory contracts and diagnostics.
- [MoE](docs/moe.md), [sparse attention](docs/sparse-attention.md), and [MLA](docs/mla.md) — reference mechanism behavior.
- [Experiments](docs/experiments.md), [model scaling](docs/model-scaling.md), and [metrics](docs/metrics.md) — controlled comparison practice.
- [Experiment evidence](docs/evidence.md) — verified checkpoints, held-out observations, controlled comparisons, and hardware-reference discipline.
- [Capability experiments](docs/capabilities.md) — versioned narrow hypotheses, matched baselines, and scale-series evidence.
- [Instruction reference training](docs/instruction-training.md) — optional synthetic curriculum, 100M reference configuration, and evaluation boundary.
- [Withheld facts](docs/withheld-facts.md) — deterministic data-separation evidence.
- [Architecture decisions](docs/decisions/README.md) — durable design context and current scope decisions.
- [Deferred roadmap](TODO.md) — native kernels, target-hardware acceptance, and distributed work intentionally outside the core.

## Data, licensing, and contribution

Synthetic data is an offline fixture. TinyStories artifacts retain pinned source/revision and license metadata locally; do not commit downloaded corpora, prepared arrays, run directories, or checkpoints. Project source is MIT licensed; downloaded datasets retain their own terms.

Contributions should preserve explicit causal, configuration, artifact, and comparison contracts. Read [CONTRIBUTING.md](CONTRIBUTING.md), verify the affected local behavior, and add a focused regression only when it protects an observable contract.
