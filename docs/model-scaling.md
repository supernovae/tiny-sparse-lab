# Model scaling and accounting

For the practical progression from a tiny exercise to a useful task, start with [From flashcards to a useful local assistant](from-toy-to-useful.md). More parameters do not by themselves turn story continuation or alias memorization into instruction-following chat.

`total` counts every unique parameter storage once. `trainable` is the trainable subset. `active_per_token` is a direct-use convention, not FLOPs or a context-wide unique-weight measurement.

For a tied dense decoder, all parameter storage is active per token. For an untied dense decoder, the convention excludes the full input-embedding table and includes one input row. Tied output-head storage is attributed to `embedding`; `output_head` is zero.

For local Top-K MoE, `total` includes router, shared expert when configured, and every routed expert. `expert` reports all expert SwiGLU storage; `active_per_token` includes router storage, the selected experts, and a shared expert where enabled. This is an accounting convention, not a claim that router work, indexing, or dispatch has zero cost. `ffn` remains the dense SwiGLU category and is zero for a fully MoE model. When an Engram mode is enabled, its table, projection, and gate are reported as memory storage; they are ordinary local model state, not an external-memory exemption.

Model-weight bytes sum unique parameter tensors once. Estimated checkpoint tensor bytes describe persisted tensor payloads for the configured optimizer after its first update; they exclude gradients, RNG, metadata, and serialization overhead. Inspect the configuration rather than extrapolating a fixed ratio across optimizer or architecture changes.

The supplied 8192-vocabulary tied dense presets are 3,344,064 (`micro_dense.yaml`), 6,917,376 (`dense_7m.yaml`), 10,244,160 (`dense_10m.yaml`), 29,893,120 (`dense_25m.yaml`), and 50,274,752 (`dense_50m.yaml`) parameters. `sparselab inspect` computes their tensor shapes without allocating a model or opening tokenizer/portable-package assets. Its JSON includes a named-category inventory, memory estimate, and passive runtime readings; the human view separates Model, Estimated training memory, Device, and Result. All scale presets hold TinyStories revision, tokenizer identity, seed, sequence length, token budget, and optimizer settings constant. `configs/smoke_moe_cpu.yaml` is a small local-router plumbing configuration; it is not a quality or scaling benchmark.

`instruction_starter.yaml` reuses the 3,344,064-parameter backbone with the instruction tokenizer/curriculum and a 256,000-target ceiling. `instruction_100m.yaml` has 104,843,648 parameters and a 20-million-target ceiling. These are different development budgets, not a controlled size comparison or a promise of conversational quality. Growing a backbone requires a new run; promotion only accepts compatible architecture and tokenizer semantics.

## Pinned external reference observation

`sparselab reference pythia trajectory CONFIG --output PATH` is an explicit, optional observation adapter for the official `EleutherAI/pythia-70m-deduped` checkpoints `step0`, `step10000`, and `step143000`. It accepts only a FineWeb-Edu configuration with a non-null pinned dataset revision and config, evaluates the same bounded validation documents for each selected checkpoint, and writes a machine-readable identity report outside Git.

The adapter installs with `tiny-sparse-lab[reference]`, downloads only the fixed registered Hub commits on this explicit command, and permits only safetensors weights plus fixed model/tokenizer/license/README metadata. It rejects pickle-style weights, Python code, remote-code mappings, and quantization metadata; loading uses built-in `GPTNeoXForCausalLM` and `GPTNeoXTokenizerFast` with remote code disabled.
FineWeb-Edu is identified in the output as ODC-BY-1.0; that database license does not settle rights in independently sourced web content.

Pythia weights are Apache-2.0. This is observation only: it does not download, package, or reproduce The Pile, and it is not a controlled comparison with SparseLab. Pythia and SparseLab can have different tokenizers, data histories, architectures, and training procedures, so their losses must not be presented as matched architecture effects.

## Phase G bounded reference study

The `engram-ffn-substitution-v1` micro study uses the pinned [FineWeb-Edu sample-10BT revision](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9). Run data is capped at 2,000,000 model tokens and validation at 65,536 tokens, with an 8,192-entry tokenizer. Before the BPE vocabulary exists, tokenizer fitting conservatively caps its UTF-8 input at 2,000,000 bytes; it used that full byte budget across 464 training documents. Each prepared memory-mode cache contains 2,000,000 train tokens from 1,571 documents and 65,536 validation tokens from 48 documents; the none and lexical caches have identical token-content hashes. The byte limit bounds input pieces, not final BPE tokens.

The 18-run MPS matrix uses 320 hidden dimensions, six layers, four heads, and 128-token training sequences. It varies dense FFN width (5,120 / 2,560 / 1,280), the repository's lexical `ngram` memory ([Engram reference](engram.md), off / on), and seed (17 / 41 / 73). The lexical variant uses a three-token address, an 8,191-row table, and a 32-dimensional latent value. Each arm has a 262,144-target-token ceiling (1,024 steps at the configured effective batch); seven declared contrasts are repeated across three matched seeds (21 descriptive pairings), without preregistered thresholds or inferential claims. The study also reports the fixed alias-retention, held-out-recall, and context-override cards separately as out-of-domain stress tests, not FineWeb quality metrics. Lookup adds 272,672 parameters at each width, so these are not parameter-matched controls. The micro study does not establish transfer to a 1B model.

The external [Pythia-70M-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped) observer scored 62,318 FineWeb-Edu validation targets from 60 documents: mean next-token loss 11.0523 at `step0` (commit `c913ae980de9355947d0bf73f9f10d580eb79301`), 4.0516 at `step10000` (`c890c8f6d8f86c36b2af66c3012a14ef1d35d3f3`), and 4.0215 at `step143000` (`9a7c847e93250c8f24d4b7e7134dbf369e8fc9cb`). The ignored local artifact `artifacts/phase-g/pythia-trajectory.json` records snapshot checksums and evaluation identities. This is an external training-trajectory observation, not a controlled comparison with SparseLab models or tokenizers.

Nemotron-CC was excluded: NVIDIA's [agreement](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/blob/main/LICENSE.md) permits internal AI-solution training only, bars transfer or distribution of the dataset and any use that would place it under open-source terms, and grants no rights to underlying copyrighted material. FineWeb-Edu is released under the [ODC-BY-1.0 license](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and is subject to [Common Crawl's terms](https://commoncrawl.org/terms-of-use); no blanket rights clearance for individual web documents is claimed. No raw corpus text will be included in the local weight package.

### Completed endpoints and outcomes

All 18 arms completed 1,024 steps and 262,144 training targets on MPS/fp32. The validation cache contains 65,536 tokens, but the configured six-batch checkpoint evaluation scored 1,536 targets per run; the cache size is not the evaluated target count.

Terminal next-token validation loss by width, memory mode, and seed:

| FFN width | Memory | Seed 17 | Seed 41 | Seed 73 |
|---|---|---:|---:|---:|
| 5,120 (4x) | none | 6.704315 | 6.685425 | 6.686797 |
| 5,120 (4x) | lexical | 6.704510 | 6.681582 | 6.679570 |
| 2,560 (2x) | none | 6.740580 | 6.731210 | 6.699564 |
| 2,560 (2x) | lexical | 6.739928 | 6.746710 | 6.703064 |
| 1,280 (1x) | none | 6.771940 | 6.786267 | 6.731881 |
| 1,280 (1x) | lexical | 6.740157 | 6.763137 | 6.746589 |

Descriptive paired validation-loss deltas (variant minus baseline; lower is better):

| Contrast | Mean delta | Seed 17 | Seed 41 | Seed 73 |
|---|---:|---:|---:|---:|
| lexical − none, 4x | −0.003625 | +0.000195 | −0.003843 | −0.007227 |
| lexical − none, 2x | +0.006116 | −0.000652 | +0.015500 | +0.003500 |
| lexical − none, 1x | −0.013402 | −0.031783 | −0.023131 | +0.014709 |
| 2x − 4x, none | +0.031606 | +0.036265 | +0.045785 | +0.012768 |
| 1x − 4x, none | +0.071184 | +0.067626 | +0.100842 | +0.045084 |
| 2x − 4x, lexical | +0.041347 | +0.035418 | +0.065128 | +0.023494 |
| 1x − 4x, lexical | +0.061407 | +0.035647 | +0.081554 | +0.067019 |

The narrower FFNs had higher validation loss than 4x in every matched seed. The lexical-minus-none differences changed sign across seeds or widths; they are not a consistent benefit, and lexical memory adds 272,672 trainable parameters. All three fixed stress-card scores were 0.0 for every terminal arm. These cards are out-of-domain checks, not FineWeb quality metrics. Three seeds and 1,536 evaluated targets per arm support descriptive observations only, not statistical or scaling-law claims.

Collection and report artifacts are local under `artifacts/phase-g/study-budgeted/`: terminal evidence `reports/architecture-24ec6aa377b2-5c5109b2fb96e1fcd53164ae35f387a6b385f717b99be9f2010f8e8a3fbdf222.json`; static report bundle `research-reports/31131bff510bfde30c1e2a169909870904eb5b469bab949a530a7d5a700abf1e/`. Collection explicitly selected each run's `latest.json`; intermediate checkpoints were not included in the card comparisons.

The optional local reference package is `artifacts/phase-g/micro-reference/`: three 4x/none fp32 safetensors weights for seeds 17, 41, and 73, the tokenizer, portable architecture metadata, source checkpoint manifests, and provenance hashes. The 414,909,288 weight bytes and all package hashes were verified; each source checkpoint passed `sparselab checkpoint verify`. No raw corpus text, token arrays, or optimizer states are included. Publishing remains unavailable because no external destination was supplied.
