# Papers we Love

Twenty-one papers for thinking about Tiny Sparse Lab experiments and model
building, grouped by theme. **Priority** retains the curated reading order
(1–21), not a claim about scientific merit. Titles and primary links were checked
on 2026-10-05; years denote first arXiv publication. The proposed tests below are
ideas for separately declared protocols, not changes to existing experiments or
claims that the lab reproduces these papers.

Three distinctions matter throughout:

- **Total versus active parameters:** sparse routing or lookup reduces the work
  touched per token, but inactive weights and memory tables still need storage.
  Count trainable, frozen, active, accelerator-resident and host-resident bytes
  separately, including gradients, optimizer state and caches where applicable.
- **Inference versus training offload:** moving inference weights or KV entries
  does not establish that training fits. The lab's [activation offload](docs/offload.md)
  moves saved tensors for backward; it is not expert, embedding or SSD offload.
- **Model quality versus software proof:** correct hashes, successful smoke tests
  and cache parity establish bounded software behavior. Held-out capability,
  retention and generalization require controlled model evaluations. Large-model
  or NVIDIA kernel results do not predict small-model ROCm performance.

The current [architecture](docs/architecture.md) provides dense, sliding-window,
block-sparse and reference MLA attention, local Top-K MoE, and token/byte/portable
Engram paths. [Semantic memory](docs/research/semantic-memory.md) accepts supplied
vectors through a direct API; it does not ship a natural-language query encoder.
New mechanisms below require explicit implementations where indicated. Use the
[research lifecycle](docs/research/experiment-learning-cycle.md) to preregister
controls, retain negative outcomes and separate compilation, training, prefill
and decoding costs. Hold data, tokenizer, effective batch, token budget and seeds
fixed unless they are the declared variable; report unavoidable budget asymmetry.

## Conditional memory, transfer and adaptation

### Priority 1 — When to Adapt: Conditional Memory Adapters for Retention-Preserving Domain Specialization

**2026 · [Primary paper](https://arxiv.org/abs/2608.29327)**

- **Contribution:** Engram Adapter adds occupancy-aware n-gram matching and gated
  residuals to a frozen model. Qwen3-4B/8B experiments report domain gains with
  strong retention on the tested out-of-domain tasks.
- **Proposed lab test:** extend the existing gated memory path with explicit
  occupancy masking; compare frozen baseline, always-on adapter and conditional
  adapter at matched adaptation budgets. Measure domain accuracy, unrelated-task
  retention, gate activity and output-distribution drift together.
- **Limit:** local matches also occur out of domain; retention depends on learned
  attenuation and the tested distributions, not a guarantee of zero interference.

### Priority 2 — Memory Grafting: Scaling Language Model Pre-training via Offline Conditional Memory

**2026 · [Primary paper](https://arxiv.org/abs/2605.20948)**

- **Contribution:** an offline donor supplies hidden vectors for frequent n-grams;
  longest-suffix lookup, projections and gates feed a recipient, with hashed
  Engram fallback for misses. The paper reports gains over matched pretraining
  baselines.
- **Proposed lab test:** use the [compiled-initialization bridge](docs/research/compiled-engram-initialization-v1.md)
  as a design reference for random, compiled-frozen and compiled-refined arms.
  A donor-vector compiler and exact suffix path would be new work. Include
  shuffled-value controls and charge donor compilation separately.
- **Limit:** recipients still train for 50B/100B tokens in the main settings.
  This is not demonstrated cheap retrofitting of an arbitrary frozen model.

### Priority 3 — Tokenizer-Agnostic Engram Module

**2026 · [Primary paper](https://arxiv.org/abs/2607.29065)**

- **Contribution:** polynomial hashing over bytes and a shared embedding space
  across n-gram orders make byte-equivalent sequences address the same memory
  despite different tokenizations; experiments train recipients using pretrained
  Engram embeddings.
- **Proposed lab test:** build a separate cross-tokenizer study from the
  [byte/portable interfaces](docs/portable-engram.md), comparing transferred,
  random and permuted tables with matched recipient training. Check byte-key
  agreement separately from held-out transfer accuracy.
- **Limit:** address equivalence is not universal vector alignment. Recipient
  training remains necessary in the reported transfer setup; the existing
  [learned portability protocol](docs/research/learned-engram-portability.md)
  tests widths within one model family, not this entire claim.

### Priority 6 — Scaling Embedding Layers in Language Models

**2025 · [Primary paper](https://arxiv.org/abs/2502.01637)**

- **Contribution:** SCONE learns contextual n-gram input embeddings with a separate
  model, then precomputes and offloads them for inference without enlarging the
  output vocabulary. It explores capacity outside the decoding backbone.
- **Proposed lab test:** use embedding-site token memory as a starting comparison,
  then separately implement contextual embedding compilation. Compare ordinary
  embeddings, random tables and compiled tables at fixed backbone/token budgets;
  measure compilation cost, lookup coverage, loss and inference bytes.
- **Limit:** fixed accelerator usage during inference excludes external storage
  and the embedding model's training cost. Existing Engram lookup is not a SCONE
  reproduction or proof of its latency results.

### Priority 7 — Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

**2026 · [Primary paper](https://arxiv.org/abs/2601.07372)**

- **Contribution:** Engram combines hashed n-gram memory with context-dependent
  gating, reallocating sparse capacity between stored patterns and MoE compute.
  Matched large-model experiments report quality improvements.
- **Proposed lab test:** use the [Engram/MoE capacity study](docs/research/engram-moe-capacity.md)
  to motivate separately budgeted memory-versus-expert ablations; report total
  parameters, active work, table utilization, held-out loss and generation quality.
- **Limit:** the under-3% host-offload throughput penalty uses host DRAM, an NVIDIA
  H800 and 512 sequences. It is not an SSD, batch-one, small-ROCm or training
  result; the lab's local reference components do not inherit that performance.

### Priority 11 — Larimar: Large Language Models with Episodic Memory Control

**2024 · [Primary paper, v4](https://arxiv.org/html/2403.11901v4)**

- **Contribution:** a learned encoder/decoder interface couples a language model
  to an episodic memory with least-squares reads and writes. After initial
  training, new edits and selective forgetting can occur without per-edit
  gradient training; fact-editing benchmarks test speed and locality.
- **Proposed lab test:** add a distinct episodic controller beside the existing
  semantic-memory API. Compare write/read/delete behavior with retrieval-only
  and no-memory controls, including paraphrases and unrelated facts.
- **Limit:** training-free edits assume an already trained memory interface.
  Removing a memory entry does not establish that the underlying language model
  has forgotten every occurrence of that fact.

### Priority 19 — User as Engram: Internalizing Per-User Memory as Local Parametric Edits

**2026 · [Primary paper](https://arxiv.org/abs/2606.19172)**

- **Contribution:** local hash-table edits hold individual facts while a shared
  adapter supplies reasoning skills; experiments separate direct recall from
  reasoning over inserted content and examine interference.
- **Proposed lab test:** use synthetic identities and facts in a new row-edit
  extension to byte memory. Compare table edits, an adapter baseline and
  retrieval, with collision, paraphrase, deletion and cross-identity controls.
- **Limit:** it assumes Engram pretraining and suitable injection depth. Locality
  for untouched addresses does not guarantee collision-free storage, robust
  paraphrase recall or reliable multi-hop reasoning as fact density grows.

## Attention, recurrence and model architecture

### Priority 5 — Optimizing Native Sparse Attention with Latent Attention and Local Global Alternating Strategies

**2025 · [Primary paper](https://arxiv.org/abs/2511.00819)**

- **Contribution:** combines latent compression with local/global sparse-attention
  alternation across layers, exploring a cheaper attention/cache design in
  340M–1.3B models.
- **Proposed lab test:** extend the current uniform attention configuration with
  explicit layer schedules; compare alternating and fixed schedules at matched
  parameters, context and token budgets. Measure retrieval, loss and cache bytes.
- **Limit:** reported training uses 15B/100B tokens. Compression and reduced cache
  size do not establish a small-model ROCm latency win, and the lab's combined
  MLA/MoE/memory smoke does not implement this alternating architecture.

### Priority 8 — DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

**2024 · [Primary paper](https://arxiv.org/abs/2405.04434)**

- **Contribution:** MLA compresses attention's key/value representation, alongside
  sparse DeepSeekMoE computation, to reduce inference costs at large scale.
- **Proposed lab test:** isolate latent attention against dense and grouped-query
  controls using the lab's reference paths. Compare quality, cache growth and
  decode timing; implementing the paper's compressed-cache design is distinct
  from the current [expanded-key cache](src/sparselab/model/attention/latent.py).
- **Limit:** the 236B-total/21B-active model trains on 8.1T tokens and combines
  several changes. Its headline savings are not predictions for the simplified
  lab MLA implementation or tiny-model quality.

### Priority 9 — Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention

**2026 · [Primary paper](https://arxiv.org/abs/2605.22791)**

- **Contribution:** separates channel-wise erase and write gates in recurrent
  memory, with channel-wise decay, to improve selective state updates; evaluates
  a 1.3B model trained on 100B tokens.
- **Proposed lab test:** add a new recurrent mixer, then compare tied and separate
  gates on repeated-key replacement, interference and retrieval at equal state
  size and training budget, followed by the existing dense language baseline.
- **Limit:** bounded recurrent state still compresses history. Better retrieval
  on the reported tasks implies neither exact arbitrary recall nor efficient
  kernels on the lab's target runtime.

### Priority 12 — On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability

**2026 · [Primary paper](https://arxiv.org/abs/2608.30320)**

- **Contribution:** studies hybrid Gated DeltaNet/QSA mixing, widened gated
  residual streams, host-prefetched n-gram embeddings and training choices as
  interacting architecture components.
- **Proposed lab test:** begin with an isolated memory-capacity sweep using the
  lab's embedding memory; compare held-out task scores as well as loss and total
  storage. Recurrent mixing, residual changes and host prefetch require separate
  extensions and ablations.
- **Limit:** downstream gains can saturate while loss improves. The
  [official Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
  lists 125B main parameters, 51B n-gram embeddings and 4B MTP, with 6B active:
  this is not a sub-35B-total model or a small-device performance demonstration.
  Serving support is implementation-specific: current [vLLM PLE offload
  documentation](https://docs.vllm.ai/en/latest/features/engram/#cpu-offload)
  requires UVA support on a CUDA-alike platform; [llama.cpp lazy reads](https://github.com/ggml-org/llama.cpp/blob/c06f84160a30c66d7b5a2829ae9b3ea15275cbc3/tools/cli/README.md)
  are a separate mmap-based disk path. Neither establishes Tiny Sparse Lab ROCm
  support or performance.

### Priority 14 — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

**2025 · [Primary paper](https://arxiv.org/abs/2502.11089)**

- **Contribution:** combines compressed context, selected token blocks and a
  local window in end-to-end trainable, hardware-aware sparse attention.
- **Proposed lab test:** use [block-sparse attention](docs/sparse-attention.md) as
  a reference starting point, then add explicit three-path ablations against
  dense attention. Sweep context length and selection budget; report quality,
  selected-key counts, end-to-end time and peak memory.
- **Limit:** the paper's 27B-model/260B-token experiment and A100 kernel timings
  do not validate the lab's simpler selector or predict benefits at short context
  on ROCm. Kernel correctness and model-quality equivalence are separate gates.

### Priority 15 — DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models

**2025 · [Primary paper](https://arxiv.org/abs/2512.02556)**

- **Contribution:** DeepSeek Sparse Attention (DSA) uses a lightweight learned
  indexer to select historical tokens for MLA, reducing long-context attention
  work; the report also includes broader training and reasoning advances.
- **Proposed lab test:** add an indexer as a new selector and compare it with
  fixed-window, random and current block selection at equal retained-token
  budgets. Count indexer overhead, missed relevant tokens and downstream quality.
- **Limit:** DSA adaptation includes 2.1B indexer warm-up tokens and 943.7B sparse
  continued-pretraining tokens. This does not demonstrate a cheap retrofit or
  isolate all full-model gains as effects of sparse attention.

### Priority 21 — Mamba-3: Improved Sequence Modeling using State Space Principles

**2026 · [Primary paper](https://arxiv.org/abs/2603.15569)**

- **Contribution:** improves state-space sequence modeling through discretization,
  complex-valued state updates and a multi-input/multi-output formulation,
  targeting state tracking and inference efficiency.
- **Proposed lab test:** implement a separate state-space mixer; compare controlled
  state tracking and associative retrieval with recurrent and dense baselines,
  then measure language loss under matched training and state-memory budgets.
- **Limit:** the reported 1.5B-scale quality and specialized inference results
  do not establish a drop-in replacement for the current decoder attention or
  comparable speed on small ROCm workloads.

## Tool use, constrained decoding and structured reasoning

### Priority 4 — ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings

**2023 · [Primary paper](https://arxiv.org/abs/2305.11554)**

- **Contribution:** learns output embeddings for tool tokens while freezing the
  language model; selecting a token invokes argument generation and tool use.
- **Proposed lab test:** extend the tokenizer/output head and add a bounded mock
  executor around [instruction training](docs/instruction-training.md). Compare
  learned tool tokens with textual descriptions under identical demonstrations;
  score selection, arguments and execution separately.
- **Limit:** experiments use LLaMA-13B/33B, not proof for 1–2B models. Embeddings
  still need training examples, and correct selection alone does not establish
  useful execution. The lab currently has no ToolkenGPT execution pipeline.

### Priority 10 — PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models

**2021 · [Primary paper](https://arxiv.org/abs/2109.05093)**

- **Contribution:** incremental parsing rejects inadmissible next tokens during
  decoding; text-to-SQL experiments use fine-tuned T5 models.
- **Proposed lab test:** add a parser-backed decoding constraint for a tiny tool
  grammar or structured output. Compare constrained and unconstrained generation
  from the same checkpoint, recording syntax validity, task correctness and
  decoding latency independently.
- **Limit:** a valid parse cannot ensure the right query or argument meaning.
  A parser and tokenizer-aware candidate checks are new work; this is not an
  existing guarantee of the lab's generation API.

### Priority 13 — Octopus v2: On-device language model for super agent

**2024 · [Primary paper, v7](https://arxiv.org/html/2404.01744v7)**

- **Contribution:** dedicated functional tokens represent APIs in a fine-tuned
  Gemma-based 2B model, reducing repeated tool-description context; the paper
  explores full and LoRA training. The linked revision is from September 2026.
- **Proposed lab test:** add a fixed synthetic tool inventory and compare function
  tokens, ordinary names and descriptions. Keep examples and tool coverage
  matched; measure argument accuracy and task success alongside prompt savings.
- **Limit:** specialized function-calling results do not establish general agent
  reliability or zero-shot support for arbitrary new tools. New inventories need
  training coverage, and fewer prompt tokens alone do not prove faster execution.

### Priority 20 — ReaRev: Adaptive Reasoning for Question Answering over Knowledge Graphs

**2022 · [Primary paper](https://arxiv.org/abs/2210.13650)**

- **Contribution:** graph feedback revises question-derived instructions while a
  graph neural network adaptively carries out multi-step reasoning.
- **Proposed lab test:** build a separate structured-memory benchmark beside the
  semantic retrieval interface. Compare adaptive revision with fixed instruction
  order and direct graph lookup on known paths, missing edges and distractors.
- **Limit:** ReaRev assumes graph structure and relevant entities; it is not a
  demonstrated drop-in decoder memory module. The lab's exact vector retrieval
  does not itself supply a graph, entity linker or learned reasoning controller.

## Inference storage and offload

### Priority 16 — Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

**2024 · [Primary paper](https://arxiv.org/abs/2402.07033)**

- **Contribution:** orchestrates CPU expert execution versus expert-weight transfer
  to GPU, using expert popularity to manage residency during MoE inference.
- **Proposed lab test:** extend local Top-K MoE with a separate inference-only
  placement experiment. Compare fully resident, CPU-executed and transferred
  experts; measure prefill/decode time, transferred bytes and numerical parity.
- **Limit:** gains depend on expert size, token load, CPU kernels and interconnect.
  The current local MoE keeps its weights resident; neither its sparsity nor
  Fiddler's results prove ROCm speedups or feasible gradient/optimizer offload.

### Priority 17 — LLM in a flash: Efficient Large Language Model Inference with Limited Memory

**2023 · [Primary paper](https://arxiv.org/abs/2312.11514)**

- **Contribution:** selectively loads predicted active feed-forward weights from
  flash, reuses recently active neurons and bundles rows/columns to improve I/O.
- **Proposed lab test:** add an inference-only weight loader and compare naive,
  cached and bundled reads at equal memory budgets. Track actual bytes read,
  predictor misses, latency and quality relative to fully resident execution.
- **Limit:** useful activation sparsity and accurate prediction are prerequisites;
  the lab's dense SwiGLU checkpoints do not automatically qualify. This concerns
  inference weight loading, not training-state offload or disk checkpoint writes.

### Priority 18 — KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference

**2025 · [Primary paper](https://arxiv.org/abs/2511.11907)**

- **Contribution:** keeps the full KV cache on disk, using compact metadata to
  predict preloads and schedule reads around computation and storage behavior.
- **Proposed lab test:** extend the lab's request-local cache with an explicit
  disk-backed inference mode. Compare resident and offloaded caches on identical
  long prompts, recording memory, I/O, decode latency and generation quality.
- **Limit:** the main evaluation uses Jetson Orin with NVMe/eMMC. Cache offload
  leaves model weights and training state as separate costs, and its storage
  behavior requires new measurements on discrete ROCm systems.
