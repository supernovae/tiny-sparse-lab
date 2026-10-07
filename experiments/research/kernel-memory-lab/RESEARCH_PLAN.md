# Building a Fresh Language Kernel with External Memory

> Repository adoption note (2026-10-07.2): the full foundation guide below is retained from release 2026-10-07.1. Statements about extracting the pack or no PR refer to that historical delivery. Use [README](README.md), [BOOTSTRAP](BOOTSTRAP.md) and [STATUS](STATUS.md) for current repository context. The later [expert track](EXPERT_TRACK.md) is an additional proposal; foundation gates and all thirteen unstarted cards remain unchanged.

Kernel Memory Lab working title
TinySparseLab research plan for Byron • Revised 7 October 2026
Revision 2026-10-07.1

### The recommendation

Start a new, randomly initialized language kernel in TinySparseLab. Its first goal is comprehension: understanding wording, nuance, relationships and evidence well enough to answer carefully. Use newly generated tiny synthetic fixtures for plumbing. Make the proposed 342M dense decoder the conditional main research kernel after fresh data and hardware checks. No earlier experiment or checkpoint is a prerequisite.

The main architecture test connects a versioned chunk store to a trainable chunk encoder and gated read adapter. Establish language and evidence-reading competence, supply known relevant chunks, then test a simple retriever, and only then consider learned routing or virtual PKM. Reader learning and router learning are separate experimental questions.

### The first useful step

Begin with the companion execution workbook: its first prompt requests a read-only gap assessment, followed by one bounded foundation change. Declare fresh initialization, fixtures, data/tokenizer decisions, evaluation and budget gates. Reuse TinySparseLab tooling, not prior checkpoints, corpora, optimizer state or approvals. This plan authorizes neither implementation nor runtime dispatch.

### What would count as a win

- A fresh kernel understands held-out language tasks and uses external evidence to improve answers, including facts inserted after its weights were frozen

- An integrated memory reader beats or improves the trade-off of a raw-text retrieval control at matched evidence and measured latency

- The working system stays inside measured GPU and host-memory limits, with acceptable cold-cache as well as warm-cache response times

This is a staged research route to a useful local language specialist. It does not promise that a 342M model becomes a general multi-billion-parameter reasoner. An archive cannot replace learning language, inference, instruction following or evidence use. Script generation is not the primary objective; code tasks are optional later domain tests.

### How to read this plan

Known means supported by the supplied project evidence or primary sources. Proposed means a concrete engineering choice to test. Unproven means a central hypothesis whose success must be measured. Thresholds and budgets below are proposed decision rules, not results already achieved.

Scope: updated research plan, execution workbook and a synchronized Markdown operating pack. No experiment implementation, training, corpus acquisition, cloud job or pull request was completed by this document task. The user reports a 7900 XTX with 24 GB VRAM on a WSL2 GPU PC; fit and software compatibility remain unverified. Optional cloud resources require separate availability checks and explicit spending approval.

## What the evidence supports

### The architectural idea is useful but not wholly new

Separating stored evidence from trainable computation has strong precedent. RAG and REALM learn document retrieval; RETRO adds neighbor reading to language modeling; Atlas studies retrieval-augmented few-shot learning. MiniRAG also combines chunks and entities in a graph. A learned address that resolves to an external chunk is not an untouched category. The contribution here would be a reproducible small-kernel implementation and quality–memory–latency frontier on ordinary hardware. [7–10,18]

RETRO is particularly relevant: a frozen-backbone retrofit trains the neighbor encoder and cross-attention, not only a router. Its 172M and 425M backbones become approximately 247M and 564M total models. Its headline efficiency comparison concerns a particular language-modeling setup, not general reasoning equivalence. REPLUG shows retriever-only adaptation with an already capable frozen reader; that does not establish the same outcome for an undertrained from-scratch kernel. [7,16]

### What each memory family actually stores

Original product-key memory stores trainable values selected through structured learned keys. Offloading those values changes residency, but the values and their training burden do not disappear. A pointer table whose rows identify documents is a different object: its payload is external data, and the reader still has to interpret that payload. [3]

An n-gram language model stores observed continuation statistics. An Engram-style module uses deterministic n-gram lookup to access learned embeddings. A chunk archive stores text or other records. kNNLM instead stores hidden-state/next-token pairs and interpolates a neighbor distribution. These systems have different objectives, update behavior and costs. [5,6]

### Where the attached discussion overreaches

- “Train a 100M model on language, then only train routing” skips the hard question of whether the model can use retrieved evidence and follow tasks. The plan tests that capability explicitly

- “Knowledge is stored rather than trained” is true for archive content, but not for the skills needed to retrieve, interpret, reconcile and apply it

- A learned slot address does not automatically generalize to unseen documents. Insertion, reindexing, stable identifiers and content-based routing must be designed

- A 10 TB archive is a possible storage endpoint, not a reason to begin with 10 TB. Search quality, index size, provenance and random-read latency can fail long before raw capacity does

Engram reports a favorable memory–compute balance, but its reported low-overhead host-offload test used a large H800-based setup and host DRAM, not an ordinary Radeon plus cold SSD. Sparse-memory-lm reports a striking small-core/large-table Wikipedia result, but its training memory, quantization, out-of-domain behavior and limited seeds prevent interpreting it as a general reasoning replacement. [4,5,17]

## Choose the experiment before the mechanism

Can a compact, freshly trained kernel understand a request, preserve its qualifications and constraints, distinguish what evidence supports, and apply newly supplied information? This is the primary question. Compare memory mechanisms only after measuring comprehension and separating reader learning from retrieval errors and hardware limits.

| Approach | What lives outside the dense kernel | Best first use | Main confound |
| --- | --- | --- | --- |
| Count n grams | Counts or continuation distribution | Cheap lexical and loss baseline | Exact overlap can look like knowledge use |
| kNN language model | Hidden state keys and next token labels | Nonparametric continuation control | Datastore and nearest neighbor costs |
| Trainable PKM | Learned value vectors and structured keys | Parameter efficient learned memory control | Optimizer state and training still scale |
| Engram style | Learned n gram embeddings with deterministic addressing | Conditional lookup experiment | Not the same as external raw knowledge |
| Raw text retrieval | Versioned text chunks and search index | Reader and retrieval control | Added context increases compute |
| Integrated chunk reader | Chunks or cached representations plus a learned adapter | Main architecture experiment | Encoder and fusion must learn to use evidence |
| Virtual PKM router | Structured address index resolving to chunk IDs | Later routing efficiency ablation | Discrete routing, stale addresses and missed evidence |

### Main path

Use raw-text retrieval as an intentionally strong control and a fast diagnostic. The main experiment is an integrated chunk reader: the kernel produces a query, an external index returns chunk IDs, a bounded read budget supplies evidence, and a gated cross-attention adapter returns a residual to the kernel. This preserves the architectural ambition while keeping failures interpretable.

### What to defer

Do not begin by combining count n-grams, Engram, PKM, a chunk graph, multiple learned routers and SSD paging. That makes a negative result impossible to diagnose. Defer graphs until simple chunk retrieval has a documented failure that graph structure can address; defer compressed latent-only storage until raw-text evidence can be read reliably.

Keep ordinary dense and no-memory controls throughout. Optional published pretrained models can provide an external reference ceiling, but they are not substitutes for the user’s from-scratch deliverable and must be labeled separately.

## Select a kernel without hiding the cost

### Keep fresh fixtures separate from the research model

New tiny synthetic models and datasets validate labels, masks, loss, gradients, serialization and resume. Generate them in a fresh project namespace with deterministic seeds and no imported weights. Passing these fixtures proves plumbing only. The main research model is separately initialized from scratch and must earn every language and memory capability claim. [2]

The main candidate is a roughly 342M native dense decoder. A proposed 124,275,456 parameter fallback uses width 768, 12 layers, 12 heads, feed forward width 2,560 and vocabulary 32,768. It is an optional response to measured fit or iteration cost, not a model that must be trained first. Do not automatically train both.

| Item | Main candidate and reason |
| --- | --- |
| Backbone | Native TinySparseLab dense decoder; proposed 341,885,952 parameter shape; random initialization |
| Transformer | 24 layers; width 1,024; feed forward width 2,816 |
| Attention | 16 attention heads; head dimension 64; use the native dense attention path first |
| Core components | RMSNorm, SwiGLU, rotary positions, tied input/output embeddings |
| Vocabulary | 32,768 entries; fit and version a new tokenizer on approved training-only data; fixtures use their own tokenizer; recount parameters if vocabulary changes |
| Context | Start at 1,024 tokens for profiling and pipeline tests; test 2,048 after measured fit; memory tokens count against the chosen reader budget |
| Memory insertion | One read adapter first, near the upper middle of the stack; exact layer is an ablation, not an established optimum |
| Precision | Use a numerically checked precision supported by the actual ROCm/PyTorch stack; do not assume every fast kernel works on the 7900 XTX |

This proposed shape was inspected in the earlier audit but remains unauthored for this project, untrained and unproven on the GPU. Reference dense attention still materializes scores and uses float32 softmax. Current main also ships opt-in native SDPA for PyTorch dense attention with equal query and KV heads. Reuse and benchmark that path only under a bounded approval; an SDPA call does not guarantee a fused backend. [2]

All neural weights start from scratch. Any imported pretrained encoder, retriever or teacher must be disclosed as a separate variant. Use lexical retrieval first if the entire trainable system must be from scratch. SmolLM2 is a useful external design and training-data reference, not the implementation to reproduce wholesale. [12]

### A research kernel is not published model quality

Official SmolLM2 reports training small models on trillions of tokens. This plan’s conditional 1B, 3B and 10B milestones are much smaller experiments, not a reproduction of that capability. Parameter count alone does not establish comprehension or reasoning. [12]

## Train useful behavior and test it early

A new tiny oracle-reader fixture can run alongside pipeline design, but it tests wiring rather than language understanding. Substantive reader comparisons need a fresh kernel with measured language and instruction competence. If oracle evidence fails, diagnose base capability and reader learning before funding a router or larger archive.

| Stage | Training content and objective | Decision before expansion |
| --- | --- | --- |
| Plumbing | New tiny synthetic model and corpus; verify labels, masks, loss, gradients and strict same-experiment resume | Loss can deliberately overfit; checkpoint/resume and generation are consistent |
| Base calibration | Approved clean general prose and chosen-domain mixture; code is optional; next-token prediction | Profile first; authorize a token cap only after measured speed, fit and data readiness |
| Reader curriculum | Oracle evidence for paraphrase, inference, negation, conditions, conflict and missing-evidence tasks | The model uses evidence and abstains on absent answers rather than memorizing question templates |
| Retrieval aware training | Same tasks with retrieved positives, hard negatives, shuffled evidence and memory dropout | Quality follows evidence relevance; wrong retrieval has a bounded failure mode |
| Instruction and domain use | Instruction following, source support, contextual interpretation, careful uncertainty and concise explanations | Useful behavior improves on held-out prompts without hiding language-model regression |
| Scale or specialize | Conditional 1B then 3B then 10B total target-token milestones, with replay and evaluation | Each extra tranche has an explicit expected gain, wall-time estimate and stop rule |

### Choose a bounded first evidence domain

Choose one coherent, licensed, versioned prose domain for the first evidence store. It should contain distinctions, conditions, exceptions and claims with short supporting passages. Authoritative technical documentation is one option; it is not a requirement to build a code assistant. Select sources and train/evaluation partitions explicitly before bulk acquisition.

The base training mixture must support reading, comprehension and task following. Propose clean general prose plus the chosen domain, with optional code only if it serves a declared test. External storage supplies specific facts and provenance; the model must still learn how negation, context, inference and uncertainty affect an answer. Do not inherit an old corpus or mixture.

### Freeze and unfreeze deliberately

First establish a measured language-capable fresh checkpoint. Then freeze that checkpoint for the initial encoder/adapter comparison to isolate reader capacity. A frozen random core is only a plumbing control. If the trained frozen base plateaus while a bounded joint-training control succeeds, allow separately approved joint training with language replay. RA-DIT and RAFT motivate reader adaptation; router-only learning remains an ablation. [19,20]

## Define language understanding as measurable behavior

The kernel should learn to interpret a passage and the user’s question, preserve distinctions, combine supported facts and say when the evidence is insufficient. Fluent wording or successful script generation alone does not establish that goal. Use the following axes before declaring the fresh kernel ready for a frozen-reader experiment.

### Meaning and reference

Paraphrase with changed wording; resolve pronouns and entity references; distinguish a statement from a question or an attributed claim.

### Logical and contextual nuance

Test negation, quantities, exceptions, conditionals, temporal order and version changes with minimal pairs whose correct answers differ.

### Evidence and inference

Separate extraction from supported inference; combine two short sources when both are needed; identify relevant passages and avoid importing unsupported assumptions.

### Uncertainty and conflicting sources

Use the declared source/version precedence; report unresolved conflicts; abstain when evidence is absent; do not reward confident unsupported prose.

### Instructions and generalization

Follow answer constraints on held-out wording, entities and document families. Keep style/format scores separate from semantic correctness.

### Evaluation contract before training

Propose a 300–500 item evidence suite with at least 200 answerable and 50 missing/conflict items, plus a separate 200-item language diagnostic with at least 20 items per axis. Include curated natural language; split synthetic templates/compositions and entities, not just rows. Freeze rubrics and a human-adjudicated sample before testing. Allow multiple valid answers and score needed clarification fairly; debugging examples never become held-out evidence.

For the initial frozen-reader eligibility gate, propose at least 60% aggregate diagnostic correctness and at least 50% on each axis, with finite held-out loss and stable instruction formatting. These are adjustable engineering thresholds, not a claim of general understanding: approve changes before a run, report denominators and uncertainty, and require a reviewer to inspect errors. A checkpoint that fails remains a training diagnostic, not a qualified frozen reader.

## Prove the reader before the router

### A small test with diagnostic power

Build 300–500 initial evaluation items from a small licensed prose corpus, separate from training. Include multiple documents, paraphrased questions, held-out entities and document families. Each answerable item needs an answer rubric, supporting chunk IDs and a version; missing-evidence items need an abstention target. Score comprehension and evidence use by category. These proposed sizes are not power guarantees.

Use the same query, answer budget and decoder for four conditions: no evidence; the gold evidence; a plausible but wrong chunk; and shuffled or absent evidence. Run both raw-text evidence and the proposed integrated reader. Keep the answer itself out of query construction and retrieval keys at evaluation time.

### What this separates

- No evidence versus oracle evidence measures whether external information can help this model at all

- Oracle versus actual retrieval measures the retrieval gap

- Raw text versus integrated reading, using identical chunks, measures the value or cost of the architectural adapter

- Correct versus wrong evidence reveals whether answers are grounded, ignored, blindly copied or fabricated

### Proposed first gate

On a held-out set of at least 200 answerable items, seek a 15 percentage-point absolute accuracy gain from oracle evidence and a clear reduction in unsupported answers. Also inspect 50 or more missing-evidence and conflict cases. Treat these as initial engineering thresholds: report confidence intervals, denominators and per-category errors, and revise a threshold before a run if the baseline makes it inappropriate.

A useful integrated reader should retain at least 90% of the raw-text oracle condition’s task score at the same evidence budget, or earn its quality cost through a measured memory or latency improvement. If raw text works and the adapter fails, fix the adapter. If neither works, improve the kernel or training curriculum. Do not scale the router to solve either failure.

### Insertion is the decisive test of external knowledge

Freeze model weights. Add new documents or a held-out documentation version to the store, build only the required index entries, and evaluate questions about the new material. Success is useful answers with evidence from the new records, without answer training on those facts. Keep this score separate from old-corpus retrieval and from any subsequent fine-tuning.

A failed held-out reader test on an undertrained fresh checkpoint is evidence about that checkpoint, not a final rejection of external memory. A tiny-set overfit validates the gradient path only. Record which competence or integration failure the next bounded test would resolve.

## Design the integrated chunk reader

### A concrete first architecture

1. Create immutable records with stable chunk IDs, document IDs, versions, token offsets, text hashes and source locations. Store raw text/tokens independently from any learned representation

2. At one read point, derive a query from the current prompt or causal hidden state. During the first experiment, supply oracle IDs; next use a lexical or hybrid retriever with a fixed top k

3. Read a bounded number of chunks. Start with four chunks of 128 tokens each, with sentence or code-boundary handling. Keep the total evidence budget fixed across comparisons

4. Proposed reader, not yet implemented: a two-layer encoder of width 256, four heads and feed forward width 1,024, plus one four-head cross-attention adapter after block 16. Project its output back to kernel width through a small gated residual. Randomly initialize and count these additional weights separately

5. Train answer-token loss together with any explicitly reported retrieval loss. Include missing evidence, negatives, contradictions and memory dropout; log gate activation and evidence utilization

6. At inference, initially retrieve once per question or prefill. Only add periodic retrieval when evaluation demonstrates that the extra reads improve multi-step answers enough to justify their latency

### Make the system causal

For language-modeling evaluation, retrieval queries may depend only on the visible prefix. A datastore containing the continuation or a near-duplicate of the test document can create a false memory gain. Technical question answering and next-token perplexity need separate retrieval policies and leakage checks. RETRO’s chunked causal setup is a relevant design reference, not a detail to copy without tests. [7]

### The virtual PKM option

Define virtual PKM narrowly: a structured, possibly product-key address index maps a query to candidate chunk IDs or posting lists. The chunk contents remain external and versioned. The index is not the knowledge itself. Compare its recall, memory, training cost and end-to-end latency against lexical search and a conventional vector index under the same candidate and read budgets.

One concrete optional prototype uses two 256-entry codebooks over halves of a 128-dimensional query, giving 65,536 address combinations. Store chunk-ID posting lists at those addresses, probe the best 8 keys per half, cap candidate reranking at 256 chunks and read the best 4. These unimplemented starting values require design review. First train aligned query/chunk embeddings with contrastive positive and hard-negative pairs; then freeze chunk embeddings, fit the product codebooks and build posting lists. Measure recall lost through bucket assignment. New chunks enter by the same content assignment; changing codebooks requires a versioned rebuild. The addresses alone provide no semantic mapping. This is a learned inverted-index prototype, not a proven PKM improvement.

Train a content-based router rather than a classifier over every fixed document ID. Use known supporting chunks and in-batch or mined negatives. Hard top-k selection is not automatically differentiable through the archive: use supervised contrastive routing first, and document whether gradients reach the query encoder, chunk encoder, routing keys and reader. More elaborate estimators are later research, not the first implementation.

If a learned address loses recall after insertion or needs expensive retraining for each update, it has failed the intended separation. Keep the simple retriever as a control. A fixed ID classifier cannot address a new document unless the system has a defined insertion and candidate-scoring mechanism.

## Make technical knowledge maintainable

### Separate content from its representations

The canonical store should retain source text or code, not only opaque learned vectors. Use a manifest that records source URL or repository revision, license, document version, extraction tool version, tokenization version and checksum. Every retrieval result must resolve back to this provenance. Never use a mutable array offset as the only durable chunk identity.

Begin with deterministic chunking near 128–256 tokens, preserving headings, code blocks and API signatures. Record overlap rather than silently counting overlapping chunks as independent evidence. Longer chunks reduce fragmentation but increase read cost; shorter chunks may lose definitions. Evaluate chunk size instead of introducing learned phrase discovery immediately.

### A minimal record

| Field group | Required content |
| --- | --- |
| Identity | chunk_id, document_id, content_hash, version or revision |
| Provenance | source locator, license, extraction date, section or token offsets |
| Payload | raw text or code; token IDs tied to a tokenizer version |
| Index metadata | embedding or lexical index version; optional routing bucket IDs |
| Lifecycle | active or superseded state; replacement relationship; access scope if needed |

### Adding or replacing facts

Ingest and validate new material into a new store version, compute its representations, update the index, run targeted insertion and conflict checks, then switch the manifest pointer. Keep old versions reproducible. A changed encoder requires representation migration or an explicit dual-version path; cached embeddings do not remain valid merely because chunk text is unchanged.

For corrected facts, retrieve the authoritative version and test whether the model overrides stale parametric knowledge. For contradictory sources, require the model to identify the conflict or abstain. A store update is not equivalent to editing the model’s beliefs: evidence precedence is learned behavior that must be evaluated.

### What should remain in the kernel

Language comprehension, syntax, contextual interpretation, inference and evidence-use skills remain distributed through neural weights. Store long-tail facts, definitions, versioned details, examples and provenance externally. The experiment measures a useful separation of capabilities and stored content; it does not assume reasoning and knowledge are perfectly separable.

Learned latent stores have extra compatibility costs. SCONE jointly trains the encoder and recipient before precomputing n-gram features. Memory Grafting uses teacher-derived representations with learned recipient integration; it is not free post-hoc knowledge transfer. Tokenizer-agnostic addressing can align text across tokenizers without making latent spaces interchangeable. Keep raw records and version every encoder, projection and tokenizer. A teacher-based variant must be labeled separately from the strict from-scratch system. [11,13,15]

## Map the architecture to real memory tiers

The user-reported GPU is an AMD Radeon RX 7900 XTX with 24 GB VRAM. Host RAM, free memory under load, SSD model and random-read performance are unverified. Use a fresh inventory and measured limits; the attached discussion’s hypothetical host-memory capacities are not facts about this machine.

| Data or state | Illustrative size | Placement and caution |
| --- | --- | --- |
| 342M dense weights | About 0.684 GB in 16 bit form | GPU; weights alone are not peak training memory |
| Dense training state | About 5.47 GB at 16 bytes per parameter | Illustrative Adam style weights, gradients and states; implementation varies; activations are additional |
| Decoder KV cache | About 201 MB at 2,048 tokens and batch one for 24 layers and 16 KV heads | GPU at inference; excludes reader, workspaces and fragmentation |
| 1M chunks of 128 tokens | 0.512 GB with uint32 token IDs | RAM or SSD; excludes text copies, offsets and metadata |
| 1M pooled 256 dimensional embeddings | 0.512 GB at 16 bit precision | Index storage before graph, quantization or metadata overhead |
| 1M chunks of token level features | 65.5 GB for 128 by 256 at 16 bit precision | One feature tensor only; separate keys and values can double it |

These are decimal-byte arithmetic estimates, not measured device allocations. A 10M-chunk token-plus-pooled-embedding store is about 10.24 GB before index and provenance overhead; a 100M-chunk store is about 102.4 GB. Caching token-level reader features can dominate both. Count the full representation rather than advertising only the raw archive size.

### A staged physical layout

- GPU: dense kernel, active reader and current evidence, with a bounded hot cache only if it helps

- Host RAM: retrieval index or its hot portion, chunk offsets, bounded content cache and staging buffers; retain a configurable operating-system reserve

- NVMe: canonical immutable chunk records, cold index partitions and checkpoints; batch reads and prefetch where the actual access pattern permits

First demonstrate the logical reader with the store in RAM or a small memory-mapped file. Then force a dataset larger than the permitted cache and measure cold reads. Memory mapping is an access method, not proof that data stayed off RAM: account for the operating-system page cache and resident set.

### Measure the cost where it occurs

Report retrieval time, page faults, cache hit rate, host-to-device transfer, reader compute, time to first token and sustained decode separately. Compare batch one and the intended service batch; warm and cold cache; sequential and random access. Learned hidden-state queries may arrive too late to hide their reads, unlike deterministic token-address lookups that can be prefetched earlier. [5]

SCONE reports NVMe lookup measurements, but on a 64-core, 512 GB host. That establishes feasibility in its setting, not this machine’s latency. Its large feature tables also illustrate why raw text and latent storage must be budgeted separately. Do not add optimizer-state offload unless profiling requires it; it solves a different problem from evidence residency. [11]

## Use controls that survive a skeptical review

| Condition | What it establishes |
| --- | --- |
| Dense kernel without memory | Base ability at each training milestone |
| Count n gram interpolation | A cheap lexical continuation baseline |
| PKM at a bounded table size | Whether learned sparse values help at a fair training budget |
| Raw text with oracle chunks | Reader ceiling without retrieval mistakes |
| Integrated reader with oracle chunks | Fusion cost or benefit on identical evidence |
| Raw text with lexical or vector retrieval | A practical external memory baseline |
| Integrated reader with the same retrieval | The main architecture comparison |
| Learned router or virtual PKM | Whether learned addressing improves recall or total system cost |
| Wrong, shuffled, absent and updated memory | Grounding, robustness, leakage and insertion behavior |

### Match the expensive things

For each comparison, declare kernel parameters, total trainable parameters, number of target tokens, optimizer steps, corpus mixture, evidence tokens, retrieval candidates, batch size and training/inference wall time. An architecture can win at equal dense parameters while losing at equal total training compute. Show both comparisons when feasible; do not force a single parameter-count ranking.

Use a dense baseline at every checkpoint. Add memory variants sequentially rather than running the entire matrix at full scale. Reuse frozen evaluation items and store versions, with multiple seeds for the small decisive experiments. Record all exclusions and failed runs.

### Score capability and the system separately

- Language modeling: held-out loss or perplexity on uncontaminated text, with a causal retrieval policy

- Comprehension and evidence use: paraphrase, coreference, negation, quantifiers, temporal conditions, inference, contradiction, source support and appropriate abstention; executable code is optional

- Retrieval: recall at k for gold evidence, latency and index size; evaluate the router independently of generated answers

- Resources: peak VRAM, peak host RSS, cache residency, disk footprint, cold/warm p50 and p95 latency, and wall time

- Economics: total training and indexing cost, incremental ingestion cost, and energy only when a defensible measurement is available

### Guard against false wins

Split by source document or project before chunking; deduplicate across train, memory and evaluation with exact and near-duplicate checks. Permit the supporting reference documents in a designated open-book evaluation, but exclude question–answer pairs and hidden answers from training and query construction. Label that setting clearly; it is different from a closed-book benchmark.

A quality-per-VRAM number is useful only alongside absolute quality and latency. A system that uses almost no VRAM but fails the task is not a win; a fast warm-cache result does not establish SSD feasibility.

## Advance through bounded decision gates

### Gate 0   Establish a trustworthy starting point

Create a fresh project namespace, random-initialization specification, new tiny fixture manifest, provisional data/tokenizer decision and hardware inventory. Verify actual hashes and provenance when artifacts exist. Pass fresh-fixture overfit and same-experiment resume tests. No old artifact, corpus, optimizer state or runtime approval may be required to reach this gate.

### Gate 1   Profile before committing tokens

After explicit authorization, profile one selected configuration with a bounded local run: proposed 20 warm-up and 100 measured steps, with a 30-minute wall cap, stopping at the first cap. Start at context 1,024 and microbatch one. Measure target tokens/s, peak VRAM/RSS, finite loss and stalls. More contexts, repeats or cloud use need their own stated cap; these defaults are not permission.

### Gate 2   Prove evidence reading

Run raw-text oracle evidence and wrong/missing-evidence controls on a language-capable fresh checkpoint. Require meaningful oracle gain and evidence-sensitive behavior. A toy overfit is not this gate. Reader training, base training and each later integration experiment require separately bounded approval; do not infer permission from an approved design.

### Gate 3   Close the retrieval gap

Start with lexical retrieval, then a conventional embedding index if needed. A proposed initial goal is at least 85% gold-evidence recall in the chosen top-k budget and retention of at least 80% of the oracle accuracy gain over no evidence. If recall is poor, fix chunking, query construction or the retriever before blaming the reader. Measure new-document insertion here.

### Gate 4   Test the architectural improvement

After raw-text oracle and lexical controls, validate the explicit same-project dense-to-augmented transfer contract and integrated reader on identical chunks. Continue only for better quality or a measured efficiency gain. Consider learned routing when lexical retrieval is a measured limitation, and virtual PKM only as a controlled index alternative.

### Gate 5   Enforce physical limits

Set approved VRAM, host-memory and latency targets from the actual machine. First compare the same immutable store in RAM and forced-cold NVMe with identical queries, evidence and model. Track the operating-system cache. Expand beyond the allowed cache only under a separate storage/data cap. Without a latency target, report the frontier rather than declaring a pass.

### Gate 6   Authorize the next training tranche

A 1B-target-token milestone is a possible first substantial run, not the next automatic action. Advance toward 3B or 10B only when validation gains, reader gains, data quality and forecast wall time justify it. Keep the best checkpoint, stop on a predeclared plateau or regression, and preserve enough budget to evaluate the result.

No calendar promise is possible before throughput, data preparation effort and spending limits are measured. A useful early outcome is a clear falsification or a small reproducible reader win, not a large checkpoint with an unexplained score.

## Map the plan onto TinySparseLab

The 7 October 2026 source audit pins main 06efc4db82ecf3da97b50cff518cba605ad27b33, including PR49. This pass inspected source contracts; no repository tests or GPU experiments were executed. Reuse the shipped infrastructure below while keeping fresh-project research cards unexecuted. [2]

| Existing surface | Use or limitation |
| --- | --- |
| Config and experiment derivation | Reuse typed declaration authoring and derivation receipts. It writes fresh outputs; it is not checkpoint transfer or a runtime proof. |
| Reference dense attention and native SDPA | Reference remains default. SDPA is opt-in for PyTorch dense/equal-head configs; measure the selected backend and actual fit. |
| Semantic probe and trainer sidecars | Reuse read-only supplied-vector controls and verified query/mask sidecars. Raw-text encoding and learned retrieval remain separate work. |
| Sealed staging and training state digest | Reuse authenticated prepared bundles and the Python exact-state comparator for appropriate tests. Neither is a generic cache or checkpoint-authentication shortcut. |
| Source overlap and preparation benchmark | Reuse descriptive continuation/source overlap. Synthetic CPU preparation benchmarks are mutating and do not measure real-corpus or GPU throughput. |
| Dataset and Campaign contracts | Reuse configuration, preparation and campaign gates. Fresh data/tokenizer identities, reviewed state and explicit runtime authority are still required. |
| Hosted execution and checkpoint relays | Reuse verified transport/receipt contracts if later approved. Provider allocation and live loss-recovery acceptance are not established by this audit. |

Relevant sources are linked in [2]; module paths are under src/sparselab/. In particular, engines/pytorch.py implements verified sidecar-built semantic batches despite stale semantic-memory prose.

### The current semantic path cannot learn the desired router

The retriever supports exact CPU top-k from 1 to 16; the adapter uses top-1. Queries detach to CPU float64, so language-model loss cannot train the query encoder through lookup. Default limits are 65,536 entries, 256 MiB tensor assets, 4,194,304 comparisons and 64 MiB results. Supplied-vector injection does not provide raw-text encoding, a learned content router or SSD-backed knowledge paging. [2]

### Initialization and transfer need explicit contracts

Keep fresh random initialization, strict same-experiment resume and explicit same-project dense-to-new-reader transfer distinct. PR46 config derive and experiment derive author typed declarations and receipts, not checkpoint tensors. The proposed transfer still needs dense-tensor mapping, tokenizer identity, new-module initialization, reset training state and disabled-memory equivalence. The exact training-state digest is a Python comparison API, not a checkpoint verifier or resume authorization. [2]

The corpus pipeline still treats requested_mixture as reporting metadata and counts the records actually retained; the delta adds no enacted mixture sampler. Specify realized ordering and target-token accounting. Derivation and preparation benchmarks do not establish a sampler. Sample artifacts and model-size filenames are interfaces, not selected data, trained capability or permission for this fresh experiment. [2]

## Use the companion workbook to execute the plan

### Task 1   Assess and declare the fresh experiment

Owner: Luna for a read-only assessment of the actual checkout, then one separately approved foundation change; Sol reviews nontrivial contracts. Reuse the shipped surfaces in the current audit. Declare only missing project-specific protocol, namespace and inputs; do not recreate existing probes, staging, derivation or campaign tooling. Preserve unrelated experiments.

Acceptance: schemas and applicable CPU tests pass; native inspection reports 341,885,952 parameters; artifacts are verified or explicitly unresolved. Record user approval before GPU/cloud work or campaign mutations. A requested bounded CPU task can authorize its stated tests. This is an operator rule: verify native enforcement rather than inventing an approval field. Missing main inputs block execution, not honest design review. Return diff, receipts and open inputs.

### Task 2   Make the small control reproducible

Owner: Luna with Sol escalation. Generate new tiny data/model fixtures, then reuse the native semantic probe for supplied-vector oracle, wrong-vector and no-memory controls. Check overfit, gradients, memory-disabled equivalence and strict resume; use the exact-state digest where applicable without treating it as artifact authentication. Correct the stale trainer-sidecar prose rather than duplicating its implemented batch construction. These receipts prove plumbing only.

Acceptance: deterministic inputs and outputs; the expected parameters receive gradients; frozen parameters do not; the disabled path matches the dense control within a stated numerical tolerance; failures leave inspectable evidence. Escalate to Sol for initialization, optimizer selection or semantic-batch alignment bugs.

### Native command templates and side effects

Set CONFIG and CAMPAIGN only to real, validated files in the selected environment. These pinned interface templates do not mean fresh files already exist. Inspect the installed CLI and locked environment first; if dependencies are absent, report that prerequisite. Status, next and explain report campaign state; they do not authorize a transition.

uv run --locked --no-sync sparselab inspect "$CONFIG" --json

Preparation only: uv run --locked --no-sync sparselab workspace preflight "$CONFIG"

uv run --locked --no-sync sparselab campaign validate "$CAMPAIGN" --json

uv run --locked --no-sync sparselab campaign status "$CAMPAIGN" --json

uv run --locked --no-sync sparselab campaign next "$CAMPAIGN" --json

uv run --locked --no-sync sparselab campaign explain "$CAMPAIGN" --json

Inspect and campaign validate/status/next/explain are observational templates. Workspace preflight checks capacity but may initialize the work-root/scratch, so keep it outside a strict read-only audit. Campaign apply/resume can prepare, pilot, evaluate or generate without execute-runs. Derive writes fresh declarations/receipts; stage may warm up. Prep benchmarks/readiness smoke are active work. Preflight and stage have no json flag. Prepared-inputs needs a sealed staging bundle, not an array cache. Each active operation needs its own authorized scope. [2]

## Keep implementation and escalation manageable

### Task 3   Prove initialization and profile the native kernel

Owner: Luna prepares the fresh initializer, native configuration and bounded fit request; Sol reviews contracts and numerics. Reuse config derivation for authoring and the shipped attention paths where appropriate. Only after exact profile approval, save throughput, actual attention backend, peak VRAM/RSS and failure evidence. Stop on nonfinite loss, OOM, identity mismatch or the first time/step/spend cap.

Acceptance: initialization is reproducible; verified tensor mappings cannot silently drop dense weights; parameter groups are explicit; approved timing excludes warm-up. Dense-to-reader equivalence is required before fusion. If reference attention limits fit, propose a bounded comparison using the shipped SDPA path and report the actual backend. Recomputation remains a separate design change. No available attention option establishes local GPU compatibility.

### Task 4   Build the reader on a small versioned store

Owner: Luna handles fixtures, provenance and evaluation; Sol handles encoder/adapter integration. Establish raw-text oracle then lexical retrieval baselines first. Add one gated reader with identical evidence and an explicit same-project dense transfer. Train only within a separate approved cap. Keep strict-from-scratch and any pretrained-component controls labeled separately.

Acceptance: the held-out oracle gate passes or produces a clear failure diagnosis; wrong/missing evidence is scored; insertion tests use frozen model weights; runtime and memory are measured. Escalate to Astra only if evidence points to an unresolved architectural question rather than a coding defect.

### Task 5   Optimize a demonstrated bottleneck

Owner: Luna for benchmark operation, Sol for retrieval and paging, Astra for experimental design. Add a content router only after the simple retriever has a measured limitation. Compare virtual PKM at the same query/candidate budgets. Once the reader is useful, compare the same store in RAM and NVMe before separately approved scale tests.

Acceptance: compare recall, task quality, insertion behavior, index maintenance, cold/warm p95 latency and total memory. A new mechanism must win a declared trade-off, not merely add parameters or reduce one isolated metric.

### Use escalation as a decision rule

| Escalate to | Trigger | Bring along |
| --- | --- | --- |
| Sol | Nontrivial loader, gradient, optimizer, device, memory or numerical correctness problem | Minimal failing case, exact revision/config, logs, expected behavior and a narrow question |
| Astra | Competing mechanisms with ambiguous evidence; failed reader despite validated plumbing; redesign of learning objective | Ablations, oracle/retrieval gap, error categories, cost measurements and the decision that must be made |
| Byron | Budget, target latency, hardware choice, corpus/license scope or a consequential scope change | Options with cost and benefit; one concrete approval request |

Keep each change independently reviewable. Do not delegate “build the whole memory hierarchy” as a single task. A compact evidence bundle and one decisive next question make a smaller model useful without asking it to resolve every research uncertainty at once.

## Budget from measurements and keep a stop rule

### Token counts are commitments only after throughput is known

Estimate training wall time by dividing target tokens by measured sustained target tokens per second, then add checkpointing, validation, restarts, data preparation and idle time. The following figures are arithmetic scenarios, not predictions for the 7900 XTX or for the proposed native attention implementation.

| Target tokens | 1,000 tokens per second | 5,000 tokens per second | 10,000 tokens per second |
| --- | --- | --- | --- |
| 1 billion | 11.6 days | 2.3 days | 1.2 days |
| 3 billion | 34.7 days | 6.9 days | 3.5 days |
| 10 billion | 115.7 days | 23.1 days | 11.6 days |

A rough dense-training compute proxy of six times parameters times target tokens gives about 2.05 × 10¹⁸ operations for 1B tokens at 341,885,952 parameters, 6.15 × 10¹⁸ for 3B and 2.05 × 10¹⁹ for 10B. This excludes reader/retriever work and does not model quadratic attention, recomputation, utilization or I/O. Use measured wall time to make decisions. [14]

Local operation is the default. Cloud calibration or training is optional; usable budget, expiry, GPU type, credit burn and runtime limits must be verified for any proposed job. Before any cloud run, verify those terms, estimate credits for setup, run and evaluation, and obtain an explicit cap. Credits do not establish affordable hours or authorization to spend.

### The most important failure modes

- The kernel cannot read: oracle evidence does not help. Improve data, instruction/reader training or capacity; do not add a smarter index

- Retrieval misses: oracle helps but real retrieval does not. Fix chunking, query construction, recall and versioning

- Integrated reading loses to raw text: retain the useful raw-text baseline and diagnose the adapter before further complexity

- New documents cannot be used without weight updates: the desired update separation is not yet achieved

- Cold-cache latency or host memory exceeds the target: reduce the working set, read frequency or representation cost; report the actual frontier

### The decision at the end of the first cycle

Continue only when fresh initialization, measured local fit and held-out comprehension are established, and an external-memory gain survives wrong/missing-evidence controls, frozen-weight insertion and the declared resource budget. Revise a localized failure with one cheap decisive test. Stop a mechanism when it loses to a simpler control at the intended operating point.

Start with the workbook’s first read-only prompt, then one approved foundation change. Progress through fresh fixtures, data and evaluation contracts, native configuration and measured fit before budgeted training. Oracle and lexical evidence controls precede fusion, learned routing and offload. Resume from artifact receipts and the execution record, not an assistant’s recollection.

## Use Codex in WSL2 with durable project context

Use Codex CLI from the WSL2 distribution that holds the project’s Python and GPU environment. Keep the checkout in the Linux filesystem, for example ~/code/tiny-sparse-lab, rather than under /mnt/c. Codex running successfully does not establish that the Radeon, driver, ROCm and framework are ready for training. Inventory those separately before an approved smoke test. [21,22]

### Keep the theory and record in files

The supplementary ZIP contains faithful Markdown versions of this plan and the complete workbook, plus a short bootstrap and state templates. Its proposed home is experiments/research/kernel-memory-lab. No folder has been installed or committed. After adoption, keep one versioned operating record there: RESEARCH_PLAN.md for theory, EXECUTION_WORKBOOK.md for the thirteen cards, STATUS.md for the current state, and append-only result and decision entries for evidence.

This delivery’s DOCX and Markdown are synchronized at revision 2026-10-07.1. The DOCX files remain editable human references to that revision. Do not maintain a second live backlog inside them. Reconcile any later human DOCX edits into the reviewed Markdown record, record the revision, and refresh both references before treating the change as accepted. Native Campaign receipts and actual artifacts establish execution facts; the written record indexes them.

### Start every session from verified state

Preserve the repository’s existing AGENTS.md. The pack provides a proposed short pointer for review, not a replacement or installed instruction file. Start at the repository root and explicitly tell Codex to read the pack’s BOOTSTRAP.md, context, current status and selected card. Instructions in a deeper directory are not automatically loaded merely because a prompt mentions files there. Keep AGENTS guidance short and restart a session after instruction-file changes. [23]

Use the workbook’s first read-only launch to compare the actual checkout with the audit, inspect the current native campaign state and verify hashes before proposing one next step. Read-only campaign recommendations are not permission to apply them. A resumed chat is convenient, but a fresh chat must be able to reconstruct the same state from files and receipts. [21]

### Review one bounded task at a time

Luna can handle clear operational cards. Ask Sol to review loader, optimizer, gradient, numerical or device issues using a minimal failing case. Ask Astra when validated controls leave competing research explanations or a design decision. Neither reviewer grants permission to spend, train or enlarge scope. Close each card with exact artifact hashes, tests, failed attempts, limits and one next decision. The workbook and ZIP include copy-ready prompts.

The Windows desktop app can also place its agent in WSL through Settings → Agent environment → Windows Subsystem for Linux, followed by a restart. The integrated terminal shell setting is separate; changing only that shell does not move the agent into WSL. This is an optional route, not a setup change performed by these documents. [24]

## Sources and evidence boundaries

Primary papers support the research rationale. The repository was freshly inspected on 7 October 2026 at the revision below; official Codex guidance was checked the same day. Architecture choices, thresholds and budgets remain proposals, and no source establishes the complete system on Byron’s hardware. Paper links and literature claims retain the original 6 October research boundary.

[1] Owner-supplied architecture discussion (private, not redistributed) The attached Copilot discussion is a hypothesis source, not proof; project facts also come from the audit.

[2] [TinySparseLab audited revision](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/README.md) 7 October audit at 06efc4db82ec. Direct component and contract references: [inspection](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/model/inspection.py) · [transformer](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/model/transformer.py) · [attention](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/model/attention/dense.py) · [semantic memory](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/engram/semantic.py) · [training engine](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/engines/pytorch.py) · [configuration](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/config/models.py) · [optimizer](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/training/optimizer.py) · [checkpoints](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/checkpointing.md) · [datasets](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/datasets.md) · [campaigns](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/campaigns.md) · [iteration](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/iteration.md) · [local API](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/local-api.md) · [instruction training](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/instruction-training.md) · [agent instructions](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/AGENTS.md) · [state digest](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/src/sparselab/training/state_digest.py) · [semantic probes](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/research/semantic-memory.md) · [hosted environments](https://github.com/supernovae/tiny-sparse-lab/blob/06efc4db82ecf3da97b50cff518cba605ad27b33/docs/hosted-environments.md)

[3] [Large Memory Layers with Product Keys](https://arxiv.org/abs/1907.05242) PKM structured keys and trainable values

[4] [Memory Layers at Scale](https://arxiv.org/abs/2412.09764) Sparse learned memory at larger scale

[5] [Conditional Memory via Scalable Lookup](https://arxiv.org/html/2601.07372v1) Engram; deterministic lookup and conditional offload results

[6] [Generalization through Memorization](https://arxiv.org/abs/1911.00172) kNN language models; datastore interpolation

[7] [Improving language models by retrieving from trillions of tokens](https://arxiv.org/html/2112.04426v3) RETRO; reader integration and frozen-backbone retrofit

[8] [Retrieval Augmented Generation for Knowledge Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) RAG

[9] [REALM Retrieval Augmented Language Model Pre Training](https://arxiv.org/abs/2002.08909) Learned retrieval during pretraining

[10] [Atlas Few shot Learning with Retrieval Augmented Language Models](https://www.jmlr.org/papers/v24/23-0037.html) Retrieval and reader training

[11] [Scaling Embedding Layers in Language Models](https://arxiv.org/html/2502.01637v1) SCONE; joint training, precomputed features, measured host and NVMe costs

[12] [SmolLM2 technical report](https://arxiv.org/abs/2502.02737) Small-model data and training-scale reference

[13] [Memory Grafting](https://arxiv.org/html/2605.20948v1) Offline conditional memory with learned recipient integration

[14] [Training Compute Optimal Large Language Models](https://arxiv.org/abs/2203.15556) Chinchilla; compute and data scaling reference, not a quality guarantee

[15] [Tokenizer Agnostic Engram Module](https://arxiv.org/html/2607.29065v1) Address compatibility and representation compatibility are distinct

[16] [REPLUG Retrieval Augmented Black Box Language Models](https://arxiv.org/html/2301.12652v3) Retriever adaptation around an already capable frozen reader

[17] [sparse memory lm author repository](https://github.com/re133/sparse-memory-lm) Reported small-core/large-table results and their experimental limitations

[18] [MiniRAG](https://aclanthology.org/2026.acl-long.1721/) Prior work on chunk/entity graph retrieval for smaller models

[19] [RAFT Adapting Language Model to Domain Specific RAG](https://arxiv.org/abs/2403.10131) Training with evidence and distractors

[20] [RA DIT Retrieval Augmented Dual Instruction Tuning](https://arxiv.org/abs/2310.01352) Reader and retriever adaptation

[21] [Codex CLI commands and resume](https://learn.chatgpt.com/docs/developer-commands?surface=cli)

[22] [Codex in WSL](https://learn.chatgpt.com/docs/windows/wsl)

[23] [Codex AGENTS instruction discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md)

[24] [Windows desktop agent environment](https://learn.chatgpt.com/docs/windows/windows-app)
