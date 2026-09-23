# Using Tiny Sparse Lab

SparseLab is a small-model experimentation workbench. Train causal models, chat with verified checkpoints, compare versioned task capabilities, and retain evidence across architectural changes and scales. Each experiment stays on one host/device; a controller can schedule whole independent experiments on local or SSH workers. It does not provide a hosted service or distributed training.

If the example tasks or terminology are unfamiliar, start with [From flashcards to a useful local assistant](from-toy-to-useful.md). It explains the invented aliases, offers a runnable instruction learner, and shows how data, prompts, evaluation, and scale work together.

## Prepare, inspect, and train

```sh
uv sync --locked --dev
uv run sparselab tokenizer train configs/tokenizer_smoke.yaml
uv run sparselab data prepare configs/smoke_cpu.yaml
uv run sparselab inspect configs/smoke_combined_cpu.yaml --json
uv run sparselab train configs/smoke_combined_cpu.yaml --run-id combined-smoke
uv run sparselab eval combined-smoke
uv run sparselab generate combined-smoke --prompt "Once upon a time" --max-new-tokens 24
```

`inspect` reports a shape-only architecture/parameter inventory, conservative memory estimates, runtime information, and recommendations without constructing the model. `train` writes run-local immutable inputs and checkpoint generations. `eval` measures next-token loss over valid supervised targets in the **run-owned** validation data and saves the exact checkpoint identity. `generate` prints prompt plus continuation (greedy by default). A smoke run proves execution, not useful language ability; the [capability workflow](capabilities.md) defines narrow measured tasks.

## Chat with a saved run

```sh
uv run sparselab chat combined-smoke --max-new-tokens 12
uv run sparselab chat chat-engram --checkpoint best.json --message "What value belongs to the alias amber?" --system "Answer the requested alias with only its value." --max-new-tokens 12 --json
uv run sparselab chat chat-engram --temperature 0.6 --top-k 20 --seed 42 --transcript /tmp/conversation.json
```

Commands accepting `--runs-dir` default to `runs/` beside the nearest `pyproject.toml` found by walking upward from the current directory. Chat, evaluation, generation, and other run consumers therefore work from repository subdirectories such as `src/`. Outside a project, the default is `./runs`; the installed package location is never used as a data root. An explicit `--runs-dir` is used as supplied, with relative paths anchored to the current directory and no fallback search. For a custom or relocated run store, pass `--runs-dir /absolute/path/to/runs`.

Interactive chat accepts one turn at a time; `/exit` or `/quit` finishes, `/reset` clears context. `--message` performs one scripted turn; `--json` emits structured responses. `--transcript` saves actual model prompts/replies, dropped-turn counts, generation settings, and a frozen checkpoint digest; existing files are never overwritten. Sampling is opt-in and locally seeded. EOS and generated role boundaries stop the assistant turn.

Chat reserves the requested response budget within the model context. It drops only complete oldest user/assistant turns; the system and current user message are never silently truncated. Oversized current turns fail with an actionable error. The default response allowance is 16 tokens; adjust it to the task and available context.

Inference resolves `latest.json` or `best.json` once, verifies the selected generation and run artifacts, and uses the run-owned tokenizer/package. Moving the run directory or removing the original training cache does not change the model's vocabulary. `--checkpoint` can select a generation explicitly for chat, generation, evaluation, and capabilities. Chat supports the implemented PyTorch architectures and supported native MLX configurations; MLX execution requires its optional runtime and uses full-prefix decoding. A base model still needs chat-oriented training to follow these transcripts.

## Choose a mechanism deliberately

- `model.ffn: dense|moe` selects dense SwiGLU or local Top-K MoE.
- `model.memory: none|ngram|byte|portable` selects no memory, token n-gram memory, prepared raw-UTF-8 byte memory, or a portable frozen-table adapter.
- `attention.kind: dense|sliding_window|block_sparse|mla` selects full causal attention, a causal window, block-selected keys, or multi-head latent attention.

Each configuration is concrete. Do not infer a result from labels or parameter count alone; compare completed runs only when source, tokenizer, device, sequence length, token budget, optimizer, and seed match. See [architecture](architecture.md), [training](training.md), and [experiments](experiments.md).

These architecture choices describe the PyTorch engine. MLX supports dense feed-forward blocks with dense or native block-sparse attention, FP32 AdamW, and optional block recomputation; it rejects the other architecture combinations rather than silently replacing them. See [runtime policy](runtime.md).

Local conversations support explicit v2 `all_tokens` or `assistant_only` supervision and validated inert tool-call transcripts. Historical unversioned records retain whole-transcript loss. See [conversation format and objectives](instruction-training.md#local-conversation-corpora); storing a tool transcript does not execute the tool.

## Inspect and verify a checkpoint

```sh
uv run sparselab checkpoint inspect runs/combined-smoke/checkpoints/latest.json --json
uv run sparselab checkpoint verify runs/combined-smoke/checkpoints/latest.json --json
```

PyTorch and MLX share immutable generations, run-owned inference assets, checkpoint-bound evaluation, recovery, and promotion. Native MLX execution requires the optional pinned runtime; offline checkpoint inspection/verification does not. Supported PyTorch generation uses a bounded request-local KV cache; the Python `generate(..., use_cache=False)` API provides the full-prefix reference. MLX and unsupported cache configurations use full-prefix decoding. These execution checks are not model-quality evidence.

## Stage and schedule independent experiments

```sh
uv run sparselab stage configs/runtime_smoke_cpu.yaml --through warmup --output /tmp/sparselab-guide-stage
uv run sparselab run configs/runtime_smoke_cpu.yaml --store /tmp/sparselab-guide-controller
uv run sparselab experiment list --json --store /tmp/sparselab-guide-controller
uv run sparselab experiment submit --matrix tests/fixtures/runtime-matrix.yaml \
  --dry-run --store /tmp/sparselab-guide-controller
```

The standalone stage command produces isolated pilot evidence; it does not initialize a later experiment from pilot weights. Direct `train` never silently runs pilots. Composed `run` prepares and dispatches through the same worker queue, including worker-side validation/pilots, then waits for terminal ingestion. Without `--worker`, it registers a local endpoint; an existing controller may drive the store while the command waits.

Use a fresh stage directory, or reuse only an identical verified bundle. Matrix dry-run expands all three coordinates without preparation or enqueueing. Actual submission seals every coordinate before the queue transaction. `experiment cancel RUN_ID` requests safe-boundary cancellation; `experiment resume RUN_ID` creates an explicit child from verified local full state. Neither a controller disconnect nor a dead executor authorizes automatic optimizer restart. See [worker operation](workers.md) for registration, foreground controllers, leases, transfer deadlines, and provisioned SSH targets.

## Verify withheld-fact fixture evidence

```sh
uv run sparselab facts manifest --seed 0 --output artifacts/withheld-facts-seed-0.json
uv run sparselab facts audit artifacts/withheld-facts-seed-0.json
```

The audit proves the deterministic fixture’s data separation only. It is not a model score or transfer result. Use `sparselab facts evaluate RUN_ID MANIFEST` for retained completions from a real checkpoint, and `facts transfer-evaluate SOURCE_RUN TARGET_RUN MANIFEST` only for the explicit byte-memory adapter-transfer boundary. Read [withheld facts](withheld-facts.md) before drawing conclusions.

## View recorded local runs

```sh
uv run sparselab dashboard --runs-dir runs
```

The read-only, localhost-only viewer includes Overview, Training, Evaluation, Architecture, Runtime, Memory, Checkpoints, Stages, and searchable Learn pages. Session-scoped refresh preserves selections and marks stale reads. Runtime shows actual worker/backend/precision/optimizer conditions; memory distinguishes native peaks from sampled lower bounds; checkpoint views separate local best from inherited lineage.

For worker results, use `--runs-dir /tmp/sparselab-guide-controller`. The controller imports telemetry and verified files into that local projection; the dashboard neither schedules training nor mounts a remote database. Compare recorded conditions and observed budgets, not worker labels or normalized curves. See the [single-host](../artifacts/acceptance/single_host_gate_2026_09_22.json) and [worker](../artifacts/acceptance/independent_workers_2026_09_23.json) acceptance records for exercised behavior and verification limits.
