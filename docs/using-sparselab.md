# Using Tiny Sparse Lab

SparseLab is a small-model experimentation workbench. Train causal models, chat with verified checkpoints, compare versioned task capabilities, and retain evidence across architectural changes and scales. Each experiment stays on one host/device; a controller can schedule whole independent experiments on local or SSH workers. It does not provide a hosted service or distributed training.

If the example tasks or terminology are unfamiliar, start with [From flashcards to a useful local assistant](from-toy-to-useful.md). It explains the invented aliases, offers a runnable instruction learner, and shows how data, prompts, evaluation, and scale work together.

For an end-to-end story model, follow the [TinyStories microlab](tinystories-microlab.md).
For YAML plans and checkpoint chaining, use [training programs](experiment-programs.md).

## Prepare, inspect, and train

```sh
uv sync --locked --extra cpu --dev
export SPARSELAB_WORK_DIR=/data/sparselab
uv run --locked --extra cpu sparselab tokenizer train configs/tokenizer_smoke.yaml
uv run --locked --extra cpu sparselab data prepare configs/smoke_cpu.yaml
uv run --locked --extra cpu sparselab inspect configs/smoke_combined_cpu.yaml --json
uv run --locked --extra cpu sparselab train --runs-dir "$SPARSELAB_WORK_DIR/runs" configs/smoke_combined_cpu.yaml --run-id combined-smoke
uv run --locked --extra cpu sparselab eval combined-smoke
uv run --locked --extra cpu sparselab generate combined-smoke --prompt "Once upon a time" --max-new-tokens 24
```

`inspect` reports a shape-only architecture/parameter inventory, conservative memory estimates, runtime information, and recommendations without constructing the model. `train` writes run-local immutable inputs and checkpoint generations. `eval` measures next-token loss over valid supervised targets in the **run-owned** validation data and saves the exact checkpoint identity. `generate` prints prompt plus continuation (greedy by default). A smoke run proves execution, not useful language ability; the [capability workflow](capabilities.md) defines narrow measured tasks.

For remote datasets, Corpus Forge Hub sources, or the Pythia reference adapter,
configure a read-only account token as described in the
[Hugging Face access guide](huggingface-access.md). Local/offline fixtures need no token.

When changing code, run the nearest tests first; the [fast test feedback
guide](test-speed.md) gives local commands and explains the full PR gates.

## Native diagnostic interfaces

These commands expose operational or descriptive observations, not model-quality
gates or authorization to train:

```sh
uv run --locked --extra cpu sparselab semantic probe probe.yaml --json
uv run --locked --extra cpu sparselab memorization analyze overlap.yaml --json
uv run --locked --extra cpu sparselab data benchmark-preparation benchmark.yaml --json
```

`semantic probe` authenticates packs and runs one bounded CPU/PyTorch forward with
explicitly supplied vectors, encoder identities, masks and times. Initialized
backbones/adapters are labeled untrained; checkpoint backbones retain their native
generation identity, including declared restored allocation adapters. The
[semantic lesson](research/semantic-memory.md) scaffolds a ready `probe.yaml`.
No implicit text encoder, training callback or run store is created.

`memorization analyze` binds an exact UTF-8 file digest or an explicit completed
row from a natively authenticated generation panel. It compares only supplied
source passages and preserves raw hashes, normalized overlap and unavailable
bounded-edit values. It emits no eligibility threshold or memorization verdict;
see [the declaration and provenance contract](memorization.md).

`data benchmark-preparation` exclusively claims a declared absent workspace,
generates bounded offline data, trains a tiny tokenizer and prepares each case
in a fresh thread-bounded worker. It retains failures, memory-coverage labels and
exact per-size prepared-identity comparisons, without model training or automatic
batch recommendations. See [runtime observations](runtime.md).

`data prepare`, `stage` and Campaign `plan|status|next|explain|apply|resume` accept
`--observations-output PATH.json`. The parent must already exist; the destination
must be absent and outside scientific inventories. The optional envelope records
the complete handler separately from nested native phases, actual counter coverage,
null/unavailable measurements and operation-scoped snapshot verifier statistics.
Diagnostic publication failures warn without changing the workflow result.
See [phase-observation boundaries](capacity-aware-execution.md).

Generic snapshots reuse full authentication only within one preparation, staging
or Campaign operation; every invocation rechecks config bindings and inventory.
`--cold-verify` disables reuse on preparation, staging and Campaign
`plan|status|next|explain|apply|resume|approve`. Imports and publication remain cold;
see [snapshot verification lifetime](datasets.md#campaign-and-existing-input-reuse).

## Derive a validated config variant

Use `config derive` to author a new standalone v2 run configuration without
rewriting YAML by hand. Each `--set` value is strict JSON, so strings must be
quoted. The command publishes a new YAML file and adjacent operational
`.derivation.json` receipt; neither an existing payload nor receipt is
replaced. The receipt is not an artifact identity or trusted experiment lock.

```sh
mkdir -p "$SPARSELAB_WORK_DIR/handoffs"
uv run --locked --extra cpu sparselab config derive configs/smoke_cpu.yaml \
  --set optimizer.peak=0.001 --set training.micro_batch_size=2 \
  --output "$SPARSELAB_WORK_DIR/handoffs/smoke-peak.yaml" --json
```

The derived configuration preserves path targets relative to the source
configuration, validates the complete typed configuration, and records the
requested assignments plus observed field delta. It does not verify prepared
data, tokenizer bytes, checkpoints, continuation compatibility, or runtime
execution; use the normal preparation, experiment, and checkpoint commands for
those boundaries.

Paths supplied by an assignment, including a whole replacement object, are
anchored to the source YAML directory, never the output directory or cwd.
Optional default fields such as `optimizer.decay_steps` remain patchable.
Equal assignments are allowed; only actual normalized changes enter the
observed delta. The output parent must exist and have no symlinked components.
Changing a token budget or batch setting never adjusts other scientific
settings implicitly or turns a config into an approved continuation.

## Scratch and artifact locations

Implicit persistent state defaults to `${XDG_DATA_HOME}/sparselab` when `XDG_DATA_HOME` is absolute and nonempty, otherwise `~/.local/share/sparselab`. For substantial campaigns, set `SPARSELAB_WORK_DIR=/data/sparselab` on a sufficiently large filesystem. Global `--work-dir PATH` (before the subcommand) overrides that environment variable; an explicit relative path stays relative to the current directory. Durable `experiments/`, `runs/` and artifact/receipt stores share that root; temporary files go to its disposable `scratch/`, and optional `cache/` is reconstructable, not evidence. Read-only inspection does not create the root. Long-lived payloads inside **any** Git checkout trigger a containment warning but are not redirected; check explicit destinations too.

Explicit `--runs-dir`, `--store`, `--output`, `dataset.cache_dir` and `logging.root_dir` retain their literal paths, including historical receipts. Selecting an external root does **not** move an existing `sparselab-work/` directory or reinterpret prior data; intentional relocation requires a new location binding/manifest and re-verification of referenced bytes. Study submission and research scaffolding derive implicit workspaces from `<work-dir>/experiments/<study-name>`; SSH hosts need their own adequately sized persistent root.


For a replicated experiment, use one [experiment workspace](workspaces.md) and pass its persistent `runs/` directory to every producer and consumer. `train --runs-dir PATH` overrides an execution destination without editing the YAML; historical config paths remain unchanged.

## Declaration-to-archive commands

```sh
uv run --locked --extra cpu sparselab research snapshot <plan-or-campaign> --json
uv run --locked --extra cpu sparselab recovery inspect <recovery.yaml> --json
uv run --locked --extra cpu sparselab recovery plan <recovery.yaml> --json
uv run --locked --extra cpu sparselab recovery reconstruct <recovery.yaml> --json
uv run --locked --extra cpu sparselab research evidence export --kind corpus_release <verified-release> \
  --declaration <recovery.yaml> --output <small-evidence.json> --json
uv run --locked --extra cpu sparselab campaign status <campaign.yaml> --json
uv run --locked --extra cpu sparselab campaign reconstruct <campaign.yaml> --json
uv run --locked --extra cpu sparselab campaign apply <campaign.yaml> --execute-runs --json
uv run --locked --extra cpu sparselab evaluation suite run <suite.yaml> <run-id> \
  --checkpoint <generation> --runs-dir "$SPARSELAB_WORK_DIR/runs" --json
uv run --locked --extra cpu sparselab readiness model <policy.yaml> <evaluation-index.json> --json
uv run --locked --extra cpu sparselab readiness review <evaluation-index.json> --reviewer <id> \
  --decision approve --note '<reason>' --output <review.json>
uv run --locked --extra cpu sparselab family show <family.yaml> --json
uv run --locked --extra cpu sparselab family graph <family.yaml> --json
uv run --locked --extra cpu sparselab family compare <family.yaml> <node-a> <node-b> --json
uv run --locked --extra cpu sparselab family verify <family.yaml> --json
uv run --locked --extra cpu sparselab archive create <recovery.yaml> --mode thin --output <new.tar>
uv run --locked --extra cpu sparselab archive verify <new.tar> --json
```

`research evidence export` also accepts `tokenizer_selection`, `prepared_data`, `runtime_probe`, `experiment_lock`, `checkpoint` and `evaluation_index`. Recovery `inspect`/`plan` are read-only; `reconstruct` never trains. `campaign apply|resume` without `--execute-runs` cannot enqueue a new training run. Reviewed `family promote|reject|supersede` require `--readiness`, `--evaluation`, `--approval`, `--note`; supersede also requires `--successor`. See the [complete recovery and promotion protocol](research/lifecycle-recovery.md) for commit-before-compute, checksummed evidence, human decisions, missing checkpoint limits and archive rights.

## Chat with a saved run

```sh
uv run --locked --extra cpu sparselab chat combined-smoke --max-new-tokens 12
uv run --locked --extra cpu sparselab chat chat-engram --checkpoint best.json --message "What value belongs to the alias amber?" --system "Answer the requested alias with only its value." --max-new-tokens 12 --json
uv run --locked --extra cpu sparselab chat chat-engram --temperature 0.6 --top-k 20 --seed 42 --transcript sparselab-work/transcripts/conversation.json
```

Run readers without an explicit `--runs-dir` use the same selected persistent root's `runs/` directory. Explicit run stores remain as supplied, relative to the current directory; for a retained historical in-checkout store pass its old exact location. Study collection may infer a receipt's sibling `runs/` when its run-directory argument is omitted.

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

## Local serving and model debugging

Follow the [TinyText model guide](tinytext-model-guide.md) for a complete
locate → verify → raw completion → chat → checkpoint comparison workflow, and
[MODEL-0](model-0.md) for the retained DevMind base checkpoint's exact identity
and availability. `chat` and `generate` support `--show-prompt`, repeatable
`--stop`, `--context-length` and `--no-cache`; `generate --json --strict-context`
provides structured raw debugging without silent prompt cropping.

[`sparselab serve RUN_ID`](local-api.md) loads one verified checkpoint and exposes
nonstreaming OpenAI-compatible models, raw-completion and chat-completion routes
on loopback. Use the API guide for curl examples, bounded request semantics and
optional Open WebUI setup. Unsupported features fail explicitly.

## Inspect and verify a checkpoint

```sh
uv run --locked --extra cpu sparselab checkpoint inspect "$SPARSELAB_WORK_DIR/runs/combined-smoke/checkpoints/latest.json" --json
uv run --locked --extra cpu sparselab checkpoint verify "$SPARSELAB_WORK_DIR/runs/combined-smoke/checkpoints/latest.json" --json
```

PyTorch and MLX share immutable generations, run-owned inference assets, checkpoint-bound evaluation, recovery, and promotion. Native MLX execution requires the optional pinned runtime; offline checkpoint inspection/verification does not. Supported PyTorch generation uses a bounded request-local KV cache; the implementation retains a full-prefix reference for parity checks. MLX and unsupported cache configurations use full-prefix decoding. These execution checks are not model-quality evidence.

## Stage and schedule independent experiments

```sh
WORK="$SPARSELAB_WORK_DIR/experiments/runtime-smoke"
mkdir -p "$WORK"
uv run --locked --extra cpu sparselab stage configs/runtime_smoke_cpu.yaml --through warmup --output "$WORK/staging/guide-stage"
uv run --locked --extra cpu sparselab run configs/runtime_smoke_cpu.yaml --store "$WORK/runs"
uv run --locked --extra cpu sparselab experiment list --json --store "$WORK/runs"
uv run --locked --extra cpu sparselab experiment submit --matrix tests/fixtures/runtime-matrix.yaml \
  --dry-run --store "$WORK/runs"
```

The standalone stage command produces isolated pilot evidence; it does not initialize a later experiment from pilot weights. Direct `train` never silently runs pilots. Composed `run` prepares and dispatches through the same worker queue, including worker-side validation/pilots, then waits for terminal ingestion. Without `--worker`, it registers a local endpoint; an existing controller may drive the store while the command waits.

Use a fresh stage directory, or reuse only an identical verified bundle. Matrix dry-run expands all three coordinates without preparation or enqueueing. Actual submission seals every coordinate before the queue transaction. `experiment cancel RUN_ID` requests safe-boundary cancellation; `experiment resume RUN_ID` creates an explicit child from verified local full state. Neither a controller disconnect nor a dead executor authorizes automatic optimizer restart. See [worker operation](workers.md) for registration, foreground controllers, leases, transfer deadlines, and provisioned SSH targets.

## Verify withheld-fact fixture evidence

```sh
uv run --locked --extra cpu sparselab facts manifest --seed 0 --output artifacts/withheld-facts-seed-0.json
uv run --locked --extra cpu sparselab facts audit artifacts/withheld-facts-seed-0.json
```

The audit proves the deterministic fixture’s data separation only. It is not a model score or transfer result. Use `sparselab facts evaluate RUN_ID MANIFEST` for retained completions from a real checkpoint, and `facts transfer-evaluate SOURCE_RUN TARGET_RUN MANIFEST` only for the explicit byte-memory adapter-transfer boundary. Read [withheld facts](withheld-facts.md) before drawing conclusions.

## View recorded local runs

```sh
uv run --locked --extra cpu sparselab dashboard --runs-dir "$SPARSELAB_WORK_DIR/runs"
```

The read-only, localhost-only viewer includes Overview, Training, Evaluation, Architecture, Runtime, Memory, Checkpoints, Stages, and searchable Learn pages. Session-scoped refresh preserves selections and marks stale reads. Runtime shows actual worker/backend/precision/optimizer conditions; memory distinguishes native peaks from sampled lower bounds; checkpoint views separate local best from inherited lineage.

For worker results, use `--runs-dir "$WORK/runs"`. The controller imports telemetry and verified files into that local projection; the dashboard neither schedules training nor mounts a remote database. Compare recorded conditions and observed budgets, not worker labels or normalized curves. See the [single-host](../artifacts/acceptance/single_host_gate_2026_09_22.json) and [worker](../artifacts/acceptance/independent_workers_2026_09_23.json) acceptance records for exercised behavior and verification limits.
