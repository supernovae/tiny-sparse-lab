# Using Tiny Sparse Lab

SparseLab is a local small-model experimentation workbench. Train causal models, chat with verified checkpoints, compare versioned task capabilities, and retain evidence across architectural changes and scales. It does not provide a hosted service or distributed training.

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

`inspect` reports configured architecture and parameter counts. `train` writes run-local artifacts and checkpoints. `eval` measures next-token loss on the **run-owned** validation data and saves the exact checkpoint identity. `generate` prints prompt plus continuation (greedy by default). A smoke run proves wiring, not useful language ability; the [chat-native capability pair](capabilities.md) targets a measured narrow learned task.

## Chat with a saved run

```sh
uv run sparselab chat combined-smoke --max-new-tokens 12
uv run sparselab chat chat-engram --checkpoint best.json --message "What value belongs to the alias amber?" --system "Answer the requested alias with only its value." --max-new-tokens 12 --json
uv run sparselab chat chat-engram --temperature 0.6 --top-k 20 --seed 42 --transcript /tmp/conversation.json
```

Commands accepting `--runs-dir` default to `runs/` beside the nearest `pyproject.toml` found by walking upward from the current directory. Chat, evaluation, generation, and other run consumers therefore work from repository subdirectories such as `src/`. Outside a project, the default is `./runs`; the installed package location is never used as a data root. An explicit `--runs-dir` is used as supplied, with relative paths anchored to the current directory and no fallback search. For a custom or relocated run store, pass `--runs-dir /absolute/path/to/runs`.

Interactive chat accepts one turn at a time; `/exit` or `/quit` finishes, `/reset` clears context. `--message` performs one scripted turn; `--json` emits structured responses. `--transcript` saves actual model prompts/replies, dropped-turn counts, generation settings, and a frozen checkpoint digest; existing files are never overwritten. Sampling is opt-in and locally seeded. EOS and generated role boundaries stop the assistant turn.

Chat reserves the requested response budget within the model context. It drops only complete oldest user/assistant turns; the system and current user message are never silently truncated. Oversized current turns fail with an actionable error. The default response allowance is 16 tokens; adjust it to the task and available context.

Inference resolves `latest.json` or `best.json` once, verifies the selected generation and run artifacts, and uses the run-owned tokenizer/package. Moving the run directory or removing the original training cache does not change the model's vocabulary. `--checkpoint` can select a generation explicitly for chat, generation, evaluation, and capabilities. Current chat supports PyTorch architectures that fit one host, not native MLX checkpoints. A base model still needs chat-oriented training to follow these transcripts.

## Choose a mechanism deliberately

- `model.ffn: dense|moe` selects dense SwiGLU or local Top-K MoE.
- `model.memory: none|ngram|byte|portable` selects no memory, token n-gram memory, prepared raw-UTF-8 byte memory, or a portable frozen-table adapter.
- `attention.kind: dense|sliding_window|block_sparse|mla` selects full causal attention, a causal window, block-selected keys, or multi-head latent attention.

Each configuration is concrete. Do not infer a result from labels or parameter count alone; compare completed runs only when source, tokenizer, device, sequence length, token budget, optimizer, and seed match. See [architecture](architecture.md), [training](training.md), and [experiments](experiments.md).

## Inspect and verify a checkpoint

```sh
uv run sparselab checkpoint inspect runs/combined-smoke/checkpoints/latest.json --json
uv run sparselab checkpoint verify runs/combined-smoke/checkpoints/latest.json --json
```

PyTorch and MLX share immutable generations, run-owned inference assets, checkpoint-bound evaluation, recovery, and promotion. Native MLX execution requires the optional pinned runtime; offline checkpoint inspection/verification does not. Supported PyTorch generation uses a bounded request-local KV cache; the Python `generate(..., use_cache=False)` API provides the full-prefix reference. MLX and unsupported cache configurations use full-prefix decoding. These execution checks are not model-quality evidence.

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

The read-only, localhost-only viewer shows run status, data source, architecture settings, metrics, events, lineage, and retained evaluation evidence. It never launches or modifies training.
