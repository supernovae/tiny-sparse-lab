# SparseLab agent guidance

Use this file for coding-agent behavior. Scientific truth comes from versioned
configs, run artifacts, evidence records, and reviewed research decisions—not
from an agent's confidence or a passing smoke test.

## Start safely

1. Before substantial changes, read the relevant project guidance and nearby
   documentation. Use `README.md` for project boundaries, `TODO.md` for code
   work, and `docs/research/` for scientific work. Read only what the task needs.
2. Inspect `git status` and relevant history before editing. Preserve user
   changes and active run directories. Do not edit configs, code, checkpoints,
   or manifests underneath an experiment that is currently running.
3. For substantial work, identify whether it changes code, a sample experiment,
   a research experiment, or evidence. Do not quietly turn one into another.

## Python environment

- This project uses uv. Run Python, tests, and project commands through the locked
  environment: `uv run --locked python ...`, `uv run --locked pytest ...`, and
  `uv run --locked sparselab ...`.
- Do not assume `python`, `pytest`, or `sparselab` is available directly on
  `PATH`. Do not create another virtual environment or run `pip install` unless
  the user explicitly requests it.
- For CUDA, ROCm, XPU, or MLX work, follow the documented worker environment
  instead of replacing its framework packages with the default locked CPU stack.
  Use `uv run --locked --no-sync ...` for an already provisioned vendor environment.
  Treat host OS/environment (Linux, macOS, WSL2) independently from backend/device;
  do not infer a GPU vendor or capability from the host environment.

## Workspaces and storage

- Use a descriptive, task-owned directory such as
  `sparselab-work/runtime-forecasting/` or set `SPARSELAB_WORK_DIR` to a named
  directory on a filesystem with enough capacity. Pass `--work-dir` when the CLI
  supports it.
- Name workspaces for the task or experiment, not the CPU/GPU backend. Keep
  backend/device choices in runtime parameters and run metadata, with distinct
  run IDs sharing the task's `runs/` store.
- Do not use anonymous `/tmp` paths for long preparation, training, checkpoints,
  downloads, or campaign output. OS temp space may be small even when the project
  filesystem is large. Python and child-process temp paths are redirected after
  SparseLab initializes its work directory, but external commands launched
  before that still need an explicit work path.
- Before expensive work, check free bytes and inodes for the actual workspace and
  estimate checkpoint/cache growth. Stop before launch if the safe margin is
  inadequate. Never delete or prune data that the current task does not own.
- Keep mutable or large outputs under `sparselab-work/`, which is ignored. Check
  in only small protocols, configs, summaries, and evidence references that are
  intended to be durable and reviewable.

## Configuration and performance

- Inspect the effective config before running it. Use `sparselab inspect`, then a
  disposable `sparselab stage --through smoke` or `--through warmup` when the
  backend is available. A memory estimate is not proof of fit and a warmup is not
  a scientific result.
- Treat effective batch (`micro_batch_size * gradient_accumulation`) as a
  scientific setting. When optimizing execution, first test divisible
  microbatch/accumulation pairs that preserve it. Do not silently change sequence
  length, optimizer, precision, token budget, seed, data, or architecture.
- Prefer measured proposals over guesses. For each bounded candidate, record
  target tokens/sec, step-time distribution after initialization, peak memory,
  backend/device identity, and failures. Select the fastest stable candidate with
  adequate headroom; write a new config/proposal instead of rewriting the source.
- Do not chase a cosmetic 100% utilization number. Low CPU use can be normal for
  accelerator-bound work; low accelerator use can reflect input stalls, tiny
  kernels, synchronization, evaluation, checkpoint I/O, or memory pressure.
  Diagnose the limiting phase before changing batch size or worker counts.
- Keep preparation, optimizer updates, validation, checkpointing, evaluation,
  generation, and reporting timings separate. Never present component timing as
  end-to-end time or performance as model quality.

## Experiments and evidence

- Copyable teaching material belongs in `experiments/samples/`. A real scientific
  campaign belongs in `experiments/research/<campaign>/` and must bind its source
  configs/protocol, acceptance gates, and evidence references. Mutable execution
  output belongs in `sparselab-work/experiments/<campaign>/`.
- Register scientific questions and next tests in the research lifecycle or
  `docs/research/roadmap.md`; keep `TODO.md` for missing or defective code only.
- Preserve failed, negative, censored, and interrupted runs. A completed command
  proves execution; it does not prove usefulness, portability, causality, or
  superiority. Report what changed, what stayed fixed, and what remains unproven.
- Never alter acceptance criteria after viewing final outcomes without recording
  a new protocol identity. Do not promote a baseline or finding automatically
  from metrics; promotion is a reviewed decision.

## Collaboration

- Keep small, local tasks with one agent. For substantial independent work, use
  parallel helpers when the benefit justifies their context and coordination cost.
- Give helpers narrow objectives and relevant paths and invariants. Ask for
  concise findings with evidence, changed paths, tests, and uncertainties.
- Split writable work by owned files or components; use one writer per file and
  review the integrated diff. Do not launch overlapping accelerator jobs on the
  same physical device unless the experiment studies concurrency and the worker
  lease path is in use.
- Let the client and user choose models, reasoning effort, and concurrency.
- Keep commits narrow and descriptive. Do not push, rewrite history, or delete
  remote branches unless the user explicitly asks.

## Verification

- Run focused tests for changed behavior first, then the relevant broader suite.
  Use the locked environment (`uv run --locked ...`) and the repository's Ruff
  configuration.
- Avoid wall-clock-sensitive assertions. Test raw counters, state transitions,
  schemas, deterministic calculations, failure behavior, and null/unavailable
  handling.
- Before handoff, review `git diff --check`, `git status`, generated paths, and
  documentation examples. State what was tested and identify every hardware or
  long-run gate that was not exercised.
