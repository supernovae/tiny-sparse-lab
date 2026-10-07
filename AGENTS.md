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

For Kernel Memory Lab tasks, explicitly read
[the scoped bootstrap](experiments/research/kernel-memory-lab/BOOTSTRAP.md) and
its current status before acting. Follow the selected card and current authorized
scope; the planning documents grant no runtime or spending approval.

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

- Select an external persistent root independent of checkout, branch, and
  worktree. The default is `${XDG_DATA_HOME}/sparselab` when XDG is absolute
  and nonempty, otherwise `~/.local/share/sparselab`; substantial campaigns
  should use `export SPARSELAB_WORK_DIR=/data/sparselab` on an adequately sized
  filesystem. Global `--work-dir` wins over the environment. A legacy relative
  `SPARSELAB_WORK_DIR=sparselab-work` is an explicit override, not the default;
  do not automatically migrate or delete it.
- Name task workspaces under the persistent root, such as
  `$SPARSELAB_WORK_DIR/experiments/runtime-forecasting/`, not for the CPU/GPU
  backend. Keep backend/device choices in runtime parameters and run metadata,
  with distinct run IDs sharing the task's `runs/` store.
- The root contains downloads, immutable evidence, prepared data, checkpoints,
  receipts and logs. Its `scratch/` contains disposable temporary files, and
  optional `cache/` holds only reconstructable caches. Neither scratch nor cache
  is scientific evidence. Do not use anonymous `/tmp` for long preparation,
  training, checkpoints, downloads, or campaign output.
- In-checkout overrides and configured long-lived output/cache paths there are
  respected but warn `STORAGE_INSIDE_GIT_CHECKOUT`. Treat branch, checkout and
  root paths as operational locations, not scientific identities; record source
  commit and declaration/content digests. Explicit output/cache destinations
  and old receipt paths retain their meaning. Relocation needs a new location
  binding and verification, not rewriting old receipts.
- Before expensive work, check free bytes and inodes for actual output locations
  and estimate checkpoint/cache growth. Stop before launch if the safe margin is
  inadequate. Never delete or prune data that the current task does not own.
- Check in only small protocols, configs, summaries, and evidence references
  intended to be durable and reviewable; keep mutable or large outputs in the
  external persistent root.

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

### Use the lab for rapid iteration

- Follow [the native iteration workflow](docs/iteration.md). Use the same native
  commands and declarations a human would use; select `--json` only where the
  command supports it. Do not write Python to reimplement artifact hashing,
  counter checks, checkpoint selection, runtime checks or Campaign state.
- At a new session boundary, use the bounded `readiness smoke --family dense`
  when lab wiring needs checking; select additional affected families for code
  changes. This runs tiny CPU training and resume, so use an isolated output.
  It does not certify an accelerator or the real experiment's inputs.
- Identify the actual delta first: declaration, data/tokenizer, code, runtime or
  location. Use nearest tests for changed code, actual-config `inspect` and
  `workspace preflight`, and config-specific staging where needed. A tested
  unchanged revision does not need the full suite before every model iteration.
- Prefer Campaign `status`, `next` and `explain` for declared workflows; these
  expose blockers and bound inputs without launching training. Use composed
  `run` for a single fresh teaching run and ExperimentPlan phases for checkpoint
  chains. An interrupted attempt needs reconciliation or explicit resume, not
  an invented fresh retry.
- Reuse authenticated unchanged inputs through supported CLI verification
  reuse. Use cold verification at a new trust boundary or when explicitly
  requested; inspect proof misses and fallbacks rather than trusting file size
  or remembered hashes. Family/archive/recovery checks retain their documented
  cold behavior. Do not repeatedly rebuild a frozen tokenizer or dataset.
- After a run, read native `evidence` and `triage`, verify the selected immutable
  checkpoint and inspect completed ingestion separately. Triage reads retained
  advice; it does not approve training, choose a new experiment or promote a
  model. Keep raw negative and unavailable observations visible.
- If a necessary check exists only as a Python API, record the exact missing
  CLI/DSL operation with input/output and failure acceptance criteria in
  `TODO.md`. Prefer a small typed adapter over a task-specific harness or a
  second orchestration engine. Do not present a proposed command as shipped.

- Copyable teaching material belongs in `experiments/samples/`. A real scientific
  campaign belongs in `experiments/research/<campaign>/` and must bind its source
  configs/protocol, acceptance gates, and evidence references. Mutable execution
  output belongs under the external persistent root's `experiments/<campaign>/`.
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
- Before a long remote or accelerator run, finish the lab code change, run its
  focused tests and local readiness smoke, commit it, and push the tested branch
  when the task authorizes publishing it. Record the commit, dirty-tree status,
  installed package/source identity, and effective config with the run. Launch
  from that fixed checkout; use another checkout for later lab development.
- A runner using an already tested code revision can perform the focused
  `sparselab readiness smoke` and config-specific `inspect`/`stage` gates at a
  new session boundary. Repeat broad tests when code changes or a required gate
  calls for them, rather than spending accelerator time on unrelated suites.
- Keep periodic, best, latest, and immediately previous verified checkpoints
  according to the declared config. Verify a finalized generation before a
  child resume, and never edit or prune an active run's files.

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
