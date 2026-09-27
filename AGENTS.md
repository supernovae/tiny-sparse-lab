# SparseLab agent guidance

Use this file for coding-agent behavior. Scientific truth comes from versioned
configs, run artifacts, evidence records, and reviewed research decisions—not
from an agent's confidence or a passing smoke test.

## Start safely

1. Read `README.md`, `TODO.md`, and the documentation nearest the code you will
   change. For research work, also read `docs/research/README.md` and the relevant
   protocol or lifecycle record.
2. Inspect `git status` and recent commits. Preserve user changes and active run
   directories. Do not edit configs, code, checkpoints, or manifests underneath
   an experiment that is currently running.
3. State whether the task is a code change, sample experiment, research
   experiment, or evidence review. Do not quietly turn one category into another.

## Workspaces and storage

- Use a descriptive, task-owned directory such as
  `sparselab-work/runtime-forecasting/` or set `SPARSELAB_WORK_DIR` to a named
  directory on a filesystem with enough capacity. Pass `--work-dir` when the CLI
  supports it.
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

- Split work by owned files or read-only investigations. Use one writer per file
  at a time, communicate shared assumptions, and review the integrated diff.
- Give helper agents exact paths, invariants, and expected outputs. Ask them to
  return evidence and uncertainties, not just conclusions.
- Do not let parallel agents launch overlapping accelerator jobs on the same
  physical device unless the experiment explicitly studies concurrency and the
  worker lease path is in use.
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
