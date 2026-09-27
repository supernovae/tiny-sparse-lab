# Implementation backlog

This file tracks only repository changes that require code. Scientific questions,
campaign status, and proposed experiments belong in
[`docs/research/roadmap.md`](docs/research/roadmap.md), the versioned research
lifecycle, or a checked-in [`experiments/research/`](experiments/research/)
record. Completed work remains discoverable in Git history and evidence records;
it is not retained here as a second changelog.

When a task moves to GitHub, use the **Code task** issue template and replace the
item below with its issue link. A code item is complete only when its tests and
documentation land. An experiment result, including a negative result, does not
close a code item unless the named software acceptance criteria also pass.

## Runtime forecasting and progress

- [ ] Replace preprocessing heartbeat-only events with a common, versioned
  progress record containing phase, completed/total work when known, elapsed
  time, last meaningful progress, and raw counters. Preserve JSON Lines on
  stderr and do not change scientific inputs or stdout result payloads.
- [ ] Add phase-aware planning, warmup-calibrated, live, and final-observed
  timing records. Keep estimates separate from observations, represent missing
  telemetry as unavailable rather than zero, and persist prediction snapshots at
  a bounded cadence.
- [ ] Add training target progress, robust recent/long-window throughput, a
  non-negative live ETA range, and explicit unstable/stalled states. Keep
  optimizer-only time separate from validation, checkpoint, preparation,
  evaluation, and report overhead.
- [ ] Add a read-only runtime-estimate/status CLI JSON contract and surface the
  same records in the dashboard without making the dashboard a scheduler.
- [ ] Add explainable historical calibration that refuses incompatible backend,
  precision, architecture, optimizer, sequence-length, recomputation, or
  offload observations and identifies every contributing run.

The bounded implementation brief is
[`docs/prompts/runtime-forecasting-and-throughput.md`](docs/prompts/runtime-forecasting-and-throughput.md).

## Throughput and resource proposals

- [ ] Extend disposable staging to benchmark safe microbatch/accumulation
  candidates that preserve the declared effective batch, rank candidates by
  measured target throughput and headroom, and write a separate proposal rather
  than mutating the requested config.
- [ ] Record why a candidate was selected or rejected, including OOM, unsupported
  precision, excessive memory pressure, unstable timing, and insufficient
  observations. Initialization/transient steps must not dominate the result.
- [ ] Add optional bottleneck observations that distinguish accelerator-bound,
  input/host-bound, memory-pressure, and unknown cases. Low CPU or less than
  100% device utilization is diagnostic evidence, not itself a failure.

## Workspace reliability

- [ ] Add a named-workspace preflight that reports filesystem capacity and inode
  headroom before long preparation or training operations, includes projected
  checkpoint/cache growth where available, and fails before partial publication
  when the configured work area is clearly insufficient.
- [ ] Ensure every remaining temporary-file path honors `--work-dir` or
  `SPARSELAB_WORK_DIR`; add regression coverage for subprocesses and external
  tool fallbacks. Long-running workflows must not depend on the platform's
  anonymous `/tmp` capacity.
- [ ] Add bounded retention/cleanup proposals for campaign-owned checkpoints and
  caches. Never delete unowned paths or required registered checkpoints, and
  keep cleanup a separate explicit action.

## Experiment ergonomics

- [ ] Make a scaffold capable of targeting the documented
  `experiments/samples/<name>` or `experiments/research/<campaign>` layouts while
  placing mutable assets under a named `sparselab-work/experiments/<campaign>`
  root.
- [ ] Emit a copyable command transcript and source/output path map in scaffolded
  experiment README files so another user can reproduce the plan without
  inheriting local absolute paths.
- [ ] Validate that checked-in research records bind protocol/config identities
  and evidence references while excluding checkpoints, caches, datasets, logs,
  and other mutable run output.

## Backend implementation

- [ ] Implement native CUDA sparse attention, then add hardware-gated correctness
  and component benchmarks. CPU or another accelerator cannot close this item.
