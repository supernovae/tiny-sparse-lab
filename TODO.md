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

- [ ] [Runtime forecasting and progress (#3)](https://github.com/supernovae/tiny-sparse-lab/issues/3).
  [Implementation brief](docs/prompts/runtime-forecasting-and-throughput.md).

## Throughput and resource proposals

- [ ] Add optional bottleneck observations that distinguish accelerator-bound,
  input/host-bound, memory-pressure, and unknown cases. Low CPU or less than
  100% device utilization is diagnostic evidence, not itself a failure.

## Experiment ergonomics

- [ ] Validate that checked-in research records bind protocol/config identities
  and evidence references while excluding checkpoints, caches, datasets, logs,
  and other mutable run output.

## Backend implementation

- [ ] Implement native CUDA sparse attention, then add hardware-gated correctness
  and component benchmarks. CPU or another accelerator cannot close this item.
- [ ] Keep single-device execution interfaces compatible with a future explicit
  distributed training design. Define optimizer ownership, data partitioning,
  checkpoint identity, and MoE expert placement before adding multi-device
  backward or sharding; independent workers are not a substitute.

## Spot-instance recovery

- [ ] Add a separately configured spot-safety policy using observed checkpoint
  write time, expected interruption notice, workspace capacity, and measured
  restart cost. Keep the scientific config fixed and record the chosen cadence.
- [ ] Validate abrupt worker loss, verified-generation selection, explicit child
  resume, remote artifact transfer, and storage retention on a real spot-like
  executor before claiming cost or recovery benefits.
