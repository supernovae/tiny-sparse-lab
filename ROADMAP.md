# Feature roadmap

This is a short, human-readable directional snapshot of pending features, not a
second checklist or a record of shipped capabilities. See the
[README feature matrix](README.md#feature-matrix) for available workflows.

## Where work is tracked

Notion owns actionable task status: [Mission Control](https://app.notion.com/p/3f63b68cca22814abcfdc5904e6cb1bd)
and the [Engineering Queue](https://app.notion.com/p/6234ef8e736e4d6187f74206d38ecc09)
are the **owner workspace and require access**. GitHub owns code, reviews and CI.
Public contributors can report gaps and propose work through
[GitHub issues](https://github.com/supernovae/tiny-sparse-lab/issues); Notion access
is not required to contribute. Maintainers reconcile issue/PR outcomes into
Notion rather than maintain task status in this file.

## Current direction

- **Trustworthy evaluation.** Evaluation-integrity work is in progress in
  [PR #72](https://github.com/supernovae/tiny-sparse-lab/pull/72), still open at
  this snapshot and not shipped on main. Its scope includes a held-back final
  split, text-level needle comparisons, recall diagnostics and honest verdicts.
  Suite-v4 reference resealing and across-seed uncertainty intervals remain
  follow-up work, beyond that PR.
- **Richer comparisons and inspection.** Broader reference trajectories,
  fixed-device latency and benchmark coverage, plus MLX and semantic-pack
  support in probes and the explorer.
- **A simpler, reliable lab loop.** Resolve known test failures, reduce CLI
  choices around the existing experiment journeys, consolidate result lookup
  and reports, and simplify configuration and legacy code while preserving
  evidence identities and useful regression coverage.
- **Stronger experiment workflows.** Support larger local comparisons,
  memory-offload/portability studies and agent iteration with protected final
  evaluation. Scientific readiness and conclusions remain in research records.

## Longer-term directions

Shared read-only experiment-ledger views, served-checkpoint provenance and
broader hardware/provider qualification remain conditional on concrete needs.
Native CUDA sparse attention, a real TPU backend and managed-provider lifecycle
support need their own implementation and measured validation. Distributed
training remains deferred under the
[single-host decision](docs/decisions/0015-single-host-extension-boundaries.md).

The distinct [scientific research roadmap](docs/research/roadmap.md) and
[original research records](experiments/research/) retain their evidence and
history. Historical mentions of the former `TODO.md` are preserved in those
records; its prior contents remain in Git history. This snapshot grants no
runtime, dataset acquisition, paid-compute or release authority.
