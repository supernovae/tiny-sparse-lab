# Runtime and workflow acceptance

This page indexes bounded software and hardware acceptance. For feature
availability, use the [README matrix](../README.md#feature-matrix); for scientific
findings, use the [experiment ledger](research/experiment-ledger.md). Acceptance
records retain their tested source, hardware, commands and limitations. They are
not a rolling test count, model-quality result or guarantee on another host.

| Evidence | What was exercised |
| --- | --- |
| [Host CLI acceptance](../artifacts/acceptance/host_cli_2026_09_22.json) | CPU FP32/BF16, Adafactor continuation and native MLX checkpoint/resume checks. |
| [Single-host acceptance](../artifacts/acceptance/single_host_gate_2026_09_22.json) | CPU, MPS and MLX execution, safe interruption/recovery, promotion, offline installed-wheel operation, staging and populated dashboard checks. |
| [Independent-worker acceptance](../artifacts/acceptance/independent_workers_2026_09_23.json) | Overlapping logical CPU workers, controller loss, replay, cancellation, child resume, executor loss, matrices, source/capacity rejection and an MLX worker. |
| [Registered runtime acceptance](../artifacts/acceptance/runtime-contract-rocm-20261001T134938Z.json) | A ROCm BF16 Campaign on one RX 7900 XTX, completed ingestion, verified checkpoint, and explicit ROCm/CPU evaluations preserving training identity. |
| [Research report checks](research/sample-report.md) | Catalog scaffolding, collection and report surfaces; scientific observations retain their separate qualifications. |

The worker UI record includes rendered curve pixels and runtime-table checks
through the app harness; conventional browser screenshot capture was incomplete.
Keep that limitation with the observation.

See [runtime support](runtime.md) for current implementation and hardware
boundaries, [workers](workers.md) for operation, and [TODO.md](../TODO.md) for
missing software. New hardware and runtime revisions require their own acceptance.
