# Bounded Campaign engineering acceptance

On 2026-10-06, the native dataset and Campaign interfaces completed acquisition,
tokenizer fitting, preparation, locking, three selected training cells, ingestion,
heldout evaluation and descriptive generation. All 17 selected Campaign stages
finished `COMPLETE`; all three controller entries separately report execution
and ingestion `COMPLETE`, with no ingestion error. This establishes the bounded
teaching workflow, not model quality or full-corpus feasibility. Follow the
[walkthrough](../../../docs/tinystories-microlab.md) to create a new workspace.

## Provenance and retained evidence

The launch checkout was commit `fd5b4d501e84d98308faa2fb4e7ba484f57df5b3`.
Tracked source was clean; a pre-existing unrelated untracked research directory
was left untouched. Final pilots and training used package source identity
`1a41cd5276a7f4624b273b00aff10d4f57286597402c443d218c7edb751cb0e2`.
Worker manifests retain null Git commit/dirty fields because execution occurred
outside the checkout; the launch commit is a separate observation, not a
replacement for those fields. External declaration provenance likewise remains
`UNKNOWN`, explicitly admitted for these copied teaching declarations.

For this record, `$WORK` denotes the persistent root's
`experiments/dataset-contract-acceptance-20261006/` directory. Payloads are external
and are not shipped with this sample. The successful declaration is
`$WORK/campaign-phased.yaml`, referencing `inputs/plan-phased.yaml`; these names
preserve an earlier failed declaration separately. Their semantics match the
checked-in teaching Campaign and plan.

| Identity | SHA-256 |
| --- | --- |
| Campaign declaration | `b77744111465b51439ebba3534fb57390f1ef9cb2be6b8e1896e3636b06947bf` |
| ExperimentPlan lock | `8a5362601e82c3dc861d2009d6567463bc58af1d72aad1c47a5a99c0b1460db5` |
| Snapshot manifest | `d29cda007eaad1fefbefb1473a595ee7fefd9b09ba3633d7107d3dbe96e472e5` |
| Frozen tokenizer bytes | `0a68c68df31f793fb8d0bf215f2392d6ab6dd78b3abed636a1b5e6e3b52d2357` |
| Prepared manifest | `fac9d86904fa87556b972f40fa6af5a490bef26d1bd40bdd58716119a0bdc914` |

Campaign receipts live under `$WORK/campaigns/stories-learning/<declaration>/`.
Its `experiments/plan-stories-program/controller/` contains each run, checkpoints
and evaluation/panel indexes; `locks/<lock>.json` binds the resolved plan.
`$WORK/acceptance/` retains native final Campaign, controller, coverage, budget,
runtime doctor, evidence, triage and full-checkpoint verification JSON plus test
and orchestration logs. `stages/{cpu-final,mps-final}/` retains pilot bundles and
reports, and `readiness-reviewed/readiness.json` retains dense readiness.

## Inputs and coverage

The Hub source is `roneneldan/TinyStories`, default configuration, pinned to
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. Generic declarations acquired 2,000
training and 100 validation documents. Both splits stopped at their document
target, with `source_exhausted: false`; there were no duplicate, overlap, empty
or null exclusions in this selection. The 2,048-token vocabulary was fitted on
training documents only. Preparation retained every acquired document without
truncation.

| Split | Packed tokens | Complete 64-target blocks | Usable supervised targets | Dropped tail targets |
| --- | ---: | ---: | ---: | ---: |
| Train | 470,200 | 7,346 | 470,144 | 55 |
| Validation | 19,015 | 297 | 19,008 | 6 |

Each split also has one initial token without a target. Native coverage correctly
reports bounded, unproven full-source coverage. Native `data budget` with an
explicit bounded-source override produced `inputs/bounded-pass-final.yaml`:
one pass, 1,837 updates, 470,144 supervised targets, effective batch four and two
blocks in the final update. This proposal was not trained. Full-source coverage
is required by default; the teaching selection does not satisfy it.

## Runtime selection

The same tiny baseline configuration was staged on the available PyTorch CPU
and MPS devices on macOS arm64, using PyTorch 2.14.0 and Python 3.14.8. Both
completed two smoke updates and five warmup updates. Scientific settings were
fixed: FP32, sequence length 64, microbatch two, accumulation two, seed 42,
hidden width 128, two layers, four heads and FFN width 256.

| Execution device | Warmup targets/s | Update time after first update (four observations) | Sampled host peak | Sampled device peak |
| --- | ---: | --- | ---: | ---: |
| CPU | 23,959 | 7.73–8.33 ms | 471,187,456 bytes | unavailable |
| MPS | 8,149 | 16.71–27.72 ms | 613,072,896 bytes | 37,519,360 bytes |

These are native five-update pilot measurements, not end-to-end throughput.
Memory peaks are sampled lower bounds; host/device measurements are separate
and must not be added as independent allocations on unified memory. CPU was
selected for this small configuration. The observation does not rank CPU and
GPU capability generally. Physical device identifiers unavailable through the
runtime API remain null. Preflight reported adequate storage headroom.

## Completed iteration

| Cell | Run ID | Cumulative updates | Cumulative targets | Terminal heldout loss |
| --- | --- | ---: | ---: | ---: |
| Baseline | `17a51c67-51be-411e-b3f8-94917fff84b1` | 40 | 10,240 | 7.084726 |
| Exposure child | `16fb47e6-fc5a-4417-9a47-e1c0aabcaac4` | 80 | 20,480 | 6.993053 |
| Fresh wider FFN | `e7533dc2-0d2c-43dd-887c-eb09a6c62e2e` | 40 | 10,240 | 7.027953 |

The child resumed the baseline's verified full optimizer state and preserved the
40-update decay horizon. Its counters include the parent. The fresh contrast
changed FFN width from 256 to 384 at the baseline budget. The plan also declares
a wider child; this Campaign did not select or execute it. No seed replication
was performed.

Full native checkpoint verification succeeded for all selected terminal
checkpoints, including the unchanged baseline after continuation:

| Cell | Terminal checkpoint SHA-256 |
| --- | --- |
| Baseline | `b907b0c6ba660c987f8bf313289186961562acc6ffeb2c086d27acd99aa73480` |
| Exposure child | `1365103b5fc181ede4f892079397fd57c7eca318371ad6575ec225f05a5c38fe` |
| Fresh wider FFN | `ac98642172074185afc65363281e6094a571d126e086ea4796836af6c4c58380` |

Every endpoint used the same heldout suite (four batches, 512 valid targets) and
three-prompt descriptive panel (greedy generation, at most 32 new tokens, seed
42). This evaluation covers a small sample, not the entire validation snapshot.
Native evidence reports `checkpointed_held_out`, verified checkpoints and no
missing or rejected reports. Panel completion does not promote a model.

Native triage's cumulative optimizer seconds were 0.32149 for the baseline,
0.64217 for the child including its parent, and 0.34616 for the fresh contrast.
These counters exclude preparation, orchestration, evaluation and checkpoint
costs; they cannot be presented as total workflow time or summed as independent
runs. Raw triage limitations remain retained alongside separate evidence and
full-checkpoint verification reports.

## Recovery, verification and remaining limits

- Earlier bounded snapshot commands published valid outputs but hung during
  Arrow shutdown. The generic synchronous Parquet reader corrected process
  termination; a subsequent live acquisition exited successfully and reproduced
  the published bytes. Earlier outputs and attempts remain retained.
- The first Campaign declaration failed before training because its comparison
  named configuration sections rather than supported invariant leaves. The
  corrected declaration uses valid leaves and explicitly scopes the fresh-width
  comparison to pretraining. The original failed declaration and receipts remain
  under their original identity.
- The successful declaration's first apply returned while the baseline was
  running after its bounded controller wait, with an RPC warning retained in the
  log. Native resume reconciled the same attempt and completed the remaining
  stages without inventing a replacement baseline.
- Source acquisition tests passed (46); corrected serial integration regressions
  passed (59); final comparison/legacy-submission/authority tests passed (56).
  These scopes overlap. The earlier broad run had 623 passes, one skip and five
  failures; the failures were resolved and rerun in focused scopes. The complete
  broad command was not rerun. Ruff checks and final-source dense readiness
  passed. Offline Campaign coverage includes reconciliation, unchanged parent
  hashes and reapplication without redispatch.
- Repeated cold snapshot verification adds substantial orchestration time.
  [The implementation backlog](../../../TODO.md#native-diagnostic-interfaces)
  retains a typed verification-reuse task with mutation/invalidation gates.
- Full-source acquisition/training, large models, seed replication and execution
  on CUDA, ROCm, XPU or MLX were not exercised. These require their own runtime,
  storage and scientific gates. No model-quality or portability claim follows
  from this engineering acceptance.
