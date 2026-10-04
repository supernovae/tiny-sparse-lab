# Capacity-aware deterministic execution v1

Operational code work only. MODEL-0 science and retained artifacts are immutable;
no MODEL-1, training or distributed execution is part of this work.

## Historical baseline

[Compact baseline](../artifacts/benchmarks/capacity-aware-execution-v1-baseline.json)
binds exact JSON pointers into versioned
[execution records](../experiments/research/devmind-pretrain-v5/model0-execution.md),
[preparation](../experiments/research/devmind-pretrain-v5/model0-preparation-verification.json),
[lock acceptance](../experiments/research/devmind-pretrain-v5/model0-plan-lock-acceptance.json),
[current-source readiness](../experiments/research/devmind-pretrain-v5/model0-current-source-full-shape-readiness.json)
and [completed result](../experiments/research/devmind-pretrain-v5/model0-result.json).
These are retrospective operational evidence, not new scientific results.

| Boundary | Observed wall seconds | Interpretation |
| --- | ---: | --- |
| Source denominator | 29.325819 | Native encoding/scan; 1,469,599,744 B peak tree RSS |
| Preparation | 2,274.680755 | Inclusive; 8,834,408,448 B peak tree RSS |
| Lock / independent cold explain | 1,604.321869 / 1,538.799164 | Repeated full input closure authentication |
| Pilot input validation | 621.631 / 628.283 | Smoke/warmup inclusive authentication |
| Pilot materialization | 638.220 / 609.643 | Smoke/warmup inclusive materialization |
| Optimizer updates | 1,979.606588 | 5,525 updates; 22,863.533 targets/s |
| Training operation | 2,459.213342 | Excludes worker staging and Campaign reconciliation |
| Native heldout | 7.823494 | Not a demonstrated host bottleneck |
| Four training checkpoints | 13.698850 | Nested in training operation |
| Reconciliation censored windows | 25,550.210 / 21,601.291 | Separate interrupted windows, not successful elapsed |
| Final closure | 8,309.562 | Successful same-attempt closure |

Preparation records 4,081,965,445 logical input B, 4,031,007,060 logical
output B and 812,974 records. Native tokenizer encoding is 598.125071 s;
final fsync is 17.091629 s. These are nested counters, not disjoint phases.
The retained prepared root has four files (two arrays, manifest and ownership
marker), totaling 4,031,011,511 B. Per-file sizes and frozen array SHA-256
are in the baseline. CPU and physical I/O counters absent from the records
are null. Disk free delta is not physical write traffic. Historical preparation
swap is host-wide used swap; later monitored swap is owned process-tree swap.

## Call graph and data-flow boundaries

| Entry | Repeated downstream checks/copies | Identity authority | Proposed reuse boundary |
| --- | --- | --- | --- |
| `resolve_plan` / `open_lock` | `verify_artifact` for each input and corpus variant | Typed kind/version/identifier/domain SHA plus package source | Explicit trusted store; direct calls remain cold |
| Tokenizer domain verifier | Export, release, build, snapshots via provenance | Tokenizer bytes and exact export/release closure | Unchanged typed ancestor closure |
| Prepared domain verifier | `load_prepared_data`, array SHA and header validation | Manifest/settings/tokenizer/source SHA | Host-local signed proof; new process-local seal |
| `stage` / pilot `input_validation` | Tokenizer closure, stage inventory, prepared input checks | Stage inventories and prepared seals | Explicit mode passed into pilot child |
| Pilot `input_materialization` | Source arrays into private prepared and assets trees | Canonical SHA, source fingerprint, destination verification | Private copies/clones; private stage-only links |
| Dispatch install/materialize | Asset SHA cache install, cache rescan, private copy | Existing dispatch asset/bundle CAS SHA | TRANSFER cold; same-store signed reuse |
| Campaign / Controller reconciliation | Artifact checks, RPC status, receipt ingestion and collection | Durable attempt state and typed output identities | Unchanged local evidence only; terminal ingestion gating retained |
| Archive / recovery / family | Independent byte and semantic verification | Final durable evidence authority | Always COLD |

## Ranked measured opportunity, before tuning

1. Repeated typed closure authentication: 1,539–1,604 s lock boundaries and
   622–628 s pilot gate. Highest eligible operational reuse opportunity; byte
   hashing versus semantic parsing share is not yet isolated.
2. Private materialization: 610–638 s inclusive pilot phase, roughly 4 GB
   canonical arrays. Clone/copy mechanisms require measured destination safety.
3. Preparation native encoding: 598 s nested component. Already Rayon-backed;
   no Python parallel producer or larger thread default is justified yet.
4. Reconciliation: largest observed windows but censored and unattributed.
   Profile before claiming improvement; 0.1 s sleep does not explain hours.
5. Independent multi-file cold SHA: real prepared deep check 15.172 s; eligible
   only when comparable medium timings demonstrate benefit. Single-file SHA
   remains native sequential SHA-256.

Heldout (7.823 s), checkpoint writes (13.699 s), ISA-specific hashing, GPU
parallelism, source parsing parallelism and new package dependencies are
unmeasured/ineligible optimizations, not claimed gains.

## Trust modes

- **COLD**: independent full byte and domain-semantic verification. Default for
  direct Python APIs, external paths and final archival.
- **VERIFIED_REUSE**: unchanged bytes and typed upstream closure in the same
  authenticated, owner-only registered persistent store, using signed local
  receipts. Not protection against a compromised same-UID owner.
- **STRUCTURAL**: same-operation, process-owned sealed `VerifiedFile` or
  `_VerifiedArtifact`; fingerprints, inventory and headers still checked.
- **TRANSFER**: full SHA and semantic authentication once upon entry to a new
  worker/cache trust domain, before publication.

Operational hardware, selected workers, proof/key paths and trust modes never
enter scientific, prepared, artifact or plan digests. Relocation, missing or
foreign receipts, changed fingerprints/closure or implementation, unsafe paths
and signature failure require COLD fallback; failed cold verification is fatal.
