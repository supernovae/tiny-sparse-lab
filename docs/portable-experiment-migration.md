# Portable experiment migration acceptance

The ownership boundary is declarative user recipes plus reusable Lab functions.
Recipes may live in any repository or local directory; Workshop is one recipe
collection, not a required service. Workshop experiment folders contain DSL,
configuration, docs/cards and artifact references, without Python launchers or
per-experiment Python tests. Lab owns generators, parsers, planning, execution
and generic Python tests with neutral fixtures. Internal runner coordination is
not part of the public reproduction contract.

Migration acceptance uses verified relocation, resolved dependencies and native
metadata-only dry-run validation outside both checkouts. It does not require or
authorize acquisition, training, hardware qualification or scientific claims.
Remove replaced personal science from the current Lab tree; Git history retains
provenance. Do not keep duplicate active copies under a historical label.

## Stage 1: local snapshot preparation

- [x] Distinct native `dataset_snapshot` artifact identity and existing typed
  preparation snapshot ports, with explicit version-2 consumption admission.
- [x] Metadata-only inspection; local native integrity/config verification;
  missing-input refusal without acquisition; public apply/resume integration.
- [x] Bounded local-fixture preparation and receipt replay without repeated work.

## Stage 2: bounded Hub acquisition foundation

- [x] Inventoried native source locks, bound output/private cache, version-3
  admission, whole-file HTTP reservations, typed Git/LFS content authentication.
- [x] Native snapshot journal, interruption/cancellation recovery and published
  artifact replay; loopback Parquet → tokenizer → pack integration.
- [x] Cumulative operational transfer/metadata/request/retry/scan/verification
  allowances and observed owned-storage stops. Unsupported hard guarantees and
  decoder options fail closed; see [the contract](preparation-recipes.md#bounded-hub-acquisition).
- [ ] Real pinned Hub delivery and decoder-resource qualification under a separate
  authorization. Loopback wiring is not TinyStories portability evidence.

## Current acceptance reconciliation (2026-10-18)

Reconciled against Lab `7e306a96bcc617468d2a06493dab4562bff6beb0`, Workshop
`0f1bbaec53540c7d2b6ed23d7306d352ad83a7c4`, and Harness
`b1c2efbd256eceeebdef7cf42b34165d01b8ccb6`. Lab PR172–180 are merged. An
unchecked real-data result is not automatically a missing native function.

| Boundary | Shipped native path | Remaining acceptance |
| --- | --- | --- |
| Author and relocate | Section validation, config derive, explicit recipe create/export; separate bindings and Git/content admission | Validate each migrated declaration at current pins; unsupported profiles must refuse visibly |
| Preparation | Snapshot consumption, bounded acquisition contracts, native journal/receipts and apply/resume | Real external acquisition and decoder-resource qualification require separate approval; loopback tests are not dataset proof |
| Prepared CPU workflow | Native inspect/resolve/publish, explicit admission/approval, apply/resume/report; tiny synthetic and snapshot-pair coverage | Concrete generated inputs, supported runtime and recipe acceptance; broader continuation/evaluation composition remains limited |
| Derivative declarations | Checkpoint-only native phase graph preview distinguishes supplied, produced and fresh training | Inherited config, payload/tokenizer compatibility, resolved budgets and executed evaluation are not established by preview |
| Evidence/reporting | Native checkpoint-bound collection and verified publication reports | Actual paired results, final evaluations and generation panels are unexecuted evidence gates, not authority to launch |
| Internal supervision | Explicit policy binding and same-owner generated-file handoff in Harness PR22/23 | No slot argv/environment amendment; fresh admission for real use; native accounting remains Lab’s responsibility |

[Current non-executing checks](portable-current-acceptance.md) exercise actual
migrated declarations from outside both repositories. They do not establish a
qualified installed runtime, artifact availability, accelerator support or quality.
No missing scientific decisions or hashes are invented to make an inventory pass.

## Historical retirement reconciliation and current scope

The inventory below records the earlier migration scope at the pins above.
The owner's 2026-10-18 three-family direction supersedes heritage porting and
personal per-candidate acceptance obligations: Workshop retains basic CPU,
TinyStories and current DevMind. Other families, including KML, are retired from
the active showcase; fresh rerun ideas belong in the coordinated backlog with
new identities and current frameworks. Remaining Lab files are historical
inventory, not active project assignments or prerequisites for the three families.
Generic neutral tests, existing operational controls and runtime holds remain.


- EXP1/LAB86, the native TinyStories microlab and data-rich 30M declarations,
  selected packaged recipes, DevMind personal corpus declarations and report
  bundles have preservation/disposition records and merged retirement changes:
  [EXP1](exp1-retirement.json), [TinyStories](tinystories-retirement.json),
  [DevMind v0](devmind-v0-retirement.json), [v1/v3/v4](devmind-v1-v3-v4-retirement.json),
  [v2](devmind-v2-retirement.json), [v5](devmind-v5-retirement.json),
  [reports](report-retirement.json), [research resources](research-resource-retirement.json),
  [historical guides](historical-guide-retirement.json), and
  [packaged samples](sample-disposition.json).
  Keep exact manifests rather than treating archive membership as blanket
  deletion permission.
- Generic corpus tests now use neutral fixtures; generic preparation argv tests
  cover all 17 native command shapes and cross-phase/nested/directory bindings.
  Context metadata tests use neutral fixtures. These changes preserve engine
  coverage but do not retire every remaining personal family.
- KML still has 224 tracked files under `experiments/research/kernel-memory-lab/`
  (`git ls-files experiments/research/kernel-memory-lab | wc -l` at the Lab pin
  above). This count excludes `src/sparselab/evaluation/kml_card03_items.py`,
  the six `tests/test_kml_*.py` modules, historical records under
  `experiments/research/history/kernel-memory-lab/`,
  `experiments/research/history/path-map.json`,
  and historical lint pins in `src/sparselab/research/lint.py`. Map its remaining
  reusable guarantees (rights, collision/lineage, accounting, replay and refusal)
  before retiring declarations, fixed-slice dependencies or historical lint pins.
- The landed [context-family audit](context-retirement-audit.md) and
  [exact disposition map](context-retirement-audit.json) cover 60 files: Context
  Engram (34), memory injection (8) and path domain (18). The
  [context retirement record](context-retirement.json) removes 30 context assets
  and redirects the frozen guide, preserving original ledger/result bytes in
  Workshop. The [memory retirement](memory-retirement.md) now removes the four
  shared context inputs and eight memory-injection assets against merged Workshop
  preservation and reviewed fixed personal expectations. Native exact-candidate
  acceptance replaced the personal assertion at that time; this obligation is
  historical and superseded by the three-family scope. Neutral Lab tests retain
  generic regressions; original audit JSON and evidence remain unchanged.
  [Path-domain allocation coverage](path-allocation-boundary.md) now uses generated
  neutral inputs and a fixed byte tokenizer. Its eighteen personal assets remain:
  `src/sparselab/data/allocation_tasks.py` still mixes fixed task metadata, split
  sizes, budgets and assignment choices into Python. The earlier plan required explicit declaration design and native personal-input
  acceptance; that personal compatibility obligation is superseded. Any separate
  Lab retirement still needs a dependency check and retained neutral coverage,
  not a fresh acceptance of the old experiment.
  Neutral runtime examples remain in Lab. These are separate retirement slices.
- The dense-lm and learned-portability catalog entries and their specialized
  consumers still need classification. Explicit allocation inputs alone do not
  externalize all scientific policy.
- Hosted/GPU sample disposition remains open, including
  `configs/first_model_20m_cuda.yaml`, `configs/runtime_smoke_rocm.yaml`,
  `configs/runtime_smoke_rocm_bf16.yaml`, `configs/scale_dense_160m_cuda.yaml`
  and `configs/scale_dense_1b_cuda.yaml`. Classify each as a retained generic
  runtime example with explicit scope or a personal scientific declaration to
  preserve in Workshop before removal. Any claimed replacement needs its own
  qualification; CPU evidence is not accelerator proof. No qualification or
  execution of these files is part of this reconciliation.
- Workshop still has per-experiment Python and an internal operator adapter.
  Native capability candidates are not proof that every helper assertion can be
  removed. Map each guarantee to neutral Lab or internal Harness coverage; do not
  move bespoke experiment Python wholesale into either repository.
- Reconcile active guides, catalogs and authoring routes after each verified
  retirement, while preserving frozen evidence/readers and documented missing
  references. Complete the file-by-file map and native dry-run evidence before
  claiming Lab #157 closed. Lab #158 remains bounded and partial.

Retirement does not wait for model workloads. Verify original bytes and dependency
closure first, then validate current declarations without executing science.
New reproductions use current interfaces and new identities; do not implement
compatibility adapters to recreate historical execution. Preparation revisions
retain their own admitted identities; later workflow declarations bind actual
verified outputs without rewriting preparation provenance or resetting allowances.

## Real-data admission

Before real acquisition/execution, record selected immutable source revision and
rights provenance, existing verified payload versus acquisition, network/cache/
storage limits, CPU runtime, all paired/continuation update budgets, pilots and
runtime probes, evaluation/panel work, monitoring and cancellation. The native
snapshot `max_work_bytes` does not cap Hub download/cache traffic. Broader hardware,
spend or additional workloads require their own authorization.
