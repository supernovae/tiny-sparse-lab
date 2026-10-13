# Portable preparation recipes

`experiment recipe preparation inspect RECIPE --bindings LOCAL --json` compiles
an external preparation recipe without running it. This is a separate
`portable_preparation_version: 1` contract; the existing matrix and prepared-input
workflow v1 formats, identities and commands are unchanged.

The descriptor has `id`, `entrypoint`, `contracts` and `files`. Contracts must be
`{campaign: 1, run_config: 2, tokenizer_config: 1}`. Each file has a canonical
recipe-root-relative `path`, SHA-256 of its bytes, and a role: `campaign`,
`run_config`, `tokenizer_config` or `metadata`. The entrypoint is the sole Campaign
member. All declaration references use the inventory root, never CWD or Git
fallback. The existing 64-document, 1 MiB per-document and 8 MiB total limits apply.

The entrypoint is a native CampaignPlan restricted to `tokenizer_train`,
`data_prepare` and external tokenizer `artifact_reference` stages. CampaignPlan
remains the sole graph validator/order authority; `compile_preparation` produces
its existing immutable operation projection. No second DAG or temporary YAML is
created. Snapshot acquisition/edges, recovery and other stage kinds are refused
before their validators can import computation. Broader contracts remain future
work. Two supported compositions are:

- `tokenizer_train → data_prepare` with an unresolved future tokenizer output.
- `artifact_reference(tokenizer) → data_prepare` with a declared, unverified identity.

The separate local document has `preparation_bindings_version: 1`, `artifacts`
(stage ID to location for every external tokenizer producer), and `config_paths`
(config reference to a mapping of native field names to locations). For example,
a training graph's bindings can be:

```json
{
  "preparation_bindings_version": 1,
  "artifacts": {},
  "config_paths": {
    "tokenizer.json": {"output_dir": "payload/tok", "dataset.cache_dir": "payload/cache"},
    "run.json": {"tokenizer.path": "payload/tok/tokenizer.json", "dataset.cache_dir": "payload/cache", "logging.root_dir": "payload/runs"}
  }
}
```

Bindings must name exactly the compiled config references and every present
native typed Path field. Relative locations anchor to the local binding document;
canonical absolute locations are allowed. Traversal and symlinks are refused.
Native declaration path placeholders must also be canonical relative paths.
The tokenizer consumer location must match its compiled producer's output or
external binding. Opaque metadata fields named `path` are not interpreted.

Inspection snapshots and verifies inventoried declaration bytes, validates native
configs, and reports operations, typed edges, config byte digests and local paths.
It neither opens payloads nor claims their availability. Future outputs have no
fabricated content identity. Runtime capability is unknown, committed provenance
is unverified, and execution is unavailable. The separate recipe digest uses the
`sparselab-portable-preparation-v1` domain and sorted inventory; local bindings
have their own byte digest. No historical hash algorithm or seed acceptance changes.

## Iteration and operation boundaries

Read-only iteration can now reconstruct a published prepared-input workflow's
intent from its authenticated availability sidecar and revalidated descriptor,
bindings and inventory. Valid context no longer fails because the native entrypoint
omits descriptor-supplied artifacts. It instead reports the explicit
`WORKFLOW_EXECUTION_UNAVAILABLE` gate. Invalid declarations and changed identities
still report errors. The profile-presence guard in CLI/Campaign dispatch remains.
The informational `execution` annotation does not authorize anything.

Inventory integrity is not trusted authorship. Acquisition, preparation, lock
publication, provenance verification, model runtime acceptance and execution
admission remain distinct operations. The explicit preparation route below uses
CampaignEngine for a bounded local CPU subset. Prepared-input workflow execution,
model dispatch, evaluation closure and run-evidence linkage remain unavailable in
these portable profiles; their existing admission gates are unchanged.

## Proof and change-impact map

| Boundary | Regression coverage |
| --- | --- |
| Descriptor/context → native graph | Train/reuse graphs equal native compilation, ordering, references, role/contract checks, cycle/missing/wrong producer refusals |
| Bound config → native adapter | Typed request construction and native pack/verifier spies; producer-location mismatch refusal; existing CampaignEngine dispatch |
| Integrity and locations | Relocation, sorted inventory, immutable snapshots, changed bytes, binding overlap, traversal/symlink, opaque metadata |
| Inspection → effects | Fresh-process CLI/help import, payload, credentials, write, network/process and runtime fences |
| Iteration → admission | Explicit gate even without other missing readiness evidence; malformed/changed source retains errors; original CLI/shared Campaign refusal tests retained |
| Compatibility | Existing recipe v1 and campaign digest goldens, seed coercion, native iteration and verifier-authority regressions, normal safe/fast/tiny CPU lab-loop CI |

These are bounded software contracts, not actual TinyStories portability or GPU
qualification. Full LAB-73/LAB-74 acceptance remains partial.

## Guard coverage after review

The preparation-only operation requirement, exact config-reference binding set,
external-tokenizer-only profile and unsupported native path allowlist have direct
loader regressions. The allowlist case uses a valid native `local_token_mixture`
config with all required paths and provenance fields; it fails at the portable
profile boundary without reading mixture payloads. Other cases validate their
native Campaign schema first and use config-read spies where appropriate.

The workflow `declaration_hashes`, declaration source and declaration root checks
are redundant consistency checks alongside the authenticated descriptor/binding
and inventory identities. Direct helper tests vary only those redundant fields,
after a successful normal context load. These tests do not claim altered sidecars
pass `open_lock`, nor grant committed provenance or execution authority.

The private `_config` helper explicitly refuses missing role metadata or config
bindings. Normal loading already establishes both through inventory/reference and
exact binding-set checks. Direct helper regressions protect those diagnostics and
prove undeclared references are still refused first, before config reads. This is
defence in depth for future callers/reordering, not a new admission mechanism.

## Explicit local CPU preparation

The owner-reviewed execution subset adds:

```sh
sparselab experiment recipe preparation apply RECIPE --bindings LOCAL \
  --admission ADMISSION.json --work-dir /absolute/work/root --json
sparselab experiment recipe preparation resume RECIPE --bindings LOCAL \
  --admission ADMISSION.json --work-dir /absolute/work/root --json
```

`inspect` still performs no execution admission or payload verification. `apply`
and `resume` require an explicit local admission document with all these fields:

```json
{
  "preparation_admission_version": 1,
  "reviewed_commit": "<full 40-character recipe Git commit>",
  "recipe_sha256": "<recipe digest from inspection>",
  "binding_sha256": "<binding byte digest from inspection>",
  "cpu_python": "/absolute/selected/environment/bin/python",
  "max_documents": 32,
  "max_tokens": 1024,
  "max_vocab_size": 512,
  "min_free_bytes": 268435456,
  "min_free_inodes": 128
}
```

Digest placeholders above must be replaced with real hexadecimal identities.
The selected Python must be the interpreter running the command; the command
never provisions or launches another environment. Recipe descriptor and every
inventoried declaration must be clean and committed at `reviewed_commit` in their
own repository. Selecting that commit asserts that the owner reviewed the recipe.
Git proves committed bytes, **not trusted authorship** or review. No uncommitted
or model-execution override is accepted on this route. Legacy Campaign overrides
and readers retain their existing behavior for legacy records.

Only `synthetic` datasets execute in this first slice. The total train plus
validation document and token limits of each native config must fit the explicit
admission caps; tokenizer document and vocabulary limits must also fit. These are
admission ceilings, never rewritten scientific settings. Native tokenizer storage,
packing storage, host checks and artifact verifiers still run. The additional
free-space/inode and permission check is repeated before dispatch. No corpus
acquisition, remote source, model training, GPU allocation or model-runtime
qualification occurs. RunConfig runtime/model fields retain their native meaning;
this CPU preparation admission does not qualify that later model workload.

All output paths must lie below the explicit absolute work root outside Git.
Tokenizer output and cache directories must not overlap or occupy native operational
store directories. Native preparation receipt/scratch publication uses that same
work root; process location settings are restored after dispatch. External tokenizers
remain explicit bound inputs, verified through the native artifact verifier.

The engine receives the inventoried native plan and bound native configs directly.
It retains the sole existing scheduler, stage-input hashes, preparation adapters,
artifact verification and immutable receipts. One context supplies availability,
requests, pre/post-dispatch provenance and replay. Receipts record recipe/binding
identity, declaration closure and commit, locations, admission limits, installed lab
source digest, Python identity and dependency versions as operational provenance.
Scientific declaration/stage/artifact hash algorithms are unchanged. An existing
store refuses a changed context (including a differently formatted bindings file),
even when a caller explicitly selects its new digest. Use a fresh work root for a
new context; cross-binding resume and migration are future work.

Native committed receipts recover after a crash without calling the adapter again.
Completed outputs are reverified, including before receipt commit; changed outputs
are refused. An interruption without a committed receipt retains its native attempt
and interruption observation and refuses automatic rerun. Existing unreceipted
outputs are never overwritten or adopted. This conservative refusal is deliberate:
partial preparation recovery needs a separate validated contract. It does not remove
legacy cancellation, recovery or resume behavior.

| Execution boundary | Focused regression evidence |
| --- | --- |
| Selected context → admission | Wrong commit/digests/interpreter, dirty index, finite caps, environment drift, no work-store creation on refusal |
| Native plan → two tiny graphs | Train then pack and verified external tokenizer then pack both complete using native adapters |
| Dispatch → receipt | Binding change immediately before adapter or after dispatch, output change before commit, immutable receipt checks |
| Receipt → replay/recovery | No redispatch on replay, crash after immutable receipt before state save, cross-binding and legacy-route refusal |
| Interrupted work → next action | Preserved interrupted attempt, no automatic overwrite/rerun, collision retains existing bytes |
| CLI → effects | Public lightweight entry route, explicit local work-root precedence, fresh-process apply/resume help without importing execution or heavy optional packages |

The profiled lifecycle tests are marked slow and explicitly included in the existing
`lab-loop` CI job; metadata and refusal checks remain in `fast`. No new workflow or
manual workload gate is introduced.

Validation is tiny synthetic wiring only. Actual pinned CPU/GPU TinyStories
portability proof, broader local datasets, model admission, evaluation closure,
publication CLI wiring and the remaining LAB-73/LAB-74 milestone work remain open.
