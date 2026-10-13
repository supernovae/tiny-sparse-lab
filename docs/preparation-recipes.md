# Portable preparation planning

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

Inventory integrity is not trusted committed-source provenance. Acquisition,
preparation execution, lock publication, provenance verification, runtime acceptance
and execution admission remain distinct operations. This slice adds no preparation
apply command, admission route or native engine context override. Eventual execution
must revalidate artifacts, source provenance, runtime evidence, permissions,
resources, claims and holds through the existing CampaignEngine/Controller.
Evaluation closure, recovery/replay context and run-evidence linkage remain open.

## Proof and change-impact map

| Boundary | Regression coverage |
| --- | --- |
| Descriptor/context → native graph | Train/reuse graphs equal native compilation, ordering, references, role/contract checks, cycle/missing/wrong producer refusals |
| Bound config → native adapter | Typed request construction and native pack/verifier spies; producer-location mismatch refusal; no executor added |
| Integrity and locations | Relocation, sorted inventory, immutable snapshots, changed bytes, binding overlap, traversal/symlink, opaque metadata |
| Inspection → effects | Fresh-process CLI/help import, payload, credentials, write, network/process and runtime fences |
| Iteration → admission | Explicit gate even without other missing readiness evidence; malformed/changed source retains errors; original CLI/shared Campaign refusal tests retained |
| Compatibility | Existing recipe v1 and campaign digest goldens, seed coercion, native iteration and verifier-authority regressions, normal safe/fast/tiny CPU lab-loop CI |

These are bounded software contracts, not actual TinyStories portability or GPU
qualification. Full LAB-73/LAB-74 acceptance remains partial.
