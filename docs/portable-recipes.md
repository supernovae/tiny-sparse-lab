# Portable recipe inspection (v1)

`sparselab experiment recipe inspect RECIPE.yaml --json` inspects a directory
outside the installed lab checkout using the existing native configuration matrix
compiler. It reads bounded, inventoried declaration bytes and prints a preview.
It never executes the recipe or produces a resolved lock. The result explicitly
says `status: incomplete` and `executable: false`.

This is the first LAB-73 profile built on LAB-72's shared document loader, not the
complete LAB-71 portable experiment workflow. Computation stays in native Python
components. The adapter calls `expand_axes` and `compare_configs`; there is no
second scheduler, DAG, component registry, or plugin loader.

A separate [prepared-input workflow profile](workflow-recipes.md) now connects
explicit artifact bindings to native verification and lock publication. It does
not change this v1 matrix profile or grant execution.

## Descriptor and declaration context

A descriptor has these fields (replace the illustrative digests with exact
lowercase SHA-256 values before inspection):

```yaml
portable_recipe_version: 1
id: width-study
entrypoint: plans/plan.yaml
contracts:
  experiment_plan: 1
  run_config: 2
files:
  - path: plans/plan.yaml
    role: experiment_plan
    sha256: <sha256 of exact plan bytes>
  - path: configs/run.json
    role: run_config
    sha256: <sha256 of exact config bytes>
  - path: notes.yaml
    role: metadata
    sha256: <sha256 of exact notes bytes>
artifacts:
  tokenizer.path: <expected sha256 of tokenizer payload>
```

The descriptor's directory is the explicit recipe root. Its sole inventoried
`experiment_plan` entrypoint uses native `plan_version: 1`, with
`base_run: configs/run.json` relative to **that root**, regardless of the plan's
subdirectory, working directory, or enclosing Git repository. Every declaration
reference must be inventoried. All inventoried members must parse as YAML/JSON
mappings, use native duplicate/non-finite/cycle checks, and match exact bytes.
Inspection snapshots those bytes before schema validation and reference binding.
Limits are 64 members, 1 MiB per document (including descriptor), and 8 MiB total.

Native typed config paths remain relative to the **config's directory**. For
example, `tokenizer.path: artifacts/tokenizer.json` in `configs/run.json` binds to
`configs/artifacts/tokenizer.json`. Paths must use canonical relative POSIX
spellings: absolute paths, `..`, `./`, repeated separators, Windows paths and
symlinks are rejected. Input payloads need not exist; their expected identities
must be listed under `artifacts`. Output/cache locations are bound but never
created. Arbitrary metadata keys named `path` remain opaque and are not followed.

Supported typed inputs are `tokenizer.path`, `model.memory_package_path`,
`dataset.{train_path,validation_path,source_manifest_path,allocation_manifest_path,
corpus_release_path,corpus_export_path}`, and `training.portability_manifest_path`.
Supported output locations are `dataset.cache_dir` and `logging.root_dir`.
Mixture path fields and path-changing axes are unavailable in this profile.
Artifacts are requirements, not authenticated content: missing inputs report
`unavailable`, existing inputs `unverified`, and runtime capability `unknown`.
Neither payload hashes nor runtime probes are calculated.

The native matrix supports axes, derived fields, and comparisons with one cell
per arm. `--max-runs` retains the compiler's expansion limit (default 1000).
Inline base configs, artifact/phase/input/corpus-variant/evaluation composition,
evaluation suites, and attempt contracts are explicitly unavailable here. Use
existing native planning for these declarations; the inspector will not silently
omit their work or claim to have compiled a full workflow.

## Identity and compatibility

The new recipe fingerprint is SHA-256 of `sparselab-portable-recipe-v1\0` followed
by native canonical JSON of the validated descriptor, with `files` sorted by
path. Member byte digests and expected artifact digests participate. Descriptor
formatting alone does not change this fingerprint; its separate
`descriptor_sha256` retains exact descriptor bytes. Directory relocation changes
only the displayed local binding. Recipe names and declared relative paths are
part of this v1 recipe identity; it does not claim scientific equivalence.

`installed_lab` uses the existing source identity algorithm, independently of
recipe bytes. Cell `requested_config_sha256` uses the existing native config hash.
No historical record is rehashed. Existing backend/engine/precision/source
identity distinctions and scientific comparison rules remain intact (LAB-66).

This strict new descriptor does **not** tighten native RunConfig seed acceptance:
`true`, `"7"`, and `7.0` retain their historical native coercions (LAB-67), while
native strict try overrides remain strict. Descriptor versions and contract
versions reject booleans. Tightening legacy seeds would need a separate versioned
contract and owner decision. No such decision is needed for this inspector.

When no explicit `RecipeContext` is passed, `declaration_reference` retains its
existing source-relative then repository-root fallback. Existing commands,
defaults, execution, cancellation, resume, replay and history readers keep their
native routes. CLI dispatch makes this inspection/help path lightweight and
read-only even with global work-directory or token-file options; those options
are not consumed by inspection.

## Change-impact and regression map

| Boundary | Fence | What it proves |
| --- | --- | --- |
| Shared document bytes → schemas | `test_documents.py`, `test_portable_recipe.py` | Native strict parsing, bounded reads, exact member digests, immutable snapshots |
| Explicit context → native declarations | Recipe relocation, decoy-root, undeclared-reference and symlink tests | Root binding is explicit; no CWD/Git fallback; legacy fallback stays available |
| Native config → existing compiler | Matrix/comparison parity, expansion cap, seed and path-axis tests | Same native config cells/derived differences; unsupported composition fails visibly |
| Recipe/lab/local identities | Relocation, metadata-byte, lab-identity and backend tests | Separate identities; native hash/lock backend distinctions retained |
| Installed CLI → inspect/help | Guarded subprocess tests with global prefixes | No heavy optional/runtime imports, process/network calls, filesystem writes, token/payload reads |
| Existing execution/readers | Normal safe/fast/lab-loop CI; declaration, manifest, seed and verifier-authority focused suites | Compatibility fences plus existing tiny CPU end-to-end coverage |

The inventory regression fences also cover order-independent fingerprints, duplicate
non-plan entries with specific refusal reasons, the aggregate metadata limit at an
exact tiny boundary, attempt-contract refusal, descriptor self-inventory before
its downstream digest mismatch, and direct uninventoried context references.
The lightweight schema audit covers all `RunConfig` and `ExperimentPlan`
validators reachable through base loading, rebinding and native axis expansion.
Dataset source pins now live in dependency-free `local_stories_pins`, re-exported
unchanged by `data.local_stories`; plan reference validators use
`declaration_context.safe_path`. Remaining validators perform only schema,
arithmetic, string and path checks. Fresh-process fences cover local_stories and
snapshot bases and source-changing axes, local corpus/allocation inputs, remote
source declarations, portable memory/MoE/MLA/Adafactor settings, MLX declarations,
and malformed falsey evaluation-suite refusals. They also check native plan
reference validation independently of preview refusal and assert heavy modules
remain absent. These are metadata checks, not runtime capability or data proofs.

Some guards intentionally overlap: document/root symlink checks complement
`safe_path`, and config containment complements binding beneath an inventoried
config directory. Removing one can leave end-to-end rejection intact. These
checks remain as defense in depth; tests do not bypass upstream validation merely
to force every redundant branch to fail independently.

Remaining milestone work includes broader typed component interfaces and workflow
connections, artifact selection/local binding contracts, complete resolved-plan
compilation/admission through the existing executor, packaging/publication, and
bounded actual CPU/GPU portability evidence with current pins. This preview is
not that evidence. No proof stores, lock publication, new science identity,
Workshop/website changes, or release tags are introduced.
