# Try declarations

The default loop is unchanged: `sparselab try DELTA.yaml --vs BASELINE.yaml`.
Five executable options remain: `--vs`, `--config`, `--backend`, `--runtime`,
`--json`. `--config` selects `fast` (default), `standard`, `full`, `no-probes`,
or a YAML path. Presets use the same validated schema as files. Preset names
win over same-named files; use `./standard` to select such a file explicitly.
Empty, comment-only, null and `{}` files select default policy. Unknown fields,
wrong versions, coercible seeds/booleans and unsupported probe tiers fail before
runtime lookup or output creation.

```yaml
try_config_version: 1
seed: 7
probe_tier: standard
fresh_baseline: false
lab_dir: ./lab
runtime_profile: ./cpu-profile.yaml
resource_envelope: ./bounds.yaml
```

All fields are optional. `seed` is an authoritative paired override for both
arms; without it, native candidate derivation retains a delta's explicit seed
change. `fresh_baseline` bypasses reuse; ordinary reuse still requires a completed
try with matching config, code and current tokenizer/data/memory identities.
`probe_tier` selects held-out screening, never the protected final split.

Paths expand `~` and resolve relative to the declaration file. Select either
`runtime_profile` or `--runtime ID`; using both fails. Native profile loading,
interpreter checks, explicit device selection and authorization remain intact.
`--backend` retains the native explicit override behavior.

Runtime bounds belong in the existing resource envelope, separate from science:

```yaml
resource_envelope_version: 1
max_rss_bytes: 2147483648
lab_try:
  max_wall_seconds_per_arm: 600
  tokenizer_batch_documents: 32
  tokenizer_batch_source_bytes: 1048576
```

`lab_try` is optional and applies to try arms. Wall limits must be positive and
finite; there is no implicit timeout. Batching uses strict integers: documents
1–256 and source bytes 1–8388608. Omitted batching uses native defaults (256
and 1048576); smaller batches bound transient preparation memory. Direct API
limits remain supported and the tighter of explicit and declared limits wins.
Native trainer preparation, checkpointed wall stops, resource preflight,
cancellation and accounting retain their current behavior. Other envelope users
retain their existing limits; absent `lab_try` preserves legacy serialization.

The removed `--seed`, `--probe-tier`, `--fresh-baseline`, `--lab-dir`,
`--runtime-profile`, `--resource-envelope`, `--max-wall-seconds` and
`--tokenizer-batch-*` flags are rejected. Replace them with declaration fields
and envelope limits above. `--work-dir` remains the global state selector.

Try seals the validated effective request, loaded profile/envelope values and
source paths before setup. Files can change later without changing that record.
Resolved wall/batching limits are also recorded under `resources` when setup
reaches execution. `report --json` returns the entire sealed record, including
failed/interrupted results and missing observations; historical records remain
readable without rewriting them. Native checkpoint/digest binding and held-out
scoring are unchanged. One paired seed is still insufficient for a scientific
claim; this migration does not add across-seed aggregation or dashboard replay
of custom settings.
