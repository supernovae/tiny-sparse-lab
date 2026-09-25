# Disk checkpoints, recovery, and promotion

A checkpoint generation is a durable, immutable local snapshot shared by the PyTorch and MLX execution paths. It is not activation recomputation, gradient accumulation, or an inference KV cache. Checkpoints are written at successful update boundaries only; partial accumulation gradients are not a resumable state.

## Verify before reuse

```sh
uv run sparselab train configs/runtime_smoke_cpu.yaml --run-id runtime-part --stop-after-step 10
uv run sparselab checkpoint inspect runs/runtime-part/checkpoints/latest.json --json
uv run sparselab checkpoint verify runs/runtime-part/checkpoints/latest.json --json
uv run sparselab train configs/runtime_smoke_cpu.yaml --run-id runtime-resumed \
  --resume runs/runtime-part/checkpoints/latest.json
```

Both engines store generations under `runs/<run-id>/checkpoints/step_<step>_gen_<generation>/`. Each finalized generation has a manifest and canonical safetensor weight shards/index; its engine selects the native training-state codec. `latest.json` and `best.json` are atomic lookup indexes; verified generation manifests and file digests are the durable truth. `checkpoint inspect` reports inventory and lineage, not a standalone integrity claim. `checkpoint verify` validates the selected path and returns structured failures with a nonzero exit when invalid.

The checkpoint manager writes a temporary sibling, verifies the completed generation, atomically publishes it, then updates lookup pointers. A writer lease prevents concurrent writers/recovery in the same run. Recovery scans finalized generations and can repair stale projections/pointers, but an explicit corrupt `--resume` selection fails rather than silently falling back.

## Full resume

`--resume` creates a new child run. It restores canonical model weights, named optimizer state, warmup/cosine schedule, optional scaler, committed counters, data cursor, cadence watermarks, cumulative timing, and the engine's safe RNG envelope. The child records `parent_run_id` and the exact parent `checkpoint_sha256`; it does not bind itself to a moving `latest` pointer.

Full resume requires the same engine/backend, architecture semantics, tokenizer/data/package identities, optimizer settings, schedule horizon, training budget, micro-batch/accumulation, sequence length, deterministic setting, and verified run-owned artifacts. Source/runtime or device-identity drift requires `--allow-runtime-drift`; that produces an explicit best-effort record and never bypasses scientific compatibility. A completed budget cannot be resumed.

Use `--recover RUN_DIR` only to seek the latest valid finalized generation after interruption/corruption. It creates a child after checking compatible state; it is not permission to overwrite the existing run. Old or incomplete historical artifacts that lack required continuation state are not full-resumable.

## Promotion is weights-only continuation

`--promote CHECKPOINT` creates a child with compatible canonical model weights but **fresh** optimizer, schedule, scaler, RNG, cursor, and zero committed counters. Its destination config supplies the new data/budget/runtime choices. Promotion still requires the same architecture semantics, tokenizer contents, and portable package identity when present; equal tensor shapes alone are insufficient. It does not resize a backbone or turn dense weights into MoE.

Use full resume for the same experiment and promotion for a compatible
weights-only starting point. The public path is local; remote scheduling and
controller-side artifact ingestion are separate contracts.

## Import compatible external weights

`weights import` accepts closed, validated source formats:
`sparselab_legacy_v1` and `hf_llama_safetensors`. The latter is a narrow
bias-free dense Llama mapping with compatible attention/RoPE/tokenizer semantics,
including exact dense or grouped-query layouts when the destination's
`num_kv_heads` exactly matches the source `num_key_value_heads`. It is not a
general Hugging Face, quantization, or chat-template importer.

```sh
sparselab weights import SOURCE OUTPUT_RUN \
  --config COMPATIBLE_V2.yaml --format sparselab_legacy_v1 \
  --source-tokenizer SOURCE_TOKENIZER.json --provenance PROVENANCE.json
```

The destination config must preserve architecture semantics and exact compatible
tokenizer contents. Provenance format 1 requires explicit source and license
statements; importing does not grant redistribution rights. The output is a
verified, promotion-only generation with null native-state codec fields, not
invented optimizer or RNG state. Use its checkpoint with `train --promote`.

Historical `.pt` files remain inspectable/verifiable through the same checkpoint
CLI. Their hash, migrated configuration, tensor shapes/values, and tied aliases
are checked. Their reported scope is always `weights_only`; they cannot become
full-resumable by supplying `--config`.

## Evaluation, best, and lineage

Validation runs at step 0, configured evaluation intervals, and final/interruption boundaries. A new best validation loss causes a checkpoint; a cadence-only checkpoint has `validation_loss: null` rather than inheriting stale attribution. `best.json` points only to a verified evaluated generation physically present in that run. A resumed child may carry an ancestor `lineage_best` record whose checkpoint is not local; it is lineage metadata, not a fabricated local best pointer.

When selecting an older parent checkpoint, inherited best-state discovery stops
at that selected generation and step. Later same-run evaluations cannot leak
backward into the child's lineage. An ancestor best that is not copied locally
remains explicitly unavailable as a local checkpoint.

`checkpoint.keep_periodic: false` prunes only after successful publication and protects latest, best, and the immediately previous verified generation. Keeping fewer generations reduces recovery/history options; it never makes old state safer.

In [actual signal/recovery acceptance](../artifacts/acceptance/signal_recovery_2026_09_22.json),
SIGTERM ended cleanly at a committed update; deliberate latest-generation
corruption made explicit resume fail before child creation. Explicit recovery
selected the previous verified generation and completed a new child update,
leaving that selected parent's files unchanged.

## Native MLX generations

MLX uses the common format-2 generation, pointer, artifact, lineage, recovery,
and evidence contract. Its native state is `training_state.json` plus
`optimizer.safetensors`; both are hashed and fsynced before generation
publication. The reader checks the exact optimizer array inventory, named
groups, counters, schedule, primitive metadata, and Python/NumPy/MLX RNG.
Malformed containers reject the generation rather than escaping recovery.

Offline inspection and verification do not require MLX. Execution does require
the pinned optional runtime. Same-engine continuation restores native state;
cross-engine continuation is canonical-weight promotion with fresh training
state, never an optimizer conversion.
