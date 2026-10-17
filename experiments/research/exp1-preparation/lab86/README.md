# LAB86 concrete preparation package — unexecuted

This package supplies 84 exact source declarations, published content SHA-256s,
a deterministic bounded shard selection, private application policy, resource
policies, a zero-model contract and 24 native phase/job templates. It has not
acquired corpus payloads, fitted a tokenizer or established 12B usable tokens.
The initial pilot boundary is fully declared in `pilot-session.template.json`
and `pilot-jobs/`; only host/runtime/baseline bindings remain. Use a reviewed
runtime containing this PR's timestamp fix. Later data-dependent review is not
a prerequisite for safe initial acquisition. See `PILOT.md` for the exact boundary.

## Scope and metadata evidence

Canonical approval provenance: [LAB86](https://app.notion.com/p/3fb3b68cca2281a5a68fc549ca0c62d6).
The pinned source cards and CodeParrot loader text are retained in `evidence/`;
the loader is evidence only and is never imported or executed. The catalog is a
projection of complete pinned HF tree metadata (path, published bytes, LFS SHA256).
LFS content hashes are not Xet hashes. No Parquet bytes were read by this writer. External review subsequently supplied
footer-only observations, retained separately with attribution: the FineWeb pilot
has `timestamp[ms]` (the card says seconds), 812684 rows; Cosmopedia has 376289
rows. These are metadata observations, not completed acquisition evidence.

| Input metadata | SHA256 of fetched response |
| --- | --- |
| SmolLM recursive tree | a74e258a9b66b34d11e1f47de5309eed9e9b7ba75068edaed3f598db8b1b65a1 |
| CodeParrot recursive tree | e3bc4e6083600440a8b9b4562ba64bf81ce1b0a2b80c3cd2b6067a983d783db6 |

Tree endpoints are `/api/datasets/REPOSITORY/tree/REVISION?recursive=true&limit=1000`;
repository/revision pairs are explicit in `shards.json`. The relevant complete
candidate catalog has 234 FineWeb, 104 Cosmopedia and 880 CodeParrot shards.
Python-edu is excluded. The selection test reconstructs the hash-ranked prefix
from this catalog. Caps are 48/16/16 GiB compressed, not a token-mixture policy.
Selected response bodies total **85,077,359,484 bytes (79.24 GiB)**. Source cap
99 GiB leaves 21,223,081,092 bytes of source retry headroom; metadata has its own
1 GiB cap and both share the cumulative 100 GiB cap. One explicit retry per shard
is permitted by the ceiling, not automatically scheduled or guaranteed affordable.

The three pilot shards sort first by both declaration path and source ID:

| Stream | Pinned shard | Bytes |
| --- | --- | ---: |
| Web | fineweb-edu-dedup/train-00068-of-00234.parquet | 2251835383 |
| Synthetic | cosmopedia-v2/train-00000-of-00104.parquet | 1174424341 |
| Code | data/train-00570-of-00880.parquet | 345453831 |

Total 3,771,713,555 leaves 523,253,741 bytes at the 4 GiB gate before metadata
and retries. The fourth shard exceeds that remainder even with zero metadata.
Earlier failure, metadata overhead or retry charging may stop before all three;
the gate does not promise completion. Minimum-sized pilots are not representative.
Each shard scans at most a 5M-row prefix, retains all selected prefix rows (hash
modulus one), and refuses above 8 GiB emitted JSONL. These are chosen bounds, not
observed row counts. Parquet expanded bytes cannot be capped by this adapter.
The 300 GiB whole-workspace monitor remains in force; per-shard ceilings do not
imply every intermediate fits. Retained row-index receipts can themselves be large.

## Initial acquisition versus later gates

The verified initial-acquisition bug is fixed in this PR: nested Parquet timestamps
are encoded as ISO 8601 strings, retaining decoded timestamp values and numeric offsets or lack of timezone.
Arrow units and named zones are not retained in JSON. Arrow map timestamps and
`date32` remain unsupported and fail closed.
Original typed Parquet bytes remain identified by shard SHA256; emitted row hashes
identify the JSON representation. Retrieval records the encoding and adapter module
hash. Nulls and ordinary JSON values retain their meaning; other unsupported types
still refuse. Tiny mocked acquisition and cold verification cover this path.
The previously cached image lacks this fix: use a reviewed, smoke-tested updated
image and matching Harness allowed-image pin before the real pilot. No image pull,
publish or host execution was performed in this task.

The following are **later processing gates**, not initial download blockers:

- The external reviewer reports a real string `language` column in the pinned
  code pilot footer. This narrows the question to verifying every selected shard;
  see `evidence/pilot-footer-review.json` for attributed metadata. Acquisition
  retains original rows without executing the loader; later qualification still
  quarantines any missing language. No extension-based derivation is needed for
  the reported pilot schema.
- Cosmopedia seed_data remains the family key. If it is a dataset label, preserve
  coarse grouping; do not replace it with per-prompt IDs. Pilot family counts
  determine whether the >=100-family split requirement is feasible. Choosing a
  different family definition is a real scientific-policy decision, not binding.
- Later release/tokenizer/packing paths and receipt identities must be produced in
  order. The full 24-slot session skeleton cannot be sealed ahead of these outputs
  because Harness fixes exact argv. This affects continuation, not the separately
  sealable four-slot pilot session. No runtime declaration mutation is proposed.
- A downstream EXP1 base/run config is absent. Initial acquisition needs neither
  model architecture nor prepared-input config. Supply it only before export/pack;
  no model design or training authority is invented here.

## Bindings and ordered execution boundary

Use `pilot-session.template.json` for the initial boundary and the existing
`../preparation.json` native plan. `phase-bindings.template.json`
separates fixed package paths from genuinely generated evidence. `jobs/` and
`session.template.json` spell out exact native argv and one slot per phase; no
run-all script or automatic failure acknowledgement is provided. Placeholder
rendering is data substitution, never shell evaluation. Native phase compilation
rejects unresolved placeholders. Job `required_files` must be a concrete mapping
of absolute path to actual SHA256 before launch, including the source declarations,
project, policy, plan, resolved contract, baseline and bindings used by that job.
Use `package-sha256.json` plus newly generated evidence; no guessed digest entries.

Host bindings: existing Linux amd64 Docker/cgroup-v2 execution path,
actual Python executable/version/digest (and venv object if applicable), boot ID,
source/config checkout roots, work/receipt roots, existing lease namespace,
actual approval owner/reference, measured capacity and baseline. The old source engine pin was `6f8b3047b3b5f30a4778f38de40b0bf45d13bdb1`;
`HOST_REVIEWED_LAB_COMMIT` must bind this PR's reviewed merge commit or a later
reviewed release containing its timestamp fix.
The old Harness pin `9020bec32f82f0c5abe93ba0eba46d6402d97b14` allows only image
`ghcr.io/supernovae/tiny-sparse-lab@sha256:4af84411f6af27bdb0e8e0c6c9e2d64569329df4f8f48f407c94b5d39808d1d8`.
That image predates the fix. Its reviewed replacement and the matching Harness
allowlist update are concrete initial-runtime release blockers, not missing data.
No pull, build, runtime qualification or runner execution was performed here.

Generated inputs, not missing configuration decisions: workspace baseline and
content seal; acquisition/snapshot hashes; pilot reuse project/contract/ledger;
reviewed admission and receipt; duplicate/family merges; frozen splits; build
result/release; LM export; fresh tokenizer/config; measured token inventory;
unique mixture and packed-input identity. Resolve them from verified native
receipts in order. `project-pilot-reuse.template.yaml` uses only the three exact
retained snapshots, no second transport allocation. Its contract must bind that
rendered project's file/semantic hashes and set `offline_retained_sources_only`.
Retain the original whole-session clock and disk baseline across both ledgers.

The main contract binds both acquisition file SHA and native semantic identity.
Its eight model/generation/forward counters are zero, with preparation-only and
release acceptance enforcement. Before later admission-bound work, bind the
actual inspected admission under the native contract/receipt rules; a template
is not an admission receipt. Release and project templates preserve the original
acquisition identity and require newly generated admission/split evidence.
Application policy leaves rights at review-required; actual reviewed eligibility
must be evidenced before release. Inspection selection is a small spot audit,
not the tokenizer's 1000 unseen documents per stream audit.

After the transfer gate, assess schema, rights, families, qualified yield, RAM,
disk and measured/projection uncertainty. Native gate release requires the actual
report-bound review and existing ledger. Harness acknowledgement merely records
the nonzero gate outcome; it grants no continuation authority. No recovery may
erase a charge, lock, failed receipt or existing LAB39/LAB70 hold.

After reviewed 98/1/1 family splits, export with a native base run configuration
at sequence length 2048, then apply `tokenizer-sampling.yaml` to the **new** export
config with vocab32768/min-frequency1. The actual base/run configs remain a
reported missing downstream input: no EXP1 architecture config was present to
reuse, and this package does not invent a model design. Fit once, measure floors,
and use `mixture.template.yaml` with native default hash shuffle, max_exposures1.
Supply shortfall blocks 12B preparation; never repeat samples or reweight silently.

Memory is observed RSS, not hard enforcement; require 24 GiB effective available
RAM initially and stop below 4 GiB headroom. Require 310 GiB free/100k inodes;
stop below 10 GiB/10k inodes. The 300 GiB retained allowance includes receipts and
64 MiB shared reserve. Per-phase storage allowances are provisional conservative
reservations; feasibility review may require a smaller supported scope, not a
budget increase. Eight-CPU quota is not a hard thread-count limit. All native
thread libraries use one; no GPU mappings or model calls.
