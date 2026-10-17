# LAB86 package validation

Metadata/source-only configuration assembly and a narrow Parquet timestamp fix,
2026-10-17. No corpus/model payload download, tokenizer fit, container launch,
image pull/publish, package install or host modification occurred locally.

Existing Python 3.14.4 environment, WSL Ubuntu 26.04; OMP/MKL/OpenBLAS=1;
HF Hub/Datasets/Transformers offline; GPU visibility empty. Native fixtures use
the existing UV_PROJECT_ENVIRONMENT. HTTP responses are mocked or tiny local
fixtures, never real dataset payloads.

`pytest -q tests/test_lab86_package.py tests/test_corpus_acquisition.py
 tests/test_preparation_commands.py tests/test_preparation_tokenizer_dispatch.py`:
**131 passed in 43.00s**.
Repository Ruff check/format, research lint (570 records), and staged diff checks
passed. Fetched source evidence is retained as JSON text with original byte SHA256,
which preserves upstream text while conforming to the research record format.

Tests cover production project/source-effect hashes, deterministic 84-shard
selection, pilot-first ordering/gate arithmetic, cumulative/source/metadata/disk/wall
limits, policy hashes, zero-model counters, resource/tokenizer/mixture schemas,
24 native phase commands and the four-slot pilot with only host bindings remaining.
Connected mocked acquisition verifies nested timestamps, nulls, offset/precision,
original shard SHA, canonical row hashes and cold snapshot/acquisition verification.
Unknown non-JSON types still refuse. A code row without language acquires but remains
quarantined at later qualification. Existing acquisition/dispatch tests also pass.

Initial runtime still needs a reviewed release image containing the timestamp fix
and matching Harness digest allowance; the previous cached image predates it.
Host qualification/bindings belong to the chief. Downstream admission, family,
late-binding and model-config questions do not block initial pilot evidence.
No self-review clearance or operational gate release is implied.

Review 5556404189 reconciliation: B1 evidence-format and B2 uppercase-slot fixes
were already in b2fc3aa. B3 now renders every production template, loads pilot
reuse/prefreeze/build with native schemas, verifies exact source scope/budgets,
reviewed release training splits, origin/digests and both inspection selections.
Nine coordinated mutations (including all seven review mutants, slot-ID drift and
lowercase slots) are rejected even after their manifest hashes are recomputed.
Family merges now use the native JSON-array fragment slot. Pilot selection is
independently checked against the smallest catalog shard; source-text envelopes
are verified against their original byte digests. Host object assembly remains a
documented JSON data-binding step; it is not a
native renderer feature. The operator round-trip test was removed in response to review r2.

Review 5556484694: one derivation check now binds the pilot contract to the full
zero-model contract, non-init jobs to their full counterparts with only the three
permitted substitutions, the session envelope/approval to the full session, and
pilot-content hashes/transfer ceilings to their actual referenced inputs. Thirteen
coordinated pilot mutations are rejected at specifically named assertions, including
all twelve reported mutations and the phase-wall case. Mutated content seals are
propagated through contract/jobs/session and the package manifest is recomputed.

Q1 decision: remove the JSON-library round-trip test and retain the documented
operator binding step; no new renderer capability or framework. Timestamp wording
now distinguishes decoded values/offsets from Arrow units/named zones, and the
JSONL-only acquisition test requires absence of the Parquet encoding annotation.
These changes preserve runtime behavior and the approved four-slot pilot boundary.

Review 5556635993: attempt-init's entire job and slot now derive from the already
validated pilot transport-init, with only their explicit command/ID and 300-second,
64-MiB initialization allowances changed. Six additional initial-job/session mutants
cover receipt collision, RSS, offline flags, CUDA, slot profile and extra session keys.
Shared invariant checks cover all 28 jobs and both sessions: observed RSS matches
the 16-GiB monitor, environment profiles/flags and CUDA stay within approved policy,
and root storage plus the 64-MiB shared evidence reserve equals the 300-GiB ceiling.
Five paired full+pilot mutations are rejected even when both sides and manifest
hashes are changed consistently. All old template mutations now match their
intended assertion or native validation message. The first focused run exposed an
expected-message mismatch for the held-out test split; matching the native
"test is held out of training" message resolved it. Runtime/package behavior did
not change, and no new qualification or workload was run.
