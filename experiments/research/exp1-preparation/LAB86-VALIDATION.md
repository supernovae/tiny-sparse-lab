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
**106 passed in 41.71s**.
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
are verified against their original byte digests. Host object assembly is explicitly
JSON parsing, documented and tested; no quoted mapping is submitted to Harness.
