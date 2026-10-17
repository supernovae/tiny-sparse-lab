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
**94 passed in 38.05s**.
Repository Ruff check/format, research lint (569 records), and staged diff checks
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
