# LAB86 package validation

Metadata/source-only configuration assembly on 2026-10-17. No corpus/model
payload download, tokenizer fit, model workload, container launch, image pull,
package install or host modification occurred.

Installed Python 3.14.4 in the existing Lab environment, WSL Ubuntu 26.04;
OMP/MKL/OpenBLAS=1, HF Hub/Datasets/Transformers offline, GPU visibility empty.
Native subprocess fixtures use the existing UV_PROJECT_ENVIRONMENT, no install.

`pytest -q tests/test_lab86_package.py tests/test_preparation_commands.py
 tests/test_preparation_tokenizer_dispatch.py`: **31 passed in 35.84s**.
The added manifest-integrity assertion was subsequently run separately.
Ruff format/check and `git diff --check` passed.

Coverage includes real production project/source-effect hashes, deterministic
84-shard reconstruction from pinned metadata, pilot ordering/gate arithmetic,
cumulative/source/metadata/disk/wall budgets, policy document hashes, zero model
counters, resource/tokenizer/mixture schemas and all 24 native argv compilations.
Existing tests cover native tokenizer dispatch and template command wiring.
A tiny generated Parquet fixture confirms the native datetime serialization
boundary; a tiny code row confirms missing language is quarantined. These are
boundary evidence, not tests asserting the real corpus is usable.

Execution remains blocked as documented in lab86/README.md. CI and external
review are independent checks; this document grants no clearance or job approval.
