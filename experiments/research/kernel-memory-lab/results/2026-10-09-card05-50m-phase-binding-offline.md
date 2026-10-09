# C05-T4 — 50M native phase binding, offline qualification

**Result:** The revised prospective packet has an explicit native command
grammar, late binding for the actual training config and evaluation-output
baseline, and a metered fixed-slice entry path. This is offline execution-path
evidence, **not** a replacement 50M attempt or corpus admission. C05-B11's
stopped ledger remains spent and unchanged: its SHA-256 is
`09a5b2a4348417bb2b90d2d56603e5a02208c07c7ea46863795c0f0ae0ea3d2d`,
matching its retained inventory.

The shared classifier accepts the reviewed `uv run --locked --no-sync sparselab
--work-dir ROOT ...` and direct native forms, checks every operation's options,
and validates one nested native monitor and its inner command. It rejects
arbitrary shell/Python, hidden training/staging, duplicate or unknown options,
and an unfunded live acquisition. The native snapshot alias cold-verifies the
retained source and current source declaration without copying or transferring
bytes. Optional new contract fields are absent from legacy serialization.
The packet contract pins the original fixed profile and family-inventory
digests; substituted expected digests fail before dispatch, while the native
fixed-slice binder cold-checks the profile's release and tokenizer identities.
The actual config digest is sealed once in the same ledger after preparation;
the static packet identity remains the phase contract identity. A separate
cold-bound baseline pins the evaluation-output monitor. Fixed-slice scoring
and continuations require an active owned evaluation phase with sufficient
durable forward/generation allowance before model load, and the native leaf
charges before forward/decoding. Historical standalone evaluation remains
explicitly outside this new attempt-only pre-load check.

The selected no-model public-path fixture executed:

`uv run --locked --no-sync sparselab --work-dir <isolated-root> attempt run ... -- uv run --locked --no-sync sparselab --work-dir <isolated-root> monitor ... -- uv run --locked --no-sync sparselab --work-dir <isolated-root> corpus budget-init <fixture-project>`.

This ran the real CLI, v2 attempt budget, owned supervisor, whole monitor,
preparation monitor and persistent native transport-budget initialization.
The bounded Git metadata fixture caused no network request or model work.
The fixture's owned completion and both monitor completions are under
`/srv/sparselab/state/experiments/kernel-memory-lab/card05-50m-binding-offline-20261009/pytest12/test_public_cli_owned_whole_an0/owned/`.
The owned completion reports zero living descendants; both monitors report
`COMPLETE`; the fixture ledger records zero optimizer updates, target positions,
nontraining forwards, generation calls and requested generation tokens.

The exact selected regression command, inside native monitor
`test-monitor12`, was:

```sh
env -u UV_PROJECT_ENVIRONMENT uv run --locked --no-sync pytest -vv --tb=long -p no:cacheprovider --basetemp /srv/sparselab/state/experiments/kernel-memory-lab/card05-50m-binding-offline-20261009/pytest12 tests/test_attempt_packet_dispatch.py tests/test_corpus_acquisition.py::test_verified_snapshot_alias_reused_across_project_ids_without_transfer tests/test_attempt_contract.py tests/test_attempt_contract_cli.py tests/test_attempt_campaign_guard.py
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and `PYTHONDONTWRITEBYTECODE=1` were set.
The enclosing `timeout 300s` and native test policy capped wall time at 300 s,
tree RSS at 4 GiB, added apparent storage at 512 MiB and entries at 5,000
against one persistent baseline. **79 passed** in 26.77 s. The monitor receipt
`test-monitor12/completion.json` reports `COMPLETE` in 28.03 s, peak tree RSS
1,888,915,456 B, peak added storage 4,819,906 B, peak added entries 2,812,
and no violations. Prior failed local test/monitor receipts remain retained
under the same offline root. Ruff check, Ruff format check, research lint and
`git diff --check` passed.

The leaf-map test parses every packet shape through the real CLI and the shared
classifier without execution. Rejection tests cover hidden commands,
malformed flags, source/project substitution, transport-ledger absence,
monitor/baseline identity, unbound or changed config, missing model allocations,
selector declaration substitution, deadline/resource failures and cleanup
receipts. The only model-leaf fixture
uses a fake `load_run` that fails if reached; it establishes a pre-load rejection,
not a real model score. No tokenizer fitting, forwards, generation, optimizer
updates, GPU work, acquisition or production-corpus processing occurred.

**Readiness:** the offline native dispatch mismatch is corrected for review.
A fresh 50M attempt would need its own approval and exact-head launch packet.
Its execution-time gates still include source transfer/admission, protected
lineage, post-cleaning unique-token floors, exact mixture and exposure proof,
headroom, ROCm fit, model-bearing fixed-profile evaluation and scientific
review. This qualification does not establish any of those outcomes.
