# C03 offline continuation test allowlist

This is the positive, reviewed node-ID allowlist for the preparation-only
continuation. Invoke these exact nodes only, with
`SPARSELAB_PREPARATION_ONLY=1`; do not use a file or corpus-test glob.

- `tests/test_preparation_guard.py::test_preparation_only_blocks_model_training_and_staging`
- `tests/test_corpus_split_freeze.py::test_freeze_splits_is_deterministic_and_validates_integer_family_counts`
- `tests/test_corpus_split_freeze.py::test_finalized_family_inventory_matches_kept_release_documents`
- `tests/test_corpus_mixture.py::test_exact_quotas_repeats_and_cold_replay`
- `tests/test_corpus_mixture.py::test_accepted_65_25_10_target_shares_are_realized`
- `tests/test_corpus_mixture.py::test_shortfall_never_publishes`
- `tests/test_corpus_mixture.py::test_card03_order_policy_uses_contract_document_hash`
- `tests/test_kml_card03_items.py::test_reviewed_heldout_draft_binds_content_and_reports_incomplete`
- `tests/test_kml_card03_items.py::test_two_source_inference_accepts_distinct_heldout_families`
- `tests/test_kml_card03_items.py::test_two_source_rejects_other_category_and_mixed_split`
- `tests/test_corpus_export_min_frequency.py::test_default_export_request_keeps_historical_hash_and_two_is_distinct`
- `tests/test_corpus_export_min_frequency.py::test_export_request_rejects_invalid_minimum_frequency`
- `tests/test_corpus_token_export_binding.py::test_cold_measurement_passes_matching_export_dataset`
- `tests/test_corpus_token_export_binding.py::test_cold_measurement_rejects_foreign_export`
- `tests/test_corpus_token_denominator.py::test_explicit_four_mib_document_bound_counts_one_large_document`

Review: each selected test and its local `_fixture` use `tmp_path` and
`monkeypatch`, construct synthetic documents, and call corpus/evaluation
functions only. `tests/conftest.py` only sets CPU thread counts. These selected
functions and their fixtures contain no subprocess, `stage`, `train`, warmup,
optimizer or GPU calls. The guard test invokes entry points with invalid
placeholder arguments and requires rejection before validation or work. No
parameterized tests or module-level glob is selected.
The two export-request tests are pure canonical-hash checks and have no fixtures
or subprocesses.
The two denominator-binding tests use only `tmp_path`, `monkeypatch`, and
in-memory fake verifier/config objects. They invoke no subprocess or trainer.
The explicit large-document node uses its `_synthetic`, `_row`, and `_tokenizer`
helpers, which create only local fixture files and a static WordLevel tokenizer;
it does not request the module's acquisition fixture or launch subprocesses.
The Card 03 order-policy node is pure hash arithmetic and has no fixtures.
The two cross-family evaluation nodes reuse `test_kml_card03_items._fixture`,
which writes only local synthetic test/train documents and monkeypatches release
verification. Neither node or helper launches a subprocess or optimizer work.
