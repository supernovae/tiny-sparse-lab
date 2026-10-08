# Card 03 evaluation continuation nontraining test allowlist

Run only these exact node IDs with `SPARSELAB_PREPARATION_ONLY=1` under the new
continuation deadline/monitor wrapper. No file glob or broad corpus suite is
authorized.

- `tests/test_preparation_guard.py::test_preparation_only_blocks_model_training_and_staging`
- `tests/test_kml_card03_items.py::test_reviewed_heldout_draft_binds_content_and_reports_incomplete`
- `tests/test_kml_card03_items.py::test_two_source_inference_accepts_distinct_heldout_families`
- `tests/test_kml_card03_items.py::test_two_source_rejects_other_category_and_mixed_split`
- `tests/test_kml_card03_context_check.py::test_context_screen_counts_each_condition_and_preserves_generation_space`
- `tests/test_kml_card03_context_check.py::test_context_screen_flags_long_question_and_answer_bearing_wrong_context`
- `tests/test_kml_card03_context_check.py::test_context_screen_flags_partial_train_phrase_overlap`
- `tests/test_kml_card03_review_receipts.py::test_review_receipt_requires_other_reviewer_and_exact_final_semantics`
- `tests/test_kml_card03_review_receipts.py::test_review_receipt_rejects_stale_blind_solution`

Review: `tests/conftest.py` only limits CPU thread counts. The guard node uses
invalid placeholders and requires rejection before any model work. The three
existing evaluation nodes use their local `tmp_path`/`monkeypatch` fixture,
synthetic source text and native draft validator without a subprocess, trainer,
stage or optimizer. The two new context nodes import only the pure offline
screen module; their `_inputs` helper uses `tmp_path`, monkeypatches cold
release/tokenizer reads with static in-memory fixtures and calls `measure`
directly. The third context node swaps only its synthetic train text to check
partial phrase overlap. The two review-receipt nodes import only the pure offline receipt
validator and build one synthetic item in memory. They launch no subprocess or
model entry point. The separate Ruff check targets only the changed scripts
and their exact test files. A read-only `ruff format --check` over those same
five exact paths is also allowed; it does not import or execute their code.
