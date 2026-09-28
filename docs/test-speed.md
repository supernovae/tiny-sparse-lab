# Fast test feedback

From the repository root, install the locked development environment once:

```sh
uv sync --locked --dev
```

Run the tests nearest the files you changed first. Examples:

```sh
# Corpus acquisition, normalization, generation, exports, and provenance
uv run --locked pytest -q tests/test_corpus_acquisition.py tests/test_corpus_pipeline.py tests/test_corpus_generators.py tests/test_corpus_exports.py tests/test_corpus_provenance.py
# Dataset/config/tokenizer/packing integration
uv run --locked pytest -q tests/test_config.py tests/test_datasets.py tests/test_conversations.py tests/test_local_chat.py tests/test_tokenizer.py tests/test_packing_manifest.py
# Surface changes
uv run --locked pytest -q tests/test_surface_overlay.py tests/test_surface_review.py tests/test_surface_triage_import.py
# Hub credentials (without a public endpoint)
uv run --locked pytest -q tests/test_hf_auth.py
```

Before opening a PR, run the full CPU suite; the focused checks do **not** replace either CI full-suite gate. Use two workers, distributing whole test files so module-scoped fixtures remain together. Pytest's `tmp_path`/`tmp_path_factory` keep test artifacts in worker-specific directories. Do not share a mutable work directory between independently launched suites.

```sh
uv run --locked pytest -q -n 2 --dist loadfile -m 'not mps and not mlx and not cuda and not rocm and not xpu and not network'
uv run --locked ruff check .
uv run --locked ruff format --check .
```

On an Apple Silicon machine with the optional MLX runtime installed (`uv sync --locked --dev --extra mlx`), run hardware tests in **one serial process**, after CPU workers finish:

```sh
uv run --locked pytest -q -m 'mlx and not mps and not network'
# Opt-in MPS tests when the device is available; the current CI gate excludes these.
uv run --locked pytest -q -m 'mps and not network'
```

Never use `-n` for MLX/MPS or run another accelerator test/training process against the same device at the same time. Network-marked tests require explicit separate opt-in.

## Why the gate is structured this way

The [successful CI run 36380519748](https://github.com/supernovae/tiny-sparse-lab/actions/runs/36380519748) measured 331 seconds for Linux's serial CPU pytest step and 407 seconds for macOS's serial pytest step; locked installs took 4 and 7 seconds respectively. CI now runs an independent short focused job **concurrently** with both full Linux and macOS CPU suites on pushes and PRs, so fast feedback does not delay the full gate. The macOS job runs MLX-marked cases only after its CPU run finishes, serially, without sharing a hardware lease with a second job. Existing MPS exclusions stay in place until hardware stability has been verified. Locked uv downloads are cached through `setup-uv`. On this Linux workstation, the two-worker CPU gate completed in 128.30 seconds (695 passed, 3 skipped); hosted-run speedup remains unmeasured. The fast gate contains corpus/config/ingestion examples; use the local commands above for the closest tests to each commit.
