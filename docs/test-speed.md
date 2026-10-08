# Fast test feedback

From the repository root, install the locked development environment once:

```sh
uv sync --locked --extra cpu --dev
```

Run the tests nearest the files you changed first. Examples:

```sh
# Corpus acquisition, normalization, generation, exports, and provenance
uv run --locked --extra cpu pytest -q tests/test_corpus_acquisition.py tests/test_corpus_pipeline.py tests/test_corpus_generators.py tests/test_corpus_exports.py tests/test_corpus_provenance.py
# Dataset/config/tokenizer/packing integration
uv run --locked --extra cpu pytest -q tests/test_config.py tests/test_datasets.py tests/test_conversations.py tests/test_local_chat.py tests/test_tokenizer.py tests/test_packing_manifest.py
# Surface changes
uv run --locked --extra cpu pytest -q tests/test_surface_overlay.py tests/test_surface_review.py tests/test_surface_triage_import.py
# Hub credentials (without a public endpoint)
uv run --locked --extra cpu pytest -q tests/test_hf_auth.py
```

Pushes to `main` and all PRs run one explicit Linux zero-model-work job:
Ruff, the frozen decoding test hash, research lint, and selected safety,
compatibility, packing, panel and readiness nodes. The selected nodes include
no optimizer update or model generation. The hosted qualification guard is
retained as experimental tooling but is not part of ordinary CI. The
`workflow_dispatch` **Run workflow** menu defaults to `safe`; its
`integration-linux`, `platform-macos` and `release-candidate` choices are
manual model-bearing checks and require a separately reviewed allocation.
No tag automatically starts model work. The release-candidate choice runs the
broader Linux and macOS CPU suites. No tests were deleted; ordinary PR success
does not certify optimizer/generation integration or macOS behavior.

For an explicit local full-suite run, use two workers, distributing whole test
files so module-scoped fixtures remain together. Pytest's `tmp_path` and
`tmp_path_factory` keep artifacts in worker-specific directories. Do not share
a mutable work directory between independently launched suites.

```sh
uv run --locked --extra cpu pytest -q -n 2 --dist loadfile -m 'not mps and not mlx and not cuda and not rocm and not xpu and not network'
uv run --locked --extra cpu ruff check .
uv run --locked --extra cpu ruff format --check .
```

## CPU worker limits and stall diagnosis

The root test `conftest.py` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to `1` during pytest
startup. This matches the CI environment and happens before collection and
xdist worker creation, so workers and subprocesses they launch inherit one
thread per CPU library. If a pytest plugin imported Torch before the
conftest, the conftest also caps Torch's intra-op pool without importing Torch
for Torch-free tests. Like CI, it leaves the inter-op pool unchanged. These limits
apply only to test processes; they do not change lab runtime defaults.

Campaign status projection verifies each completed dependency-DAG stage once
per traversal, including shared upstream nodes. Later projections and checks
before dispatch verify again; no completed-stage cache survives the traversal.
This is not generic snapshot proof reuse across projections or commands.

Verifier authority walks deduplicate module scheduling and file discovery only
within one closure walk. Each later authority lookup still reads and hashes the
verifier source bytes. Proof lookup diagnostics use the same authority that was
compared with the receipt, rather than walking the closure again for telemetry.

The CPU gate includes native Campaign integrations with real worker startup,
staging, training, checkpoint transfer, continuation, and evaluation. These can
take several minutes. A whole-command deadline must allow all separately
bounded runs and the subsequent verification; a faulthandler stack report is
not a test failure or a run deadline.

For a suspected CPU stall, retain the normal two-worker layout and ask
pytest's built-in faulthandler plugin for thread stacks after a bounded interval:

```sh
uv run --locked --extra cpu pytest -vv -n 2 --dist loadfile \
  -o faulthandler_timeout=120 \
  -m 'not mps and not mlx and not cuda and not rocm and not xpu and not network'
```

`faulthandler_timeout` reports stacks; it does not skip, kill, or otherwise
turn a slow test into a passing result. Use the reported test name to rerun it
serially with the same option when isolating the blockage.

On an Apple Silicon machine with the optional MLX runtime installed (`uv sync --locked --dev --extra cpu --extra mlx`), run hardware tests in **one serial process**, after CPU workers finish:

```sh
uv run --locked --extra cpu --extra mlx pytest -q -m 'mlx and not mps and not network'
# Opt-in MPS tests when the device is available; the current CI gate excludes these.
uv run --locked --extra cpu pytest -q -m 'mps and not network'
```

Never use `-n` for MLX/MPS or run another accelerator test/training process against the same device at the same time. Network-marked tests require explicit separate opt-in.

## Why the gate is structured this way

The full CPU jobs on Linux and macOS retain their tests, hardware exclusions,
and existing 30-minute safety limits, but are manual-only while their long
native integrations are paused as automatic build gates. Automatic lint and
focused/evidence/serving jobs retain their 5- and 10-minute limits. The manual hosted
CPU jobs use up to four workers with work-stealing; the local command above
uses whole-file scheduling for fixture reuse.

The [historical CI run 36380519748](https://github.com/supernovae/tiny-sparse-lab/actions/runs/36380519748)
measured 331 seconds for Linux's serial CPU step and 407 seconds for macOS's
serial CPU step. Those measurements are not a bound for today's larger suite.
Locked uv downloads are cached through `setup-uv`. Accelerator and network
checks remain separate explicit opt-ins.
