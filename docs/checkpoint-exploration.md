# Explore retained checkpoints

These examples use retained research artifacts on a specific workstation.
For a new training run, start with the [TinyStories microlab](tinystories-microlab.md).

To explore the two retained seed-42 endpoints on this ROCm workstation, run
the [data-rich checkpoint comparator](../experiments/research/tinystories-dense-30m-data-rich-v1/compare.py)
from the separate data-rich worktree. It verifies both checkpoint identities
and uses each model's native tokenizer without silently truncating prompts:

```sh
cd /home/byron/src/tiny-sparse-lab-data-rich
export UV_PROJECT_ENVIRONMENT=/home/byron/src/tiny-sparse-lab/.venv
export PYTHONPATH="$PWD/src"
export SPARSELAB_WORK_DIR="$PWD/sparselab-work/experiments/tinystories-dense-30m-data-rich-v1"
uv run --locked --no-sync python experiments/research/tinystories-dense-30m-data-rich-v1/compare.py
uv run --locked --no-sync python experiments/research/tinystories-dense-30m-data-rich-v1/compare.py --prompt "Mia carried the little red boat to the pond." --decoder sampled --seed 11
```

The first command opens a one-line-per-prompt terminal session; an empty line
or Ctrl-D exits. `--prompt-file FILE` accepts a multiline UTF-8 prompt;
`--prompt "..." --json` prints raw completions, native token counts/IDs and
checkpoint digests. The same text and decoder settings are used for both
models, but their vocabularies and contexts differ. On the frozen 256-story
same-text control the data-rich 30M had lower bits/byte than the earlier
50M; hand-entered continuations are exploratory, not a prose-quality score.
The 50M also differs in width, depth, training data, budget, context and
precision, so this comparison **cannot attribute an outcome to data variety
instead of depth**. Checkpoints and source snapshots must remain available
in the two sibling worktrees; no new training is run.

For a local **self-blind review of already generated text**, import a verified
study's original outputs into a new, ignored, task-owned bundle, then open the
dedicated review page (not the read-only telemetry dashboard):

```sh
uv run --locked sparselab surface import data-rich-v1 \
  --campaign-root /home/byron/src/tiny-sparse-lab-data-rich \
  --sample quick --selection-seed 2026 --presentation-seed 2027 \
  --output sparselab-work/experiments/surface-review-v1/bundles/data-rich-quick-2026
uv run --locked sparselab surface review \
  sparselab-work/experiments/surface-review-v1/bundles/data-rich-quick-2026 --port 8502
```

The import reads existing, hash-verified campaign JSONLs; it does not run models,
replace missing files or change old study evidence. Submit all A/B judgments,
complete the write-once blind review, then explicitly reveal identities and
descriptive counts. `sparselab surface chat` with two to four
`--cell ALIAS=GENERATION_PATH` options instead generates *exploratory* local
checkpoint replies; its votes are not sealed review evidence. See the
[Surface Review v1 guide](research/surface-review-v1.md)
for old decoding and triage imports, profiles, dimensions, seed rules, privacy
limits and the optional verified triage/dashboard overlay. One person's review
does not establish population preference or promote a model.
