# TinyStories sample declarations

Start with the [lab walkthrough](../../../docs/tinystories-microlab.md).
Copy only `source.yaml`, `tokenizer.yaml` and `run.yaml` into a fresh external
workspace's `inputs/` directory, prepare the snapshot/tokenizer, and use `try`
for the width comparison. Relative paths resolve against those copied YAML files.
Lab runs live in `WORK_DIR/lab/runs`; direct-training configs name `../runs`.

The sample pins a TinyStories revision and excludes whole-document duplicates
and train/validation overlap. Each baseline arm has 40 updates / 10,240 targets.
Source acquisition uses network access; model work is a small teaching exercise,
not a quality benchmark. Preserve prior data and negative or interrupted results.

## Advanced declarations

These remain available for explicitly chosen exercises; they do not import a
previous lab baseline automatically.

| File | Purpose | Native guide |
| --- | --- | --- |
| `continued.yaml` | PyTorch AdamW child: 80 cumulative updates / 20,480 targets, original 40-update decay horizon | [Checkpointing](../../../docs/checkpointing.md) |
| `matrix.yaml` | Two fresh FFN widths at the baseline budget | [Experiments](../../../docs/experiments.md) |
| `plan.yaml` | Width comparison with checkpoint-bound child phases | [Training programs](../../../docs/experiment-programs.md) |
| `campaign.yaml` | Acquisition through evaluation for three selected cells; optional wider child unselected | [Campaigns](../../../docs/campaigns.md) |

For those routes copy the required declarations into a separate workspace;
`campaign.yaml` belongs at its root with the other YAML files in `inputs/`.
Keep matched scientific settings and backend selections consistent before locking.
Larger-source exercises use [dataset coverage and budgets](../../../docs/datasets.md).
The original [acceptance record](acceptance.md) retains the completed Campaign's
identities, measured results and limitations. It describes that execution, not
the beginner lab walkthrough.
