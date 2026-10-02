# Semantic kernel MVP v1

Status: staged research proposal plus an executable, read-only **substrate pilot**.
No model is trained, no neural benefit is established, and this is not a new
Engram portability result. See [protocol.md](protocol.md),
[implementation-plan.md](implementation-plan.md), and
[project-context.md](project-context.md). The
[Stage 0 observation](substrate-observation.md) records software verification.
Existing DevMind and Engram work is
unchanged. This pilot is not registered in the research catalog or campaign DSL.

The first real domain is this repository's Python source. It is understandable,
locally available, versioned, and MIT licensed. The compiler reads syntax without
importing or executing source. It records definitions, signatures, docstrings,
and **syntactic call sites**, not resolved dependency or causal edges.

Run from the repository root, with outputs outside the checkout:

```sh
export KERNEL_WORK="$HOME/.local/share/sparselab/experiments/semantic-kernel-mvp-v1"
uv run --locked python experiments/research/semantic-kernel-mvp-v1/substrate.py compile \
  --repo . --output "$KERNEL_WORK/source-bank.json"
uv run --locked python experiments/research/semantic-kernel-mvp-v1/substrate.py query \
  --bank "$KERNEL_WORK/source-bank.json" \
  --action '{"op":"find","args":{"name":"format_chat_prompt"}}'
uv run --locked python experiments/research/semantic-kernel-mvp-v1/substrate.py replay \
  --bank "$KERNEL_WORK/source-bank.json" \
  --actions experiments/research/semantic-kernel-mvp-v1/demo-actions.json \
  --output "$KERNEL_WORK/demo-trace.json"
```

Outputs fail if the destination already exists. Use a new output path to retain
prior attempts. `query` prints a result; `replay` publishes a bounded trace with
each action, result/error, bank identity, and component timing. It **does not run
a model**. Demo actions are human-authored wiring probes, not an evaluation set
or training corpus. The AST bank is a new interface artifact, not an EngramPack,
embedding space, or teacher compilation.

Available operations:

| Operation | Exact arguments | Meaning |
|---|---|---|
| `find` | `name` | Exact final-name match; returns source IDs and locations |
| `get` | `id` | Returns one definition, signature and bounded docstring |
| `call_sites` | `name` | Finds syntax spelled `name` or ending in `.name` |

Example question: **Where is `format_chat_prompt` defined, and where is that
spelling called?** The runtime can ground the answer in source lines. It cannot
prove those calls resolve to that definition or establish behavior from syntax.

To carry this into your existing ChatGPT Project, add `project-context.md`,
`protocol.md`, and `implementation-plan.md` as project sources. The context file
contains a starting instruction and the next bounded coding task. These files
were not automatically attached to a ChatGPT Project by this change.

## What to run next

Review the pilot records manually, then implement Stage 1's independently scored
task bundle and model adapter. Do not start a 1–2B scratch pretraining run yet.
The first model comparison can use an explicitly selected, rights-reviewed local
pretrained checkpoint. Training comes only after a frozen-model comparison shows
what the interface does and where the controller fails.
