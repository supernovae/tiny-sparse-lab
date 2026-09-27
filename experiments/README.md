# Experiment workspace

This directory contains durable, reviewable experiment definitions—not mutable
run output.

| Path | Purpose | Expected contents |
|---|---|---|
| [`samples/`](samples/) | Copyable teaching and starter experiments | Small configs, walkthroughs, expected artifact shapes, and explicitly non-scientific example output |
| [`research/`](research/) | Actual SparseLab research campaigns | Identity-bound protocols/config references, preregistered gates, iteration notes, findings, and evidence links |
| `../sparselab-work/experiments/` | Local execution workspace (ignored) | Downloads, prepared data, run stores, checkpoints, logs, temporary files, and generated reports |

Do not promote sample output into research evidence. To turn a sample into a
campaign, copy its editable inputs into a named `research/<campaign>/` directory,
declare the question and gates, and use a separate named work directory.

Existing packaged lessons and catalog recipes remain under
`src/sparselab/research/resources/`; existing reference configs remain under
`configs/references/`. This directory indexes how they are used rather than
duplicating them.
