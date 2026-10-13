# Native model-lab smoke inputs

Locally generated JSONL, with distinct training and held-out sentences. No corpus
or model download. The native workflow trains a 260-token byte-level tokenizer,
prepares data, initializes and trains two one-layer/16-wide dense CPU models for
3 updates and 96 targets each, scores 64 held-out targets per arm, runs fast
probes and reports the sealed comparison. The declared change is FFN width
32 -> 48; paired seed is 7. Lower, higher and tied loss are valid outcomes.

The selected subsets are eight training documents and four validation documents;
unused lines are still included in the input file identity. This demonstrates
native lifecycle and numerical wiring, not language-model quality.

All output paths are relative to the declarations. `lab-job` copies these exact
input bytes into its durable `inputs` directory before native loaders resolve
paths, so `../outputs` and `../native/lab` belong to that job's evidence root.
Do not run these declaration files directly from this repository checkout if you
intend to keep outputs outside Git. See [lab jobs](../../docs/lab-jobs.md).
