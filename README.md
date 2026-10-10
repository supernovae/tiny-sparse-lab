<p align="center">
  <img src="docs/assets/sparselab-banner.svg" alt="Tiny Sparse Lab — Small models. Real experiments." width="100%">
</p>

<p align="center">
  <a href="https://github.com/supernovae/tiny-sparse-lab/actions/workflows/ci.yml"><img src="https://github.com/supernovae/tiny-sparse-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&amp;logoColor=white" alt="Python 3.14"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-7ceac4" alt="MIT license"></a>
  <a href="docs/experiment-programs.md"><img src="https://img.shields.io/badge/Experiments-YAML%20DSL-a7a2ff" alt="YAML experiment DSL"></a>
</p>

<p align="center">
  <a href="docs/tinystories-microlab.md">Start with TinyStories</a> ·
  <a href="docs/lab-mode.md">Try an idea</a> ·
  <a href="docs/using-sparselab.md">Find a guide</a>
</p>

Tiny Sparse Lab is a workbench for small language-model experiments. Change one
setting, train a baseline and candidate, and inspect the evidence. Native lab
mode keeps inputs, checkpoints and results together without requiring a research
Campaign for an ordinary local experiment.

## First run

Install **Python 3.14** and [uv](https://docs.astral.sh/uv/), then:

```sh
git clone https://github.com/supernovae/tiny-sparse-lab.git
cd tiny-sparse-lab
uv sync --locked --extra cpu --dev
export SPARSELAB_WORK_DIR="$HOME/.local/share/sparselab"
```

Follow **[the TinyStories microlab](docs/tinystories-microlab.md)**: copy three
existing declarations, prepare a small pinned story dataset and tokenizer, then
compare one FFN-width change with `try`. It teaches the workflow on CPU; the tiny
training budget does not promise fluent stories. The walkthrough labels source
downloads and model work before their commands.

Already have a RunConfig and its tokenizer? Start with
**[lab mode](docs/lab-mode.md)**:

```sh
cat > wider-ffn.yaml <<'YAML'
question: Does a wider FFN help at the same token budget?
set:
  model.ffn_dim: 384
YAML
uv run --locked --extra cpu sparselab try wider-ffn.yaml --vs BASELINE.yaml
uv run --locked --extra cpu sparselab report TRY_ID
```

Replace `BASELINE.yaml` with your config and `TRY_ID` with the printed record ID;
choose a width different from your baseline (the TinyStories baseline is 256).
`try` prepares data as needed, trains both arms, scores the same held-out protocol
and runs fast probes. It reuses only a verified matching completed baseline.

## The everyday lab loop

| Question | Existing command | Work performed |
| --- | --- | --- |
| Does this change help? | `try DELTA.yaml --vs BASELINE.yaml` | Preparation, training, scoring and fast probes. |
| What happened? | `report TRY_ID` | Reads the sealed record. |
| What should I check next? | `probe CANDIDATE_RUN --vs BASELINE_RUN --backend cpu --tier standard` | Loads checkpoints and scores probes. |
| How do recorded results compare? | `compare TRY_ID --references` | Reads records; no download or scoring. |
| What is inside this checkpoint? | `explore CANDIDATE_RUN` | Loads a small checkpoint and computes diagnostics on CPU. |

Use the **run IDs** in the report for `probe` and `explore`; use **try/probe
record IDs** for `report` and `compare`. The [probe guide](docs/probe-battery.md)
explains verdicts and optional full-tier evaluation. Missing evidence and
`NOT_COMPARABLE` are not wins. Keep a separate untouched evaluation and repeat
promising changes with more seeds before making quality claims.

`sparselab dashboard` opens the [lab dashboard](docs/dashboard.md): Home,
Experiments, Models, Behaviors and Explorer, with detailed run telemetry below.

![Lab home: latest verdict and next steps](docs/assets/dashboard-home.png)

## Bring your machine

Choose the compute available to your installed framework. PyTorch is the
reference engine; MLX is an optional, separate engine for Apple Silicon.
These settings describe where an experiment runs. Its question and comparison
come from the model, data, training and evaluation declarations.

| Compute | Engine / backend | Lab capability and requirements |
| --- | --- | --- |
| **CPU** | PyTorch / `cpu` | CPU execution on Linux, WSL2 and macOS. |
| **NVIDIA GPU** | PyTorch / `cuda` | Requires a compatible CUDA framework and driver installation. Dense SDPA is opt-in; [probe](docs/hosted-environments.md#boundaries-and-gates) the actual selected kernel rather than assuming FlashAttention. |
| **AMD GPU** | PyTorch / `rocm` | Requires a compatible ROCm framework and driver installation. A native HIP sparse-attention path is also available within its documented device limits. |
| **Intel GPU** | PyTorch / `xpu` | Requires a compatible XPU framework and driver installation. |
| **Apple Silicon GPU** | PyTorch / `mps` | Uses the PyTorch Metal backend on a supported Mac. |
| **Apple Silicon GPU, optional engine** | MLX / `metal` | FP32 dense/native block-sparse training, AdamW and block recomputation; a smaller feature set than PyTorch. |

Host OS, engine and device are separate choices. Local runtime discovery and
warmup check whether your installed environment can execute the selected workload.
See [runtime support](docs/runtime.md) for precision, feature restrictions and
[recorded software checks](docs/lab-status.md) for the environments used in those
checks. A study described as running on an AMD GPU is still a study of its
declared model or training change.

For a larger model after this lesson, [the first-model guide](docs/first-model.md)
uses the 17.3M FineWeb-Edu configs, repeated seeds and runtime-specific setup.

## Advanced workflows

Choose these when the question needs them; they are not setup steps for `try`.

- **Inspect saved models:** [checkpoint exploration](docs/tinytext-model-guide.md),
  [local API](docs/local-api.md), and the retained [MODEL-0](docs/model-0.md) evidence.
- **Control training and continuation:** [training](docs/training.md),
  [checkpointing](docs/checkpointing.md), [datasets](docs/datasets.md), and
  [instruction objectives](docs/instruction-training.md).
- **Schedule independent runs:** [workers](docs/workers.md) and
  [hosted environments](docs/hosted-environments.md); one host/device per run.
- **Declare release evidence:** [training programs](docs/experiment-programs.md),
  [Campaigns](docs/campaigns.md), [iteration checks](docs/iteration.md), and
  [corpus preparation](docs/corpus-preparation.md).
- **Study mechanisms and findings:** [architecture](docs/architecture.md),
  [papers](papers.md), [research workbench](docs/research/README.md), and
  [experiment ledger](docs/research/experiment-ledger.md).

Keep large outputs under the external work root. Explicit YAML output/cache
paths retain their destinations; setting `SPARSELAB_WORK_DIR` does not move them.
Keep retained data, failed runs and original evidence. Release promotion and
paid compute require their own explicit authorization; a lab verdict grants none.

## Feature matrix

Support describes the shipped lab interfaces; scientific outcomes live in the
[experiment ledger](docs/research/experiment-ledger.md). Backend restrictions are
listed below and in the [runtime guide](docs/runtime.md).

| Feature | Available workflow | Boundary / guide |
| --- | --- | --- |
| Tokenizer, preparation, training and generation | Native CLI and YAML RunConfig | [First story model](docs/tinystories-microlab.md); data and weights need local storage. |
| Attention comparisons | Dense, sliding-window, block-sparse and reference MLA | [Architecture](docs/architecture.md); reference mechanisms are not full paper reproductions. |
| Conditional capacity | Local Top-K MoE; token, byte and portable Engram | [Capability experiments](docs/capabilities.md); resident and active parameters differ. |
| Training programs | Matrices, immutable ExperimentPlan locks and checkpoint phases | [DSL](docs/experiment-programs.md); child dispatch follows verified parent completion. |
| Campaign orchestration | Declared dependencies, readiness, approvals and reconciliation | [Campaigns](docs/campaigns.md); independent single-device runs. |
| Corpus provenance | Acquire, shape, freeze, export and verify source identities | [Corpus Forge sample](corpora/devmind-sample-v0/README.md); source terms remain separate. |
| Declarative text datasets | Pin Hub sources, snapshot, resume, report coverage and propose pass budgets | [Dataset interfaces](docs/datasets.md); explicit resource limits and immutable inputs. |
| Instruction objectives | Whole-transcript or assistant-only loss on local conversations | [Instruction training](docs/instruction-training.md); tool transcripts are inert. |
| Evaluation and review | Held-out loss, capability cards, checkpoint-bound suites, self-blind review | [Evidence](docs/evidence.md); promotion requires review. |
| Native diagnostics | Supplied-vector semantic probes, continuation overlap, bounded preparation benchmarks and opt-in phase observations | [CLI guide](docs/using-sparselab.md#native-diagnostic-interfaces); mechanism/descriptive/operational evidence, not quality or promotion gates. |
| Local, SSH, and supplied hosted execution | Explicit runtime selection, staging, independent worker queues, and optional relay-backed Colab | [Hosted environments](docs/hosted-environments.md); one device per run, no provider allocation or distributed training. |
| Dashboard | Lab home with next steps, experiments and verdicts, model catalog and Pareto frontier, behaviors (generations, recall misses, calibration), a visual model explorer, plus training telemetry and research views | [Dashboard tour](docs/dashboard.md); [research views](docs/research/dashboard.md); read-only. |
| Local model exploration | Raw completion, transcript chat, Streamlit comparisons and a loopback API | [Model guide](docs/tinytext-model-guide.md); [nonstreaming API](docs/local-api.md); verified local run required. |

Contributions: [CONTRIBUTING.md](CONTRIBUTING.md), [code backlog](TODO.md),
[agent guidance](AGENTS.md). Code is [MIT licensed](LICENSE); source datasets
retain their own licenses and attribution.
