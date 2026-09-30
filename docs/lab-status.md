# Lab status and evidence


The dated **2026-09-22/23 acceptance records** capture **335 passing tests**, Ruff lint/format checks, installed-wheel execution outside the checkout, and real CPU, MPS, and MLX/Metal scenarios. Later regression results remain with the relevant evidence records and commits rather than the implementation backlog. These are implementation checks, not a claim that a tiny checkpoint is a useful general assistant.

| Evidence | What was actually exercised |
|---|---|
| [Single-host acceptance](../artifacts/acceptance/single_host_gate_2026_09_22.json) | CPU FP32/BF16 and Adafactor continuation, actual MPS/MLX execution, safe interruption/recovery, promotion, offline artifacts, staging, and populated dashboard checks. |
| [Independent-worker acceptance](../artifacts/acceptance/independent_workers_2026_09_23.json) | Three overlapping logical CPU workers, progress through controller loss, idempotent launch replay, cancellation and bitwise child-resume comparison, forced executor loss, CLI matrices, source/capacity rejection, and an actual MLX worker. |
| [Scientific studies](../artifacts/acceptance/scientific_studies_2026_09_22.json) | Preregistered multi-seed/multi-budget context/Engram comparisons and domain adaptation with retention checks. Untouched context overrides remained **0/8** at every endpoint; adaptation did not establish reliable held-out domain behavior and caused severe forgetting. |
| [Research workbench evidence](research/sample-report.md) | Completed CPU/offline smoke and nano campaigns plus an MPS/FineWeb-Edu micro study; the observed alias-card scores were zero. The MLA report smoke exercised factorial, nondominance, allocation, and boundary outputs, not model quality or speedup. |
| [Dense-LM token-budget study](../experiments/research/dense-lm-token-budget-v1/results.md) | On one ROCm device, three full-state continuations of the fixed TinyStories reference lowered held-out loss at 8.39M and 16.78M target exposures. Some fixed-prompt continuations worsened or contradicted the prompt; no broader text-quality claim or automatic promotion follows. |
| [Dense-LM scale comparison](../experiments/research/dense-lm-scale-v1/results.md) | On one measured ROCm device, three ~50M fully dense seeds at 16.78M supervised targets lowered held-out loss relative to like-seeded mature ~30M references. Six-case fixed generation remained mixed, with higher update time and memory; no architecture or broad text-quality claim and no automatic promotion. |
| [Dense-LM decoding study](../experiments/research/dense-lm-decoding-v1/results.md) | Reused six mature 30M/50M checkpoints without training. Greedy regression remains frozen; 22 development prompts selected a sampled decoder, followed by 990 generated cells on 55 separate prompts. Sampling reduced mechanical repetition but caused visible drift; independent subjective text quality remains unreviewed. |

The [research roadmap](research/roadmap.md) separates implemented paths,
smoke evidence, task-level results, and open questions about
lexical/portable/semantic memory and useful task models. [`TODO.md`](../TODO.md)
contains only pending implementation work.

The latest worker UI check captured actual rendered curve pixels; standard browser screenshots stalled, so runtime-table values were additionally verified through Streamlit's app harness. The earlier populated single-host dashboard screenshots remain in the acceptance record.


See [runtime support](runtime.md) for platform boundaries and the
[project page](../README.md) for first use. These records are dated evidence,
not a rolling test count or a promise of model quality.
