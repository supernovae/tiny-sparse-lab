# Foundation contract for Card 01

Status: **READY FOR REVIEW** for the approved documentation scope. This is a
protocol, not an executable Campaign, a completed Gate 0, or a runtime receipt.
The acceptance gates and card order in `RESEARCH_PLAN.md` and
`EXECUTION_WORKBOOK.md` remain controlling.

## Project identity and artifact boundaries

The research declaration belongs under
`experiments/research/kernel-memory-lab/`. Use native schemas, typed derivation,
staging, Campaign observation and evidence commands when their required inputs
exist. Keep mutable data, logs, receipts and checkpoints in the external work
root's `experiments/kernel-memory-lab/` namespace. The observed session setting
was `SPARSELAB_WORK_DIR=/srv/sparselab/state`; verify that location anew before
an active task. Paths identify storage locations, not scientific identities.

| Mode | Origin and identity | Required check before claiming success |
| --- | --- | --- |
| Tiny correctness fixture (Card 02) | New deterministic synthetic text, fixture tokenizer and tiny random model with their own versions, seeds and hashes. | Check labels, masks, gradients, deliberate overfit and strict save/resume. Fixture weights cannot initialize the main model. |
| Main dense kernel (Cards 03–05) | A separately authored config, approved data and tokenizer, and fresh random initialization. The proposed 341,885,952 parameter shape needs native recount. | Bind and verify actual inputs, inspect the effective config, then obtain separate fit and training approvals. A schema-valid declaration is not a trained or capable model. |
| Strict same-experiment resume | A verified finalized checkpoint generation from the same model, tokenizer, config and run lineage. | Authenticate the parent and check full training state, including optimizer, scheduler, counters and RNG, with the native resume contract. The state-digest API is an exact comparison aid, not checkpoint authentication. |
| Optional dense-to-reader transfer (Card 08) | One selected, verified same-project dense checkpoint after the Card 06 controls; new reader modules have a distinct initialization identity. | Map compatible tensors, reject unexplained missing/unexpected keys, verify tokenizer identity and disabled-memory equivalence, and reset optimizer, scheduler and counters. Typed config/experiment derivation does not transfer tensors. |

No MODEL0/MODEL-0 artifact, earlier corpus, tokenizer, checkpoint, optimizer state
or approval is a prerequisite. Missing inputs block only the dependent operation;
they are not filled by another experiment. The proposed main architecture and
GPU fit remain unmeasured.

## Unresolved inputs and dependent gates

| Input or decision | Current evidence | Dependent work |
| --- | --- | --- |
| Fixture specification, seed, tokenizer and hashes | No project fixture or receipt observed. | Card 02 CPU correctness tests and Gate 0. |
| Main config and parameter recount | No project config or native inspect receipt observed. | Card 04 fit calibration and any main run. |
| Corpus selection, rights, split and realized mixture | No approved project source or data manifest observed. | Card 03 preparation and Card 05 training. |
| Main tokenizer identity and training-only provenance | No project tokenizer or digest observed. | Main preparation, training and checkpoint identity. |
| Frozen language and evidence evaluation contracts | No item manifests, split hashes or reviewed rubrics observed. | Comprehension qualification and Cards 05–09 comparisons. |
| Device, framework, storage capacity and measured fit | User-reported hardware only; no project-specific runtime receipt. | Card 04 GPU profile and later runtime. |
| Steps, tokens, time, memory/disk and spending bounds | No approved project runtime envelope. | Every active training, GPU, cloud or Campaign transition. |

The evaluation contract must cover paraphrase, reference, negation,
conditions, quantities, temporal order, inference, missing evidence, conflict,
source support and answer constraints. Freeze item IDs, source/chunk versions,
split boundaries, rubrics and denominators before using results as a gate.
Oracle, wrong, shuffled and absent evidence are distinct controls. Card 03
selects the actual data, tokenizer and evaluation identities; this document
does not approve sources or change numerical thresholds.

## Checks for the next bounded task

After review of this foundation record, Card 02 may be separately scoped to
create only fresh tiny fixtures. Its required checks are deterministic fixture
bytes and hashes, hand-checked target masks, finite intended gradients,
non-updating frozen parameters, deliberate overfit, and strict resume under the
card's stated CPU/time/step bounds. Use the native semantic probe for supplied
vector controls and the existing trainer sidecar construction. A passing fixture
will prove plumbing only; it will not satisfy the main model's comprehension,
fit or external-memory gates.
