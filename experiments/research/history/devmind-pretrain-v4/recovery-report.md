# DevMind v4 recovery audit

## Audit boundary

Audited after `git switch main; git pull --ff-only` in the existing checkout;
`origin/main`, local main and independently queried remote main were
`cbd4d559b600f21408797515c50f1fb6f661db7e`. Implementation uses the normal branch
`feat/lifecycle-recovery-v1`, not a new worktree. Initial tree was clean. The
process inventory contained no SparseLab training/acquisition process. Existing
workspace outputs were preserved.

`git diff e260ea866ded8ec5beb23cbacec1fb3c120c1d16 HEAD -- corpora/devmind-v4`
was empty. The recovery declaration pins that reviewed corrected input commit.
Read-only reflog/object audit found older unreachable Campaign/runtime-preparation
commits (`dd5b1d2`, `1539f35`, `d3247a7`), not MODEL-0 declarations or evidence.
Unavailable earlier conversation text is not scientific evidence.

## Durable and absent inventory

| Item | Durable declaration/evidence | Current availability and limit |
| --- | --- | --- |
| Corpus intent | `corpora/devmind-v4/corpus.yaml`, 123 source declarations, LM transform, splits and release | Checked-in declarations; source bytes are external pins, not bundled data |
| Corrected project | Historical project digest `ee95707a159d4fbc62e8f22cce6301010cbf86a500db6c010a2d282e1166e6b6` | Recorded protocol expectation; not regenerated here |
| Build | Historical digest `79b568629fc6682c33bf1212eef0653cdb05d26a772654019458f44af2ceaee5` | Mutable build bytes not verified here |
| Frozen release | Expected `0f06976e1c90309c7920cd23b1079179ba3cd026fdde55e934c1320fe6dc5fd4` | Expected legacy workspace `sparselab-work/experiments/devmind-pretrain-v4` absent; home-default v4 workspace also absent |
| Corpus measurements | `source-quality.json`, `pool-measurement.json`, `near-overlap.json`, `protocol.md` | Durable historical evidence, not fresh verification of release bytes |
| Tokenizer | No v4 tokenizer selection declaration or selected artifact observed | External scientific decision required; do not infer vocabulary from mixture metadata |
| MODEL-0 | No concrete v4 architecture, token/update budget, base run or ExperimentPlan observed | External declaration required; v1 conditional intent is not a v4 model recipe |
| Runtime | No v4 portable requirement/accepted runtime binding observed | Required declaration and operational device verification |
| Evaluation | No v4 checkpoint-bound evaluation suite observed | Required declaration before model execution |
| Checkpoints/family | No verified v4 MODEL-0 checkpoint, evaluation index or family observed | No evidence of creation; do not label a hypothetical checkpoint reconstructable |

The absent local paths prove only absence at these inspected locations, not loss
of every external copy. Lost-versus-never-created mutable state cannot be inferred
from a historical digest. The protocol records 735,281,099 retained developer train
bytes and `READY_FOR_TOKENIZER`; that corpus decision is neither model readiness
nor tokenizer/model execution. Its descriptive mixture is not a training sampler.

## Negative history and reconstruction boundary

The v4 protocol retains the failed pinned-symlink acquisition and an invalid early
build, followed by corrected historical acquisition/build/freeze. The v1 evidence
records an unauthorized MODEL-0 and a failed workstation runtime gate, no CPU
substitute and no selected tokenizer. These histories remain separate and immutable.

Deterministic prerequisites can be rebuilt only from the committed declarations,
retrievable pinned sources and recorded rights policy, then reverified against
expected domain digests. Remote source availability and rights remain external
requirements. Missing tokenizer/model/evaluation choices require explicit reviewed
new declarations. Training bytes cannot be reconstructed from plans or parent
availability: any actually lost checkpoint requires an external verified copy or
an explicitly authorized new run with a new execution identity.

The v4 release is `metadata_reconstruction_only`: a thin metadata archive can
preserve the declaration and expectations; portable source-text publication is
not authorized. No v4 corpus build, download, tokenizer fit, training, accelerator
job or weight-publication gate was performed in this audit.

## Read-only CLI observations

Executed with the absent task-owned root
`$HOME/.local/share/sparselab/lifecycle-recovery-audit`:

```sh
uv run --locked sparselab --work-dir "$HOME/.local/share/sparselab/lifecycle-recovery-audit" \
  recovery inspect experiments/research/devmind-pretrain-v4/recovery.yaml --json
uv run --locked sparselab --work-dir "$HOME/.local/share/sparselab/lifecycle-recovery-audit" \
  recovery plan experiments/research/devmind-pretrain-v4/recovery.yaml --json
```

Both exited successfully with valid JSON and empty stderr. The root and scratch
remained absent. Corpus status was `MISSING_EXTERNAL` because pinned acquisition
bytes were unavailable; the historical release digest remained **expected**, not
verified. `tokenizer-choice`, `model-zero-intent`, `runtime-requirement` and
`evaluation-declaration` were independently `MISSING_EXTERNAL`. The plan contained
no executable commands. This is an audited blocker report, not a recovered corpus,
tokenizer, MODEL-0 checkpoint or model-readiness finding.

## Lab-code verification boundary

The task's focused locked-environment regression set passed **196 tests**.
The tiny local CPU fixture commits its declarations, deletes only its own
persistent state, reconstructs the five expected release/export/tokenizer/prepared/
lock identities, and verifies immutable receipt replay. Dirty scientific input
blocks without creating a new workspace; an unrelated dirty log does not.
Campaign reconstruction shares the selected external root and does not enqueue
training. A separate explicit submission is reconciled without authorizing a
replacement; its checkpoint, declared suite and readiness result are verified.

Model-family inspection and test-only reviewed lifecycle publication were exercised
through the CLI. Explicit thin and portable tiny-fixture archives were created
and verified; updated-index omission, checkpoint-role/weight tampering and forged
action eligibility regressions passed. These are implementation checks on bounded
teaching data, not DevMind evidence, model-quality findings or scientific promotion.
Final locked full regression: **1,137 passed, 12 skipped**. Ruff lint and formatting
for all 45 changed Python files passed. At the user's request, unrelated baseline
formatting was preserved; whole-repository formatting still flags 22 unchanged
files. No real v4 payload or MODEL-0 execution is implied by these checks.
