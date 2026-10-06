# MODEL-1: equal additional base exposure

Status: **PRE_MODEL1_BLOCKED**. No MODEL-1 attempt, approval, optimizer execution,
checkpoint, evaluation outcome or promotion exists. This protocol records intent
before outcomes; it does not change any MODEL-0 declaration or evidence.

## Question and intervention

Does another equal amount of base exposure improve this exact undertrained model
when all other scientific settings remain fixed? The native
[ExperimentPlan](model1-plan.yaml) uses `extend_budget` from the accepted terminal
MODEL-0 full-state generation, not fresh training or weight promotion.

| Setting | Accepted parent | Proposed cumulative child |
| --- | ---: | ---: |
| Optimizer updates | 5,525 | 11,050 |
| Committed supervised targets | 45,260,800 | 90,521,600 |
| Original optimizer decay horizon | 5,525 | 5,525 |

Additional budget: exactly 5,525 updates and 45,260,800 supervised targets. This
reaches approximately the existing preregistered `3D` boundary with whole-update
accounting. The historical D measurement and its receipt are not recalculated.
The only authored phase patches are `training.max_steps` and
`training.max_tokens`; the original explicit `optimizer.decay_steps: 5525` is
inherited. Warmup is not restarted.

## Immutable parent and invariants

Parent generation: `step_00005525_gen_000004`.

Parent checkpoint SHA-256:
`52aba43c269204ada5ce4101414f71aa7a8bbe116d5d1d58780eb2e2717d2709`.
Its exact persistent location is bound in both the Plan and
[Campaign](model1-campaign.yaml). Native checkpoint/iteration verification reports
valid, terminal, format-2 full state at step 5,525 and 45,260,800 targets. The
retained MODEL-0 readiness remains `READY_FOR_NEXT_STAGE`; it is not a MODEL-1
result.

Native continuation comparison reports configuration compatibility and these
scientific differences only: the two cumulative caps and derived effective target
exposure. Tokenizer and logging paths differ from the saved worker-local paths;
these are existing operational location bindings, not new tokenizer bytes or
training settings. The parent-owned tokenizer is authenticated against its run
inventory before accepting an identical-byte location change.

Keep the 69,317,760-parameter dense architecture, v5 release/export identities,
32,768 tokenizer, prepared-data identity, sequence length 1,024, microbatch 2,
accumulation 4, effective batch 8, BF16, ROCm device 0, transformer-block
activation checkpointing, seed 42, optimizer family/hyperparameters and original
scheduler horizon. Preserve full model/AdamW/scheduler/cursor/RNG/counter state
under the existing continuation contract. No reseed, reset, source-data change,
SFT or promotion is authorized.

The Plan binds the same [heldout suite](suite.yaml). The Campaign binds the same
[descriptive panel](generation-panel.json), ordered prompts/decoder and
[readiness policy](readiness.yaml), using the native `generation_panel` stage.
All four Plan retention settings remain true, matching MODEL-0. MODEL-1 uses its
own persistent execution workspace; it must not write into the parent run.

## Native evidence and immutable authorization boundary

The tested infrastructure commit is
`c1ce31263b3089265b8698a127800285a1cb208e`, pushed before runtime inspection.
Actual package SHA-256:
`295e4eb1a2ce0a97f5a3706e7ac47b169ffe03b618ded73e3ba3301357acd242`.
Mutable command evidence is retained under
`/srv/sparselab/state/experiments/pre-model1-native-iteration/`; see the compact
[preparation reference](model1-preparation-reference.json).

- Native `bind-inputs` published this full Plan from the unchanged MODEL-0
  RunConfig and authenticated existing inputs: five hits, zero misses/records,
  44.66 seconds, no tokenizer fitting or data preparation.
- Native Plan validate/inspect succeed. Native lock publication fails with
  `prepared cache does not bind the locked dataset/tokenizer/packing/source`.
- Native `iteration check` authenticates the exact full-state parent and
  mechanically checks the cumulative-budget delta, but returns `BLOCKED`.
- Parent execution source is
  `2f911e6d2a833e748971bd3621f95307b05ac8e536e23ebcc447368a84a88b4f`;
  prepared inputs bind historical source
  `b58975066ea9625d6d6a71f1b1835fd2d8e3320dc840bdceb83a3951c4df38af`.
  Neither is the tested current package. Verifier-proof eligibility is not
  execution-source authorization.
- No eligible, independently reviewed source-compatibility record for this
  current revision has been authorized. Such an operational lock/cache mapping
  would still **not** authorize full-state optimizer source drift: the existing
  budget-continuation contract requires the exact parent execution source.
- No MODEL-1 lock or scientific SHA exists. Do not replace either with a YAML
  byte digest or a guessed future identity.
- The intended ModelFamily edge is `model-0 -> model-1`, exposure budget only,
  with all invariants above. A valid child node requires a genuine pinned Plan
  lock SHA. That prerequisite is unavailable, so no placeholder Family node is
  published and the existing MODEL-0 node is unchanged. This is not a completed
  Family declaration closure.

A reviewed decision/native source policy capable of this cross-source
full-state continuation, or a separately reviewed exact-source workflow, is
required before resolving a child lock. No currently supported approval flag
removes this blocker. Do not silently broaden `OperationalSourceCompatibility`
or use `allow_runtime_drift` to bypass it.

## Runtime and storage gates

After the tested commit/push, registered `rocm-7900xtx` doctor reports `READY`,
RX 7900 XTX and PyTorch `2.13.0+rocm10.0.0`, loading this checkout's package.
One native `stage --through inspect` records the unchanged 69,317,760-parameter
shape and zero steps/targets. Inspection runs in the CPU control environment;
its report explicitly notes the CPU framework's unavailable HIP execution.
It is **not** accelerator warmup, device-fit proof or pilot acceptance.

A new current-source full-shape smoke/warmup is required by the native iteration
gate. It has not run. Resolve source/input authorization first: the ordinary
public `stage --through warmup` path prepares inputs, and launching it now could
create a new source-bound prepared dataset rather than reuse the frozen input.
No repack or source-policy bypass is authorized by this protocol.

Native workspace preflight reports adequate capacity for the unchanged parent
RunConfig: 705,353,482,240 available bytes, 64,981,199 available inodes and
331,056,403,356 projected bytes. This is not MODEL-1 cumulative-retention
headroom proof. After the genuine child lock exists, use native `export-config`
for `main:single`, then run `inspect` and `workspace preflight` on that exact
export before the one required current-source accelerator pilot.

## Campaign state and next commands

Campaign declaration SHA-256:
`738895791668fb90f85865b9bbc796e57540e9828cd7908d1ba3bfe912c556b2`.
Native status/next/explain report zero attempts. Parent/release/export/tokenizer/
prepared stages are `READY`; their dependent stages remain `BLOCKED` because no
stage receipts exist. Native `next` selects `apply` for `parent`; it does not
claim the child is dispatchable.

Read-only inspection:

```sh
uv run --locked --no-sync sparselab --work-dir /srv/sparselab/state \
  iteration check experiments/research/devmind-pretrain-v5/model1-plan.yaml \
  --parent /srv/sparselab/state/campaigns/devmind-v5-model0/417d5245f753873012f2d29b0c31a366bbe586d7ed9f12ef067456f3ad192fda/experiments/plan-devmind-v5-model0/controller/762c4104-3a3b-4226-afa6-402cc350356e/checkpoints/step_00005525_gen_000004 --json
uv run --locked --no-sync sparselab --work-dir /srv/sparselab/state \
  campaign next experiments/research/devmind-pretrain-v5/model1-campaign.yaml --json
```

The next Campaign mutation, **not run in this task**, is `campaign apply` without
`--execute-runs`, to authenticate deterministic references. It will not solve
the source blocker. Subsequent prerequisites are a source-authorized genuine
lock, valid Family child pin, exact-config capacity/pilot evidence, fresh
Campaign runtime acceptance, and explicitly bound human approval.

Only after every prerequisite is satisfied is the eventual training command:

```sh
uv run --locked --no-sync sparselab --work-dir /srv/sparselab/state \
  campaign apply experiments/research/devmind-pretrain-v5/model1-campaign.yaml \
  --runtime rocm-7900xtx --execute-runs --json
```

That command is conditional, not authorized or executed here. A later model is
not automatically superior: retain negative/mixed heldout and descriptive
outcomes, and make any advancement decision through the unchanged reviewed
readiness/lifecycle contracts.
