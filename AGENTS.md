# SparseLab agent invariants

## 0. Default to lab mode

Most work is a **normal authorized experiment**: a local or owned-SSH run that
the user asked for, on hardware they already control, inside its resource
envelope. Use the fast path for it and do not write ceremony:

```sh
uv run --locked --extra cpu sparselab try DELTA.yaml --vs BASELINE.yaml
uv run --locked --extra cpu sparselab report TRY_ID
```

`try` derives the candidate through native `config derive`, trains both arms
with the native trainer, scores both on the same held-out split and writes one
sealed record (config, seed, code revision, data identity, resource envelope,
result). See [lab mode](docs/lab-mode.md). In lab mode, do **not** write
proposals, bindings, ledgers, allocations or stop/result documents, do not lock
ExperimentPlans, and do not request Campaign approvals. Iterate: read the
record, change one setting, try again. A lab record is evidence of one local
comparison, not a release, promotion or scientific conclusion.

For a new user, follow the [TinyStories lab walkthrough](docs/tinystories-microlab.md):
prepare the three sample inputs, then use `try` and `report`. Keep advanced
continuation, matrices and release instructions in their own guides. Run IDs
belong to `probe`/`explore`; record IDs belong to `report`/`compare`. Standalone
`probe` inherits the saved backend unless explicitly overridden; the CPU extra
installs a framework but does not override a saved CUDA/ROCm backend.

Every try also runs the fast [probe battery](docs/probe-battery.md) tier; use
`sparselab probe RUN --vs BASE --backend cpu --tier standard --json`
(or `--tier full`) for more. Act on
`probe.verdict.action`: `abandon` (a hard probe failed, including a NaN/inf
loss reported as a numerical failure: drop or fix the idea),
`tweak` (change one setting; `suggestion` names what the failing probe
implicates), `rerun` (the battery is incomplete: `verdict.missing` names each
check that produced no evidence, such as lm-eval not installed, a probe error,
or a cancel/signal/resource/OOM stop; fix it and re-run, never read it as a
pass), `escalate` (re-probe at `next_tier`), `longer_run` (all tiers pass and loss
improved: worth a larger budget, still in lab mode), `compare` (add `--vs`) or
`report` (a completed `--final` verdict: terminal, report it as is; a final
battery with any missing evidence, with or without `--vs`, says `rerun` the
same `--final` unchanged unless a NaN/inf or hard failure decided it).
Probes are screening signals, not proof that an idea is useful. Selecting on
the held-out probes again and again turns them into development data, so any
claim that an idea is better needs a separate, untouched final evaluation
(fresh split, longer run, more seeds). Iteration verdicts use only the held-out
item split. The **held-back final split** is that untouched evaluation for
probe items: never select on it. Only `sparselab probe CANDIDATE --vs BASE
--tier standard --final` reads it (`try` and ordinary probes cannot; a test
enforces this), once, for the final verdict on a candidate you have already
chosen; never use it to compare or rank candidates, and never re-run it after
changing the idea. Final results never enter ordinary `compare`, enrichment,
Pareto or dashboard history; `sparselab compare --final A B` reports them
side by side only. A single-seed "beyond noise" loss gain is within-run eval
noise only: confirm it with paired seeds before claiming a win. A try whose
comparison is `NOT_COMPARABLE` is never promoted, whatever its probes say. Do not edit probe
items, thresholds or prompts to make an idea pass, and do not iterate against
the dev split: `guard.overfit_suspected` means you are fitting the probes; change
the idea instead. Any suite change needs a `SUITE_VERSION` bump, and results are
comparable only within one suite digest.

To place a result on a known curve, run `sparselab compare RESULT
--references` (records only, no compute). It compares a metric only within one
comparison group (held-out loss: same `eval_group`; lm-eval: same
`benchmark_group`; fact recall: same `item_group`) and otherwise reports
`not_comparable` with the reason or `missing_evidence` with the command that
produces it; never compare such pairs by hand. Reference models (`ref:NAME`)
are scored on the text-level probes only (both fact recalls and lm-eval;
`sparselab probe ref:NAME --tier full`, optional `reference` + `lmeval`
extras) and are never a `--vs` baseline. `--lm-eval-tasks`/`--lm-eval-limit`
make a new benchmark group, so such results never compare with the default
slice.

To look inside a small checkpoint (≤ 60M parameters), run `sparselab explore
RUN [--text "…"] [--json]` (CPU by default; `--backend`/`--runtime`,
`--resource-envelope` go through the same runtime preparation as `probe`). It
reports the architecture, per-token loss and top-k, attention maps, weight
statistics, expert routing and memory lookups, cached under `LAB/explorer`. It is descriptive, not a verdict.
`sparselab dashboard` shows the same data for people (Home, Experiments,
Models, Behaviors, Explorer; see `docs/dashboard.md`).

Write proposal, binding and stop documents **only** for:

- **release runs**: results that will be promoted, published, cited as a
  scientific milestone or used as a parent for a release family; these keep the
  full ExperimentPlan/Campaign path with approvals, locks and reconciliation;
- **paid compute above an explicit budget** the user has stated (rented GPUs,
  hosted providers). Spending always needs the user's explicit authorization;
  never infer it from a plan, a missing receipt or an earlier allocation.

If you are unsure whether a run is lab or release, run it in lab mode and say
so; promoting a result later is a separate, reviewed step.

## 1. Establish scope before execution

Read `README.md`, relevant nearby guidance, `TODO.md` for code work, and
`docs/research/` for scientific work. Inspect Git state/history before editing;
preserve other work and use an isolated checkout when experiments have pinned
another checkout. Distinguish code, fixtures, experiments and evidence.
Kernel Memory Lab *release* work also requires its
[bootstrap](experiments/research/kernel-memory-lab/BOOTSTRAP.md), current status
and selected scope; its historical allocations grant no runtime or spending
authority for new release work.

## 2. Use the native, current public path

Reuse native dispatch, renderers, verifiers and monitors described in
[the iteration guide](docs/iteration.md); `try`/`report` are the native lab
path. Keep one current public command/schema path; update callers together
instead of adding executable version forks, compatibility aliases or a second
runner. Necessary historical readers belong in private, read-only compatibility
code. Do not reimplement hashing, checkpoint selection, accounting or workflow
state in task scripts. Before building new training, evaluation or quantization
machinery, check existing libraries (nanoGPT/modded-nanogpt, litgpt,
lm-evaluation-harness, llm-compressor) and wrap them. Record a genuinely
missing operation with inputs and failure criteria in `TODO.md`.

For release corpus preparation, use the
[canonical corpus-to-bundle interface](docs/corpus-preparation.md) and its
authenticated phase map; lab mode may use any dataset already available
locally, with its digests recorded in the try record.

## 3. Preserve identities, evidence and scientific settings

Never rewrite historical hashes, normalizer identities, receipts, failed runs or
acceptance criteria to fit new code. Preserve negative, censored, interrupted and
unavailable observations, including interrupted lab tries. A changed protocol
needs a new identity; promotion requires review. Paths are locations, not
scientific identities. Keep preparation, updates, evaluation, generation and
reporting measurements separate. Do not silently change effective batch, data,
tokenizer, seed, architecture, precision, token budget or optimizer: a lab delta
states its change explicitly. Never change held-out evaluation data or settings
inside a comparison; `try` refuses such comparisons as `NOT_COMPARABLE`.
Fixtures establish wiring only; successful commands are not scientific
conclusions.

## 4. Bound resources and retain owned outputs

Use a persistent work root outside Git: global `--work-dir` overrides
`SPARSELAB_WORK_DIR`; otherwise use absolute nonempty `XDG_DATA_HOME/sparselab`
or `~/.local/share/sparselab`. Lab tries live under `WORK_DIR/lab`. Respect
explicit legacy locations. Only disposable fixture scratch belongs in `/tmp`.
Check bytes, inodes and expected growth before expensive work; pass a
`--resource-envelope` for anything large. Do not add arbitrary short wall-time
limits to normal experiments; use `--max-wall-seconds` only when the user asks
for one. Stop runs safely (Ctrl-C, SIGTERM or the printed `CANCEL` file), never
by killing a process mid-checkpoint. Never mutate an active run, reset a spent
ledger, prune unrelated data, or infer safe shutdown from a missing receipt.
Check in only small durable declarations, summaries and evidence references.

## 5. Verify the changed behavior and its actual callers

Use the locked uv environment (`uv run --locked ...`), repository Ruff settings,
and the documented vendor environment for accelerator work (`--no-sync` where
already provisioned). Do not install packages or create another environment
without authorization. Host OS does not identify the accelerator. Inspect test
fixtures and transitive calls before choosing a bounded explicit test selection;
expand only for a changed behavior or required gate. While iterating, run
the fast suite (`-m 'not slow and not mps and not mlx and not cuda and not rocm
and not xpu and not network'`) plus targeted slow tests for the touched area;
leave the full suite to the nightly workflow (docs/test-speed.md). Prefer deterministic
identities, counters, transitions and failure checks to wall-clock assertions
(the generous lab-loop timer is the deliberate exception).

Changes to release corpus preparation keep connected coverage: load the
checked-in production templates, use public baseline rendering/initialization
and canonical phase arguments, and reach a cold prepared bundle on tiny
fixtures, declaring fixture substitutions explicitly.

Before pruning a test, map its useful guarantees to retained coverage,
including provenance, zero-network reuse, review/sample binding, protected
families, collisions, accounting and shutdown. Review `git diff --check`, Git
status and current documentation examples before handoff; report exact passed,
failed and unrun checks.

## 6. Keep changes and collaboration reviewable

Use one agent for small tasks; substantial independent work may use parallel
helpers with narrow owned files and concise evidence-backed handoffs. Avoid
overlapping device jobs. Let the user choose models, reasoning effort and
concurrency. Keep local commits narrow; publishing, history rewriting and remote
deletion require authorization. Lab records already capture commit, dirty state
and effective config; release runs additionally run from a fixed tested
revision. Keep scientific milestones in research records and missing code in
`TODO.md`.
