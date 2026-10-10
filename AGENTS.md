# SparseLab agent invariants

## 1. Establish scope before execution

Read `README.md`, relevant nearby guidance, `TODO.md` for code work, and
`docs/research/` for scientific work. Inspect Git state/history before editing;
preserve other work and use an isolated checkout when experiments have pinned
another checkout. Kernel Memory Lab work also requires its
[bootstrap](experiments/research/kernel-memory-lab/BOOTSTRAP.md), current status,
and selected scope. Planning documents and historical allocations grant no
runtime or spending authority. Distinguish code, fixtures, experiments and
evidence. An offline task permits only inspected static checks and explicitly
bounded zero-model fixtures; readiness smoke, initialization, training and
generation require their own authorization.

## 2. Use the native, current public path

Reuse native dispatch, renderers, verifiers and monitors described in
[the iteration guide](docs/iteration.md). Keep one current public command/schema
path; update callers together instead of adding executable version forks,
compatibility aliases or a second runner. Necessary historical readers belong
in private, read-only compatibility code. Do not reimplement hashing, checkpoint
selection, accounting or workflow state in task scripts. Record a genuinely
missing operation with inputs and failure criteria in `TODO.md`.

For corpus preparation, use one authenticated phase map for command arguments,
paths, bindings and template references, shared by operators and connected tests.
Capture a common-root baseline once and render its **verified embedded identity**
with public `corpus render-declaration --workspace-baseline`; initialize through
public `attempt init` only after validation. Derive output, receipt and monitor
paths through native `attempt phase-command` / `attempt run-phase`, preserving disjoint namespaces,
pre-reservation collision checks and exclusive writes. Reuse-only inputs must
fail closed without acquisition. Inspection before admission must not depend on
an admitted release.

## 3. Preserve identities, evidence and scientific settings

Never rewrite historical hashes, normalizer identities, receipts, failed runs or
acceptance criteria to fit new code. Preserve negative, censored, interrupted and
unavailable observations. A changed protocol needs a new identity; promotion
requires review. Paths are locations, not scientific identities: relocation needs
a verified binding, not edited receipts. Keep preparation, updates, evaluation,
generation and reporting measurements separate. Do not silently change effective
batch, data, tokenizer, seed, architecture, precision, token budget or optimizer.
Fixtures establish wiring only; synthetic review decisions are never production
admission, and successful commands are not scientific conclusions.

## 4. Bound resources and retain owned outputs

Use a persistent work root outside Git: global `--work-dir` overrides
`SPARSELAB_WORK_DIR`; otherwise use absolute nonempty `XDG_DATA_HOME/sparselab`
or `~/.local/share/sparselab`. Respect explicit legacy locations. Store durable
outputs by task, with backend/device in runtime metadata. Only disposable fixture
scratch belongs in `/tmp`; retain evidence separately from reconstructable cache.
Check bytes, inodes and expected growth before expensive work. Never mutate an
active run, reset a spent ledger, prune unrelated data, or infer safe shutdown
from a missing receipt. Reconcile interrupted attempts through native interfaces.
Check in only small durable declarations, summaries and evidence references.

## 5. Verify the changed behavior and its actual callers

Use the locked uv environment (`uv run --locked ...`), repository Ruff settings,
and the documented vendor environment for accelerator work (`--no-sync` where
already provisioned). Do not install packages or create another environment
without authorization. Host OS does not identify the accelerator. Inspect test
fixtures and transitive calls before choosing a bounded explicit test selection;
do not launch broad tests blindly. Expand only for a changed behavior or required
gate. Prefer deterministic identities, counters, transitions and failure checks
to wall-clock assertions.

Connected preparation coverage must load checked-in production templates, use
public baseline rendering/initialization and canonical phase arguments, and reach
a cold prepared bundle on tiny fixtures. Declare fixture substitutions explicitly;
never repair production literals silently. Before pruning a test, map its useful
guarantees to retained coverage, including provenance, zero-network reuse,
review/sample binding, protected families, collisions, accounting and shutdown.
Review `git diff --check`, Git status and current documentation examples before
handoff; report exact passed, failed and unrun checks plus remaining qualification.

## 6. Keep changes and collaboration reviewable

Use one agent for small tasks; substantial independent work may use parallel
helpers with narrow owned files, relevant invariants and concise evidence-backed
handoffs. Review the integrated diff and avoid overlapping device jobs. Let the
user/client choose models, reasoning effort and concurrency. Keep local commits
narrow; publishing, history rewriting and remote deletion require authorization.
Run experiments from a fixed tested revision and record commit, dirty state,
package identity and effective config. Keep scientific milestones in research
records and missing code in `TODO.md`; optional hardware/provider qualification
is a separate allocation, not a reason to expand an offline refactor.
