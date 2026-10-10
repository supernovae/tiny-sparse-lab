# C05-T3 — offline 50M launch packet

- Record: C05-T3, 2026-10-09 UTC; code and prospective declaration work on
  `codex/kernel-base-50m-proposal`, starting from `561f2661e9f0c0a5e8e208812e91a71580d7ee18`.
- Authorization: the owner's bounded offline packet request only. No attempt
  ledger, source request, production corpus processing, tokenizer fit, model
  construction/forward, generation, optimizer update, GPU work or cloud use.
- Result: [the command and binding packet](../CARD05_BASE_50M_LAUNCH_PACKET.md)
  gives one prospective acquisition declaration, unchanged retained source
  identities, v3 release/admission/family/split/mixture/run templates,
  original tokenizer/evaluation bindings, monitor/contract caps, explicit
  phase ordering and execution-time artifact resolution. None is an actual
  new corpus admission or runtime receipt.
- Code: `evaluation fixed-slices select` verifies all eleven declared native
  checkpoints and one-batch validation events, every exact 12-window/
  3,072-target fixed-validation score, finite token-weighted losses and
  run/profile/checkpoint/config identities. It distinguishes the requested
  source config from the staged effective config, selects the earliest
  minimum, and publishes an immutable receipt before test/prose outputs.
  `verify-selection` cold-reconciles the receipt after downstream evaluation.
  A reusable read-only `corpus audit-protected-lineage` replaces the
  B7-path-hardcoded comparison, binds both releases/inventories and the
  original frozen profile/suite, and retains a BLOCKED receipt for collisions.
  The monitor CLI now exposes its existing persistent-baseline capture and
  baseline-path argument, making the separate 8 GiB preparation, 72 GiB
  cumulative and 64 MiB evaluation-output policies executable without a new
  resource engine.
- Offline tests: exact selected zero-model fixtures cover selector completeness,
  duplicates, substitutions, ordering/ties, nonfinite values, wrong identities,
  requested/effective config separation, cold receipt reuse and post-selection
  score changes; protected-lineage pass/failure and CLI binding; prospective
  source/rights/scientific/resource declarations; and mocked transport-free
  snapshot alias reuse across project IDs. Test collection and transitive
  fixtures were inspected; the mock HTTP fixture uses local bytes only.
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked --no-sync pytest -vv
  --tb=long -p no:cacheprovider` on the five new/changed selected nodes and
  the seven prepared-input CLI nodes passed **41/41**. `ruff check .`,
  `ruff format --check .`, research lint (`375` tracked files, zero errors)
  and staged `git diff --check` passed. Research lint correctly rejected a
  preliminary unresolved YAML run-config; the final packet keeps it in
  Markdown as a deliberately non-runnable template, then lint passed.
- Gate: **READY FOR REVIEW as an offline packet**, not a 50M attempt. The
  historical C05-T2 selector blocker is closed in code. Runtime still requires
  an exact-head decision, live identity/headroom preflight, one independent
  ledger, acquisition of the declared additional Wikimedia selection,
  rights/admission and family review, protected-lineage and leakage checks,
  post-cleaning unique floors, exact mixture/exposure verification and all
  proposal resource/phase stops. The prior conditional allocation is
  unconsumed; no new source, release, checkpoint or score digest is invented.
- Known limits: no real v3 release or 50M artifacts exist, and no live ROCm,
  long-run, acquisition or model-evaluation behavior was exercised. B7/B8/B9,
  all historical attempts and frozen evaluations remain unchanged; B9 is
  unadmitted and Card 06 remains blocked.
- Next decision: review this exact tested packet/head for conditional runtime
  authority; preserve the existing numerical proposal ceilings unchanged.
