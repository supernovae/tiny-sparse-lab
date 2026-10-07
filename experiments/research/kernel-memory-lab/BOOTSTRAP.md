# Reconstruct the project before acting

This protocol does not authorize an implementation, installation, data acquisition, GPU run, paid job, remote PR or campaign mutation. Follow the current user's exact task and the repository's existing instructions.

1. Read the active `AGENTS.md` instructions and this directory's `README.md`, `CONTEXT.md`, `STATUS.md`, `DECISIONS.md`, `REVISION_LOG.md` and `SOURCE_AUDIT.md`; for an expert-track task also read `EXPERT_TRACK.md` and the relevant entries in `SOURCE_CATALOG.md`. On a first session, read the full `RESEARCH_PLAN.md` and inspect the workbook's stage checklist. On later sessions, reread the affected theory sections and exactly one selected card from `EXECUTION_WORKBOOK.md`.
2. Verify the real repository location, branch, dirty state and revision. Compare to the documented audit `06efc4db82ecf3da97b50cff518cba605ad27b33`. Report later changes relevant to the selected card instead of assuming this audit describes an updated checkout.
3. Verify cited native Campaign manifests, status, run IDs, evidence bundles and input/output hashes. Use only verified observational commands. Missing project files at the initial state are honest unresolved inputs, not permission to inherit another experiment. A documentation claim is not a runtime receipt.
4. Report a concise state: what infrastructure exists, which project work is NOT STARTED, which evidence is VERIFIED, and which dependent task is BLOCKED. Keep a selected card separate from permission to execute it. Resolve contradictions between status prose and receipts before proceeding.
5. Propose one smallest next task, its inputs, exact acceptance gate and required scope/resource approval. Stop after the read-only assessment unless the current message explicitly authorizes that exact operation. Do not use `campaign apply` as a read-only check.

## For an authorized task

Use the workbook's handoff envelope and selected card without broadening them. Verify approved data, model, checkpoint, device, steps/tokens, wall time, memory/disk and money/credit caps. Stop at the first cap or failure condition. If approval or a prerequisite is missing, name it and stop dependent work; never invent a hash, substitute a checkpoint, fetch a new corpus or retrain a missing artifact.

Return the useful result first, with actual revision/diff, artifact paths/hashes, tests and their outcomes, native run/receipt IDs, failures, resource use, limitations and one next decision. Within authorized write scope, copy `RESULT_TEMPLATE.md` to a new uniquely named entry under `results/` and update `STATUS.md` to link it. Never overwrite an older result. A read-only session returns proposed record text without writing it. Mark READY FOR REVIEW until a real reviewer verifies the gate.

Ask Sol for correctness problems such as tensor mapping, loading, gradients, optimizer membership, numerics or device behavior. Ask Astra for an unresolved experimental decision after correctness checks, with matched ablations, error categories, uncertainty and cost. Both reviews are bounded and read-only unless explicitly requested otherwise. Neither review authorizes runtime or spend.

When the owner says "proceed" after a reviewable card result, treat that as acceptance of the completed, evidenced step and permission to carry its decisions and artifacts into the next card. Record the owner review in a new decision/result and update `STATUS.md`; do not leave that step pending review. State any unmet full-card gate separately. Acceptance of a specification, candidate, code fix or read-only preparation does not create missing source rights, acquired rows, tokenizer, measured fit or runtime receipts. Honor any separately stated acquisition, training and resource-approval boundaries.
