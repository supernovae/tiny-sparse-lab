# Stable research context

Repository edition 2026-10-07.2; foundation source revision 2026-10-07.1. This is a short index of invariants; the full rationale and definitions remain in `RESEARCH_PLAN.md` and the full cards in `EXECUTION_WORKBOOK.md`.

- Goal: a compact language kernel that understands wording, nuance, relationships, constraints, instruction intent and evidence. Script writing is optional, not the primary objective.
- Fresh start: no MODEL0/MODEL-0 checkpoint, tokenizer, corpus, optimizer state, runtime approval or reproduction is required. New deterministic tiny fixtures test plumbing; their weights never initialize the main model.
- Main proposal: native dense decoder with vocabulary 32,768, width 1,024, 24 layers, 16 heads, FFN 2,816 and tied embeddings, provisionally 341,885,952 parameters. Recount against the actual code/config. This is a proposed shape, not a trained model or proven GPU fit. A smaller fallback is conditional, not another mandatory training project.
- Local environment: user-reported Radeon RX 7900 XTX with 24 GB VRAM on a WSL2 GPU PC. Actual Windows driver, WSL distro/kernel, ROCm/framework compatibility, host RAM/storage, throughput and fit are UNVERIFIED. Codex availability does not establish GPU readiness.
- Optional cloud: any provider balance, expiry, GPU availability and burn rate must be verified before a separately approved bounded job. No spending approval is implied.
- Separate fresh initialization, strict same-experiment resume, typed config or experiment derivation and the proposed explicit dense-to-new-reader transfer. Config and experiment derive author fresh validated declarations; they do not transform checkpoint tensors or prove transfer equivalence.
- Order: fresh fixtures and data/evaluation contracts → measured shape/fit → bounded language training → raw-text oracle evidence → lexical retrieval → explicit transfer and integrated reader → optional router/index experiment → frozen-weight knowledge update and same-store RAM/NVMe comparisons.
- Reader learning and router learning are separate. No frozen random model may be called a capable reader. Correct/wrong/missing evidence controls and held-out comprehension gates remain mandatory.
- Preserve raw source provenance, stable chunk IDs, versions, tokenizer identity, exact hashes, split boundaries and causal query construction. Evaluate inserted and corrected facts with frozen weights.
- Reuse shipped native infrastructure and record actual Campaign/run receipts. It does not prove this experiment's capability. The source audit is pinned to `06efc4db82ecf3da97b50cff518cba605ad27b33`; recheck newer code.
- All thirteen project cards start NOT STARTED. No GPU run, model training, evaluation, installation, corpus acquisition, cloud dispatch or PR is authorized by this pack.
- One card and one approved scope at a time. Proposed numeric ceilings are not approval. Technical verification, research judgment and human permission are separate.
- Stop on a missing artifact, identity mismatch, nonfinite loss, OOM, first resource cap or predeclared regression. Report failed attempts. Do not repair a missing checkpoint by retraining without approval.

## Future behavior north star

A later expert/cognitive-systems-engineering track should scale effort to uncertainty, risk and reversibility, question consequential assumptions, verify recovery and evidence, and keep simple tasks simple. See `EXPERT_TRACK.md` and `SOURCE_CATALOG.md`. It does not displace language/comprehension foundations, authorize training, or establish current competence. External stored facts cannot replace learned comprehension or judgment.

## Durable state

After adoption, this Markdown directory is the versioned operating record. `STATUS.md` is the sole current checklist and points to append-only results/decisions plus native receipts. The DOCX files remain editable snapshots of the 2026-10-07.1 source release; the repository edition and new expert track are recorded in `REVISION_LOG.md`. Reconcile any human Word edits through a reviewed revision before changing the accepted theory. Never promote state based only on chat recollection.
