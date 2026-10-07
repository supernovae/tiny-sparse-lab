# Kernel Memory Lab agent pack

Repository edition 2026-10-07.2 · 7 October 2026

**Planning only.** These guides are now proposed in this repository; no campaign declaration, fixture, dataset, model, reader or run is created by this documentation change. The future [calibrated expert track](EXPERT_TRACK.md) and [source catalog](SOURCE_CATALOG.md) extend the north star without changing the foundation gates.

Use this directory as durable context for one proposed fresh TinySparseLab experiment. It includes the complete research plan and all thirteen execution cards, with practical Codex CLI guidance for the WSL2 GPU PC. No experiment has been implemented or run by this document task.

## Start here

1. Open the WSL2 distribution containing the project's Linux environment. Prefer a checkout under `~/code/tiny-sparse-lab`, not `/mnt/c`.
2. Read the repository's existing `AGENTS.md`. This pack is located at `experiments/research/kernel-memory-lab` on this documentation branch. After checkout, verify the revision and actual files before proceeding. Do not replace root instructions or copy a prior experiment's data or weights.
3. From WSL, check `echo "$WSL_DISTRO_NAME"`, `pwd` and `command -v codex`. These identify the context and executable, not GPU readiness. If setup is absent, use the [official WSL guidance](https://learn.chatgpt.com/docs/windows/wsl) separately.
4. Substitute the real repository path in the initial command:

```sh
codex --cd "$HOME/code/tiny-sparse-lab" --sandbox read-only --ask-for-approval never "Read AGENTS.md and experiments/research/kernel-memory-lab/BOOTSTRAP.md. Reconcile project state without edits, installs, downloads, GPU work or campaign apply. Report missing evidence and one next task for approval."
```

The command is a template, not something this task executed. The initial shell sandbox is read-only; the `never` approval setting disables escalation requests. The prompt also forbids mutations through other tools. For later explicitly approved work, choose the appropriate scoped write and approval settings in a separate launch. A tool's permission mode is not a training or spending budget.

## What to read

- `BOOTSTRAP.md`: the first-session and fresh-session protocol
- `CONTEXT.md`: stable intent and constraints; read every session
- `RESEARCH_PLAN.md`: the full theory, experiments, criteria and bibliography; read in full initially and reread affected sections after changes
- `EXECUTION_WORKBOOK.md`: the full thirteen-card backlog, exact prompts and approval templates; select one card at a time
- `STATUS.md`: the single current execution checklist after adoption
- `DECISIONS.md` and `results/`: decisions and append-only evidence entries
- `SOURCE_AUDIT.md`: what current main already provides and what remains unimplemented
- `REVISION_LOG.md`: synchronization and accepted document changes
- `EXPERT_TRACK.md`: proposed uncertainty-calibrated expert behavior, training stages and evaluation gates
- `SOURCE_CATALOG.md`: candidate papers, datasets and bounded generators with explicit rights gates
- `PACKAGE_PROVENANCE.md`: source-release checksums and documented repository adaptations
- `AGENTS_POINTER_PROPOSAL.md`: rationale for the minimal root pointer; not an instruction file

## One working record

The complete plan and thirteen-card workbook originated in the synchronized 2026-10-07.1 Markdown/DOCX release. This repository edition adds adoption notes and the future expert track; see `REVISION_LOG.md` and `PACKAGE_PROVENANCE.md`. The DOCX reading copies remain in the owner’s Library at the source revision: research plan version 2 and workbook version 1. They are historical reference snapshots, not a second live backlog or a mirror of the new expert track. Repository research policy rejects binary document payloads, so only reviewable Markdown is checked in. Reconcile future Word edits into reviewed Markdown and refresh reference documents from the accepted repository revision when needed. Native campaign state, run receipts and actual artifact hashes establish execution facts; `STATUS.md` indexes them and never overrides them.

The existing root `AGENTS.md` is preserved with one short task-scoped bootstrap pointer. Codex instruction discovery follows global and project-to-launch-directory guidance. A nested instruction file does not become active merely because a task mentions it. The explicit bootstrap path avoids that ambiguity. Keep any later reviewed pointer small; the default combined instruction cap is 32 KiB. Restart after accepted instruction-file changes.

## Resuming

Use `codex resume` to choose the correct session or `codex resume --last` from the same repository. A known session ID is safer when several sessions are plausible. Repeat the intended sandbox and approval flags. Reconcile the files and receipts before doing new work, even in a resumed session. Chat history is not the experiment record.

The Windows desktop app is an optional alternative: Settings → Agent environment → Windows Subsystem for Linux, then restart. Its integrated-terminal shell setting is separate. CLI operation and 24 GB VRAM do not prove Radeon/ROCm training readiness. Verify actual driver, WSL kernel/distribution, framework build and device visibility before a separately approved small GPU test.

## Official references

- [CLI flags and resume](https://learn.chatgpt.com/docs/developer-commands?surface=cli)
- [WSL workflow](https://learn.chatgpt.com/docs/windows/wsl)
- [AGENTS discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
- [Windows desktop environment](https://learn.chatgpt.com/docs/windows/windows-app)

These official pages were checked on 7 October 2026. Local setup and GPU compatibility were not inspected by this document task.
