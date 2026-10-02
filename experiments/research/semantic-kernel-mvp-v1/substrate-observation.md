# Stage 0 substrate observation — 2026-10-02

**Software-only pass; no model/controller result.** The compiler and four
human-authored replay probes executed against clean source commit
`ec78fab554cd2bcf23e07b1dab601a71431fb26c`.

| Observation | Value |
|---|---|
| Source files | 199 Python files under `src/` |
| Definitions | 2,765 |
| Syntactic call sites | 37,809; targets unresolved |
| Serialized bank bytes | 13,340,967 |
| Compiler + publication component wall time | 1.5071 s on this host; one sample, not a benchmark |
| Bank SHA-256 | `36691fcdff1a6305e042a81f88e14dad718a9381e14b4c3715afbbcd4e5860d3` |
| Compiler source SHA-256 | `582aae31438b59797685f8d9d51dfc53dfc5b16a859a8d2a7a9cad61267a9912` |
| Replay SHA-256 | `bdba1c3644983ad80d1f4b975f5dcfa54e7a12708905be4ffa4f18e275fd6725` |
| Replay steps / invalid actions | 4 / 0 |

Replay found one `format_chat_prompt` definition, ten matching call spellings,
one `SemanticRetriever` definition, and no definition for the deliberately absent
probe. Independent source inspection confirmed `format_chat_prompt` at
`src/sparselab/evaluation/chat.py:30` and `SemanticRetriever` at
`src/sparselab/engram/semantic.py:117`. This is a spot-check, not complete semantic
validation or function-call resolution.

The full local bank and trace were retained outside Git under the task-owned
external work-root subdirectory `experiments/semantic-kernel-mvp-v1/ec78fab/`.
Those bytes are not bundled in this PR and their availability on another host
is not asserted. Regenerate with the README commands from the pinned commit;
wall times and replay digests containing timing will differ.

Verification: 12 focused substrate tests pass; Ruff check/format and
`git diff --check` pass. Adjacent `test_conversations.py` and `test_local_chat.py`
could not collect because this Torch-free environment has no PyTorch. No existing
training runtime, accelerator gate, pretrained import, or neural benchmark was
exercised. No source data was downloaded for this pilot.

Decision: continue to Stage 1's independently authored task bundle, explicit
local-model adapter and baseline comparisons. Do not infer useful language
ability, policy learning, semantic portability or cost superiority from this
compiler/replay observation. Bank identity checks detect changed bytes; they
are not signatures authenticating an untrusted publisher.
