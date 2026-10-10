# Current corpus preparation interface

This directory is the sole current preparation packet. Its public baseline is
`sparselab-preparation-v1`; existing contract v1, release schema 2, admission
review v2 and `normalizer-structure-v3` retain their real artifact identities.
The parent directory's versioned packets are historical inputs, not alternative
operator routes. No production execution is authorized by this refactor.

`preparation.json` owns the 32 phase labels, argument vectors, output names,
template references and project-template path bindings. `@phase` means that
phase's leaf, always beneath the new attempt's `prep/`. There are no aliases.
The compiler injects relative project source/split/release/transform references
from those same leaves; callers cannot override them. Ordinary named bindings
are late values from verified receipts or reviewed decisions. Missing and unused
bindings fail. The compiler does no acquisition, review, artifact discovery,
execution or scheduling.

Before an authorized attempt, capture one common-root baseline with native
`monitor-baseline`. Render `attempt-contract.template.json` using public
`corpus render-declaration --workspace-baseline BASELINE`; its reserved slot is
the verified embedded baseline identity, never the receipt's file hash. Initialize
using `attempt init --contract CONTRACT --contract-sha256 SHA --ledger LEDGER
--policy WHOLE --baseline BASELINE --workspace ROOT`. The pinned production input
hashes and monitor policies require review and verification on the actual host;
do not replace an unavailable input with a guess.

Read one phase without executing it:

```sh
uv run --locked --no-sync sparselab attempt phase-command \
  --plan experiments/research/kernel-memory-lab/card05-base-50m/current/preparation.json \
  --attempt-root /absolute/fresh-attempt --label verify-snapshots
```

It returns the exact leaf arguments plus separate `leaf` (when applicable),
`completion` and `inner_monitor` paths. To execute a single phase after separate
runtime authorization, use the same declaration through the native adapter:

```sh
uv run --locked --no-sync sparselab --work-dir /absolute/common-root attempt run-phase \
  --plan experiments/research/kernel-memory-lab/card05-base-50m/current/preparation.json \
  --attempt-root /absolute/fresh-attempt --label verify-snapshots \
  --bindings-json '{}' --ledger /absolute/fresh-attempt/attempt-ledger.sqlite \
  --workspace /absolute/common-root --policy /absolute/whole-policy.yaml \
  --preparation-policy /absolute/preparation-policy.yaml \
  --baseline /absolute/fresh-attempt/workspace-baseline.json \
  --content-identity-sha256 VERIFIED_CONTENT_IDENTITY
```

`run-phase` builds the existing `attempt run` → native monitor → leaf invocation
with zero counters. The existing dispatcher still validates contracts, source
bindings, output collisions, reservations and owned shutdown. It dispatches only
one explicitly selected phase; it never advances or approves the next phase.
The low-level `attempt run` remains for other native work. The old `phase-paths`
CLI and writable `python -m sparselab.training.attempt_budget` CLI are removed.

Render phases default to their checked-in template and an empty value map. Supply
`--bindings-json '{"VALUES_JSON":"{...}"}'` with exact template slots after
review; the native renderer remains the only writer. `TEMPLATE` overrides are
explicit alternate fixture declarations, not silent production repairs. Build
and release paths, tokenizer origin, reviewer decisions and hashes remain
late-bound, never synthesized by the compiler. `attempt bind-artifact` still
seals pre-freeze/build/release acceptance at their existing gates.

The connected test loads this packet, uses tiny offline snapshots and a prebuilt
WordLevel vocabulary, explicitly replaces fixture identities/resource limits,
and follows the public path to cold bundle verification. Synthetic decisions
prove binding and coverage only. Real source admission, family review, supply,
mixture/bundle qualification, reviewer independence and all model/hardware work
remain unqualified. B11–B16 evidence is unchanged; B16's outer receipt-based
shutdown remains unverified. B17 has no production ledger or consumed allocation.
