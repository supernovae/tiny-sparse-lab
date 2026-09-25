# Architecture decision records

These ADRs preserve historical design context, not a current feature roadmap. Some later implementations extend an earlier decision without rewriting its original rationale. Use the [README](../../README.md), [runtime policy](../runtime.md), [independent-worker contract](../workers.md), and [capability backlog](../../TODO.md) for current capabilities, verified acceptance, and blocked hardware gates.

| ADR | Decision |
|---|---|
| [0001](0001-dense-first.md) | Dense decoder baseline before sparse mechanisms |
| [0002](0002-local-moe-routing.md) | Local Top-K MoE feed-forward routing |
| [0003](0003-token-ngram-memory.md) | Causal token n-gram memory before byte hashing |
| [0004](0004-byte-addressing.md) | Raw UTF-8 byte addressing as a preprocessing contract |
| [0005](0005-byte-generation.md) | Greedy byte-memory generation from the raw prompt stream |
| [0006](0006-withheld-facts.md) | Withheld-fact diagnostics as data-separation tests |
| [0007](0007-diagnostic-manifest.md) | Immutable split manifests for diagnostics |
| [0008](0008-manifest-verification.md) | Fixture-verified diagnostic manifests |
| [0009](0009-manifest-audit.md) | Separate verified split evidence from results |
| [0010](0010-trained-portability.md) | Real trained runs for byte-address portability |
| [0011](0011-sliding-window-attention.md) | Causal sliding-window attention reference |
| [0012](0012-latent-attention.md) | Reference multi-head latent attention mode |
| [0013](0013-combined-architecture.md) | Explicit composition of independent mechanisms |
| [0014](0014-scale-experiments.md) | Recorded, bounded scale comparisons |
| [0015](0015-single-host-extension-boundaries.md) | Historical local-core boundary; later extended by independent-worker scheduling |
| [0016](0016-evidence-before-claims.md) | Verified checkpoint and held-out evidence before claims |
| [0017](0017-versioned-capability-cards.md) | Versioned capability cards for cumulative claims |
| [0018](0018-chat-native-experiments.md) | Chat-native, artifact-bound small-model experiments |
