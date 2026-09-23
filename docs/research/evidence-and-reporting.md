# Evidence and static reports

A report consumes one explicit collected report; it does not choose the latest/best result, train, load a model, generate, or evaluate cards.

```sh
sparselab study report experiments/ffn-memory/study.yaml RECEIPT.json \
  --evidence COLLECTED.json --research experiments/ffn-memory/research.json \
  --output artifacts/research-reports
```

The result is a content-addressed static bundle under the output directory. It retains supplied evidence and presents declared cards, endpoints, comparisons, calculated architecture/cache quantities, and limitations. It has no automatic winner, milestone, threshold, frontier, amortization, or monetary-cost conclusion; monetary cost is unavailable without billing or energy-meter evidence.

Without `--runs-dir`, the verification scope is `checksummed_report_only`: report values are checksummed supplied evidence, not a new experiment attestation. Supplying `--runs-dir RUNS` additionally performs local checkpoint/manifest validation and reads endpoint-bounded telemetry read-only; it does not re-collect cards. Missing local runs are unavailable, while unsafe/corrupt/mismatched local evidence is rejected rather than replaced with an inferred zero.

For research-scaffolded studies, the configured endpoint policy is the first configured limit. An endpoint is complete only when it reaches either configured `max_steps` or `max_tokens` without exceeding either cap and has complete identities. Earlier interruption is partial. Pair deltas require complete matched endpoints; partial, unavailable, invalid, and inconclusive contexts stay visible and are not converted to a score or zero. Raw card cases, including zero and negative deltas, are retained.

Bundles include `report.json`, original receipt/collected bytes, manifest, escaped static HTML/Markdown, deterministic SVG charts, and available inert inputs. They exclude model weights, prepared arrays, downloaded corpora, credentials, and database copies. Hashes establish bundle byte identity, not authenticity, license clearance, or removal of machine-local paths from copied evidence.
