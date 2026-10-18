# Fixed study identity acceptance

Use the existing read-only `study report` route with optional fixed expectations:

```sh
sparselab study report /absolute/study.yaml /absolute/receipt.json \
  --evidence /absolute/collected-report.json --output /absolute/new-report-root \
  --identity-expectations /absolute/expected.json \
  --candidate-commit FULL_LAB_CHECKOUT_COMMIT
```

The JSON/YAML expectation record has exactly `format` (value
`sparselab-study-identity-expectations-v1`), `matrix_sha256`, `study_sha256`,
`config_identity_mode` (`current` or `legacy_default_fields`), and ordered
`config_sha256` (one SHA256 per run, at most 1000). These are fixed authored
expectations, not values to regenerate from the candidate under test.

Native report verification still checks the study, receipt and collected evidence
first, including all run identities. The optional check compares its verified
identities and receipt-selection mode against the declaration. A mismatch or
invalid record refuses before report output creation. Both flags are required
together. No historical dispatch, model loading or new identity algorithm exists.

The emitted report includes `identity_acceptance`, the expectation record and its
file hash, the actual verifier checkout commit, and the native package source
hash/algorithm. The supplied commit must match the verifier's checkout. A wheel
without Git provenance cannot claim checkout-bound acceptance. The package hash
identifies actual source bytes; the commit alone is not proof of a clean package
or installed-runtime equivalence. `verifier_file_status` describes only that file.

Lab CI uses neutral local fixtures offline. It does not fetch external collections
or silently skip personal acceptance. An external collection must explicitly run
this command against each candidate it intends to accept and retain the report.
Passing once at an old pin is preservation evidence, not ongoing regression
coverage. Do not remove an existing personal historical assertion until the
replacement guarantee mapping and candidate acceptance have passed review.
