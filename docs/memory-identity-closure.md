# Memory-injection historical identity: retirement prerequisite

At Lab `2a0342de653b766f87b2297e12c4fb9f741bfb6d`, all eight memory-injection
assets and four shared context inputs remain unchanged. The
[exact closure record](memory-identity-closure.json) verifies each against landed
Workshop `b74dff53259343aafd336cd00e3480246d70133b` by original bytes, SHA256,
size, Git blob and preservation manifest. This is not an asset retirement.

The existing historical-identity test needs seven files: the study, its matrix,
the shared base config, the three shared context cards, and the two-hop card.
Native `plan_study`, `_historical_study_identity` and `_identity_for_receipt`
reproduced the original matrix/study identities and all nine config digests from
unchanged Workshop copies outside both repositories, from an unrelated cwd.
No YAML references or historical digests were rewritten. Dataset/tokenizer
payloads were not loaded, reconstructed or qualified.

## Generic coverage strengthened without replacing the historical assertion

`tests/test_study_identity.py` uses the existing neutral runtime-smoke report
fixture. Fixed matrix, study and config digest expectations now detect a reader
change even when that same reader also constructs the synthetic receipt. The
checks cover relocation, selection between current and historical identities,
rejection of either mismatched receipt digest, and refusal to erase a nondefault
training setting. No personal study declarations/cards were copied into tests.
The original personal-study identity assertion remains unchanged and still runs.

## Remaining design boundary

Deleting the seven inputs today would break the original exact-identity test.
Copying them into a test fixture directory would retain the personal study under
another path; the neutral vectors alone do not prove the original personal study's
identity. Neither approach is claimed as full retirement here. The existing
historical reader remains untouched; no execution adapter or compatibility path
is added. Full retirement requires a separately accepted ownership design for
that exact historical-reader input set and assertion. The five remaining evidence
files are preserved, but retiring them alone would not resolve this input closure.

Lab #157/#158 remain partial. The next independent safe slice is neutralizing the
path-domain allocation test's corpus dependency while preserving its provenance,
normalization and accounting assertions; scientific constants and actual asset
retirement remain separate decisions. The pre-existing nightly #79 failures are
outside this change. Validation here is metadata only, not model work or a claim
of execution readiness.
