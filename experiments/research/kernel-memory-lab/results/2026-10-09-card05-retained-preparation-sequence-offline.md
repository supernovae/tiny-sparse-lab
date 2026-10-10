# C05-T6 — retained-snapshot preparation sequence, offline qualification

**Status: READY FOR REVIEW, fixture qualification only.** This corrects the
post-C05-B13 preparation packet. B11–B13 and their spent ledgers, receipts and
working files were not changed or reused as acceptance evidence. No new
production ledger, source request, tokenizer fit, model forward, generation,
optimizer update, GPU operation or cloud work occurred.

The [corrected packet](../CARD05_BASE_50M_RETAINED_PREPARATION_PACKET.md)
requires cold verification of all four retained snapshots, reviewed admission,
one admission-bound schema-2/v3-normalizer pre-freeze project, split inventory,
reviewed family freeze, a distinct final build project, cold release and lineage
checks, and explicit acceptance of the exact release **before** supply,
mixture or prepared-input work. The pre-freeze project's native acquisition
identity remains `12604ffb8ccfe0351f6379ea0f781b3f9b45f6fb4ad9c8cfa80dd0b744f8b83c`;
the original B13 acquisition lock remains the input, not a new declaration.
The optional contract fields preserve legacy serialization when absent. The
new offline-only contract rejects transport-budget initialization, aliasing and
online acquisition; `corpus budget-status` reads an existing budget without
initializing it. The supervised, exclusive 64 MiB `corpus render-declaration`
operation covers deterministic JSON/YAML writes. Its hashes are receipts, not
semantic review. Reviewer active time and item decisions must be recorded
separately against the 8-hour preparation and 2-hour evaluation allowances;
process monitoring cannot measure reviewer deliberation.

The tiny fixture uses three source strata, one quarantined rights exception per
source, exact duplicate content, and a prior held-out family. It runs the real
native admission, split, release and protected-lineage verifiers, without
mocking their outcomes. The candidate leaves have the exact public path
`python -m sparselab --work-dir ROOT attempt run ... -- python -m sparselab
--work-dir ROOT monitor --policy PREP ... -- python -m sparselab --work-dir ROOT
LEAF`; the attempt path starts the whole-attempt monitor and owned supervisor.
The exact argv construction, phase labels and leaves are in
`tests/test_kml_50m_preparation_sequence.py` (`_native`, `_phase` and the
integration test). A `sitecustomize` tripwire disables socket connections.
The fixture's prebuilt authenticated WordLevel tokenizer is created before its
candidate attempt baseline; no fitting occurs. It permits cold verification of
the 100-target mixture and prepared bundle, without implying 50M supply.

Exact integration invocation:

```sh
timeout 300s env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked --no-sync pytest -q --tb=short --basetemp=/srv/sparselab/state/experiments/kernel-memory-lab/offline-preparation-qualification-c05-t8-20261009-2040 tests/test_kml_50m_preparation_sequence.py::test_offline_preparation_sequence_through_native_supervision
```

It passed **1/1 in 178.42 seconds** on the corrected release/config binding, with a 300-second local deadline,
4 GiB child-tree RSS, 512 MiB added workspace and 5,000 added-entry fixture
caps. The fixture root is the command's `--basetemp` path; its candidate work
root is `test_offline_preparation_seque0/work/`. The native attempt status
reports 24 zero-counter reservations and bindings for `pre_freeze_project`,
`build_project` and `release_acceptance`; all charged and actual model counters
are zero. The terminal whole and preparation owned-completion receipts each
report `returncode: 0`, `reason: completed`, `living_descendants: 0`. The
terminal preparation monitor reports `COMPLETE`, no violations, 433,102,848
peak process-tree RSS bytes and 2,132,386 peak added workspace bytes/541
entries. These are terminal-phase readings, not claimed attempt-wide peaks.

Retained fixture identity references (SHA-256): contract `9a99452ed57e306adc3557bbd51eae6798acdd36de8ddb61fff50e9909dbe586`;
admission review `970c544f9c53b1862e28fcd1f007090007b50b20d49f3250c3791abb56a23328`;
family-freeze receipt `19aedf649f369e20984b10da977ca1b8018ee62e70a61fd6a6005f45199cca99`;
candidate release manifest `dc50b9fbb4285758381d3fe4ef9a5addcf310d996fc7dc545f0f64193d908e32`;
protected-lineage receipt `cfcddaa2a2049707cadf98c4bb942644f34b1ccba223f2b9bf9edaa75fdfb6df`;
release acceptance `bf8b78dcfdf0172a49093ea16e82e7dbe8b4bbe1314d5c57e25228fe5dcd0898`;
mixture receipt `761b9748f9a4f04bab1a54839095a0915600212836700b9fc26640006d64ac1e`;
prepared-bundle `inputs.json` `462db4b44eb78f22f758f98d7e5a0b3be37f9a51ba9f8202c79b862c195d3b3b`;
terminal owned completion `5f2ffeb0744268aa0d26fd5f2a101ddb780cadcb907b347029fe70b7a4d7f7d9`.

Inspected zero-model negatives reject a missing admission, an acquisition-only
normalizer, inventory substitution, an edited draft presented as reviewed,
unsafe declaration output, a prior held-out family shifted into training, and a
prepared-input config substituted to name the prior release. The substituted
config is rejected before dispatch and creates no bundle.
The focused supplementary selection passed **39/39** in 3.19 seconds. Ruff
rule/format, research lint (`392` tracked records) and `git diff --check`
passed. No production source was processed in this qualification.

**Readiness verdict:** the corrected offline preparation *command sequence* is
qualified on tiny local fixtures. A new production allocation is still needed;
it must cold-verify the four actual snapshots and lock, conduct real independent
record/family/release review, pass rights/leakage and protected-lineage checks,
measure all three post-cleaning token floors, and cold-verify the exact
50M-position bundle before any model work. The fixture does not qualify the
large corpus, actual review decisions, ROCm, runtime limits or model quality.
The [next proposal](../CARD05_BASE_50M_RETAINED_PROPOSAL.md) permits **zero**
acquisition and retains every original scientific and resource ceiling.
