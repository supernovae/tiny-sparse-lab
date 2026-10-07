# Card 02 bounded confirmation protocol after the failed cap

Status: **EXECUTED ONCE under [KML-D02](DECISIONS.md#kml-d02--bounded-card-02-confirmation-approved-2026-10-07); [A2 result](results/2026-10-07-card02-confirmation.md) READY FOR REVIEW, no remaining runtime authority**. The original
[Card 02 attempt](results/2026-10-07-card02-tiny-fixtures.md) remains FAILED,
and Gate 0 remains unverified. This proposal does not amend that result.

## One bounded confirmation

Use a fresh external root at
`/srv/sparselab/state/experiments/kernel-memory-lab/card02-confirm-20261007-v1`.
It must not exist before launch. Verify free bytes and inodes at that location
and confirm the locked CPU environment and exact test-source hashes. No old
weights or tokenizer initialize the fixture. Only new generated local text and
supplied vectors are used.

| Work | Maximum optimizer updates |
| --- | ---: |
| Fixed-batch overfit, gradients and optimizer membership | 80 |
| Native uninterrupted full run | 4 |
| Native interrupted parent | 2 |
| Native resumed child | 2 |
| Existing trainer sidecar regression, selected by exact test name | 1 |
| Supplied-vector probe and artifact checks | 0 |
| **Total planned maximum; each part charged before its work** | **89** |

One persistent SQLite ledger has `max_updates=89` and
`max_wall_seconds=120`. The wall clock begins at ledger creation and includes
test setup, data/tokenizer preparation, all training and retries, probe,
validation, evidence and checkpoint checks. No automatic retry is planned.
The project fixture tests reserve 80, 4, 2 and 2 updates before their
respective optimizer work; the separate sidecar command reserves one before
launch. A failed or interrupted reservation remains charged. Every command
uses the same ledger and the wrapper stops the process group at the deadline.
If a command fails, inspect the remaining ledger budget and preserve its
output; do not start a replacement ledger or exceed either cap. With this
exact allocation, any training-bearing retry needs a new explicit decision.

Maximum generated text remains 1 MiB and maximum selected test cases 64;
the expected fresh local JSONL is 2,226 bytes and the existing sidecar
regression generates about 1,016 bytes. CPU only, no GPU, external corpus,
cloud or spend. The previous successful project test tree occupied about
3 MiB; check actual output storage before execution and retain every attempt.

## Approved commands

Run from the repository root with the already available locked environment.
The command names below are shipped by this proposed code diff; they are not
runtime receipts. Set `CONFIRM_ROOT` to the exact path above and verify it is
new before creating it. Use a separate fresh base directory if a later attempt
is approved; pytest can remove an existing `--basetemp` directory.

```sh
set -e
CONFIRM_ROOT=/srv/sparselab/state/experiments/kernel-memory-lab/card02-confirm-20261007-v1
test ! -e "$CONFIRM_ROOT"
mkdir "$CONFIRM_ROOT"
uv run --locked --no-sync python -m sparselab.training.attempt_budget init \
  --path "$CONFIRM_ROOT/budget.sqlite" --max-updates 89 --max-wall-seconds 120
uv run --locked --no-sync python -m sparselab.training.attempt_budget run \
  --path "$CONFIRM_ROOT/budget.sqlite" -- \
  uv run --locked --no-sync pytest tests/test_kernel_memory_lab_fixture.py -q \
  --basetemp "$CONFIRM_ROOT/project-tests"
uv run --locked --no-sync python -m sparselab.training.attempt_budget run \
  --path "$CONFIRM_ROOT/budget.sqlite" --reserve-updates 1 -- \
  uv run --locked --no-sync pytest \
  tests/test_semantic_probe.py::test_real_restored_allocation_checkpoint_probe -q \
  --basetemp "$CONFIRM_ROOT/sidecar-test"
uv run --locked --no-sync python -m sparselab.training.attempt_budget status \
  --path "$CONFIRM_ROOT/budget.sqlite"
```

After the test commands, read the actual native run paths and generation
names. Within the same 120-second deadline, invoke `sparselab evidence` for
`full`, `part` and `resumed`, `sparselab checkpoint verify` on each selected
finalized generation, and `sparselab triage` for completed runs through the
same budget wrapper with zero additional updates. Do not guess checkpoint
generation IDs. Preserve the ledger, command output, native receipts and
failed artifacts under the fresh root.

## Acceptance and stop rule

The ledger must show at most 89 charged updates, with no work after the
deadline. Its reservations and the native run counters must reconcile to the
declared 80 + 4 + 2 + 2 + 1 allocation. All four selected tests must pass.
The fixed batch must reduce loss by at least 80% in 80 updates, intended
gradients must be finite, frozen parameters must have no gradients, and the
hand-checked target shift must match. The native full and resumed model,
optimizer, scheduler, RNG and counters must match exactly. Oracle, wrong
and masked vector controls must return the declared retrieval states;
disabled-memory logits must match the dense path exactly (`rtol=0`,
`atol=0`). Verify source/tokenizer hashes and all cited native checkpoint
generations. A passing command alone cannot promote Card 02: append a new
result, keep the first attempt FAILED, and request reviewer acceptance.
Gate 0 remains unverified until its own full requirements are met.
