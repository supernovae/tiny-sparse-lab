# Bounded shared-host lab jobs

`lab-job` is an admission/evidence adapter over the existing Linux
`runner_control` and owned operational monitor. Its first supported payload is
**one to five explicitly named CPU pytest functions**, sufficient for the CPU
attention pilot below. It has no shell command field, scheduler, dependency sync,
worker registration, release ledger, automatic retry, or deletion operation.
Ordinary experiments still use `try` and `report`; queue work still uses native
workers. GPU admission is not implemented. A later GPU pilot remains gated by
coordinator review of the real CPU result.

## Select the immutable handoff

Before admission, the coordinator selects a **merged, validated full SHA** that
contains this adapter, checks out a clean authoring/controller checkout at that
SHA, and gives the operator a JSON manifest. `validation_reference` records the
review/CI evidence; the adapter verifies the local source SHA and ancestry in
`refs/remotes/origin/main`, but does not fetch GitHub or independently attest CI.
The selected SHA is never refreshed or substituted during admission.

All host directories must already exist, be canonical absolute paths, owned by
the operator and not group/other writable. Source, durable evidence, native
state and disposable work roots must be disjoint. Confirm actual host paths
before filling the manifest: CPU uses its existing **state-v2** directory and
`.lab-cpu.blocked.json`; ROCm uses `.lab-rocm.blocked.json`. There are no defaults,
renames, migrations, marker removal or state-directory creation. Use the exact
canonical lock already agreed for the host, shared by every contender. Do not
wrap `lab-job` in another `flock` or a legacy launcher that takes that lock.

The interpreter must already have the required packages. Record its exact Python,
Torch and pytest versions; no installation or runtime registry mutation is done.
Invocation uses that interpreter directly, isolated from Python path overrides,
and verifies the SparseLab import from the fresh checkout. Accelerator visibility
is disabled, thread counts are one, Hugging Face/Transformers offline flags and
pytest plugin-autoload suppression are set. These are execution controls, not an
OS network sandbox or protection against hostile code under the same UID.

Create a normal JSON file outside the checkout. Replace every angle-bracket
placeholder; resource values below are an example bounded recipe for coordinator
approval, not proof of available host capacity:

```json
{
  "format": "sparselab-lab-job-v1",
  "job_id": "cpu-attention-001",
  "attempt_id": "cpu-attention-authorized-001",
  "source_sha": "<exact merged validated 40-character SHA>",
  "validation_reference": "<review and exact-head CI URLs>",
  "backend": "cpu",
  "host": {
    "repository": "<absolute clean checkout at selected SHA>",
    "lock": "<absolute existing canonical CPU lock>",
    "state": "<absolute existing CPU state-v2 directory>",
    "blocked": "<absolute parent>/.lab-cpu.blocked.json",
    "evidence": "<absolute precreated durable job evidence root>",
    "work": "<absolute precreated disposable work root>",
    "python": "<absolute provisioned interpreter>"
  },
  "python_version": "<exact version>",
  "torch_version": "<exact version including suffix>",
  "pytest_version": "<exact version>",
  "nodes": [
    "tests/test_sparse_attention.py::test_sparse_backend_dispatches_cpu_and_torch_devices",
    "tests/test_sparse_attention.py::test_block_sparse_attention_is_causal_and_reports_selection",
    "tests/test_sparse_attention.py::test_sparse_matches_dense_when_every_causal_block_is_selected",
    "tests/test_sliding_attention.py::test_sliding_window_mask_excludes_distant_past_and_future",
    "tests/test_latent_attention.py::test_latent_attention_is_causal_and_shape_preserving"
  ],
  "policy": {
    "monitor_policy_version": 1,
    "max_wall_seconds": 180,
    "max_tree_rss_bytes": 2147483648,
    "min_disk_free_bytes": 1073741824,
    "min_disk_free_inodes": 1000
  },
  "max_evidence_bytes": 16777216
}
```

The two external identifiers are owner-authorized labels, not authorization
credentials. The evidence root is the local namespace for unique job and attempt
IDs: keep using the same root. Reusing either identity with changed content is
refused; a corrected or reattempted workload needs a newly authorized identity.
Manifest identity is the native domain-separated canonical digest, independent of
input JSON whitespace. Unknown fields, duplicate keys and nonfinite values refuse.

## Admit, inspect and collect

From the selected clean checkout, using its already provisioned interpreter:

```sh
/absolute/provisioned/bin/python -m sparselab lab-job admit /absolute/job.json
/absolute/provisioned/bin/python -m sparselab lab-job status /absolute/job.json
/absolute/provisioned/bin/python -m sparselab lab-job collect /absolute/job.json
```

`admit` durably reserves the job directory and unique attempt index, and seals
`manifest.json` plus `prepared.json` binding both external IDs, manifest digest,
and a native runner UUID **before any workload launch**. It creates a fresh owned
scratch directory and detached Git worktree at the selected SHA. The same binding
is in the native runner's `pending.json` before monitor spawn. No experiment,
optimizer-attempt or worker IDs are invented for a pytest payload. Native queue
`PREPARED`/claim/replay remains the contract for actual queued training; this
adapter uses the runner's exclusive pending/completion mechanism and one host
flock for the supported test payload.

A duplicate admission never reaches launch. It returns `UNKNOWN_OR_RUNNING`,
`COLLECTABLE`, or a reverified `FINALIZED` result. Partial handoffs refuse; even a
crash before native pending is written does **not** authorize retrying execution.
A claimed or uncertain launch remains for inspection. Interrupted native launches
retain pending state and, when possible, the native blocked marker. PID absence
is never evidence of completion. Existing native history, receipt-referenced logs,
lock inode, state and markers remain authoritative. Collection currently verifies
all native runner history, so incomplete unrelated native state also blocks it.

Inside the native owned supervisor, the fixed source, runtime and pytest gates
record actual exit codes and explicit executed/skipped states. Native completion
and owned cleanup evidence stay separate from those gates. Final collection checks
manifest/source identity, gate receipts, named-node JUnit coverage (including
failures and skips), final byte cap, elapsed-time cap, native receipt hashes and
cleanup. Missing evidence refuses finalization. A failed gate or skipped test
cannot become `PASS`. JSON output includes job ID, attempt ID, manifest hash,
native UUID and status. `controller_completion` is this adapter's verified
operational result; `coordinator_acceptance` remains `PENDING` even for `PASS`.

On final-publication failure, keep everything and retry **collect**, not a new
admission. `COLLECTABLE` is not acceptance. Publication is immutable and collection
has no workload-launch code path. No logs or receipts are moved into scratch.
`evidence/<job-id>/summary.json` indexes retained file hashes and sizes; keep the
entire evidence root, including the attempt index. No email bundle is required.

## CPU acceptance and cleanup preview

After external review, merge and successful CI for the chosen merged source:

1. Parent/operator verifies the host's actual lock/state-v2/blocked paths, retained
   prior receipts, free space/inodes and existing interpreter. This authoring PR
   performs no host qualification. Shellcheck is irrelevant to this Python path;
   its host availability is not inferred from the cloud environment.
2. Approve and admit the five-node manifest once within the recorded resource
   envelope. Verify `PASS`, all five named nodes represented, zero skips/failures,
   all gates executed with zero exits, native `COMPLETE`, cleanup verified, and
   final manifest/source/size/time checks true. Inspect retained stdout/stderr and
   monitor samples as well as the compact summary.
3. Redeliver the identical manifest and collect again. Confirm the same native
   UUID, same summary, and exactly one native pending/completion pair. The
   coordinator separately records acceptance against job ID and manifest hash.
4. Only then consider a separately reviewed GPU implementation/pilot. CPU evidence
   does not qualify ROCm or another accelerator.

Optional cleanup planning is **preview only**:

```sh
/absolute/provisioned/bin/python -m sparselab lab-job cleanup-preview /absolute/job.json \
  --candidate /absolute/disposable-work/scratch/cpu-attention-authorized-001
```

Only the exact owned scratch path is eligible; linked/foreign content, ancestor
paths, evidence/state/lock/marker targets and unverified completion refuse. Output
has `delete_implemented: false`. No directory, file, Git worktree or history is
removed. Retain all runner history, state, locks, markers and referenced logs;
future admission reads them. Scratch retention is required for source rechecking
in this first slice; preview does not authorize later deletion.
