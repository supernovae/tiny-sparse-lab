# C05-T7 — verified baseline binding before attempt initialization

**READY FOR REVIEW; offline code and fixture evidence only.** Starting from
clean `e7d8d72654e0121392282793fd901828643682a9` on
`codex/kernel-base-50m-proposal`, this focused correction addresses the
operator binding error preserved in [C05-B14](2026-10-09-card05-50m-attempt4-baseline-stop.md).
It creates no production attempt, baseline or corpus result. B14's ledger remains
SHA-256 `55240d0268f438bc8b18a05d5f9469a03007b63a20e697dbc4924efb58e512da`;
its supervisor receipt remains
`2f8ac88a569b96b050d11fa8240830b7ea6016af427e54a19d6d558a3b434f82`.

Native `corpus render-declaration` now accepts `--workspace-baseline <receipt>`
for the reserved `${VERIFIED_WORKSPACE_BASELINE_IDENTITY_SHA256}` slot. It uses
`load_workspace_baseline(receipt, --work-dir)` to validate the receipt digest,
root, device and root inode, then inserts the **embedded** identity. Caller JSON
cannot supply or override that slot; missing, unsafe or invalid receipt input
fails before the rendered file is published. The existing renderer and its
exact-slot/output checks remain in use.

The shared `validate_contract_monitor_inputs` check verifies the actual policy
digest and loaded policy plus that same actual baseline against the fully
rendered contract. The v2 packet contract requires the optional
`require_preledger_monitor_binding` field; `attempt init` therefore requires
`--policy`, `--baseline` and `--workspace`, validates them before creating a
ledger, and `attempt run` repeats the same check before reservation/dispatch.
Absent optional fields preserve legacy serialization and initialization. The
packet and v2 template name the verified identity explicitly and forbid using
the baseline file hash or recapturing a mismatched baseline. No scientific or
numerical cap changed.

The selected zero-model checks used only temporary fixture roots and ledgers.
The inspected fixture has no autouse model setup; its real public dispatch runs
CLI → native attempt budget → whole monitor → preparation monitor → owned
supervisor → harmless offline transport-budget initialization. It performed
zero model updates or generations and reported zero surviving descendants.
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked --no-sync pytest -q -p
no:cacheprovider` on the eight explicitly selected nodes in
`test_attempt_packet_dispatch.py`, `test_kml_50m_preparation_sequence.py`,
`test_kml_50m_launch_packet.py` and `test_attempt_contract.py` passed **12/12**.
The five invalid pre-ledger cases covered the baseline file hash, tampered
receipt, wrong root, changed policy and changed identity; each rejected before
`ledger.sqlite` existed. Missing pre-ledger flags and attempted JSON override
also rejected; the verified binding passed actual existing dispatch. Legacy
optional-field serialization and policy drift at the child boundary passed.

This qualifies only the offline binding path and tiny fixture, not retained
production snapshot authentication, rights/admission, post-cleaning supply,
ROCm, 50M training or model quality. The next decision is review of this change
and, only if warranted, a separately bounded fresh attempt. B11–B14 stay
historical and Card 06 stays blocked.
