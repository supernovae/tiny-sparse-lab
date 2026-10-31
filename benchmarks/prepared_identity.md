# Prepared identity regression (offline CPU)

Base: `6c31d7f9be516f2adc3ed92849bf4e8057a9169e`. The only top-level
`data/*.py` or `config/models.py` change between `b3a1630` and this base is
`sources.py` in #286. Its changed code is snapshot acquisition batching and
capacity admission; snapshot verification and document iteration are unchanged.
The former preparation fingerprint includes that entire file, including for
synthetic datasets which never call acquisition.

Bounded reproduction used `tests/test_training.py::config` (synthetic CPU data,
no training), the same tokenizer and preparation implementation, and injected
each revision's original preparation fingerprint. The fingerprints were computed
from `git show` source bytes with shared `sha256_file` and `canonical_digest`,
using the former sorted `data/*.py` plus `config/models.py` inventory algorithm.
This isolates identity selection; it is not an execution of two historical
packages, a real dataset benchmark, or confirmation of the reported 122 seconds.

| Revision inventory | Preparation fingerprint | Cache directory |
| --- | --- | --- |
| `b3a1630` | `cdf175840a00033853e5d9f32711ec2cb8592b85c56aff1f8f7af1fb4522972b` | `2a0a2ddf95576c35` |
| `6c31d7f` | `dfc29d53186d7d9bd4d40d7e0ac64f6c348482930830881ed2c34f929d2c1ad8` | `cb69db160bd56ce8` |

Both preparations produced these identical file digests:

- `train.npy`: `654e9b2d4e4f240b536c79095c3d85070a9711147fab34dcc3b33b989ef43f41`
- `validation.npy`: `c53b61d2eba721af0997914363ff7b7996e4c6fd5c7dafb7fe2dfbde20d95054`

The fix extracts publication/admission into `data/snapshot_writer.py` and excludes
only that module from the preparation fingerprint. Shared source selection,
snapshot verifier and iterators stay hashed; published content remains bound by
the authenticated snapshot manifest. All other data modules, including unknown
new modules, stay included. Allocation attestations retain package identity.

`tests/test_prepared_identity.py` exercises writer-only source changes, model
widths, batch settings, physical cache copies and negative invalidations using
bounded fixtures. Reuse forbids document iteration, so byte equality alone cannot
conceal repacking. Existing dataset-source tests cover interrupted acquisition,
resource stops, resumed journal equality and snapshot corruption; existing
packing tests cover modified manifests, unexpected arrays and dtype mismatches.

See [prepared identity and migration](../docs/datasets.md#prepared-content-identity-and-replicas)
for the deliberate one-time new identity and limits of this conservative boundary.
