# Card 03 Scoutflo pinned metadata inventory

This is a **metadata-only** inventory for the approved scaled preparation
attempt. It records the exact source file selection but does not assert that
any file's contents have been acquired, screened, admitted or released.

- Repository: `https://github.com/Scoutflo/Scoutflo-SRE-Playbooks`
- Commit: `acc55da0224fda5eda580939a15bc60990e8e69f`
- Commit tree: `1e77048f12715248fd7a1cab6c4d0bdc4eb2ec86`
- Recursive tree metadata: `truncated: false`; 511 blobs in the repository.
- Selection: every Markdown blob beneath `AWS Playbooks/`, `K8s Playbooks/`
  and `Sentry Playbooks/`, plus root `LICENSE` and `README.md`. No other
  repository path is selected. The [source declaration](corpus-scale/sources/scoutflo.yaml)
  pins every selected path, Git blob OID and exact byte length.

| Selection | Blobs | Declared bytes |
| --- | ---: | ---: |
| AWS Playbooks | 159 | 837,127 |
| K8s Playbooks | 245 | 1,097,884 |
| Sentry Playbooks | 27 | 122,267 |
| Root LICENSE and README.md | 2 | 30,225 |
| **Total** | **433** | **2,087,503** |

The declaration's SHA-256 is
`e73f0f00c997cacb7130115e6ba01815237d2778a8c8a4228b99e85a84a979f9`.
The official [commit metadata](https://api.github.com/repos/Scoutflo/Scoutflo-SRE-Playbooks/git/commits/acc55da0224fda5eda580939a15bc60990e8e69f)
response was 1,633 bytes (SHA-256
`265a69ebc7a35f29762d392271e82c170990aa739bdecedf3b313ec990c3f3a8`);
the [recursive tree metadata](https://api.github.com/repos/Scoutflo/Scoutflo-SRE-Playbooks/git/trees/1e77048f12715248fd7a1cab6c4d0bdc4eb2ec86?recursive=1)
response was 156,989 bytes (SHA-256
`a5e695771b386fbaa5caeba3633cd323f29403a076debfa1f1b1eab2c4fbde93`).
These metadata bodies were fetched for preflight; no source blob body was
requested, and no acquisition ledger was initialized.

The pinned [LICENSE](https://github.com/Scoutflo/Scoutflo-SRE-Playbooks/blob/acc55da0224fda5eda580939a15bc60990e8e69f/LICENSE)
states MIT. That repository claim is source-level rights context under accepted
policy v1, subject to source-policy review and file-level exception screening.
Individual third-party, privacy, secret, or coverage exceptions remain
quarantined without blocking other covered files; a source-wide conflict stops
this source. Exact blob IDs do not prove file-level eligibility.

An offline mock of the native bounded-Git path with 433 blobs and paths
containing spaces passed in `tests/test_kml_scoutflo_scale.py`: all 433 files
verified, the source body charge equaled the mocked sum, the ledger contained
433 complete transfers, and metadata was charged separately. It exercised no
network, rights decision, or real content transfer.
