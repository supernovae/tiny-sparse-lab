# ROCm container delivery status

The ROCm image is **not available yet**. Python 3.14 wheels exist for the
candidate below, but a complete verified dependency lock is still missing.
`container/backends/rocm/status.json` records the block; there is deliberately
no enabled image configuration or `requirements.lock`.

## Candidate for Radeon RX 7900 XTX

Metadata checked on 2026-10-15:

| Component | Published candidate |
| --- | --- |
| Interpreter/platform | Stable CPython 3.14, Linux x86-64 |
| ROCm | 10.1.0 |
| AMD PyTorch | `2.14.0+rocm10.1.0`, extra `device-gfx1100` |
| AMD device packages | `amd-torch-device-gfx1100` and `amd-torch-device-gfx110x`, both `2.14.0+rocm10.1.0` |
| AMD Triton index entry | `3.8.0+git669b31ac.rocm10.1.0` |
| Lab build backend | `uv_build==0.12.17` |

These are published candidates, not a resolved dependency closure. The planned
target is gfx1100 (RX 7900 XTX); it is not an all-AMD image or a claim of Windows
or WSL qualification. Compatible host drivers and device access remain required.
See AMD's [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

AMD's [current installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
explicitly selects `torch[device-gfx1100]==2.14.0+rocm10.1.0` from its
`stable.repo.amd.com/rocm/whl-next/` index. The directory name alone does not
mean nightly. This is the latest vendor build observed in that index, not an
assertion that upstream PyTorch's latest patch has an equivalent AMD wheel.
The optional torchvision alpha in AMD's example is not needed by Lab and is
not part of this candidate.

## What blocks publication

The official [Torch index](https://stable.repo.amd.com/rocm/whl-next/torch/)
lists the CPython 3.14 Linux wheel, without SHA256 fragments or separate
PEP 658 metadata. The corresponding `.metadata`, `.sha256`, and `.sha256sum`
URLs returned 404. Metadata-only inspection therefore did not establish its
full dependencies or a verified wheel hash. No accelerator wheel was fetched,
installed, built, or executed for this change.

The official [ROCm runtime index](https://stable.repo.amd.com/rocm/core/whl-next/rocm/)
lists `rocm-10.1.0.tar.gz` rather than a wheel. A binary-only installer must not
silently relax that rule or run an unreviewed source build. The package's actual
role and any required build treatment need verification during lock preparation.

To enable delivery, obtain and verify the complete Linux CPython 3.14 dependency
metadata and SHA256 closure, including Lab's common dependencies, AMD device
libraries, and the build backend. If the publisher does not expose that metadata,
wheel acquisition and hashing require a separately authorized packaging step.
Resolve the source-archive requirement explicitly. Then add the verified lock
and enabled configuration, build the immutable dependency image, and smoke-test
the thin Lab layer before publication. Do not substitute Python 3.13, a nightly,
or a CPU wheel to make the build pass.

## Existing Lab support

Lab already implements ROCm through PyTorch's HIP-backed `cuda` device API and
records the backend as `rocm`. An unavailable explicit backend refuses; it does
not retry on CPU. See [runtime selection](runtime.md).
The existing `rocm-gfx1100-v1` provisioning recipe remains a distinct ROCm 10.0.0
candidate; this document does not upgrade or qualify that environment.

After a future image is qualified, use native `sparselab try`, `report`, and
`probe` with the same persistent state conventions as the [CPU image](containers.md).
No new experiment runner or backend implementation is required. No ROCm hardware
test, GPU workload, or image build was performed for this metadata-only change.
