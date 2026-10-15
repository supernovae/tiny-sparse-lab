# ROCm image for Radeon RX 7900 XTX

This recipe targets stable CPython 3.14 on Linux x86-64 with AMD gfx1100
(RX 7900 XTX). It uses the existing Lab ROCm backend. It is not an all-AMD image
or a claim of Windows/WSL qualification. A compatible host driver and device
access are prerequisites; see AMD's
[compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

## Exact runtime pins

| Component | Pin |
| --- | --- |
| Python/base OS | Python 3.14.4, Debian Bookworm, immutable base digest in the Dockerfile |
| AMD PyTorch | `2.14.0+rocm10.1.0`, extra `device-gfx1100` |
| ROCm SDK | `10.1.0` |
| Embedded HIP build | `7.16.26385` (checked separately from SDK marketing version) |
| AMD Triton | `3.8.0+git669b31ac.rocm10.1.0` |
| AMD device packages | gfx1100 and gfx110x, both `2.14.0+rocm10.1.0` |
| Metapackage builder | `setuptools==84.0.0` |
| Lab build backend | `uv_build==0.12.17` |

AMD's [official guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
selects `torch[device-gfx1100]==2.14.0+rocm10.1.0` from the stable publisher's
`whl-next` directory. This is the latest vendor build observed on 2026-10-15;
upstream PyTorch 2.14.1 does not imply an equivalent AMD wheel. No nightly,
Python downgrade, optional torchvision alpha, or CPU substitution is used.

## Dependency reuse

The shared accelerator workflow builds this backend independently. Its immutable
dependency identity includes the backend recipe, exact wheel/source locks,
configuration, Python/OS base, and shared build inputs. Ordinary application
updates reuse the resolved dependency digest and add only the Lab application
layer. Missing dependency images bootstrap once; heavyweight dependencies are
not rebuilt just because the application source changed. See the
[accelerator image guide](container-builds.md) for publication and refresh.

`requirements.lock` contains the complete binary closure for Lab's common
dependencies, PyTorch and gfx1100 libraries. Package-specific AMD `--find-links`
entries avoid letting AMD's general index replace unrelated PyPI dependencies.
All distributions are exact-pinned and SHA256-checked.

AMD intentionally distributes `rocm==10.1.0` as a small source metapackage.
`source-requirements.lock` pins the inspected 27,536-byte archive and its SHA256.
Its build uses only the already hash-pinned setuptools, with build isolation and
dependency resolution disabled. `ROCM_SDK_TARGET_FAMILY=gfx1100` prevents its
build-time hardware discovery; no GPU is exposed. Its required SDK core and
`libraries` extra dependencies are included explicitly in the binary lock.
The Dockerfile runs `uv pip check`, verifies the exact Torch/HIP build, and uses
a tiny CPU tensor for packaging validation. This exception does not allow other
source distributions or unpinned build dependencies.

## Running Lab

After reviewed publication, use the published immutable ROCm application digest
with the [container state and CLI conventions](containers.md). On an already
configured Linux Docker host, a minimal CLI check is:

```sh
# Replace this placeholder with the published ROCm application digest.
IMAGE=ghcr.io/supernovae/tiny-sparse-lab@sha256:REPLACE_WITH_ROCM_DIGEST
docker run --rm --network=none --cap-drop=ALL \
  --security-opt=no-new-privileges --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -v sparselab-rocm-state:/state "$IMAGE" sparselab --help
```

Use the render node belonging to the intended GPU if it differs from
`renderD128`; the example assumes an existing Docker host and existing device
permissions. Rootless Podman needs the host's supported supplemental-group/device
mapping. Do not weaken host security or change device ownership to make it work.
The help command checks CLI wiring only. Actual `try`/`probe` work still needs
your own inputs and resource envelope, with `backend: rocm` in the declaration.
The separate CPU viewer can observe the same persisted evidence read-only.

Lab records HIP-backed PyTorch execution as `rocm`; an unavailable explicit
backend refuses rather than silently retrying on CPU. See [runtime selection](runtime.md).
The existing `rocm-gfx1100-v1` provisioning recipe remains a distinct ROCm 10.0.0
environment; this image does not mutate it.

## Validation boundary

The lock was resolved for Python 3.14/Linux x86-64 from official wheel payloads
and published metadata. The required AMD source metapackage was inspected and
hashed. Hardware execution is untested: no GPU workload, model, dataset, driver
installation, or host configuration change was part of this packaging work.
Image build and smoke results must be reported separately from hardware support.
