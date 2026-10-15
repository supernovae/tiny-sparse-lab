# Intel XPU delivery

The latest stable upstream XPU wheel verified for CPython 3.14 on Linux x86_64
is `torch==2.14.1+xpu`, with `triton-xpu==3.8.0` and the Intel 2026.1 runtime
closure. The [complete wheel lock](../container/backends/xpu/requirements.lock)
includes public Lab dependencies and `uv-build==0.12.17`. No nightly packages,
Python downgrade, IPEX or alternative Lab engine are introduced.

**An XPU container image is not available from this change.**
[Machine-readable status](../container/backends/xpu/status.json) records the
remaining userspace-driver blocker. This backend deliberately has no
`config.json`, so the shared image pipeline cannot build or publish it.

## Why wheels alone are insufficient

PyTorch's [stable Intel GPU instructions](https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html)
require a driver independently of the XPU binary wheels. A container shares the
host kernel but has its own userspace libraries. The pinned Intel SYCL runtime
packages are not evidence that the container has a compatible Level Zero loader
and GPU device driver. Mapping `/dev/dri` alone does not supply those libraries.

The current [Intel compute-runtime release](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10)
is 26.35.39758.10, with gmmlib 22.10.0 and IGC 2.41.5; its package procedure and
validation target Ubuntu 26.04. The separately released
[Level Zero loader](https://github.com/oneapi-src/level-zero/releases/tag/v1.34.0)
is 1.34.0. These versions are research evidence, not installed or tested image
pins. The shared Python 3.14.4 Debian Bookworm base has not been validated with
that Ubuntu driver package closure. Copying arbitrary host libraries or mixing
these distributions would not establish compatibility.

To enable container publication, pin a compatible OS/Python base and the full
userspace loader/driver package closure, verify their checksums and import/link
behavior, then validate device discovery on an already provisioned Intel host.
Include all those inputs in the immutable dependency-image identity. Ordinary
Lab updates should reuse that exact dependency digest and add only the public
application layer. This preparation does not require an automatic accelerator
build on every application change.

## Use an existing native Intel environment

The supported current delivery path is the native [Intel acceleration guide](intel-xpu.md)
and its provision/register/doctor commands. Use stable Python 3.14 and an already
provisioned supported Intel driver stack. Provisioning downloads packages and
runs an optimizer pilot; it is a separate explicit action, not a check performed
by this packaging change. An existing interpreter can be registered without
replacing the checkout's CPU environment. Never sync the CPU extra into it.

For a future container on native Linux, expose only the intended render node
and the existing group access required for it. Keep ordinary container security
confinement; do not use privileged mode, unconfined profiles or world-writable
devices. Windows/WSL device and driver forwarding needs separate validation;
this preparation does not claim WSL XPU container support. The CPU web viewer can
still read completed output from native XPU jobs using the storage layout in
[the container guide](containers.md).

## Reproducing the wheel lock

From the repository root, with the existing uv executable:

```sh
uv pip compile pyproject.toml requirements/xpu-arc.txt container/build-requirements.txt \
  --no-sources --python-version 3.14 --python-platform x86_64-manylinux_2_28 \
  --only-binary :all: --generate-hashes --no-annotate --no-header \
  --emit-index-url --emit-find-links -o /tmp/xpu-requirements.lock
```

This resolves wheel metadata without installing an environment or building source
packages. uv 0.12.19 omitted the two PyTorch-hosted find-links hashes in its
output. The checked-in lock adds their CPython 3.14 wheel SHA256 fragments from
the official [Torch index](https://download.pytorch.org/whl/xpu/torch/) and
[Triton XPU index](https://download.pytorch.org/whl/xpu/triton-xpu/); exact values
are also recorded in status.json. Do not replace the checked-in lock with output
containing unhashed entries. Installation must require hashes and binary wheels.

Validation performed: wheel-only metadata resolution and strict static checking
of every locked package/hash, JSON and documentation links. No image was built,
no XPU wheels installed, and no Intel hardware workload or model download ran.
GPU availability, precision support and performance remain untested.
