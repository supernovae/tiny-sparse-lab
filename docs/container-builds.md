# Accelerator image builds

Linux accelerator variants are independent images in the public
`ghcr.io/supernovae/tiny-sparse-lab` package. A backend participates only when
`container/backends/NAME/config.json` and its complete hashed wheel lock exist.
Blocked backend status documents do not activate builds. macOS Metal needs native
delivery and is not provided by these Linux images.

Pull requests run standard-library planner tests and validate configured input
identities without downloading framework wheels. On trusted main, available
backends run independently in parallel; a manual workflow dispatch on main can
retry publication while reusing existing dependency digests; dispatch does not refresh them. Each job has a 60-minute ceiling. A first dependency build can
require several GB of disk/network even though no GPU or experiment runs.

Dependency images use `deps-BACKEND-SHA256` tags. The identity includes the complete
hashed wheel lock, backend config, dependency Dockerfile (including pinned Python
3.14.4/OS and uv 0.12.17 image digests), build backend hash lock, validation program,
build-context allowlist and project dependency declaration. Registry lookup resolves
an existing identity to an exact digest; only an explicit manifest-unknown response
allows bootstrap. Authentication and transport failures stop the job. Workflow
concurrency serializes main runs, preventing competing writes of the same tag.
Existing dependency tags are never overwritten. To refresh dependencies, review a
change to their pinned inputs; there is no unpinned refresh switch.

Application source, README, license, startup and application Dockerfile changes do
not change the dependency identity. The application starts from its exact registry
dependency digest and installs only Lab, offline and without dependency resolution.
It does not copy the entire environment into another application layer. `uv pip
check` verifies the installed package requirements. Failed dependency compatibility,
framework-build verification or application smoke prevents application publication.

The smoke checks installed framework version/build identity, a two-element CPU
tensor and installed CLI help in a network-disabled, nonroot, read-only container
with finite memory/CPU/PID/tmpfs limits. It provides no GPU-device or driver
compatibility evidence. No GPU passthrough or model workload runs in CI. The same
tested local application image is pushed as `BACKEND-sha-COMMIT`; the retained JSON
artifact records source commit, dependency key/digest and application digest.
Use the digest for deployment. CPU delivery and its viewer smoke remain separate.

Publication uses the workflow's built-in GitHub token on main with package write
permission confined to the publication job. Only public Lab sources enter the
allowlisted context. The package's existing visibility setting applies to these
tags; no workflow changes repository or package visibility.

Local lightweight checks (no installs):

```sh
python3 -m unittest discover -s container -p 'test_image_plan.py'
python3 container/image_plan.py plan cuda
```

The planner command needs that backend's committed configuration and lock. Building
an image does not establish that the worker's driver, hardware or native Lab
backend supports every operation; use the backend-specific delivery notes before
an independently authorized device test.
