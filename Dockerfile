FROM ghcr.io/astral-sh/uv:0.12.17@sha256:10787c682e4184e4f290de1171fd4703dc63de99221f10fe1c99002ce7fa9acc AS uv
FROM python:3.14.4-bookworm@sha256:786d12398a4b61918e68bc30d93d0bd679a95136bbbd1dbe4d96f116636516bc AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
WORKDIR /build
# Dependency layer is independent of application source changes.
COPY pyproject.toml uv.lock README.md ./
COPY container/build-requirements.txt ./
RUN test "$(uname -m)" = x86_64 \
 && uv sync --frozen --extra cpu --no-dev --no-install-project --no-build --python /usr/local/bin/python3.14 \
 && uv pip install --python /opt/venv/bin/python --no-deps --require-hashes -r build-requirements.txt
COPY src/ src/
COPY LICENSE ./
RUN uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation . \
 && /opt/venv/bin/sparselab --help \
 && /opt/venv/bin/python -c "import torch; assert torch.version.cuda is None and torch.version.hip is None"

FROM python:3.14.4-bookworm@sha256:786d12398a4b61918e68bc30d93d0bd679a95136bbbd1dbe4d96f116636516bc
ARG SOURCE_REVISION=local
LABEL org.opencontainers.image.source="https://github.com/supernovae/tiny-sparse-lab" \
      org.opencontainers.image.revision=$SOURCE_REVISION \
      org.opencontainers.image.licenses="MIT"
COPY --from=build /opt/venv /opt/venv
COPY container/start.sh /usr/local/bin/lab-start
RUN groupadd -g 10001 lab && useradd -u 10001 -g 10001 -M lab \
 && mkdir /state /observe && chown 10001:10001 /state
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/state/home TMPDIR=/state/tmp \
    SPARSELAB_WORK_DIR=/state/work SPARSELAB_WORKER_LEASE_DIR=/state/work/worker-leases \
    HF_HOME=/state/cache/hf HF_HUB_CACHE=/state/cache/hf/hub HF_DATASETS_CACHE=/state/cache/hf/datasets \
    XDG_CACHE_HOME=/state/cache/xdg HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
USER 10001:10001
WORKDIR /state
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s \
 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=2).read()"]
ENTRYPOINT ["/bin/sh", "/usr/local/bin/lab-start"]
CMD ["streamlit", "run", "/opt/venv/lib/python3.14/site-packages/sparselab/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--", "--runs-dir", "/observe/work/runs", "--lab-dir", "/observe/work/lab", "--reports-dir", "/observe/work/reports", "--evidence-root", "/observe/evidence"]
