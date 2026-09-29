# syntax=docker/dockerfile:1
# Multi-stage build. The runtime venv is assembled from two layers:
#   1. a multi-GB third-party layer (torch + bundled CUDA userland), cache-
#      keyed on pyproject.toml + uv.lock -- rebuilt only when dependencies
#      change;
#   2. a few-KB project layer (this package, its metadata, console script).
# A code-only change therefore rebuilds -- and re-pushes to the registry --
# just the small layer; the torch/CUDA layer stays byte-identical and is
# skipped by the registry's content dedup.
#
# The server also runs fine on CPU (torch falls back automatically when the
# container has no GPU); pass --gpus all to use the NVIDIA GPU.

# --- deps: third-party packages only; never invalidated by source edits ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS deps

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# --frozen requires a committed uv.lock (regenerate with `uv lock` locally).
# README.md is required by hatchling (pyproject references it).
# --no-install-package triton: linux torch pulls Triton in as a dependency
# (~900 MB installed), but nothing here uses it -- the fast backend runs
# CUDA graphs on eager kernels and stock runs plain eager. Eager torch never
# imports Triton: TORCH_DISABLE_NATIVE_JIT=1 (see below) keeps the native-op
# registry off, and torch ships without a hard Triton import. torch.compile
# would need it back (plus a C toolchain), so don't enable QWEN3TTS_COMPILE
# in this image.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --no-install-package triton

# --- project: the project wheel on top of the deps venv --------------------
FROM deps AS project
COPY src ./src
# --no-editable: install the project as a wheel into the venv (an editable
# install would .pth-point at /app/src, which is not copied to the runtime).
# Dependencies are already satisfied, so this step only writes the project
# package; only those files are copied into the runtime image below.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --no-install-package triton
# Stage the project's own files under fixed paths. The dist-info directory
# name embeds the version, and COPY flattens directory sources (contents
# land in dest), so resolve the glob with the shell instead of in COPY.
RUN mkdir -p /code-layer/site-packages /code-layer/bin \
    && cp -a /app/.venv/lib/python3.12/site-packages/qwen3_tts_stripped_wyoming \
             /app/.venv/lib/python3.12/site-packages/qwen3_tts_stripped_wyoming-*.dist-info \
             /code-layer/site-packages/ \
    && cp -a /app/.venv/bin/qwen3-tts-stripped-wyoming /code-layer/bin/

# ---------------------------------------------------------------------------
# The PyPI linux torch wheels bundle the CUDA 12 userland (cudnn, cublas, ...)
# as pip dependencies; the NVIDIA runtime only needs to provide the driver
# (libcuda.so via the container toolkit). This base also works CPU-only.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime

# libgomp1 is required by torch; ca-certificates for HF downloads; sox quiets
# the `sox` python package's startup warning. No C/C++ toolchain and no
# Triton: this image runs eager inference only (the fast backend needs CUDA
# graphs, not compilers), which keeps it ~1 GB smaller. Ubuntu 24.04 images
# may already ship a UID 1000 user; reuse it if present.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates sox \
    && rm -rf /var/lib/apt/lists/* \
    && (getent passwd 1000 > /dev/null || useradd --create-home --uid 1000 app)

WORKDIR /app
# The venv's bin/python is an absolute symlink into the builder's
# interpreter (/usr/local/bin/python3), so the interpreter must come along.
COPY --from=deps /usr/local /usr/local
# Layer 1 (multi-GB, changes only with uv.lock): third-party packages.
COPY --from=deps --chown=1000:1000 /app/.venv /app/.venv
# Layer 2 (few KB, changes with every commit): this package + its metadata.
COPY --from=project --chown=1000:1000 \
    /code-layer/site-packages \
    /app/.venv/lib/python3.12/site-packages/
# The console script (its shebang points at /app/.venv/bin/python).
COPY --from=project --chown=1000:1000 /code-layer/bin /app/.venv/bin/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    QWEN3TTS_MODEL_DIR=/data/models \
    HF_HOME=/data/hf \
    # Triton is not installed in this image; this keeps torch's native-op
    # registry from rerouting tiny bmms (RoPE) to Triton kernels that cannot
    # load. Eager/cuBLAS is what we want (see __main__.py, which also sets
    # this as a fallback for venv users).
    TORCH_DISABLE_NATIVE_JIT=1

RUN mkdir -p /data/models /data/hf && chown -R 1000:1000 /data
USER 1000
VOLUME /data

EXPOSE 10200

# First start downloads the model source (~4.5 GB for 1.7B) and converts it
# when a converted variant is requested; give it room.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=5 \
    CMD ["python", "-m", "qwen3_tts_stripped_wyoming.healthcheck"]

ENTRYPOINT ["qwen3-tts-stripped-wyoming"]
CMD ["--uri", "tcp://0.0.0.0:10200"]
