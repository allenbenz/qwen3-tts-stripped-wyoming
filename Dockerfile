# syntax=docker/dockerfile:1
# Multi-stage build: uv installs the project into a venv, then only the venv
# and the CUDA userland land in the runtime image.
#
# The server also runs fine on CPU (torch falls back automatically when the
# container has no GPU); pass --gpus all to use the NVIDIA GPU.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first so source edits don't invalidate the layer.
# --frozen requires a committed uv.lock (regenerate with `uv lock` locally).
# README.md is required by hatchling (pyproject references it).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
# --no-editable: install the project as a wheel into the venv (an editable
# install would .pth-point at /app/src, which is not copied to the runtime).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---------------------------------------------------------------------------
# The PyPI linux torch wheels bundle the CUDA 12 userland (cudnn, cublas, ...)
# as pip dependencies; the NVIDIA runtime only needs to provide the driver
# (libcuda.so via the container toolkit). This base also works CPU-only.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime

# libgomp1 is required by torch; ca-certificates for HF downloads; sox quiets
# the `sox` python package's startup warning; gcc serves Triton (CUDA compile)
# and g++ serves Inductor's CPU C++ wrappers -- both are needed for the
# torch.compile path (see TORCH_DISABLE_NATIVE_JIT below for why the eager
# path stays compiler-free). Ubuntu 24.04 images may already ship a UID 1000
# user; reuse it if present.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates sox gcc g++ \
    && rm -rf /var/lib/apt/lists/* \
    && (getent passwd 1000 > /dev/null || useradd --create-home --uid 1000 app)

WORKDIR /app
# The venv's bin/python is an absolute symlink into the builder's
# interpreter (/usr/local/bin/python3), so the interpreter must come along.
COPY --from=builder /usr/local /usr/local
COPY --from=builder --chown=1000:1000 /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    QWEN3TTS_MODEL_DIR=/data/models \
    HF_HOME=/data/hf \
    # linux torch bundles Triton; without this its native-op registry reroutes
    # tiny bmms (RoPE) to Triton kernels and dies on first use because the slim
    # image has no C compiler. Eager/cuBLAS is what we want (see __main__.py).
    # NOTE: this kill switch only affects the eager torch._native registry --
    # torch.compile/Inductor generates its own Triton kernels independently.
    TORCH_DISABLE_NATIVE_JIT=1 \
    # torch.compile (Inductor/Triton) for the stock TTS backend: the caches
    # live on the /data volume so the multi-minute first compile happens once
    # per model, not per container. (The default fast backend needs neither.)
    QWEN3TTS_COMPILE=1 \
    TORCHINDUCTOR_CACHE_DIR=/data/compile-cache/inductor \
    TRITON_CACHE_DIR=/data/compile-cache/triton

RUN mkdir -p /data/models /data/hf /data/compile-cache && chown -R 1000:1000 /data
USER 1000
VOLUME /data

EXPOSE 10200

# First start downloads the model source (~4.5 GB for 1.7B) and converts it
# when a converted variant is requested; give it room.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=5 \
    CMD ["python", "-m", "qwen3_tts_stripped_wyoming.healthcheck"]

ENTRYPOINT ["qwen3-tts-stripped-wyoming"]
CMD ["--uri", "tcp://0.0.0.0:10200"]
