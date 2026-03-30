# syntax=docker/dockerfile:1
# ── GRPO Strict Generation — Dev Container ──────────────────────────────────
# NVIDIA CUDA base with Python, uv, and project dependencies.
# Built for GPU-accelerated ML training with Unsloth + vLLM.

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
        python3.12 python3.12-dev python3.12-venv python3-pip \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Working directory — bind-mounted at runtime
WORKDIR /workspace

# Create venv OUTSIDE /workspace (bind mount sovrascrive /workspace a runtime)
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv" \
    UV_PROJECT_ENVIRONMENT="/opt/venv"

# Install project dependencies (no editable: src viene dal bind mount a runtime)
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e ".[dev,fast]"

# Default: interactive shell
CMD ["bash"]
