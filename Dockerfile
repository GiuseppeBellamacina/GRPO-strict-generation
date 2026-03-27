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

# Create venv and put it on PATH
RUN uv venv /workspace/.venv
ENV PATH="/workspace/.venv/bin:$PATH" \
    VIRTUAL_ENV="/workspace/.venv"

# Install project dependencies via setup.sh (cached layer)
COPY pyproject.toml uv.lock* setup.sh README.md ./
RUN bash setup.sh

# Default: interactive shell
CMD ["bash"]
