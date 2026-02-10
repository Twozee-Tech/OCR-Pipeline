# syntax=docker/dockerfile:1.6
#
# Advanced OCR Pipeline — Two-venv Docker build
#
# Architecture:
#   nvcr.io/nvidia/pytorch:26.01-py3  (torch + CUDA base)
#   ├── /opt/venv_vllm/   → vLLM, qwen-vl-utils, PyMuPDF, Pillow
#   └── /opt/venv_ocr/    → transformers==4.46.3, accelerate, einops, etc.
#
# Both venvs use --system-site-packages to inherit torch/CUDA from base.
# Neither can affect the other.

ARG BUILD_JOBS=16

# =========================================================
# STAGE 1: Base (build dependencies)
# =========================================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS base

ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV VLLM_BASE_DIR=/workspace/vllm

RUN apt update && \
    apt install -y --no-install-recommends \
    curl vim ninja-build git ccache \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv && pip uninstall -y flash-attn

ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_MAXSIZE=50G
ENV CCACHE_COMPRESS=1
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

WORKDIR $VLLM_BASE_DIR

ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# =========================================================
# STAGE 2: Builder (FlashInfer + vLLM from source)
# =========================================================
FROM base AS builder

# --- FlashInfer ---
ENV FLASHINFER_CUDA_ARCH_LIST="12.1a"
WORKDIR $VLLM_BASE_DIR
ARG FLASHINFER_REF=main
ARG CACHEBUST_DEPS=1

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "flashinfer" ]; then \
        echo "Cache miss: Cloning FlashInfer from scratch..." && \
        git clone --recursive https://github.com/flashinfer-ai/flashinfer.git; \
        if [ "$FLASHINFER_REF" != "main" ]; then \
            cd flashinfer && \
            git checkout ${FLASHINFER_REF}; \
        fi; \
    else \
        echo "Cache hit: Fetching flashinfer updates..." && \
        cd flashinfer && \
        git fetch --all && \
        git checkout ${FLASHINFER_REF} && \
        if [ "${FLASHINFER_REF}" = "main" ]; then \
            git reset --hard origin/main; \
        fi && \
        git submodule update --init --recursive && \
        git gc --auto; \
    fi && \
    cp -a /repo-cache/flashinfer /workspace/flashinfer

WORKDIR /workspace/flashinfer

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' -e '/license-files/d' pyproject.toml && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    cd flashinfer-cubin && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    cd flashinfer-jit-cache && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

# --- vLLM ---
ARG CACHEBUST_VLLM=1
ARG VLLM_REF=main

RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "vllm" ]; then \
        echo "Cache miss: Cloning vLLM from scratch..." && \
        git clone --recursive https://github.com/vllm-project/vllm.git; \
        if [ "$VLLM_REF" != "main" ]; then \
            cd vllm && \
            git checkout ${VLLM_REF}; \
        fi; \
    else \
        echo "Cache hit: Fetching updates..." && \
        cd vllm && \
        git fetch --all && \
        git checkout ${VLLM_REF} && \
        if [ "${VLLM_REF}" = "main" ]; then \
            git reset --hard origin/main; \
        fi && \
        git submodule update --init --recursive && \
        git gc --auto; \
    fi && \
    cp -a /repo-cache/vllm $VLLM_BASE_DIR/

WORKDIR $VLLM_BASE_DIR/vllm

ARG VLLM_PRS=""
RUN if [ -n "$VLLM_PRS" ]; then \
        echo "Applying PRs: $VLLM_PRS"; \
        for pr in $VLLM_PRS; do \
            echo "Fetching and applying PR #$pr..."; \
            curl -fL "https://github.com/vllm-project/vllm/pull/${pr}.diff" | git apply -v; \
        done; \
    fi

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/test.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test.txt && \
    uv pip install -r requirements/build.txt

RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

# =========================================================
# STAGE 3: Runner (two isolated venvs)
# =========================================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS runner

ARG BUILD_JOBS=16
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV VLLM_BASE_DIR=/workspace/vllm

RUN apt update && \
    apt install -y --no-install-recommends \
    curl vim git libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv && pip uninstall -y flash-attn

WORKDIR $VLLM_BASE_DIR

# Download Tiktoken encodings (needed by vLLM)
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# ---------------------------------------------------------
# venv_vllm: vLLM + Qwen dependencies (Stage 1 inference)
# ---------------------------------------------------------
RUN python3 -m venv --system-site-packages /opt/venv_vllm

# Install vLLM wheels from builder
RUN --mount=type=bind,from=builder,source=/workspace/wheels,target=/mount/wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    /opt/venv_vllm/bin/pip install --no-cache-dir /mount/wheels/*.whl

# Install vLLM runtime deps + Qwen/OCR utilities
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    /opt/venv_vllm/bin/pip install --no-cache-dir \
    ray[default] \
    fastsafetensors \
    "qwen-vl-utils>=0.0.14" \
    PyMuPDF \
    Pillow

# ---------------------------------------------------------
# venv_ocr: DeepSeek-OCR-2 (Stage 2 inference)
# ---------------------------------------------------------
RUN python3 -m venv --system-site-packages /opt/venv_ocr

# transformers pinned to 4.46.3 — the proven version for DeepSeek-OCR-2
# Installed with --no-deps to avoid pulling in tokenizers/huggingface-hub
# versions that conflict with vLLM (even though they're in separate venvs,
# keeping deps minimal is cleaner).
RUN /opt/venv_ocr/bin/pip install --no-cache-dir --no-deps transformers==4.46.3 && \
    /opt/venv_ocr/bin/pip install --no-cache-dir \
    tokenizers \
    accelerate \
    einops \
    easydict \
    addict \
    Pillow \
    numpy \
    safetensors \
    sentencepiece \
    "qwen-vl-utils" \
    huggingface_hub \
    PyMuPDF \
    img2pdf

# flash-attn for DeepSeek-OCR — optional, falls back to eager attention
RUN /opt/venv_ocr/bin/pip install --no-cache-dir flash-attn --no-build-isolation || true

# ---------------------------------------------------------
# Application code
# ---------------------------------------------------------
COPY src/*.py /app/
COPY config/ocr_config.json /app/
WORKDIR /app

RUN mkdir -p /data/input /data/output

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings
ENV PATH=$VLLM_BASE_DIR:$PATH

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV OMP_NUM_THREADS=1
ENV OCR_DESCRIBE_DIAGRAMS=false
ENV OCR_DPI=200
ENV OCR_VENV_PYTHON=/opt/venv_ocr/bin/python3

ENTRYPOINT ["/opt/venv_vllm/bin/python3", "/app/entrypoint.py"]
