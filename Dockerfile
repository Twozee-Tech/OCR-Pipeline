FROM nvcr.io/nvidia/pytorch:25.12-py3

# System dependencies for DeepSeek-OCR (needs transformers==4.46.3)
RUN pip install --no-cache-dir \
    transformers==4.46.3 \
    tokenizers \
    PyMuPDF \
    einops \
    easydict \
    addict \
    Pillow \
    numpy \
    safetensors \
    accelerate \
    sentencepiece \
    huggingface_hub

# Create venv for Qwen (needs newer transformers >=4.57)
# Use --system-site-packages to inherit system torch/cuda
# Only upgrade transformers, don't reinstall torch
RUN python3 -m venv --system-site-packages /opt/venv_qwen && \
    /opt/venv_qwen/bin/pip install --no-cache-dir \
    "transformers>=4.57.0" \
    qwen-vl-utils

# Copy pipeline code
COPY *.py /app/
COPY ocr_config.json /app/
WORKDIR /app

# Create directories
RUN mkdir -p /data/input /data/output

# Environment defaults
ENV OCR_CLASSIFIER=qwen3-vl-8b \
    OCR_PRECISION=fp16 \
    OCR_DESCRIBE_DIAGRAMS=false \
    OCR_DPI=200 \
    OCR_VENV_PYTHON=/opt/venv_qwen/bin/python3

ENTRYPOINT ["python3", "/app/entrypoint.py"]
