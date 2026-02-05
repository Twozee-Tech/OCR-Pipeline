FROM nvcr.io/nvidia/vllm:26.01-py3

# Packages for Qwen (vLLM)
RUN pip install --no-cache-dir \
    qwen-vl-utils>=0.0.14 \
    PyMuPDF \
    Pillow

# Venv for DeepSeek-OCR (transformers 4.47.1)
RUN python3 -m venv --system-site-packages /opt/venv_ocr && \
    /opt/venv_ocr/bin/pip install --no-cache-dir \
    transformers==4.47.1 \
    accelerate \
    addict \
    matplotlib \
    einops \
    easydict

COPY src/*.py /app/
COPY config/ocr_config.json /app/
WORKDIR /app

# Create directories
RUN mkdir -p /data/input /data/output

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn \
    OMP_NUM_THREADS=1 \
    OCR_DESCRIBE_DIAGRAMS=false \
    OCR_DPI=200 \
    OCR_VENV_PYTHON=/opt/venv_ocr/bin/python3

ENTRYPOINT ["python3", "/app/entrypoint.py"]
