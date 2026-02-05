FROM nvcr.io/nvidia/vllm:26.01-py3

RUN pip install --no-cache-dir \
    qwen-vl-utils>=0.0.14 \
    PyMuPDF \
    Pillow

COPY src/*.py /app/
COPY config/ocr_config.json /app/
WORKDIR /app

# Create directories
RUN mkdir -p /data/input /data/output

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn \
    OMP_NUM_THREADS=1 \
    OCR_DESCRIBE_DIAGRAMS=false \
    OCR_DPI=200

ENTRYPOINT ["python3", "/app/entrypoint.py"]
