# Plan: Konteneryzacja OCR Pipeline

## Cel
Przerobić działający 3-stage OCR pipeline w kontener Docker, który:
- Przyjmuje PDF na wejściu
- Oddaje MD na wyjściu
- Modele montowane jako wolumen (nie wbudowane - za duże ~60GB)

## Wymagane modele

```bash
# Pobierz modele do /workspace/models/
hf download Qwen/Qwen3-VL-8B --local-dir /workspace/models/Qwen3-VL-8B-model
hf download Qwen/Qwen3-VL-32B-Instruct --local-dir /workspace/models/Qwen3-VL-32B-model
hf download ds4sd/DeepSeek-OCR-2 --local-dir /workspace/models/DeepSeek-OCR-2-model
```

**Uwaga:** Thinking model (Qwen3-VL-32B-Thinking) nie jest wymagany - testy wykazały brak znaczącej poprawy przy dłuższym czasie przetwarzania.

## Obecny stan
Pipeline jest już prawie gotowy do konteneryzacji:
- ✅ Wszystkie ścieżki modeli mają domyślne `/workspace/...`
- ✅ Config ma fallback do `/workspace/...`
- ✅ Architektura modułowa (3 stage'e)
- ❌ venv_qwen tworzony runtime (powinien być w obrazie)
- ❌ Brak entry point dla kontenera
- ❌ CLI args zamiast env vars

## Architektura docelowa

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container                          │
│                                                              │
│  /data/input/doc.pdf ──→ [OCR Pipeline] ──→ /data/output/   │
│                                                              │
│  Volumes:                                                    │
│  - /workspace/models (ro) - modele HF (~90GB)               │
│  - /data/input (ro) - PDF wejściowy                         │
│  - /data/output - MD + assets                               │
└─────────────────────────────────────────────────────────────┘
```

## Pliki do utworzenia/modyfikacji

### 1. `Dockerfile` (NOWY)

Lokalizacja: `OCR_Final/Dockerfile`

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.12-py3

# System dependencies for DeepSeek-OCR
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
RUN python3 -m venv --system-site-packages /opt/venv_qwen && \
    /opt/venv_qwen/bin/pip install --ignore-installed \
    "transformers>=4.57.0" \
    qwen-vl-utils \
    huggingface-hub \
    tokenizers \
    accelerate \
    pillow

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
```

### 2. `entrypoint.py` (NOWY)

Lokalizacja: `OCR_Final/entrypoint.py`

```python
#!/usr/bin/env python3
"""Docker entrypoint for OCR Pipeline."""
import os
import sys
import glob
from pathlib import Path

def main():
    # Find input PDF
    input_pdf = os.environ.get('OCR_INPUT_PDF')
    if not input_pdf:
        pdfs = glob.glob('/data/input/*.pdf')
        if len(pdfs) == 1:
            input_pdf = pdfs[0]
        elif len(pdfs) > 1:
            print("ERROR: Multiple PDFs. Set OCR_INPUT_PDF env var.")
            sys.exit(1)
        else:
            print("ERROR: No PDF in /data/input")
            sys.exit(1)

    # Output path
    input_name = Path(input_pdf).stem
    output_md = os.environ.get('OCR_OUTPUT_PATH', f'/data/output/{input_name}.md')

    # Build args
    args = [input_pdf, output_md]
    args.extend(['--classifier', os.environ.get('OCR_CLASSIFIER', 'qwen3-vl-8b')])
    args.extend(['--precision', os.environ.get('OCR_PRECISION', 'fp16')])

    if os.environ.get('OCR_DESCRIBE_DIAGRAMS', 'false').lower() == 'true':
        args.append('--describe-diagrams')

    args.extend(['--dpi', os.environ.get('OCR_DPI', '200')])
    args.append('--verbose')

    sys.argv = ['ocr_pipeline.py'] + args
    from ocr_pipeline import main as run_pipeline
    run_pipeline()

if __name__ == '__main__':
    main()
```

### 3. Modyfikacja `ocr_pipeline.py`

Zmiana w funkcji `find_venv_python()` (linia ~90):

```python
def find_venv_python() -> str:
    """Find Python executable in venv_qwen."""
    # Check env var first (for Docker)
    env_venv = os.environ.get('OCR_VENV_PYTHON')
    if env_venv and os.path.exists(env_venv):
        return env_venv

    # Existing logic continues...
    script_dir = Path(__file__).parent
    venv_python = script_dir / "venv_qwen" / "bin" / "python3"
    # ...
```

### 4. `docker-compose.yml` (NOWY, opcjonalny)

Lokalizacja: `OCR_Final/docker-compose.yml`

```yaml
services:
  ocr:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - OCR_CLASSIFIER=qwen3-vl-8b
      - OCR_DESCRIBE_DIAGRAMS=false
    volumes:
      - /workspace/models:/workspace/models:ro
      - ./input:/data/input:ro
      - ./output:/data/output
```

## Użycie docelowe

```bash
# Build
docker build -t ocr-pipeline .

# Run (auto-detect PDF)
docker run --gpus all \
    -v /workspace/models:/workspace/models:ro \
    -v $(pwd)/input:/data/input:ro \
    -v $(pwd)/output:/data/output \
    ocr-pipeline

# Run z opcjami
docker run --gpus all \
    -e OCR_DESCRIBE_DIAGRAMS=true \
    -e OCR_INPUT_PDF=/data/input/doc.pdf \
    -v /workspace/models:/workspace/models:ro \
    -v $(pwd)/doc.pdf:/data/input/doc.pdf:ro \
    -v $(pwd)/output:/data/output \
    ocr-pipeline
```

## Environment Variables

| Zmienna | Default | Opis |
|---------|---------|------|
| `OCR_INPUT_PDF` | auto-detect | Ścieżka do PDF |
| `OCR_OUTPUT_PATH` | `/data/output/{name}.md` | Ścieżka output |
| `OCR_CLASSIFIER` | `qwen3-vl-8b` | Classifier |
| `OCR_DESCRIBE_DIAGRAMS` | `false` | Stage 1.5 (Qwen3-VL-32B-Instruct) |
| `OCR_PRECISION` | `fp16` | Precision |
| `OCR_DPI` | `200` | DPI |
| `OCR_VENV_PYTHON` | `/opt/venv_qwen/bin/python3` | Python venv |

### 5. `ocr.sh` (NOWY) - Wrapper script

Lokalizacja: `OCR_Final/ocr.sh`

```bash
#!/bin/bash
# OCR Pipeline wrapper script
# Usage: ./ocr.sh input.pdf [--diagrams]

set -e

INPUT_PDF="$1"
shift || true

if [ -z "$INPUT_PDF" ]; then
    echo "Usage: ./ocr.sh input.pdf [--diagrams]"
    exit 1
fi

# Defaults
DESCRIBE_DIAGRAMS="false"

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --diagrams) DESCRIBE_DIAGRAMS="true"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

INPUT_DIR=$(dirname "$(realpath "$INPUT_PDF")")
INPUT_NAME=$(basename "$INPUT_PDF")
OUTPUT_DIR="${INPUT_DIR}/output"

mkdir -p "$OUTPUT_DIR"

echo "Processing: $INPUT_PDF"
echo "Diagrams: $DESCRIBE_DIAGRAMS"

docker run --gpus all --rm \
    -e OCR_DESCRIBE_DIAGRAMS="$DESCRIBE_DIAGRAMS" \
    -v /workspace/models:/workspace/models:ro \
    -v "$INPUT_DIR":/data/input:ro \
    -v "$OUTPUT_DIR":/data/output \
    -e OCR_INPUT_PDF="/data/input/$INPUT_NAME" \
    ocr-pipeline

echo "Output: $OUTPUT_DIR/${INPUT_NAME%.pdf}.md"
```

## Weryfikacja

1. Build: `docker build -t ocr-pipeline .`
2. Test z poc.pdf:
   ```bash
   docker run --gpus all \
       -v /workspace/models:/workspace/models:ro \
       -v /workspace/docs/poc.pdf:/data/input/poc.pdf:ro \
       -v /tmp/test:/data/output \
       ocr-pipeline
   cat /tmp/test/poc.md
   ```
3. Test z --describe-diagrams:
   ```bash
   docker run --gpus all \
       -e OCR_DESCRIBE_DIAGRAMS=true \
       -v /workspace/models:/workspace/models:ro \
       -v /workspace/docs/poc.pdf:/data/input/poc.pdf:ro \
       -v /tmp/test2:/data/output \
       ocr-pipeline
   ```
