# Advanced OCR for NVIDIA DGX Spark

AI-powered PDF to Markdown conversion using vision-language models with vLLM for fast batched inference.

## Install

One command installs everything. The interactive installer will guide you through model selection and configuration.

```bash
curl -fsSL https://raw.githubusercontent.com/Twozee-Tech/Advanced-OCR-Nvidia-DGX-SPARK/main/install.sh | bash
```

[View install.sh source](https://github.com/Twozee-Tech/Advanced-OCR-Nvidia-DGX-SPARK/blob/main/install.sh)

**What it does:**
- Checks Docker and GPU availability
- Asks where to store models (default: `~/.cache/ocr-models`)
- Downloads AI models from HuggingFace (~68GB for full setup)
- Builds the Docker image
- Installs the `ocr` command to `~/.local/bin`

No Python installation required on the host - everything runs in Docker.

## Usage

```bash
ocr document.pdf                      # Output to ./output/document.md
ocr document.pdf result.md            # Output to ./result.md
ocr document.pdf --diagrams           # With detailed diagram descriptions
```

## Requirements

- NVIDIA GPU with 48GB+ VRAM (for both models)
- Docker with NVIDIA runtime
- ~68GB disk space for models

## How It Works

```
PDF → [Stage 1: Qwen3-VL-30B] → Page Classification
              ↓
      [Optional: Qwen3-VL-30B] → Diagram Descriptions
              ↓
      [Stage 2: DeepSeek-OCR-2] → Text Extraction
              ↓
           Markdown Output
```

### Stage 1: Classification
Qwen3-VL-30B analyzes each page and classifies content:
- `text` - paragraphs, lists, headers
- `table` - data tables
- `diagram` - technical diagrams, UML, architecture
- `flowchart` - process flows, decision trees
- `figure` - photos, charts, illustrations
- `mixed` - combination of content types

### Stage 1.5: Diagram Description (Optional)
Same Qwen3-VL-30B model generates detailed descriptions for diagram/flowchart pages:
- Mermaid sequence diagrams
- Step-by-step flow descriptions
- Component relationships

Enable with `--diagrams` flag.

### Stage 2: OCR
DeepSeek-OCR-2 extracts content using classification-optimized prompts.

## Models

| Model | Size | VRAM | Purpose |
|-------|------|------|---------|
| DeepSeek-OCR-2 | ~8GB | ~10GB | Text extraction (required) |
| Qwen3-VL-30B | ~60GB | ~30GB | Classification + diagrams |

Models load sequentially via vLLM, not simultaneously.

## Project Structure

```
├── install.sh          # One-line installer
├── README.md
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   ├── entrypoint.py
│   ├── ocr_pipeline.py
│   ├── qwen_processor.py
│   └── stage2_ocr.py
├── config/
│   └── ocr_config.json
└── docs/
    └── notes.md
```

## Advanced Options

```bash
# Keep intermediate files for debugging
ocr document.pdf --keep-assets

# Check installed version
cat ~/.local/bin/ocr | head -5
```

## Reinstall / Update

```bash
curl -fsSL https://raw.githubusercontent.com/Twozee-Tech/Advanced-OCR-Nvidia-DGX-SPARK/main/install.sh | bash
```

## Uninstall

```bash
rm ~/.local/bin/ocr
docker rmi ocr-pipeline
# Optionally remove models:
rm -rf ~/.cache/ocr-models
```

## Development

For local development:

```bash
git clone https://github.com/Twozee-Tech/Advanced-OCR-Nvidia-DGX-SPARK.git
cd Advanced-OCR-Nvidia-DGX-SPARK

# Build with docker-compose
cd docker
docker-compose build
```

## License

MIT
