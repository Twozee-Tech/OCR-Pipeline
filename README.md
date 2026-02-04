# Advanced OCR for NVIDIA DGX Spark

AI-powered PDF to Markdown conversion using vision-language models. Combines Qwen3-VL for intelligent page classification with DeepSeek-OCR for content extraction.

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/Twozee-Tech/Advanced-OCR-Nvidia-DGX-SPARK/main/install.sh | bash
```

This will:
- Ask where to store models (~24GB for recommended setup)
- Download required AI models
- Build the Docker image
- Install the `ocr` command

## Usage

```bash
ocr document.pdf                      # Output to ./output/document.md
ocr document.pdf result.md            # Output to ./result.md
ocr document.pdf --diagrams           # With detailed diagram descriptions
```

## Requirements

- NVIDIA GPU with 24GB+ VRAM
- Docker with NVIDIA runtime
- ~24GB disk space for models

## How It Works

```
PDF → [Stage 1: Qwen3-VL-8B] → Page Classification
              ↓
    [Stage 1.5: Qwen3-VL-32B] → Diagram Descriptions (optional)
              ↓
      [Stage 2: DeepSeek-OCR] → Text Extraction
              ↓
           Markdown Output
```

### Stage 1: Classification
Qwen3-VL-8B analyzes each page and classifies content:
- `text` - paragraphs, lists, headers
- `table` - data tables
- `diagram` - technical diagrams, UML, architecture
- `flowchart` - process flows, decision trees
- `figure` - photos, charts, illustrations
- `mixed` - combination of content types

### Stage 1.5: Diagram Description (Optional)
Qwen3-VL-32B generates detailed descriptions for diagram/flowchart pages:
- ASCII art representations
- Step-by-step flow descriptions
- Component relationships

Enable with `--diagrams` flag (requires 64GB+ VRAM).

### Stage 2: OCR
DeepSeek-OCR extracts content using classification-optimized prompts.

## Models

| Model | Size | VRAM | Purpose |
|-------|------|------|---------|
| DeepSeek-OCR-2 | ~8GB | ~10GB | Text extraction (required) |
| Qwen3-VL-8B | ~16GB | ~16GB | Page classification (recommended) |
| Qwen3-VL-32B | ~65GB | ~64GB | Diagram descriptions (optional) |

Models load sequentially, not simultaneously.

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
│   ├── stage1_classifier.py
│   ├── stage1_5_diagram.py
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
