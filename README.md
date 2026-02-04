# OCR Pipeline - Three-Stage PDF to Markdown

A modular OCR pipeline that combines Qwen3-VL models for intelligent page classification and diagram description with DeepSeek-OCR for content extraction.

## Architecture

```
PDF → [Stage 1: Classifier] → classifications.json
              ↓
       (diagram/flowchart pages only)
              ↓
      [Stage 1.5: Diagram Describer] → diagram_descriptions.json
              ↓
PDF + classifications + descriptions → [Stage 2: OCR] → output.md
```

### Stage 1: Classification (Qwen3-VL-8B)
- Analyzes each page to determine content type
- Types: text, table, diagram, flowchart, figure, mixed
- Provides confidence scores and descriptions
- Fallback to heuristic classification if model unavailable

### Stage 1.5: Diagram Description (Qwen3-VL-32B) [Optional]
- Processes diagram and flowchart pages identified by Stage 1
- Generates detailed ASCII art representations
- Provides step-by-step flow descriptions
- Only runs when `--describe-diagrams` flag is used

### Stage 2: OCR (DeepSeek-OCR)
- Processes pages with content-aware prompts
- Extracts text, tables, and figures
- Merges diagram descriptions from Stage 1.5
- Generates clean markdown output

## Files

| File | Description |
|------|-------------|
| `ocr_pipeline.py` | Master orchestrator - main entry point |
| `stage1_classifier.py` | Page classification module (Qwen3-VL-8B) |
| `stage1_5_diagram.py` | Diagram description module (Qwen3-VL-32B) |
| `stage2_ocr.py` | OCR processing module (DeepSeek-OCR) |
| `ocr_config.json` | Configuration file |
| `setup.sh` | Environment setup script |

## Quick Start

### 1. Setup (in NVIDIA container)
```bash
./setup.sh
```

### 2. Basic Usage (heuristic classification)
```bash
python ocr_pipeline.py document.pdf
```

### 3. With VL Classification (recommended for mixed content)
```bash
python ocr_pipeline.py document.pdf --classifier qwen3-vl-8b
```

### 4. With Diagram Description (best for flowcharts/call flows)
```bash
python ocr_pipeline.py document.pdf --classifier qwen3-vl-8b --describe-diagrams
```

## CLI Options

### ocr_pipeline.py (Main)
```
python ocr_pipeline.py input.pdf [output.md] [options]

Options:
  --classifier, -c       qwen3-vl-8b | heuristic (default: heuristic)
  --precision, -p        fp16 | fp8 (default: fp16)
  --describe-diagrams    Enable Stage 1.5 diagram description with Qwen3-VL-32B
  --no-describe-diagrams Skip diagram description (faster)
  --deepseek-model       Path to DeepSeek-OCR model
  --qwen-model           Path to Qwen3-VL-8B classifier model
  --qwen-describer       Path to Qwen3-VL-32B describer model
  --dpi                  PDF rendering resolution (default: 200)
  --config               Path to config JSON
  --verbose, -v          Verbose output
```

### stage1_classifier.py (Standalone)
```
python stage1_classifier.py input.pdf -o classifications.json [options]

Options:
  --heuristic         Use heuristic only (no GPU required)
  --model, -m         Path to Qwen3-VL-8B model
  --precision, -p     fp16 | fp8
  --dpi               PDF rendering resolution
  --verbose, -v       Verbose output
```

### stage1_5_diagram.py (Standalone)
```
python stage1_5_diagram.py input.pdf -c classifications.json -o diagrams.json [options]

Options:
  --model, -m         Path to Qwen3-VL-32B model
  --precision, -p     fp16 | fp8
  --dpi               PDF rendering resolution
  --verbose, -v       Verbose output
```

### stage2_ocr.py (Standalone)
```
python stage2_ocr.py input.pdf -o output.md [options]

Options:
  --classifications, -c   JSON file from Stage 1
  --model, -m             Path to DeepSeek-OCR model
  --dpi                   PDF rendering resolution
  --verbose, -v           Verbose output
```

## Configuration

`ocr_config.json`:
```json
{
    "deepseek_model_path": "/workspace/models/DeepSeek-OCR-model",
    "qwen_model_path": "/workspace/models/Qwen3-VL-8B-model",
    "qwen_describer_path": "/workspace/models/Qwen3-VL-32B-model",
    "classifier": "heuristic",
    "precision": "fp16",
    "describe_diagrams": false
}
```

## VRAM Management

The pipeline loads models sequentially to minimize VRAM usage:

```
1. Load Qwen3-VL-8B (~16GB) → classify all pages → unload
2. Load Qwen3-VL-32B (~60GB) → describe diagram pages only → unload
3. Load DeepSeek-OCR (~8GB) → OCR all pages → unload
4. Merge results → save markdown
```

Requirements:
- Without `--describe-diagrams`: 24GB+ VRAM
- With `--describe-diagrams`: 64GB+ VRAM (128GB recommended)

## Content Type Prompts

| Type | Stage 2 Prompt Strategy |
|------|------------------------|
| text | Convert document to markdown |
| table | Convert document to markdown (preserves structure) |
| diagram | Describe image in detail (or use Stage 1.5 description) |
| flowchart | Describe image in detail (or use Stage 1.5 description) |
| figure | Parse the figure |
| mixed | Convert document to markdown (safe default) |

## Output Format

### Standard Output (without --describe-diagrams)
```markdown
# Document Name

*Converted using Qwen3-VL-8B + DeepSeek-OCR*
*Pages: 10 | Figures: 5*

---

<!-- Page 1 | Type: text | Confidence: 95% | Method: qwen3-vl-8b -->

# Introduction

This is the document content...
```

### Diagram Page (with --describe-diagrams)
```markdown
<!-- Page 15 | Type: flowchart | Confidence: 92% | Method: qwen3-vl-32b -->

## Call Flow: User Authentication

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  User   │────>│  API    │────>│  Auth   │
└─────────┘     └─────────┘     └─────────┘
     │               │               │
     │   1. Login    │               │
     │──────────────>│               │
     │               │  2. Validate  │
     │               │──────────────>│
     │               │               │
     │               │  3. Token     │
     │               │<──────────────│
     │  4. Response  │               │
     │<──────────────│               │
```

**Flow description:**
1. User sends login request to API
2. API forwards credentials to Auth service
3. Auth validates and returns JWT token
4. API returns token to user

---
*Additional OCR text:* Figure 3.1 - Authentication Flow
```

## Testing

```bash
# Test Stage 1 standalone
source venv_qwen/bin/activate
python stage1_classifier.py test.pdf -o /tmp/class.json -v
cat /tmp/class.json
deactivate

# Test Stage 1.5 standalone (in venv)
source venv_qwen/bin/activate
python stage1_5_diagram.py test.pdf -c /tmp/class.json -o /tmp/diagrams.json -v
deactivate

# Test Stage 2 standalone (system Python)
python stage2_ocr.py test.pdf -c /tmp/class.json -o /tmp/out.md -v

# Test full pipeline without diagram description
python ocr_pipeline.py test.pdf output.md --classifier qwen3-vl-8b -v

# Test full pipeline with diagram description
python ocr_pipeline.py test.pdf output.md --classifier qwen3-vl-8b --describe-diagrams -v
```

## Requirements

- Python 3.8+
- PyTorch with CUDA
- transformers >= 4.50.0
- accelerate
- qwen-vl-utils
- pymupdf
- pillow

## Models

| Model | Purpose | Size | Required |
|-------|---------|------|----------|
| DeepSeek-OCR | OCR processing | ~8GB | Yes |
| Qwen3-VL-8B | Page classification | ~16GB | Optional (can use heuristic) |
| Qwen3-VL-32B | Diagram description | ~65GB | Optional (only for --describe-diagrams) |

Download commands:
```bash
# Required
hf download deepseek-ai/DeepSeek-OCR --local-dir /workspace/models/DeepSeek-OCR-model

# Optional - for VL classification
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /workspace/models/Qwen3-VL-8B-model

# Optional - for diagram description (large, ~65GB)
hf download Qwen/Qwen3-VL-32B-Instruct --local-dir /workspace/models/Qwen3-VL-32B-model
```
