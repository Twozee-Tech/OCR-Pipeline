# OCR Pipeline - Two-Stage PDF to Markdown

A modular OCR pipeline that combines Qwen3-VL-8B for intelligent page classification with DeepSeek-OCR for content extraction.

## Architecture

```
PDF → [Stage 1: Classifier] → classifications.json
                                       ↓
PDF + classifications.json → [Stage 2: OCR] → output.md
```

### Stage 1: Classification (Qwen3-VL-8B)
- Analyzes each page to determine content type
- Types: text, table, diagram, flowchart, figure, mixed
- Provides confidence scores and descriptions
- Fallback to heuristic classification if model unavailable

### Stage 2: OCR (DeepSeek-OCR)
- Processes pages with content-aware prompts
- Extracts text, tables, and figures
- Generates clean markdown output

## Files

| File | Description |
|------|-------------|
| `ocr_pipeline.py` | Master orchestrator - main entry point |
| `stage1_classifier.py` | Page classification module |
| `stage2_ocr.py` | OCR processing module |
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

## CLI Options

### ocr_pipeline.py (Main)
```
python ocr_pipeline.py input.pdf [output.md] [options]

Options:
  --classifier, -c    qwen3-vl-8b | heuristic (default: heuristic)
  --precision, -p     fp16 | fp8 (default: fp16)
  --deepseek-model    Path to DeepSeek-OCR model
  --qwen-model        Path to Qwen3-VL model
  --dpi               PDF rendering resolution (default: 200)
  --config            Path to config JSON
  --verbose, -v       Verbose output
```

### stage1_classifier.py (Standalone)
```
python stage1_classifier.py input.pdf -o classifications.json [options]

Options:
  --heuristic         Use heuristic only (no GPU required)
  --model, -m         Path to Qwen3-VL model
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
    "classifier": "heuristic",
    "precision": "fp16"
}
```

## VRAM Management

The pipeline loads models sequentially to minimize VRAM usage:

1. Load Qwen3-VL-8B (~16GB)
2. Run classification on all pages
3. Unload Qwen3-VL-8B
4. Load DeepSeek-OCR (~8GB)
5. Run OCR on all pages
6. Unload DeepSeek-OCR

This allows running on GPUs with 24GB+ VRAM.

## Content Type Prompts

| Type | Prompt Strategy |
|------|-----------------|
| text | Convert document to markdown |
| table | Convert document to markdown (preserves structure) |
| diagram | Describe image in detail |
| flowchart | Describe image in detail |
| figure | Parse the figure |
| mixed | Convert document to markdown (safe default) |

## Output Format

The generated markdown includes:
- HTML comments with page metadata
- Extracted text and tables
- Figure references (saved to `{output}_assets/figures/`)
- VL descriptions for visual content (when available)

Example:
```markdown
# Document Name

*Converted using Qwen3-VL-8B + DeepSeek-OCR*
*Pages: 10 | Figures: 5*

---

<!-- Page 1 | Type: text | Confidence: 95% | Method: qwen3-vl-8b -->

# Introduction

This is the document content...

---

<!-- Page 2 | Type: table | Confidence: 88% | Method: qwen3-vl-8b -->

| Column A | Column B | Column C |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
```

## Testing

```bash
# Test Stage 1 standalone
python stage1_classifier.py test.pdf -o /tmp/class.json -v
cat /tmp/class.json

# Test Stage 2 standalone (with classifications)
python stage2_ocr.py test.pdf -c /tmp/class.json -o /tmp/out.md -v

# Test full pipeline
python ocr_pipeline.py test.pdf output.md --classifier qwen3-vl-8b -v
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

- **DeepSeek-OCR**: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **Qwen3-VL-8B**: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
