# OCR Pipeline Modularization Notes

## Objective
Refactoring the monolithic `pdf_to_md_transformers.py` into a modular architecture with separate files for each stage.

## Source Files Analyzed
- `qwen-classifier-investigation/pdf_to_md_transformers.py` - Main source (1021 lines)
- `qwen-classifier-investigation/test_qwen.py` - Reference for Qwen3-VL loading
- `qwen-classifier-investigation/ocr_config.json` - Configuration example

## Key Observations

### Model Loading Differences
1. `pdf_to_md_transformers.py` uses `AutoModelForImageTextToText` for Qwen3-VL
2. `test_qwen.py` uses `Qwen3VLForConditionalGeneration` from transformers
3. Need to standardize on one approach - using `Qwen3VLForConditionalGeneration` as it's more explicit

### VRAM Management
- Critical: Models need to be loaded/unloaded sequentially
- Qwen3-VL-8B and DeepSeek-OCR both require significant VRAM
- Solution: Load Stage 1 model → run classification → unload → load Stage 2 model

### Classification Pipeline
1. Convert PDF to images with metadata
2. For each page, run Qwen3-VL classification OR heuristic fallback
3. Save classifications to JSON for reusability
4. Pass classifications to Stage 2 for prompt selection

### OCR Pipeline
1. Load DeepSeek-OCR model
2. For each page, select prompt based on classification
3. Run OCR with content-aware prompts
4. Clean and format output
5. Generate final markdown

## Implementation Decisions

### File Structure
```
OCR_Final/
├── stage1_classifier.py   # Qwen3-VL classification
├── stage2_ocr.py          # DeepSeek-OCR processing
├── ocr_pipeline.py        # Master orchestrator
├── ocr_config.json        # Configuration
├── setup.sh               # Environment setup
├── README.md              # Documentation
└── notes.md               # This file
```

### CLI Design
- Each stage can run standalone for debugging
- Orchestrator manages full pipeline
- Intermediate results saved as JSON for inspection

### Error Handling
- Graceful fallback from VL to heuristic classification
- Model loading failures don't crash pipeline
- Verbose mode for debugging

## Changes Made

### Stage 1: Classifier (`stage1_classifier.py`)
- Extracted: `load_qwen_classifier()` → `load_model()`
- Extracted: `classify_page_with_vl()` → `classify_page()`
- Extracted: `classify_page_content_heuristic()` → `classify_page_heuristic()`
- Extracted: `parse_vl_response()` → `parse_response()`
- Added standalone CLI with JSON output

### Stage 2: OCR (`stage2_ocr.py`)
- Extracted: DeepSeek model loading
- Extracted: OCR page processing
- Extracted: Result cleaning and formatting
- Added classifications input support
- Added standalone CLI

### Orchestrator (`ocr_pipeline.py`)
- Manages model loading/unloading sequence
- Coordinates Stage 1 → Stage 2 flow
- Handles configuration
- Generates final output

## Testing Plan
1. Test Stage 1 standalone: `python stage1_classifier.py test.pdf -o /tmp/class.json -v`
2. Test Stage 2 standalone: `python stage2_ocr.py test.pdf --classifications /tmp/class.json -o /tmp/out.md`
3. Test full pipeline: `python ocr_pipeline.py test.pdf output.md --classifier qwen3-vl-8b -v`

## Implementation Complete

### Files Created
- `stage1_classifier.py` (395 lines) - Page classification with Qwen3-VL-8B
- `stage2_ocr.py` (456 lines) - OCR processing with DeepSeek-OCR
- `ocr_pipeline.py` (292 lines) - Master orchestrator
- `ocr_config.json` - Configuration template
- `setup.sh` - Container setup script
- `README.md` - Documentation

### Key Features
1. **Modular design**: Each stage can run independently
2. **VRAM optimization**: Sequential model loading/unloading
3. **Heuristic fallback**: Works without VL model
4. **Intermediate results**: Classifications saved as JSON
5. **Content-aware prompts**: Different prompts per content type
6. **Clean output**: Markdown with figure extraction

### API Surface

**stage1_classifier.py:**
- `load_model(model_path, precision, verbose)` - Load Qwen3-VL
- `unload_model(model, processor, verbose)` - Free VRAM
- `classify_page(model, processor, image, verbose)` - Single page
- `classify_pages(model, processor, pages, use_heuristic_fallback, verbose)` - Batch
- `classify_page_heuristic(page_info)` - Fallback classifier
- `pdf_to_page_images(pdf_path, dpi, verbose)` - PDF to images

**stage2_ocr.py:**
- `load_model(model_path, verbose)` - Load DeepSeek-OCR
- `unload_model(model, tokenizer, verbose)` - Free VRAM
- `ocr_page(model, tokenizer, image, classification, ...)` - Single page
- `ocr_pages(model, tokenizer, pages, classifications, ...)` - Batch
- `generate_markdown(results, pdf_name, classifier_method)` - Output generation
- `get_prompt(classification)` - Prompt selection

**ocr_pipeline.py:**
- `run_pipeline(pdf_path, output_path, classifier, ...)` - Full pipeline
- `load_config(config_path)` - Load configuration

### Differences from Original
1. Uses `Qwen3VLForConditionalGeneration` (explicit) instead of `AutoModelForImageTextToText`
2. Uses `qwen_vl_utils.process_vision_info` for proper input processing
3. Separate unload functions for explicit VRAM management
4. Classifications saved to `{output}_assets/classifications.json`
5. More structured CLI with argparse
