#!/usr/bin/env python3
"""
Stage 1: Page Classification with Qwen3-VL-8B

IMPORTANT: This script requires the venv with newer transformers!
    source venv_qwen/bin/activate
    python3 stage1_classifier.py input.pdf -o classifications.json

Analyzes PDF pages and classifies content type (text, table, diagram, etc.)
to enable optimal prompt selection in Stage 2 (OCR).
"""

import torch
import json
import re
import sys
import os
import time
import argparse
from pathlib import Path
from PIL import Image
import io

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


# Classification prompt for Qwen3-VL
CLASSIFY_PROMPT = """Analyze this document page and classify its content.
Respond ONLY with valid JSON in this exact format:
{
    "type": "text|table|diagram|flowchart|figure|mixed",
    "confidence": 0.95,
    "description": "Brief description of the page content",
    "has_text": true,
    "has_images": false,
    "has_tables": false,
    "has_diagrams": false
}

Type definitions:
- text: Primarily text content (paragraphs, lists, headers)
- table: Contains data tables
- diagram: Technical diagrams, architecture diagrams, UML diagrams, sequence diagrams, call flow diagrams, state machines, component diagrams
- flowchart: Process flows, decision trees, workflow diagrams, activity diagrams
- figure: Photos, screenshots, illustrations, graphs, charts with NO connecting arrows or flow
- mixed: Combination of text with significant visual elements

IMPORTANT: If the page contains boxes/nodes connected by arrows showing a flow or sequence, classify as "diagram" or "flowchart", NOT "figure".

Respond with ONLY the JSON, no explanation."""


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: str = None, precision: str = "fp16", verbose: bool = False):
    """
    Load Qwen3-VL model for page classification.

    Requires venv with transformers>=4.50.0 and qwen-vl-utils.
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    # Default to HuggingFace hub
    model_id = model_path or "Qwen/Qwen3-VL-8B-Instruct"
    local_only = model_path is not None and os.path.exists(model_path)

    if verbose:
        print(f"\nLoading Qwen3-VL classifier ({precision})...")
        print(f"  Model: {model_id}")
        print(f"  Local only: {local_only}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  VRAM: {vram:.1f} GB")

    start = time.time()

    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_only
        )

        # Configure dtype
        if precision == "fp8":
            dtype = torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16
        else:
            dtype = torch.float16

        # Force GPU loading
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="cuda:0",
            trust_remote_code=True,
            local_files_only=local_only,
        )
        model.eval()

        if verbose:
            print(f"  Loaded in {time.time() - start:.1f}s")

        return model, processor

    except Exception as e:
        print(f"ERROR: Failed to load Qwen3-VL: {e}")
        print("\nMake sure you're running in the venv:")
        print("  source venv_qwen/bin/activate")
        return None, None


def unload_model(model, processor, verbose: bool = False):
    """Unload model and free VRAM."""
    if model is not None:
        del model
    if processor is not None:
        del processor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if verbose:
        print("  Unloaded classifier, freed VRAM")


# ============================================================================
# Classification Functions
# ============================================================================

def classify_page(model, processor, image: Image.Image, verbose: bool = False) -> dict:
    """Classify a single page using Qwen3-VL."""
    from qwen_vl_utils import process_vision_info

    if model is None or processor is None:
        return None

    try:
        # Prepare conversation for Qwen3-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": CLASSIFY_PROMPT}
                ]
            }
        ]

        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(messages)

        # Process all inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # Decode (skip input tokens)
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if verbose:
            print(f"    VL response: {response[:150]}...")

        # Parse JSON response
        result = parse_response(response)
        if result:
            result['method'] = 'qwen3-vl-8b'
        return result

    except Exception as e:
        if verbose:
            print(f"    VL classification error: {e}")
        return None


def parse_response(response: str) -> dict:
    """Parse JSON response from Qwen3-VL classifier."""
    response = response.strip()

    # Handle markdown code blocks
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0]
    elif "```" in response:
        parts = response.split("```")
        if len(parts) >= 2:
            response = parts[1]

    # Try direct parse
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON object
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    # Validate and normalize
    if 'type' not in result:
        return None

    valid_types = {'text', 'table', 'diagram', 'flowchart', 'figure', 'mixed', 'document'}
    result['type'] = result['type'].lower()
    if result['type'] not in valid_types:
        result['type'] = 'mixed'

    # Ensure confidence is float
    if 'confidence' in result:
        try:
            result['confidence'] = float(result['confidence'])
        except (ValueError, TypeError):
            result['confidence'] = 0.5

    return result


# ============================================================================
# Heuristic Classification (Fallback)
# ============================================================================

def classify_page_heuristic(page_info: dict) -> dict:
    """Classify page using heuristics (fallback when VL unavailable)."""
    text_blocks = page_info.get('text_blocks', 0)
    image_count = page_info.get('image_count', 0)
    drawings = page_info.get('drawings', 0)
    has_tables = page_info.get('has_tables', False)

    if has_tables:
        page_type = 'table'
        confidence = 0.8
    elif drawings > 10 and text_blocks < 5:
        page_type = 'diagram'
        confidence = 0.7
    elif image_count > 0 and text_blocks < 10:
        page_type = 'figure'
        confidence = 0.7
    elif image_count > 0 or drawings > 5:
        page_type = 'mixed'
        confidence = 0.6
    else:
        page_type = 'text'
        confidence = 0.8

    return {
        'type': page_type,
        'confidence': confidence,
        'description': f"Heuristic: {text_blocks} text blocks, {image_count} images, {drawings} drawings",
        'has_text': text_blocks > 0,
        'has_images': image_count > 0,
        'has_tables': has_tables,
        'has_diagrams': drawings > 5,
        'method': 'heuristic'
    }


# ============================================================================
# Batch Processing
# ============================================================================

def classify_pages(model, processor, pages: list, use_heuristic_fallback: bool = True,
                   verbose: bool = False) -> list:
    """Classify all pages in a document."""
    classifications = []
    use_vl = model is not None and processor is not None

    for i, page_info in enumerate(pages):
        # Always show progress
        print(f"  Page {i+1}/{len(pages)}...", end=" ", flush=True)

        result = None

        # Try VL classification
        if use_vl:
            result = classify_page(model, processor, page_info['image'], verbose)

        # Fallback to heuristic
        if result is None and use_heuristic_fallback:
            result = classify_page_heuristic(page_info)

        if result is None:
            result = {
                'type': 'mixed',
                'confidence': 0.0,
                'description': 'Classification failed',
                'method': 'none'
            }

        result['page'] = i + 1
        classifications.append(result)

        # Always show result
        print(f"{result['type']} ({result['confidence']:.0%}) [{result['method']}]", flush=True)

    return classifications


# ============================================================================
# PDF Processing Utilities
# ============================================================================

def extract_page_metadata(page) -> dict:
    """Extract metadata from a PyMuPDF page."""
    text_dict = page.get_text("dict")
    text_blocks = len(text_dict.get("blocks", []))
    image_count = len(page.get_images())
    drawings = len(page.get_drawings())

    # Detect tables
    text = page.get_text()
    lines = text.split('\n')
    table_like_lines = 0
    for line in lines:
        segments = [s.strip() for s in line.split('  ') if s.strip()]
        if len(segments) >= 3:
            table_like_lines += 1
    has_tables = table_like_lines >= 3

    return {
        'text_blocks': text_blocks,
        'image_count': image_count,
        'drawings': drawings,
        'has_tables': has_tables,
    }


def pdf_to_page_images(pdf_path: str, dpi: int = 200, verbose: bool = False) -> list:
    """Convert PDF to list of page images with metadata."""
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required: pip install pymupdf")

    pdf = fitz.open(pdf_path)
    pages = []

    if verbose:
        print(f"Converting {pdf.page_count} pages (DPI={dpi})...")

    for i, page in enumerate(pdf):
        pix = page.get_pixmap(dpi=dpi)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        metadata = extract_page_metadata(page)
        metadata['image'] = img
        metadata['page_num'] = i
        metadata['size'] = img.size

        pages.append(metadata)

        if verbose:
            heur = classify_page_heuristic(metadata)
            print(f"  Page {i+1}/{pdf.page_count} - {img.size[0]}x{img.size[1]} [~{heur['type']}]")

    pdf.close()
    return pages


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Classify PDF pages with Qwen3-VL-8B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: Run in venv with newer transformers!
  source venv_qwen/bin/activate
  python3 stage1_classifier.py document.pdf -o classifications.json

Examples:
  # Classify with Qwen3-VL
  python3 stage1_classifier.py document.pdf -o classifications.json

  # Use heuristic only (no GPU)
  python3 stage1_classifier.py document.pdf -o classifications.json --heuristic

  # Verbose output
  python3 stage1_classifier.py document.pdf -o classifications.json -v
"""
    )

    parser.add_argument('input_pdf', help='Input PDF file')
    parser.add_argument('-o', '--output', required=True, help='Output JSON file')
    parser.add_argument('-m', '--model', help='Path to Qwen3-VL model')
    parser.add_argument('-p', '--precision', choices=['fp16', 'fp8'], default='fp16')
    parser.add_argument('--heuristic', action='store_true', help='Use heuristic only')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--dpi', type=int, default=200)

    return parser.parse_args()


def load_config():
    """Load config from ocr_config.json."""
    config_paths = [
        Path("ocr_config.json"),
        Path(__file__).parent / "ocr_config.json",
        Path("/workspace/data/ocr_config.json"),
    ]
    for path in config_paths:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return {}


def main():
    args = parse_args()

    pdf_path = Path(args.input_pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    # Load config
    config = load_config()
    model_path = args.model or config.get('qwen_model_path')

    print("=" * 60)
    print("Stage 1: Page Classification")
    print("=" * 60)
    print(f"Input:  {pdf_path}")
    print(f"Output: {args.output}")
    print(f"Method: {'heuristic' if args.heuristic else 'qwen3-vl-8b'}")
    print()

    start_time = time.time()

    # Convert PDF to images
    pages = pdf_to_page_images(str(pdf_path), dpi=args.dpi, verbose=args.verbose)

    # Load model (unless heuristic only)
    model, processor = None, None
    if not args.heuristic:
        model, processor = load_model(
            model_path=model_path,
            precision=args.precision,
            verbose=args.verbose
        )
        if model is None:
            print("WARNING: Falling back to heuristic classification")

    # Classify pages
    print(f"\nClassifying {len(pages)} pages...")
    classifications = classify_pages(model, processor, pages, verbose=args.verbose)

    # Unload model
    if model is not None:
        unload_model(model, processor, verbose=args.verbose)

    # Save results
    output_data = {
        'source': str(pdf_path),
        'pages': len(pages),
        'method': 'qwen3-vl-8b' if model else 'heuristic',
        'classifications': classifications
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Summary
    content_types = {}
    for c in classifications:
        ct = c.get('type', 'unknown')
        content_types[ct] = content_types.get(ct, 0) + 1

    elapsed = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"Completed in {elapsed:.1f}s ({elapsed/len(pages):.1f}s per page)")
    print(f"Content types: {content_types}")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
