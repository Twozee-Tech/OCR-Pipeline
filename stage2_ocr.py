#!/usr/bin/env python3
"""
Stage 2: OCR Processing with DeepSeek-OCR

Processes PDF pages with content-aware prompts based on Stage 1 classifications.

Standalone usage:
    python stage2_ocr.py input.pdf --classifications classifications.json --output output.md

As module:
    from stage2_ocr import load_model, ocr_pages
    model, tokenizer = load_model(model_path)
    results = ocr_pages(model, tokenizer, pages, classifications)
"""

import torch
from transformers import AutoModel, AutoTokenizer
import json
import re
import sys
import os
import time
import argparse
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import io

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


# Prompt strategies for different content types
PROMPTS = {
    'text': "<image>\n<|grounding|>Convert the document to markdown.",
    'document': "<image>\n<|grounding|>Convert the document to markdown.",
    'figure': "<image>\nParse the figure.",
    'diagram': "<image>\nDescribe this image in detail.",
    'flowchart': "<image>\nDescribe this image in detail.",
    'table': "<image>\n<|grounding|>Convert the document to markdown.",
    'mixed': "<image>\n<|grounding|>Convert the document to markdown.",
}


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: str, verbose: bool = False):
    """
    Load DeepSeek-OCR model and tokenizer.

    Args:
        model_path: Path to DeepSeek-OCR model
        verbose: Print loading progress

    Returns:
        tuple: (model, tokenizer)
    """
    if verbose:
        print(f"\nLoading DeepSeek-OCR model...")
        print(f"  Path: {model_path}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        _attn_implementation='eager',
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    ).eval()

    if verbose:
        print(f"  Loaded in {time.time() - start:.1f}s")

    return model, tokenizer


def unload_model(model, tokenizer, verbose: bool = False):
    """
    Unload model and free VRAM.

    Args:
        model: DeepSeek-OCR model
        tokenizer: DeepSeek-OCR tokenizer
        verbose: Print progress
    """
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if verbose:
        print("  Unloaded OCR model, freed VRAM")


# ============================================================================
# Prompt Selection
# ============================================================================

def get_prompt(classification: dict) -> str:
    """
    Get optimal DeepSeek-OCR prompt based on classification.

    Args:
        classification: Dict with 'type' and 'confidence'

    Returns:
        str: Prompt for DeepSeek-OCR
    """
    if classification is None:
        return PROMPTS['mixed']

    page_type = classification.get('type', 'mixed')
    confidence = classification.get('confidence', 0.5)

    # Low confidence -> safer mixed prompt
    if confidence < 0.7:
        return PROMPTS['mixed']

    return PROMPTS.get(page_type, PROMPTS['mixed'])


# ============================================================================
# OCR Processing
# ============================================================================

def ocr_page(model, tokenizer, image: Image.Image, classification: dict,
             temp_dir: str, output_dir: str = None, verbose: bool = False) -> dict:
    """
    Run OCR on a single page.

    Args:
        model: DeepSeek-OCR model
        tokenizer: DeepSeek-OCR tokenizer
        image: PIL Image of the page
        classification: Classification from Stage 1
        temp_dir: Temporary directory for processing
        output_dir: Output directory for extracted figures
        verbose: Print debug info

    Returns:
        dict: OCR result with 'text', 'figures', 'classification'
    """
    page_num = classification.get('page', 0)
    content_type = classification.get('type', 'mixed')

    # Save image temporarily
    temp_image = os.path.join(temp_dir, f"page_{page_num:03d}.png")
    image.save(temp_image, quality=95)

    # Select prompt
    prompt = get_prompt(classification)

    if verbose:
        print(f"    Prompt type: {content_type}")

    # Run OCR
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=temp_image,
        output_path=temp_dir,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=False
    )

    # Extract text from result
    ocr_text = ""

    # Check direct result
    if result and isinstance(result, str) and len(result.strip()) > 0:
        ocr_text = result

    # Check result.mmd file
    if not ocr_text:
        mmd_file = os.path.join(temp_dir, 'result.mmd')
        if os.path.exists(mmd_file):
            with open(mmd_file, 'r', encoding='utf-8') as f:
                ocr_text = f.read()
            os.remove(mmd_file)

    # Check 'other' directories
    if not ocr_text:
        for root, dirs, files in os.walk(temp_dir):
            if 'other' in dirs:
                other_dir = os.path.join(root, 'other')
                parts = []
                for fname in sorted(os.listdir(other_dir)):
                    fpath = os.path.join(other_dir, fname)
                    if os.path.isfile(fpath):
                        try:
                            with open(fpath, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    parts.append(content)
                        except:
                            pass
                if parts:
                    ocr_text = '\n\n'.join(parts)
                shutil.rmtree(other_dir, ignore_errors=True)
                break

    # Check for .md/.txt files
    if not ocr_text:
        for root, dirs, files in os.walk(temp_dir):
            for fname in files:
                if fname.endswith(('.md', '.txt')):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content and len(content) > 10:
                                ocr_text = content
                                os.remove(fpath)
                                break
                    except:
                        pass
            if ocr_text:
                break

    # Initialize results
    results = {
        'text': ocr_text,
        'figures': [],
        'classification': classification,
    }

    # For visual content, get description
    if content_type in ('diagram', 'figure', 'flowchart'):
        vl_desc = classification.get('description', '')
        if not vl_desc or classification.get('method') == 'heuristic':
            desc_result = model.infer(
                tokenizer,
                prompt="<image>\nDescribe this image in detail.",
                image_file=temp_image,
                output_path=temp_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=False
            )
            if desc_result:
                results['ocr_description'] = desc_result

    # Extract figures
    if output_dir:
        figures_dir = os.path.join(temp_dir, 'images')
        if os.path.exists(figures_dir):
            for fig_file in os.listdir(figures_dir):
                if fig_file.endswith(('.jpg', '.png')):
                    src = os.path.join(figures_dir, fig_file)
                    dst_name = f"page{page_num:03d}_{fig_file}"
                    dst_dir = os.path.join(output_dir, 'figures')
                    os.makedirs(dst_dir, exist_ok=True)
                    dst = os.path.join(dst_dir, dst_name)
                    shutil.copy2(src, dst)
                    results['figures'].append(dst_name)
            shutil.rmtree(figures_dir, ignore_errors=True)

    return results


def ocr_pages(model, tokenizer, pages: list, classifications: list,
              output_dir: str = None, verbose: bool = False) -> list:
    """
    Run OCR on all pages.

    Args:
        model: DeepSeek-OCR model
        tokenizer: DeepSeek-OCR tokenizer
        pages: List of page info dicts with 'image'
        classifications: List of classifications from Stage 1
        output_dir: Output directory for figures
        verbose: Print progress

    Returns:
        list: OCR results for each page
    """
    results = []
    temp_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")

    try:
        for i, (page_info, classification) in enumerate(zip(pages, classifications)):
            page_start = time.time()
            content_type = classification.get('type', 'mixed')
            confidence = classification.get('confidence', 0)

            if verbose:
                print(f"\n--- Page {i+1}/{len(pages)} [{content_type} @ {confidence:.0%}] ---")

            ocr_result = ocr_page(
                model, tokenizer,
                page_info['image'],
                classification,
                temp_dir,
                output_dir,
                verbose
            )

            # Clean result
            cleaned_text = clean_result(
                ocr_result['text'],
                i + 1,
                ocr_result.get('figures', []),
                classification
            )

            ocr_result['text'] = cleaned_text
            results.append(ocr_result)

            if verbose:
                elapsed = time.time() - page_start
                print(f"  {len(cleaned_text)} chars | {len(ocr_result.get('figures', []))} figures | {elapsed:.1f}s")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


# ============================================================================
# Result Cleaning
# ============================================================================

def clean_result(text: str, page_num: int, figures: list, classification: dict = None) -> str:
    """
    Clean OCR output text.

    Args:
        text: Raw OCR text
        page_num: Page number
        figures: List of figure filenames
        classification: Classification info

    Returns:
        str: Cleaned markdown text
    """
    if not text:
        return ""

    # Remove end token
    text = text.replace('<｜end▁of▁sentence｜>', '')

    # Track figure index
    fig_idx = [0]

    def replace_ref(match):
        content = match.group(1)
        if content == 'image':
            if fig_idx[0] < len(figures):
                fig_name = figures[fig_idx[0]]
                fig_idx[0] += 1
                return f'\n\n![Figure](figures/{fig_name})\n\n'
            return '\n\n[Figure]\n\n'
        return content

    # Replace grounding tags with figure references
    pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>.*?<\|/det\|>'
    text = re.sub(pattern, replace_ref, text, flags=re.DOTALL)

    # Fix table formatting
    text = fix_table_formatting(text)

    # Clean whitespace
    text = re.sub(r'\n{4,}', '\n\n', text)
    text = text.strip()

    return text


def fix_table_formatting(text: str) -> str:
    """Improve table formatting in markdown."""
    lines = text.split('\n')
    result = []
    in_table = False

    for line in lines:
        if '|' in line and line.count('|') >= 2:
            if not in_table:
                in_table = True
                result.append('')  # Blank line before table
            result.append(line)
        else:
            if in_table:
                in_table = False
                result.append('')  # Blank line after table
            result.append(line)

    return '\n'.join(result)


# ============================================================================
# PDF Processing Utilities
# ============================================================================

def pdf_to_page_images(pdf_path: str, dpi: int = 200, verbose: bool = False) -> list:
    """
    Convert PDF to list of page images.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering
        verbose: Print progress

    Returns:
        list: Page info dicts with 'image'
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required: pip install pymupdf")

    pdf = fitz.open(pdf_path)
    pages = []

    if verbose:
        print(f"Loading {pdf.page_count} pages (DPI={dpi})...")

    for i, page in enumerate(pdf):
        pix = page.get_pixmap(dpi=dpi)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        pages.append({
            'image': img,
            'page_num': i,
            'size': img.size,
        })

        if verbose:
            print(f"  Page {i+1}/{pdf.page_count} - {img.size[0]}x{img.size[1]}")

    pdf.close()
    return pages


# ============================================================================
# Output Generation
# ============================================================================

def generate_markdown(results: list, pdf_name: str, classifier_method: str = 'unknown') -> str:
    """
    Generate final markdown from OCR results.

    Args:
        results: List of OCR results from ocr_pages()
        pdf_name: Name of source PDF
        classifier_method: Method used for classification

    Returns:
        str: Complete markdown document
    """
    combined = []

    for r in results:
        classification = r.get('classification', {})
        page_num = classification.get('page', 0)
        content_type = classification.get('type', 'unknown')
        confidence = classification.get('confidence', 0)
        method = classification.get('method', 'unknown')
        text = r.get('text', '')

        if text:
            # Metadata comment
            meta = f"<!-- Page {page_num} | Type: {content_type} | Confidence: {confidence:.0%} | Method: {method} -->"

            # Add VL description for visual content
            vl_desc = classification.get('description', '')
            if content_type in ('diagram', 'figure', 'flowchart') and vl_desc:
                text += f"\n\n> **Page Analysis:** {vl_desc}\n"

            # Add OCR description if available
            if 'ocr_description' in r:
                text += f"\n\n> **OCR Description:** {r['ocr_description']}\n"

            combined.append(f"{meta}\n\n{text}")

    # Determine converter info
    if classifier_method == 'qwen3-vl-8b':
        converter_info = "Qwen3-VL-8B + DeepSeek-OCR"
    else:
        converter_info = "DeepSeek-OCR (heuristic classification)"

    # Build document
    total_figures = sum(len(r.get('figures', [])) for r in results)

    parts = [
        f"# {pdf_name}\n",
        f"*Converted using {converter_info}*\n",
        f"*Pages: {len(results)} | Figures: {total_figures}*\n",
        "\n---\n",
        "\n\n---\n\n".join(combined)
    ]

    return "\n".join(parts)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 2: OCR with DeepSeek-OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OCR with classifications from Stage 1
  python stage2_ocr.py document.pdf --classifications class.json -o output.md

  # Without classifications (uses mixed prompt for all pages)
  python stage2_ocr.py document.pdf -o output.md

  # Verbose output
  python stage2_ocr.py document.pdf --classifications class.json -o output.md -v
"""
    )

    parser.add_argument('input_pdf', help='Input PDF file')
    parser.add_argument('-o', '--output', required=True, help='Output markdown file')
    parser.add_argument('-c', '--classifications', help='Classifications JSON from Stage 1')
    parser.add_argument('-m', '--model', help='Path to DeepSeek-OCR model')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--dpi', type=int, default=200, help='PDF rendering DPI (default: 200)')

    return parser.parse_args()


def find_model_path(custom_path: str = None) -> str:
    """Find DeepSeek-OCR model path."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    # Common paths
    candidates = [
        './DeepSeek-OCR-model',
        '../DeepSeek-OCR-model',
        '/workspace/DeepSeek-OCR-model',
        '/workspace/models/DeepSeek-OCR-model',
        os.path.expanduser('~/DeepSeek-OCR-model'),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def main():
    args = parse_args()

    pdf_path = Path(args.input_pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    # Find model
    model_path = find_model_path(args.model)
    if model_path is None:
        print("ERROR: DeepSeek-OCR model not found.")
        print("Download with: git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR DeepSeek-OCR-model")
        sys.exit(1)

    print("=" * 60)
    print("Stage 2: OCR Processing")
    print("=" * 60)
    print(f"Input:  {pdf_path}")
    print(f"Output: {args.output}")
    print(f"Model:  {model_path}")
    print()

    start_time = time.time()

    # Load classifications (if provided)
    classifications = None
    classifier_method = 'none'
    if args.classifications:
        if os.path.exists(args.classifications):
            with open(args.classifications, 'r') as f:
                data = json.load(f)
                classifications = data.get('classifications', [])
                classifier_method = data.get('method', 'unknown')
            print(f"Loaded {len(classifications)} classifications from {args.classifications}")
        else:
            print(f"WARNING: Classifications file not found: {args.classifications}")

    # Convert PDF to images
    pages = pdf_to_page_images(str(pdf_path), dpi=args.dpi, verbose=args.verbose)

    # Create default classifications if not provided
    if classifications is None or len(classifications) != len(pages):
        if classifications is not None:
            print(f"WARNING: Classification count mismatch ({len(classifications)} vs {len(pages)} pages)")
        print("Using default 'mixed' classification for all pages")
        classifications = [
            {'page': i + 1, 'type': 'mixed', 'confidence': 0.5, 'method': 'default'}
            for i in range(len(pages))
        ]
        classifier_method = 'default'

    # Load model
    model, tokenizer = load_model(model_path, verbose=args.verbose)

    # Setup output directory
    output_path = Path(args.output)
    output_dir = output_path.parent / f"{output_path.stem}_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run OCR
    print(f"\nProcessing {len(pages)} pages...")
    results = ocr_pages(model, tokenizer, pages, classifications, str(output_dir), args.verbose)

    # Unload model
    unload_model(model, tokenizer, verbose=args.verbose)

    # Generate output
    markdown = generate_markdown(results, pdf_path.stem, classifier_method)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    # Summary
    elapsed = time.time() - start_time
    total_figures = sum(len(r.get('figures', [])) for r in results)
    total_chars = sum(len(r.get('text', '')) for r in results)

    print(f"\n" + "=" * 60)
    print(f"Completed in {elapsed:.1f}s ({elapsed/len(pages):.1f}s per page)")
    print(f"Total characters: {total_chars}")
    print(f"Figures extracted: {total_figures}")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
