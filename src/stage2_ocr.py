#!/usr/bin/env python3
"""
Stage 2: OCR Processing with DeepSeek-OCR via venv subprocess

Processes PDF pages with content-aware prompts based on Stage 1 classifications.
Uses subprocess to run DeepSeek-OCR-2 in a separate venv with transformers 4.46.3.

Usage:
    from stage2_ocr import ocr_pages, pdf_to_page_images

    pages = pdf_to_page_images(pdf_path)
    results = ocr_pages(image_paths, classifications, config)
"""

import os
import re
import json
import time
import tempfile
import subprocess
import shutil
from pathlib import Path
from PIL import Image
import io

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


# =============================================================================
# Prompts (for reference - actual prompts are in stage2_ocr_worker.py)
# =============================================================================

PROMPTS = {
    'text': "Convert this document page to markdown. Preserve all text formatting, headers, lists, and structure.",
    'document': "Convert this document page to markdown. Preserve all text formatting, headers, lists, and structure.",
    'figure': "Describe this image in detail.",
    'diagram': "Describe this diagram in detail, including all elements, connections, and flow.",
    'flowchart': "Describe this flowchart in detail, including all steps, decisions, and flow direction.",
    'table': "Convert this document to markdown. Pay special attention to table formatting with proper alignment.",
    'mixed': "Convert this document page to markdown. Preserve all text, tables, and describe any images or diagrams.",
}


# =============================================================================
# OCR Processing via Subprocess
# =============================================================================

def ocr_pages(image_paths: list, classifications: list, config: dict,
              verbose: bool = False, progress_callback=None) -> list:
    """
    Run OCR on all pages using venv subprocess with DeepSeek-OCR-2.

    Args:
        image_paths: List of paths to page images
        classifications: List of classifications from Stage 1
        config: Configuration dict with 'deepseek_model_path'
        verbose: Print progress
        progress_callback: Optional callback(page_num, total)

    Returns:
        list: OCR results for each page
    """
    if not image_paths:
        return []

    # Get venv Python path from environment
    venv_python = os.environ.get('OCR_VENV_PYTHON', '/opt/venv_ocr/bin/python3')

    # Get model path from config
    model_path = config.get('deepseek_model_path')
    if not model_path:
        # Try default locations
        for candidate in ['/workspace/models/DeepSeek-OCR-2-model', './DeepSeek-OCR-2-model']:
            if os.path.exists(candidate):
                model_path = candidate
                break

    if not model_path or not os.path.exists(model_path):
        raise ValueError(f"DeepSeek model path not found. Set deepseek_model_path in config.")

    # Ensure classifications match pages
    while len(classifications) < len(image_paths):
        classifications.append({
            'page': len(classifications) + 1,
            'type': 'mixed',
            'confidence': 0.5,
            'method': 'default'
        })

    if verbose:
        print(f"OCR processing {len(image_paths)} pages via venv subprocess...")

    # Prepare input data
    input_data = {
        'image_paths': [os.path.abspath(p) for p in image_paths],
        'classifications': classifications,
        'model_path': model_path,
    }

    # Create temp files for IPC
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(input_data, f)
        input_file = f.name

    output_file = input_file.replace('.json', '_output.json')

    try:
        # Build command
        worker_path = os.path.join(os.path.dirname(__file__), 'stage2_ocr_worker.py')
        if not os.path.exists(worker_path):
            worker_path = '/app/stage2_ocr_worker.py'

        cmd = [venv_python, worker_path, input_file, output_file]
        if verbose:
            cmd.append('--verbose')

        if verbose:
            print(f"  Running: {' '.join(cmd)}")

        # Run worker subprocess
        start = time.time()
        result = subprocess.run(
            cmd,
            check=False,  # Don't raise, we handle errors manually
            capture_output=True,
            text=True
        )

        # Print stdout if verbose or on error
        if result.stdout:
            print(result.stdout)

        # Check for errors
        if result.returncode != 0:
            print(f"ERROR: OCR worker failed with code {result.returncode}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

        if verbose:
            elapsed = time.time() - start
            print(f"  Subprocess completed in {elapsed:.1f}s")

        # Load results
        with open(output_file) as f:
            results = json.load(f)

        # Clean results
        cleaned_results = []
        for i, r in enumerate(results):
            text = r.get('text', '')
            classification = r.get('classification', classifications[i])
            cleaned_text = clean_result(text, i + 1, [], classification)

            cleaned_results.append({
                'text': cleaned_text,
                'figures': r.get('figures', []),
                'classification': classification,
            })

            if progress_callback:
                progress_callback(i + 1, len(image_paths))

        return cleaned_results

    except subprocess.CalledProcessError:
        raise  # Already printed error above

    finally:
        # Cleanup temp files
        if os.path.exists(input_file):
            os.unlink(input_file)
        if os.path.exists(output_file):
            os.unlink(output_file)


# =============================================================================
# Result Cleaning
# =============================================================================

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

    # Remove common end tokens
    text = text.replace('<｜end▁of▁sentence｜>', '')
    text = text.replace('<|im_end|>', '')
    text = text.replace('<|endoftext|>', '')

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


# =============================================================================
# PDF Processing Utilities
# =============================================================================

def pdf_to_page_images(pdf_path: str, dpi: int = 200, verbose: bool = False) -> list:
    """
    Convert PDF to list of page images.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering
        verbose: Print progress

    Returns:
        list: Page info dicts with 'image' (PIL Image)
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
            'page_num': i + 1,
            'size': img.size,
        })

        if verbose:
            print(f"  Page {i+1}/{pdf.page_count} - {img.size[0]}x{img.size[1]}")

    pdf.close()
    return pages


def save_pages_to_temp(pages: list, temp_dir: str = None) -> list:
    """
    Save PIL images to temp files for processing.

    Args:
        pages: List of dicts with 'image' (PIL Image)
        temp_dir: Optional temp directory

    Returns:
        list: Paths to saved image files
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="ocr_pages_")

    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    paths = []
    for i, page in enumerate(pages):
        path = os.path.join(temp_dir, f"page_{i+1:04d}.png")
        page['image'].save(path, "PNG")
        paths.append(path)

    return paths


# =============================================================================
# Output Generation
# =============================================================================

def generate_markdown(results: list, pdf_name: str, method: str = 'transformers',
                      diagram_descriptions: dict = None) -> str:
    """
    Generate final markdown from OCR results.

    Args:
        results: List of OCR results from ocr_pages()
        pdf_name: Name of source PDF
        method: Processing method description
        diagram_descriptions: Dict of {page_num: description} from Qwen

    Returns:
        str: Complete markdown document
    """
    combined = []
    diagram_descriptions = diagram_descriptions or {}
    diagram_types = {'diagram', 'flowchart'}
    diagrams_used = 0

    for r in results:
        classification = r.get('classification', {})
        page_num = classification.get('page', 0)
        content_type = classification.get('type', 'unknown')
        confidence = classification.get('confidence', 0)
        ocr_text = r.get('text', '')

        # Check if we have a Qwen description for this page
        diagram_desc = diagram_descriptions.get(page_num)
        has_diagrams = classification.get('has_diagrams', False)
        is_diagram_page = content_type.lower() in diagram_types or (content_type.lower() == 'mixed' and has_diagrams)

        if diagram_desc and is_diagram_page:
            # Use Qwen description for diagram pages
            text = diagram_desc
            page_method = 'qwen3-vl-30b'
            diagrams_used += 1

            # Append DeepSeek text if it found additional content
            if ocr_text and len(ocr_text.strip()) > 50:
                text += f"\n\n---\n*Additional OCR text:*\n\n{ocr_text}"
        else:
            text = ocr_text
            page_method = method

        if text:
            meta = f"<!-- Page {page_num} | Type: {content_type} | Confidence: {confidence:.0%} | Method: {page_method} -->"
            combined.append(f"{meta}\n\n{text}")

    # Build document
    if diagrams_used > 0:
        converter_info = f"Qwen3-VL-30B ({diagrams_used} diagrams) + DeepSeek-OCR (transformers)"
    else:
        converter_info = f"Qwen3-VL-30B + DeepSeek-OCR (transformers)"

    total_figures = sum(len(r.get('figures', [])) for r in results)

    parts = [
        f"# {pdf_name}\n",
        f"*Converted using {converter_info}*\n",
        f"*Pages: {len(results)} | Figures: {total_figures}*\n",
        "\n---\n",
        "\n\n---\n\n".join(combined)
    ]

    return "\n".join(parts)
