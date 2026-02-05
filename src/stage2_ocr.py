#!/usr/bin/env python3
"""
Stage 2: OCR Processing with DeepSeek-OCR via vLLM

Processes PDF pages with content-aware prompts based on Stage 1 classifications.

Usage:
    from stage2_ocr import create_ocr_llm, ocr_pages, pdf_to_page_images

    llm = create_ocr_llm(model_path, config)
    processor = AutoProcessor.from_pretrained(model_path)
    pages = pdf_to_page_images(pdf_path)
    results = ocr_pages(llm, processor, image_paths, classifications)
"""

import os
import re
import json
import time
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import io

import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


# =============================================================================
# Prompts & Parameters
# =============================================================================

# Prompt strategies for different content types
PROMPTS = {
    'text': "Convert this document page to markdown. Preserve all text formatting, headers, lists, and structure.",
    'document': "Convert this document page to markdown. Preserve all text formatting, headers, lists, and structure.",
    'figure': "Describe this image in detail.",
    'diagram': "Describe this diagram in detail, including all elements, connections, and flow.",
    'flowchart': "Describe this flowchart in detail, including all steps, decisions, and flow direction.",
    'table': "Convert this document to markdown. Pay special attention to table formatting with proper alignment.",
    'mixed': "Convert this document page to markdown. Preserve all text, tables, and describe any images or diagrams.",
}

OCR_PARAMS = SamplingParams(temperature=0, max_tokens=4096, top_k=-1)


# =============================================================================
# Model Creation
# =============================================================================

def create_ocr_llm(model_path: str, config: dict = None) -> LLM:
    """
    Create vLLM instance for DeepSeek-OCR model.

    Args:
        model_path: Path to DeepSeek-OCR model
        config: Optional config dict with gpu_memory_utilization, max_model_len

    Returns:
        LLM: vLLM model instance
    """
    config = config or {}

    return LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=config.get('ocr_gpu_memory_utilization', 0.50),
        max_model_len=config.get('ocr_max_model_len', 8192),
        tensor_parallel_size=torch.cuda.device_count(),
    )


def unload_llm(llm: LLM):
    """Unload vLLM model and free VRAM."""
    if llm is not None:
        del llm

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Prompt Selection
# =============================================================================

def get_prompt(classification: dict) -> str:
    """
    Get optimal OCR prompt based on classification.

    Args:
        classification: Dict with 'type' and 'confidence'

    Returns:
        str: Prompt for OCR
    """
    if classification is None:
        return PROMPTS['mixed']

    page_type = classification.get('type', 'mixed')
    confidence = classification.get('confidence', 0.5)

    # Low confidence -> safer mixed prompt
    if confidence < 0.7:
        return PROMPTS['mixed']

    return PROMPTS.get(page_type, PROMPTS['mixed'])


# =============================================================================
# Input Preparation
# =============================================================================

def prepare_ocr_input(image_path: str, classification: dict, processor: AutoProcessor) -> dict:
    """
    Prepare OCR input for vLLM.

    Args:
        image_path: Path to page image
        classification: Classification from Stage 1
        processor: AutoProcessor

    Returns:
        dict: Input for vLLM generate()
    """
    prompt = get_prompt(classification)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # For DeepSeek-OCR, we use a simpler approach
    # The model expects image in multi_modal_data
    return {
        "prompt": text,
        "multi_modal_data": {"image": f"file://{os.path.abspath(image_path)}"},
    }


# =============================================================================
# OCR Processing
# =============================================================================

def ocr_pages(llm: LLM, processor: AutoProcessor, image_paths: list,
              classifications: list, verbose: bool = False,
              progress_callback=None) -> list:
    """
    Run OCR on all pages using vLLM batching.

    Args:
        llm: vLLM model instance
        processor: AutoProcessor
        image_paths: List of paths to page images
        classifications: List of classifications from Stage 1
        verbose: Print progress
        progress_callback: Optional callback(page_num, total)

    Returns:
        list: OCR results for each page
    """
    if not image_paths:
        return []

    # Ensure classifications match pages
    while len(classifications) < len(image_paths):
        classifications.append({
            'page': len(classifications) + 1,
            'type': 'mixed',
            'confidence': 0.5,
            'method': 'default'
        })

    if verbose:
        print(f"OCR processing {len(image_paths)} pages...")

    # Prepare all inputs
    inputs = [
        prepare_ocr_input(path, cls, processor)
        for path, cls in zip(image_paths, classifications)
    ]

    # Batch generate
    start = time.time()
    outputs = llm.generate(inputs, OCR_PARAMS)

    if verbose:
        elapsed = time.time() - start
        print(f"  Batch completed in {elapsed:.1f}s ({elapsed/len(image_paths):.1f}s/page)")

    # Process results
    results = []
    for i, (output, classification) in enumerate(zip(outputs, classifications)):
        text = output.outputs[0].text
        cleaned = clean_result(text, i + 1, [], classification)

        results.append({
            'text': cleaned,
            'figures': [],
            'classification': classification,
        })

        if progress_callback:
            progress_callback(i + 1, len(image_paths))

        if verbose:
            print(f"  Page {i+1}: {len(cleaned)} chars [{classification.get('type', 'mixed')}]")

    return results


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
    Save PIL images to temp files for vLLM processing.

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

def generate_markdown(results: list, pdf_name: str, method: str = 'vllm',
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
        converter_info = f"Qwen3-VL-30B ({diagrams_used} diagrams) + DeepSeek-OCR (vLLM)"
    else:
        converter_info = f"Qwen3-VL-30B + DeepSeek-OCR (vLLM)"

    total_figures = sum(len(r.get('figures', [])) for r in results)

    parts = [
        f"# {pdf_name}\n",
        f"*Converted using {converter_info}*\n",
        f"*Pages: {len(results)} | Figures: {total_figures}*\n",
        "\n---\n",
        "\n\n---\n\n".join(combined)
    ]

    return "\n".join(parts)
