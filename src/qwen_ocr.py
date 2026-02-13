#!/usr/bin/env python3
"""
Qwen OCR - Single-phase OCR Pipeline (Experimental)

Uses Qwen3-VL-30B for everything: OCR, tables, diagrams, figures.
One comprehensive prompt per page, no cross-page context, batch processed via vLLM.

Usage:
    python3 qwen_ocr.py input.pdf [output.md] [--dpi 200] [-v] [--max-tokens 4096]
"""

import json
import sys
import os
import time
import argparse
import tempfile
import shutil
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from stage2_ocr import (
    pdf_to_page_images,
    clean_result,
    merge_page_boundaries,
    fix_table_formatting,
)
from qwen_processor import save_pages_to_temp, prepare_inputs


# =============================================================================
# Prompt
# =============================================================================

OCR_PROMPT = """1. Do OCR of the page
2. diagrams/images had to be included and described in very detail matter
3. Put Picture/diagram description in Exact same spot as it was on page
4. Do not add any text from yourself, that is not on page"""


# =============================================================================
# Config
# =============================================================================

CONFIG_PATHS = [
    './ocr_config.json',
    '/app/ocr_config.json',
    '/workspace/ocr_config.json',
    '/workspace/data/ocr_config.json',
    os.path.expanduser('~/.ocr_config.json'),
]


def load_config(config_path: str = None) -> dict:
    """Load configuration from JSON file."""
    paths_to_check = [config_path] if config_path else CONFIG_PATHS
    for path in paths_to_check:
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                print(f"Loaded config from: {path}")
                return config
            except Exception as e:
                print(f"Warning: Failed to load config from {path}: {e}")
    return {}


def find_model_path(config: dict, key: str, candidates: list = None) -> str:
    """Find model path from config or candidate locations."""
    if config.get(key) and os.path.exists(config[key]):
        return config[key]
    for path in (candidates or []):
        if os.path.exists(path):
            return path
    return None


# =============================================================================
# Model Creation
# =============================================================================

def create_qwen_ocr_llm(model_path: str, config: dict = None) -> LLM:
    """
    Create vLLM instance for Qwen3-VL with higher memory for single-phase OCR.

    Uses qwen_ocr_gpu_memory_utilization (default 0.80) and
    qwen_ocr_max_model_len (default 16384) since no DeepSeek phase follows.
    """
    config = config or {}

    return LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=config.get('qwen_ocr_gpu_memory_utilization', 0.80),
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=config.get('qwen_ocr_max_model_len', 40960),
        seed=0,
    )


# =============================================================================
# OCR Processing
# =============================================================================

def make_ocr_msg(image_path: str) -> list:
    """Create OCR message for a page image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]


BATCH_SIZE = 20  # Process pages in chunks to avoid OOM


def ocr_all_pages(llm: LLM, processor: AutoProcessor, image_paths: list,
                   max_tokens: int = 4096, verbose: bool = False) -> list:
    """
    OCR all pages via vLLM, in batches of BATCH_SIZE.

    Each page is an independent request — no shared context between pages.

    Args:
        llm: vLLM model instance
        processor: AutoProcessor
        image_paths: Paths to page images
        max_tokens: Max output tokens per page
        verbose: Print progress

    Returns:
        list: Raw OCR text for each page
    """
    if not image_paths:
        return []

    params = SamplingParams(
        temperature=0.25,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        max_tokens=max_tokens,
        greedy=False
    )
    total = len(image_paths)
    results = []

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_paths = image_paths[batch_start:batch_end]
        batch_num = batch_start // BATCH_SIZE + 1
        num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        if verbose:
            print(f"Batch {batch_num}/{num_batches}: preparing pages {batch_start+1}-{batch_end}...")

        inputs = [prepare_inputs(make_ocr_msg(p), processor) for p in batch_paths]

        if verbose:
            print(f"Batch {batch_num}/{num_batches}: running inference ({len(inputs)} pages)...")

        start = time.time()
        outputs = llm.generate(inputs, params)
        elapsed = time.time() - start

        if verbose:
            print(f"Batch {batch_num}/{num_batches}: done in {elapsed:.1f}s ({elapsed/len(inputs):.1f}s/page)")

        for i, output in enumerate(outputs):
            text = output.outputs[0].text.strip()
            # Strip chain of thought: remove everything up to and including <\think>
            if "<\\think>" in text:
                text = text.split("<\\think>")[-1]
            results.append(text)
            if verbose:
                print(f"  Page {batch_start+i+1}: {len(text)} chars")

    return results


# =============================================================================
# Output Generation
# =============================================================================

def generate_markdown(results: list, pdf_name: str) -> str:
    """
    Generate final markdown from Qwen OCR results.

    Args:
        results: List of dicts with 'text' key per page
        pdf_name: Name of source PDF

    Returns:
        str: Complete markdown document
    """
    combined = []

    for r in results:
        page_num = r.get('page_num', 0)
        text = r.get('text', '')
        if text:
            meta = f"<!-- Page {page_num} | Method: qwen3-vl-ocr -->"
            combined.append(f"{meta}\n\n{text}")

    # Merge page boundaries: join incomplete sentences across pages
    combined = merge_page_boundaries(combined)

    parts = [
        f"# {pdf_name}\n",
        f"*Converted using Qwen3-VL-30B (single-phase OCR)*\n",
        f"*Pages: {len(results)}*\n",
        "\n---\n",
        "\n\n---\n\n".join(combined),
    ]

    return "\n".join(parts)


# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline(
    pdf_path: str,
    output_path: str = None,
    dpi: int = 200,
    max_tokens: int = 4096,
    verbose: bool = False,
    debug: bool = False,
    config: dict = None,
) -> str:
    """
    Run the single-phase Qwen OCR pipeline.

    Args:
        pdf_path: Path to input PDF
        output_path: Path to output markdown (default: input.md)
        dpi: PDF rendering resolution
        max_tokens: Max output tokens per page
        verbose: Print detailed progress
        debug: Enable vLLM debug logging
        config: Configuration dict

    Returns:
        str: Path to output markdown file
    """
    config = config or {}
    pdf_path = Path(pdf_path)

    # Configure vLLM logging
    if debug or verbose:
        import logging
        vllm_logger = logging.getLogger("vllm")
        vllm_logger.setLevel(logging.DEBUG if debug else logging.INFO)
        print("vLLM debug logging enabled" if debug else "vLLM info logging enabled")

    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    if output_path is None:
        output_path = pdf_path.with_suffix('.md')
    output_path = Path(output_path)

    temp_dir = tempfile.mkdtemp(prefix="qwen_ocr_")

    if verbose:
        print("=" * 60)
        print("Qwen OCR - Single-Phase Pipeline (Experimental)")
        print("=" * 60)
        print(f"Input:      {pdf_path}")
        print(f"Output:     {output_path}")
        print(f"Max tokens: {max_tokens}")
        print()
    else:
        print(f"Qwen OCR: {pdf_path.name} -> {output_path.name}")

    total_start = time.time()

    try:
        # =================================================================
        # Load PDF pages
        # =================================================================
        if verbose:
            print("Loading PDF pages...")

        pages = pdf_to_page_images(str(pdf_path), dpi=dpi, verbose=verbose)
        num_pages = len(pages)
        image_paths = save_pages_to_temp(pages, temp_dir)

        if verbose:
            print(f"Saved {num_pages} page images to temp directory")

        # =================================================================
        # Load Qwen model
        # =================================================================
        if verbose:
            print("\nLoading Qwen3-VL model...")

        qwen_path = find_model_path(
            config, 'qwen_model_path',
            ['/workspace/models/Qwen3-VL-30B-model', './Qwen3-VL-30B-model'],
        )

        if qwen_path is None:
            print("ERROR: Qwen3-VL model not found.")
            print("Set qwen_model_path in config or download to /workspace/models/")
            sys.exit(1)

        if verbose:
            print(f"Model: {qwen_path}")

        llm = create_qwen_ocr_llm(qwen_path, config)
        processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)

        # =================================================================
        # OCR all pages (single batch)
        # =================================================================
        if verbose:
            print("\n" + "=" * 60)
            print("OCR Processing (Qwen3-VL batch)")
            print("=" * 60)

        raw_texts = ocr_all_pages(
            llm, processor, image_paths,
            max_tokens=max_tokens, verbose=verbose,
        )

        # Unload model
        if verbose:
            print("\nUnloading model...")
        del llm, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # =================================================================
        # Clean results
        # =================================================================
        if verbose:
            print("\nCleaning results...")

        results = []
        for i, text in enumerate(raw_texts):
            cleaned = clean_result(text, i + 1, [], None)
            results.append({
                'text': cleaned,
                'page_num': i + 1,
            })

        # =================================================================
        # Generate output
        # =================================================================
        if verbose:
            print("\nGenerating markdown output...")

        markdown = generate_markdown(results, pdf_path.stem)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        # =================================================================
        # Summary
        # =================================================================
        total_time = time.time() - total_start
        total_chars = sum(len(r['text']) for r in results)

        if verbose:
            print(f"\nCompleted in {total_time:.1f}s ({total_time/num_pages:.1f}s/page)")
            print(f"Total characters: {total_chars}")
            print(f"Output: {output_path}")
            print("=" * 60)
        else:
            print(f"Done in {total_time:.1f}s -> {output_path}")

        return str(output_path)

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen OCR - Single-phase PDF to Markdown (Experimental)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 qwen_ocr.py document.pdf
  python3 qwen_ocr.py document.pdf output.md
  python3 qwen_ocr.py document.pdf -v --max-tokens 6000
  python3 qwen_ocr.py document.pdf --dpi 300
"""
    )

    parser.add_argument('input_pdf', help='Input PDF file')
    parser.add_argument('output_md', nargs='?', default=None, help='Output markdown file')
    parser.add_argument('--dpi', type=int, default=200, help='PDF rendering DPI (default: 200)')
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Max output tokens per page (default: 4096)')
    parser.add_argument('--config', default=None, help='Path to config JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--debug', action='store_true', help='Enable vLLM debug logging')

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    run_pipeline(
         pdf_path=args.input_pdf,
         output_path=args.output_md,
         dpi=args.dpi,
         max_tokens=args.max_tokens,
         verbose=args.verbose,
         debug=args.debug,
         config=config,
     )


if __name__ == "__main__":
    main()
