#!/usr/bin/env python3
"""
OCR Pipeline v3.0 - vLLM Edition

Two-stage PDF to Markdown conversion using vLLM for fast batched inference:
  Stage 1: Qwen3-VL-30B classifies pages + describes diagrams
  Stage 2: DeepSeek-OCR-2 processes pages with content-aware prompts

Usage:
    python3 ocr_pipeline.py input.pdf [output.md] [-v] [--diagrams]

    # Basic usage
    python3 ocr_pipeline.py document.pdf

    # With diagram description
    python3 ocr_pipeline.py document.pdf --describe-diagrams

    # Verbose output
    python3 ocr_pipeline.py document.pdf -v
"""

import json
import sys
import os
import time
import argparse
import tempfile
import shutil
from pathlib import Path

from transformers import AutoProcessor

from qwen_processor import (
    create_qwen_llm,
    unload_llm as unload_qwen,
    classify_pages,
    describe_diagrams,
    save_pages_to_temp,
)
from stage2_ocr import (
    ocr_pages,
    pdf_to_page_images,
    generate_markdown,
)


# =============================================================================
# Progress Bar
# =============================================================================

class ProgressBar:
    """Simple terminal progress bar."""

    def __init__(self, total: int, width: int = 40, prefix: str = "Progress"):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0
        self.stage = ""
        self.enabled = True

    def update(self, current: int = None, stage: str = None):
        """Update progress bar."""
        if not self.enabled:
            return

        if current is not None:
            self.current = current
        if stage is not None:
            self.stage = stage

        percent = min(100, int(100 * self.current / self.total)) if self.total > 0 else 0
        filled = int(self.width * percent / 100)
        bar = "█" * filled + "░" * (self.width - filled)

        stage_text = f" | {self.stage}" if self.stage else ""
        sys.stdout.write(f"\r{self.prefix}: [{bar}] {percent:3d}%{stage_text}    ")
        sys.stdout.flush()

    def increment(self, amount: int = 1, stage: str = None):
        """Increment progress."""
        self.update(self.current + amount, stage)

    def finish(self, message: str = "Done!"):
        """Complete the progress bar."""
        if not self.enabled:
            return
        self.current = self.total
        self.update()
        print(f"\n{message}")


# =============================================================================
# Configuration
# =============================================================================

# Diagram types that trigger description
DIAGRAM_TYPES = {'diagram', 'flowchart'}

# Config file paths
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


def find_model_path(config: dict, key: str, custom_path: str = None, candidates: list = None) -> str:
    """Find model path from config, custom path, or candidate locations."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    if config.get(key) and os.path.exists(config[key]):
        return config[key]

    for path in (candidates or []):
        if os.path.exists(path):
            return path

    return None


# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline(
    pdf_path: str,
    output_path: str = None,
    describe_diagrams_flag: bool = False,
    dpi: int = 200,
    verbose: bool = False,
    keep_assets: bool = False,
    config: dict = None
) -> str:
    """
    Run the two-stage OCR pipeline with vLLM.

    Stage 1: Qwen3-VL-30B classifies pages + describes diagrams
    Stage 2: DeepSeek-OCR-2 processes pages with content-aware prompts

    Args:
        pdf_path: Path to input PDF
        output_path: Path to output markdown (default: input.md)
        describe_diagrams_flag: Generate detailed diagram descriptions
        dpi: PDF rendering resolution
        verbose: Print detailed progress
        keep_assets: Keep intermediate files
        config: Configuration dict

    Returns:
        str: Path to output markdown file
    """
    config = config or {}
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    # Default output path
    if output_path is None:
        output_path = pdf_path.with_suffix('.md')
    output_path = Path(output_path)

    # Create temp directory for page images
    temp_dir = tempfile.mkdtemp(prefix="ocr_pipeline_")

    # Show header
    if verbose:
        print("=" * 60)
        print("OCR Pipeline v3.0 - vLLM Edition")
        print("=" * 60)
        print(f"Input:            {pdf_path}")
        print(f"Output:           {output_path}")
        print(f"Describe diagrams: {describe_diagrams_flag}")
        print()
    else:
        print(f"OCR: {pdf_path.name} → {output_path.name}")

    total_start = time.time()

    try:
        # =====================================================================
        # Load PDF pages
        # =====================================================================
        if verbose:
            print("Loading PDF pages...")

        pages = pdf_to_page_images(str(pdf_path), dpi=dpi, verbose=verbose)
        num_pages = len(pages)

        # Save pages to temp files for vLLM (requires file:// URIs)
        image_paths = save_pages_to_temp(pages, temp_dir)

        if verbose:
            print(f"Saved {num_pages} page images to temp directory")

        # Calculate progress steps
        total_steps = num_pages * 2 + 1  # classify + OCR + final
        if describe_diagrams_flag:
            total_steps += num_pages // 2

        progress = ProgressBar(total_steps, prefix="OCR")
        progress.enabled = not verbose

        # =====================================================================
        # Stage 1: Qwen Classification + Diagram Description
        # =====================================================================
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 1: Page Classification (Qwen3-VL-30B)")
            print("=" * 60)

        progress.update(0, "Stage 1: Loading Qwen model")

        # Find Qwen model
        qwen_path = find_model_path(
            config, 'qwen_model_path', None,
            ['/workspace/models/Qwen3-VL-30B-model', './Qwen3-VL-30B-model']
        )

        if qwen_path is None:
            print("ERROR: Qwen3-VL model not found.")
            print("Set qwen_model_path in config or download to /workspace/models/")
            sys.exit(1)

        if verbose:
            print(f"Loading Qwen3-VL from: {qwen_path}")

        qwen_llm = create_qwen_llm(qwen_path, config)
        qwen_processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)

        # Classify pages
        progress.update(stage="Stage 1: Classifying pages")
        classifications = classify_pages(qwen_llm, qwen_processor, image_paths, verbose=verbose)

        progress.increment(num_pages, "Stage 1: Classification complete")

        # Print classification summary
        if verbose:
            content_types = {}
            for c in classifications:
                ct = c.get('type', 'unknown')
                content_types[ct] = content_types.get(ct, 0) + 1
            print(f"\nContent types: {content_types}")

        # =====================================================================
        # Stage 1.5: Diagram Description (optional)
        # =====================================================================
        diagram_descriptions = {}

        # Find diagram pages
        diagram_pages = [
            c for c in classifications
            if c.get('type', '').lower() in DIAGRAM_TYPES
            or (c.get('type', '').lower() == 'mixed' and c.get('has_diagrams', False))
        ]

        if describe_diagrams_flag and diagram_pages:
            if verbose:
                print("\n" + "=" * 60)
                print(f"STAGE 1.5: Diagram Description ({len(diagram_pages)} pages)")
                print("=" * 60)

            progress.update(stage=f"Stage 1.5: Describing {len(diagram_pages)} diagrams")

            diagram_descriptions = describe_diagrams(
                qwen_llm, qwen_processor, image_paths,
                classifications=classifications, verbose=verbose
            )

            progress.increment(len(diagram_pages), "Stage 1.5: Complete")

            if verbose:
                print(f"Generated {len(diagram_descriptions)} diagram descriptions")

        elif describe_diagrams_flag and not diagram_pages:
            if verbose:
                print("\nNo diagram/flowchart pages found - skipping Stage 1.5")

        elif diagram_pages and verbose:
            print(f"\nNote: Found {len(diagram_pages)} diagram pages. Use --describe-diagrams for better output.")

        # Unload Qwen to free VRAM
        if verbose:
            print("\nUnloading Qwen model...")
        unload_qwen(qwen_llm)
        del qwen_processor

        # =====================================================================
        # Stage 2: OCR Processing (via venv subprocess)
        # =====================================================================
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 2: OCR Processing (DeepSeek-OCR-2 via transformers)")
            print("=" * 60)

        progress.update(stage="Stage 2: Processing pages")

        def ocr_progress(page_num, total):
            progress.update(
                num_pages + page_num,
                f"Stage 2: OCR page {page_num}/{total}"
            )

        # Run OCR via venv subprocess (model loading happens in worker)
        results = ocr_pages(
            image_paths, classifications, config,
            verbose=verbose,
            progress_callback=None if verbose else ocr_progress
        )

        progress.increment(num_pages, "Stage 2: Complete")

        # =====================================================================
        # Generate Output
        # =====================================================================
        progress.update(stage="Generating output")

        if verbose:
            print("\n" + "=" * 60)
            print("Generating Output")
            print("=" * 60)

        markdown = generate_markdown(
            results, pdf_path.stem, 'deepseek-ocr',
            diagram_descriptions=diagram_descriptions
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        progress.increment(1)

        # =====================================================================
        # Summary
        # =====================================================================
        total_time = time.time() - total_start
        total_chars = sum(len(r.get('text', '')) for r in results)
        total_figures = sum(len(r.get('figures', [])) for r in results)

        progress.finish(f"Done in {total_time:.1f}s → {output_path}")

        if verbose:
            print(f"\nCompleted in {total_time:.1f}s ({total_time/num_pages:.1f}s per page)")
            print(f"Total characters: {total_chars}")
            print(f"Figures extracted: {total_figures}")
            print(f"Diagrams described: {len(diagram_descriptions)}")
            print(f"Output: {output_path}")
            print("=" * 60)

        return str(output_path)

    finally:
        # Cleanup temp directory
        if not keep_assets and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        elif keep_assets:
            print(f"Temp files kept: {temp_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OCR Pipeline v3.0 - vLLM Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 ocr_pipeline.py document.pdf

  # With diagram description
  python3 ocr_pipeline.py document.pdf --describe-diagrams

  # Verbose output
  python3 ocr_pipeline.py document.pdf -v

  # Custom output path
  python3 ocr_pipeline.py document.pdf output.md
"""
    )

    parser.add_argument('input_pdf', help='Input PDF file')
    parser.add_argument('output_md', nargs='?', default=None, help='Output markdown file')

    # Diagram description options
    diagram_group = parser.add_mutually_exclusive_group()
    diagram_group.add_argument(
        '--describe-diagrams', '--diagrams',
        action='store_true',
        dest='describe_diagrams',
        help='Generate detailed descriptions for diagram/flowchart pages'
    )
    diagram_group.add_argument(
        '--no-describe-diagrams',
        action='store_false',
        dest='describe_diagrams',
        help='Skip diagram description (faster)'
    )
    parser.set_defaults(describe_diagrams=None)

    parser.add_argument('--dpi', type=int, default=200, help='PDF rendering DPI (default: 200)')
    parser.add_argument('--config', default=None, help='Path to config JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--keep-assets', action='store_true', help='Keep intermediate files')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Determine describe_diagrams setting
    if args.describe_diagrams is not None:
        describe_diagrams = args.describe_diagrams
    else:
        describe_diagrams = config.get('describe_diagrams', False)

    # Run pipeline
    run_pipeline(
        pdf_path=args.input_pdf,
        output_path=args.output_md,
        describe_diagrams_flag=describe_diagrams,
        dpi=args.dpi,
        verbose=args.verbose,
        keep_assets=args.keep_assets,
        config=config
    )


if __name__ == "__main__":
    main()
