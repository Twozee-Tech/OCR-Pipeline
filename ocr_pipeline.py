#!/usr/bin/env python3
"""
OCR Pipeline Orchestrator

Two-stage PDF to Markdown conversion:
  Stage 1: Qwen3-VL-8B classifies pages (runs in venv_qwen)
  Stage 2: DeepSeek-OCR processes pages (runs in system Python)

Usage:
    python3 ocr_pipeline.py input.pdf [output.md] [--classifier qwen3-vl-8b|heuristic]

Setup required:
    ./setup.sh  # Creates venv_qwen for Stage 1
"""

import json
import sys
import os
import time
import argparse
import subprocess
from pathlib import Path

# Stage 2 imports (system Python)
from stage2_ocr import (
    load_model as load_ocr,
    unload_model as unload_ocr,
    ocr_pages,
    pdf_to_page_images,
    generate_markdown,
)


# Config file paths
CONFIG_PATHS = [
    './ocr_config.json',
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


def find_deepseek_model(config: dict, custom_path: str = None) -> str:
    """Find DeepSeek-OCR model path."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    if config.get('deepseek_model_path') and os.path.exists(config['deepseek_model_path']):
        return config['deepseek_model_path']

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


def find_venv_python() -> str:
    """Find Python executable in venv_qwen."""
    script_dir = Path(__file__).parent
    venv_python = script_dir / "venv_qwen" / "bin" / "python3"

    if venv_python.exists():
        return str(venv_python)

    # Try relative to cwd
    venv_python = Path("venv_qwen") / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)

    return None


def run_stage1_subprocess(
    pdf_path: str,
    output_json: str,
    model_path: str = None,
    precision: str = "fp16",
    heuristic: bool = False,
    dpi: int = 200,
    verbose: bool = False
) -> bool:
    """
    Run Stage 1 classifier as subprocess in venv.

    Returns True on success, False on failure.
    """
    script_dir = Path(__file__).parent
    stage1_script = script_dir / "stage1_classifier.py"

    # Find venv Python
    venv_python = find_venv_python()

    if venv_python is None and not heuristic:
        print("WARNING: venv_qwen not found. Run ./setup.sh first.")
        print("         Falling back to heuristic classification.")
        heuristic = True

    # Build command
    if heuristic:
        # Use system Python for heuristic
        python_exe = sys.executable
    else:
        python_exe = venv_python

    cmd = [
        python_exe,
        str(stage1_script),
        pdf_path,
        "-o", output_json,
        "--dpi", str(dpi),
    ]

    if heuristic:
        cmd.append("--heuristic")
    else:
        if model_path:
            cmd.extend(["-m", model_path])
        cmd.extend(["-p", precision])

    if verbose:
        cmd.append("-v")

    # Run subprocess
    print(f"\n--- Running Stage 1 {'(heuristic)' if heuristic else '(Qwen3-VL in venv)'} ---")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        # Run with real-time output (no capture)
        result = subprocess.run(
            cmd,
            check=True,
        )

        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Stage 1 failed with exit code {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

    except FileNotFoundError as e:
        print(f"ERROR: Could not run Stage 1: {e}")
        return False


def run_pipeline(
    pdf_path: str,
    output_path: str = None,
    classifier: str = 'heuristic',
    classifier_precision: str = 'fp16',
    deepseek_model_path: str = None,
    qwen_model_path: str = None,
    dpi: int = 200,
    verbose: bool = False,
    config: dict = None
) -> str:
    """
    Run the full two-stage OCR pipeline.

    Stage 1 runs in venv_qwen (subprocess)
    Stage 2 runs in current Python (system)
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

    # Create output directory
    output_dir = output_path.parent / f"{output_path.stem}_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OCR Pipeline - Two-Stage Processing")
    print("=" * 60)
    print(f"Input:      {pdf_path}")
    print(f"Output:     {output_path}")
    print(f"Assets:     {output_dir}")
    print(f"Classifier: {classifier} ({classifier_precision})")
    print()

    total_start = time.time()

    # =========================================================================
    # Stage 1: Classification (subprocess in venv)
    # =========================================================================
    print("=" * 60)
    print("STAGE 1: Page Classification")
    print("=" * 60)

    classifications_file = output_dir / "classifications.json"
    qwen_path = qwen_model_path or config.get('qwen_model_path')

    success = run_stage1_subprocess(
        pdf_path=str(pdf_path),
        output_json=str(classifications_file),
        model_path=qwen_path,
        precision=classifier_precision,
        heuristic=(classifier == 'heuristic'),
        dpi=dpi,
        verbose=verbose
    )

    if not success:
        print("ERROR: Stage 1 failed. Aborting pipeline.")
        sys.exit(1)

    # Load classifications
    with open(classifications_file, 'r') as f:
        class_data = json.load(f)

    classifications = class_data.get('classifications', [])
    classifier_method = class_data.get('method', 'unknown')

    # Print summary
    content_types = {}
    for c in classifications:
        ct = c.get('type', 'unknown')
        content_types[ct] = content_types.get(ct, 0) + 1
    print(f"\nContent types: {content_types}")
    print(f"Classifications saved to: {classifications_file}")

    # =========================================================================
    # Stage 2: OCR Processing (system Python)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: OCR Processing (DeepSeek-OCR)")
    print("=" * 60)

    # Find DeepSeek model
    deepseek_path = find_deepseek_model(config, deepseek_model_path)
    if deepseek_path is None:
        print("ERROR: DeepSeek-OCR model not found.")
        print("Download: git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR DeepSeek-OCR-model")
        sys.exit(1)

    # Load model
    ocr_model, tokenizer = load_ocr(deepseek_path, verbose=verbose)

    # Convert PDF to images (for Stage 2)
    pages = pdf_to_page_images(str(pdf_path), dpi=dpi, verbose=verbose)

    # Validate classification count
    if len(classifications) != len(pages):
        print(f"WARNING: Classification count mismatch ({len(classifications)} vs {len(pages)} pages)")
        # Pad or truncate
        while len(classifications) < len(pages):
            classifications.append({
                'page': len(classifications) + 1,
                'type': 'mixed',
                'confidence': 0.5,
                'method': 'default'
            })

    # Run OCR
    print(f"\nProcessing {len(pages)} pages...")
    results = ocr_pages(
        ocr_model, tokenizer, pages, classifications,
        str(output_dir), verbose
    )

    # Unload model
    unload_ocr(ocr_model, tokenizer, verbose=verbose)

    # =========================================================================
    # Generate Output
    # =========================================================================
    print("\n" + "=" * 60)
    print("Generating Output")
    print("=" * 60)

    markdown = generate_markdown(results, pdf_path.stem, classifier_method)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - total_start
    total_chars = sum(len(r.get('text', '')) for r in results)
    total_figures = sum(len(r.get('figures', [])) for r in results)

    print(f"\nCompleted in {total_time:.1f}s ({total_time/len(pages):.1f}s per page)")
    print(f"Total characters: {total_chars}")
    print(f"Figures extracted: {total_figures}")
    print(f"Output: {output_path}")
    print("=" * 60)

    return str(output_path)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OCR Pipeline - Two-Stage PDF to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup:
  ./setup.sh  # Creates venv_qwen for Qwen3-VL

Examples:
  # With Qwen3-VL-8B classifier (requires venv)
  python3 ocr_pipeline.py document.pdf --classifier qwen3-vl-8b

  # Heuristic classification (no venv needed)
  python3 ocr_pipeline.py document.pdf --classifier heuristic

  # Full options
  python3 ocr_pipeline.py document.pdf output.md --classifier qwen3-vl-8b -v
"""
    )

    parser.add_argument('input_pdf', help='Input PDF file')
    parser.add_argument('output_md', nargs='?', default=None, help='Output markdown file')

    parser.add_argument(
        '--classifier', '-c',
        choices=['qwen3-vl-8b', 'heuristic'],
        default=None,
        help='Page classifier (default from config or heuristic)'
    )

    parser.add_argument(
        '--precision', '-p',
        choices=['fp16', 'fp8'],
        default=None,
        help='Classifier precision (default: fp16)'
    )

    parser.add_argument('--deepseek-model', default=None, help='Path to DeepSeek-OCR model')
    parser.add_argument('--qwen-model', default=None, help='Path to Qwen3-VL model')
    parser.add_argument('--dpi', type=int, default=200, help='PDF rendering DPI')
    parser.add_argument('--config', default=None, help='Path to config JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Get settings
    classifier = args.classifier or config.get('classifier', 'heuristic')
    precision = args.precision or config.get('precision', 'fp16')

    # Run pipeline
    run_pipeline(
        pdf_path=args.input_pdf,
        output_path=args.output_md,
        classifier=classifier,
        classifier_precision=precision,
        deepseek_model_path=args.deepseek_model,
        qwen_model_path=args.qwen_model,
        dpi=args.dpi,
        verbose=args.verbose,
        config=config
    )


if __name__ == "__main__":
    main()
