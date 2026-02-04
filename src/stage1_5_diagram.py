#!/usr/bin/env python3
"""
Stage 1.5: Diagram Description with Qwen3-VL-32B

Processes diagram/flowchart pages identified by Stage 1 classifier and generates
detailed ASCII art representations and flow descriptions.

IMPORTANT: This script requires the venv with newer transformers!
    source venv_qwen/bin/activate
    python3 stage1_5_diagram.py input.pdf --classifications class.json -o diagrams.json

Usage:
    As standalone:
        python stage1_5_diagram.py input.pdf -c classifications.json -o diagrams.json

    As module:
        from stage1_5_diagram import load_model, describe_diagrams
        model, processor = load_model(model_path)
        descriptions = describe_diagrams(model, processor, pages, classifications)
"""

import torch
import json
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


# Diagram description prompt for Qwen3-VL-32B
DIAGRAM_PROMPT = """Analyze this diagram and provide:
1. Type of diagram (sequence diagram, flowchart, architecture, call flow, state machine, etc.)
2. Detailed ASCII art representation that captures the structure and flow
3. Step-by-step description of the flow or relationships shown

Format your response as markdown with:
- A header indicating the diagram type
- The ASCII diagram in a code block using box-drawing characters
- A numbered list describing each step or component

Use these box-drawing characters for the ASCII art:
- Boxes: ┌ ┐ └ ┘ │ ─
- Arrows: → ← ↓ ↑ ↔
- Connectors: ├ ┤ ┬ ┴ ┼

Make the ASCII representation as accurate as possible to the original diagram."""


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: str = None, precision: str = "fp16", verbose: bool = False):
    """
    Load Qwen3-VL-32B model for diagram description.

    Requires venv with transformers>=4.50.0 and qwen-vl-utils.

    Args:
        model_path: Path to local model or HuggingFace model ID
        precision: "fp16" or "fp8"
        verbose: Print loading progress

    Returns:
        tuple: (model, processor)
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    # Default to HuggingFace hub
    model_id = model_path or "Qwen/Qwen3-VL-32B-Instruct"
    local_only = model_path is not None and os.path.exists(model_path)

    if verbose:
        print(f"\nLoading Qwen3-VL-32B diagram describer ({precision})...")
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

        # Force GPU loading - don't offload to CPU when VRAM is sufficient
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
        print(f"ERROR: Failed to load Qwen3-VL-32B: {e}")
        print("\nMake sure you're running in the venv:")
        print("  source venv_qwen/bin/activate")
        print("\nAnd that the model is downloaded:")
        print("  git clone https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct /workspace/models/Qwen3-VL-32B-model")
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
        print("  Unloaded diagram describer, freed VRAM")


# ============================================================================
# Diagram Description
# ============================================================================

def describe_diagram(model, processor, image: Image.Image,
                     page_num: int = 0, verbose: bool = False) -> str:
    """
    Generate detailed description of a diagram using Qwen3-VL-32B.

    Args:
        model: Qwen3-VL-32B model
        processor: Qwen3-VL processor
        image: PIL Image of the diagram page
        page_num: Page number for logging
        verbose: Print debug info

    Returns:
        str: Markdown description with ASCII art
    """
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
                    {"type": "text", "text": DIAGRAM_PROMPT}
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

        # Generate response (longer for detailed ASCII art)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.2,
                use_cache=True,
            )

        # Decode (skip input tokens)
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if verbose:
            print(f"    Generated {len(response)} chars")

        return response.strip()

    except Exception as e:
        if verbose:
            print(f"    Diagram description error: {e}")
        return None


def describe_diagrams(model, processor, pages: list, classifications: list,
                      verbose: bool = False) -> dict:
    """
    Describe all diagram/flowchart pages.

    Args:
        model: Qwen3-VL-32B model
        processor: Qwen3-VL processor
        pages: List of page info dicts with 'image'
        classifications: List of classifications from Stage 1
        verbose: Print progress

    Returns:
        dict: {page_number: description_text}
    """
    descriptions = {}
    diagram_types = {'diagram', 'flowchart'}

    # Find diagram pages (including mixed pages with diagrams)
    diagram_pages = []
    for i, c in enumerate(classifications):
        page_type = c.get('type', '').lower()
        has_diagrams = c.get('has_diagrams', False)
        if page_type in diagram_types or (page_type == 'mixed' and has_diagrams):
            diagram_pages.append((i, c))

    if not diagram_pages:
        if verbose:
            print("No diagram pages found to describe")
        return descriptions

    print(f"\nDescribing {len(diagram_pages)} diagram pages with Qwen3-VL-32B...")

    for idx, (page_idx, classification) in enumerate(diagram_pages):
        page_num = classification.get('page', page_idx + 1)
        content_type = classification.get('type', 'diagram')
        confidence = classification.get('confidence', 0)

        print(f"  Page {page_num} [{content_type} @ {confidence:.0%}]...", end=" ", flush=True)

        start = time.time()

        if page_idx < len(pages):
            description = describe_diagram(
                model, processor,
                pages[page_idx]['image'],
                page_num,
                verbose
            )

            if description:
                descriptions[page_num] = description
                elapsed = time.time() - start
                print(f"{len(description)} chars ({elapsed:.1f}s)")
            else:
                print("FAILED")
        else:
            print(f"SKIP (page index {page_idx} out of range)")

    return descriptions


def save_descriptions(descriptions: dict, output_path: str):
    """
    Save diagram descriptions to JSON file.

    Args:
        descriptions: {page_number: description_text}
        output_path: Output JSON file path
    """
    output_data = {
        'method': 'qwen3-vl-32b',
        'pages_described': len(descriptions),
        'descriptions': descriptions
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


# ============================================================================
# PDF Processing Utilities
# ============================================================================

def pdf_to_page_images(pdf_path: str, dpi: int = 200, verbose: bool = False) -> list:
    """Convert PDF to list of page images."""
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


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 1.5: Describe diagrams with Qwen3-VL-32B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: Run in venv with newer transformers!
  source venv_qwen/bin/activate
  python3 stage1_5_diagram.py document.pdf -c classifications.json -o diagrams.json

Examples:
  # Describe diagrams identified by Stage 1
  python3 stage1_5_diagram.py document.pdf -c class.json -o diagrams.json

  # Verbose output
  python3 stage1_5_diagram.py document.pdf -c class.json -o diagrams.json -v

  # Custom model path
  python3 stage1_5_diagram.py document.pdf -c class.json -o diagrams.json \\
      -m /workspace/models/Qwen3-VL-32B-model
"""
    )

    parser.add_argument('input_pdf', help='Input PDF file')
    parser.add_argument('-c', '--classifications', required=True,
                        help='Classifications JSON from Stage 1')
    parser.add_argument('-o', '--output', required=True,
                        help='Output JSON file for diagram descriptions')
    parser.add_argument('-m', '--model', help='Path to Qwen3-VL-32B model')
    parser.add_argument('-p', '--precision', choices=['fp16', 'fp8'], default='fp16')
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

    if not os.path.exists(args.classifications):
        print(f"ERROR: Classifications file not found: {args.classifications}")
        sys.exit(1)

    # Load config
    config = load_config()
    model_path = args.model or config.get('qwen_describer_path')

    print("=" * 60)
    print("Stage 1.5: Diagram Description (Qwen3-VL-32B)")
    print("=" * 60)
    print(f"Input:           {pdf_path}")
    print(f"Classifications: {args.classifications}")
    print(f"Output:          {args.output}")
    print(f"Model:           {model_path or 'Qwen/Qwen3-VL-32B-Instruct (HuggingFace)'}")
    print()

    start_time = time.time()

    # Load classifications
    with open(args.classifications, 'r') as f:
        class_data = json.load(f)
    classifications = class_data.get('classifications', [])

    # Count diagram pages
    diagram_types = {'diagram', 'flowchart'}
    diagram_count = sum(1 for c in classifications
                        if c.get('type', '').lower() in diagram_types
                        or (c.get('type', '').lower() == 'mixed' and c.get('has_diagrams', False)))

    if diagram_count == 0:
        print("No diagram or flowchart pages found in classifications.")
        print("Creating empty output file.")
        save_descriptions({}, args.output)
        return

    print(f"Found {diagram_count} diagram/flowchart pages to describe")

    # Convert PDF to images
    pages = pdf_to_page_images(str(pdf_path), dpi=args.dpi, verbose=args.verbose)

    # Load model
    model, processor = load_model(
        model_path=model_path,
        precision=args.precision,
        verbose=args.verbose
    )

    if model is None:
        print("ERROR: Could not load Qwen3-VL-32B model.")
        print("Make sure you're in the venv: source venv_qwen/bin/activate")
        sys.exit(1)

    # Describe diagrams
    descriptions = describe_diagrams(model, processor, pages, classifications, args.verbose)

    # Unload model
    unload_model(model, processor, verbose=args.verbose)

    # Save results
    save_descriptions(descriptions, args.output)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"Completed in {elapsed:.1f}s")
    print(f"Diagrams described: {len(descriptions)}/{diagram_count}")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
