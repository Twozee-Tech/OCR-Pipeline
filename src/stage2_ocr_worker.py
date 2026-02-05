#!/usr/bin/env python3
"""
DeepSeek-OCR Worker - Runs in venv with transformers 4.47.1

This worker script is called via subprocess from stage2_ocr.py.
It loads the DeepSeek-OCR-2 model using transformers (not vLLM) and
processes pages one by one.

Usage:
    /opt/venv_ocr/bin/python3 stage2_ocr_worker.py input.json output.json [-v]
"""

import sys
import json
import argparse
import time


# =============================================================================
# Prompts (same as in stage2_ocr.py)
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


def get_prompt(classification: dict) -> str:
    """Get optimal OCR prompt based on classification."""
    if classification is None:
        return PROMPTS['mixed']

    page_type = classification.get('type', 'mixed')
    confidence = classification.get('confidence', 0.5)

    # Low confidence -> safer mixed prompt
    if confidence < 0.7:
        return PROMPTS['mixed']

    return PROMPTS.get(page_type, PROMPTS['mixed'])


# =============================================================================
# OCR Processing
# =============================================================================

def ocr_single_page(model, tokenizer, image_path: str, classification: dict) -> dict:
    """
    OCR a single page using DeepSeek-OCR-2 with transformers.

    Args:
        model: Loaded DeepSeek-OCR model
        tokenizer: Model tokenizer
        image_path: Path to page image
        classification: Classification from Stage 1

    Returns:
        dict: OCR result with 'text', 'classification', 'figures'
    """
    prompt = get_prompt(classification)

    # DeepSeek-OCR-2 uses the infer method
    result = model.infer(
        tokenizer,
        prompt=f"<image>\n{prompt}",
        image_file=image_path,
        max_new_tokens=4096,
        temperature=0.0,
    )

    return {
        'text': result,
        'classification': classification,
        'figures': []
    }


def main():
    parser = argparse.ArgumentParser(description='DeepSeek-OCR Worker')
    parser.add_argument('input_file', help='Input JSON file with task data')
    parser.add_argument('output_file', help='Output JSON file for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Load input data
    with open(args.input_file) as f:
        data = json.load(f)

    model_path = data['model_path']
    image_paths = data['image_paths']
    classifications = data['classifications']

    if args.verbose:
        print(f"DeepSeek-OCR Worker starting...")
        print(f"  Model: {model_path}")
        print(f"  Pages: {len(image_paths)}")

    # Import here so we only load transformers in the venv
    import torch
    from transformers import AutoModel, AutoTokenizer

    # Load model
    load_start = time.time()
    if args.verbose:
        print(f"Loading DeepSeek-OCR model...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    ).eval()

    if args.verbose:
        load_time = time.time() - load_start
        print(f"  Model loaded in {load_time:.1f}s")

    # Process pages
    results = []
    process_start = time.time()

    for i, (img_path, cls) in enumerate(zip(image_paths, classifications)):
        page_start = time.time()

        if args.verbose:
            page_type = cls.get('type', 'mixed')
            print(f"  Processing page {i+1}/{len(image_paths)} [{page_type}]...", end='', flush=True)

        result = ocr_single_page(model, tokenizer, img_path, cls)
        results.append(result)

        if args.verbose:
            page_time = time.time() - page_start
            chars = len(result.get('text', ''))
            print(f" {chars} chars in {page_time:.1f}s")

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False)

    if args.verbose:
        total_time = time.time() - process_start
        total_chars = sum(len(r.get('text', '')) for r in results)
        print(f"DeepSeek-OCR Worker complete:")
        print(f"  Total time: {total_time:.1f}s ({total_time/len(image_paths):.1f}s/page)")
        print(f"  Total chars: {total_chars}")


if __name__ == '__main__':
    main()
