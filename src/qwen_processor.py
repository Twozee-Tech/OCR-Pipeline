#!/usr/bin/env python3
"""
Qwen3-VL Processor for OCR Pipeline v3.0

Handles page classification and diagram description using vLLM for fast batched inference.
Based on POC: /home/lokarmus/Projects/claude/POC_NVP4/flowchart_analyzer.py

Usage:
    from qwen_processor import create_qwen_llm, classify_pages, describe_diagrams

    llm = create_qwen_llm(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    classifications = classify_pages(llm, processor, image_paths)
    descriptions = describe_diagrams(llm, processor, diagram_paths)
"""

import os
import json
import re
import tempfile
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


# =============================================================================
# Classification Prompts & Parameters
# =============================================================================

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

CLASSIFY_PARAMS = SamplingParams(temperature=0, max_tokens=256, top_k=-1)


# =============================================================================
# Diagram Description Prompts & Parameters
# =============================================================================

DIAGRAM_PROMPT = """Analyze this diagram in detail.

1. DESCRIPTION: Provide a detailed description of the diagram including:
   - Purpose/title of the diagram
   - All nodes (start, end, process, decision points)
   - All connections and flow direction
   - Any labels or text on the diagram

2. MERMAID SEQUENCE DIAGRAM: Recreate as a Mermaid sequence diagram:
   - Use sequenceDiagram syntax
   - Define participants/actors
   - Show interactions with arrows ->> or -->>
   - Include notes and labels where needed
   - IMPORTANT: Do not use parentheses () unless they are inside quoted strings

Be precise and capture all elements from the original diagram."""

DIAGRAM_PARAMS = SamplingParams(temperature=0, max_tokens=6000, top_k=-1)


# =============================================================================
# Model Creation - Tuned parameters from POC - DO NOT CHANGE
# =============================================================================

def create_qwen_llm(model_path: str, config: dict = None) -> LLM:
    """
    Create vLLM instance for Qwen3-VL model.

    Parameters tuned for DGX Spark unified memory architecture.

    Args:
        model_path: Path to Qwen3-VL model
        config: Optional config dict with gpu_memory_utilization, max_model_len

    Returns:
        LLM: vLLM model instance
    """
    config = config or {}

    return LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=config.get('qwen_gpu_memory_utilization', 0.20),
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=config.get('qwen_max_model_len', 6144),
        seed=0,
    )


def unload_llm(llm: LLM):
    """Unload vLLM model and free VRAM."""
    if llm is not None:
        del llm

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Input Preparation (from POC)
# =============================================================================

def prepare_inputs(messages: list, processor: AutoProcessor) -> dict:
    """
    Prepare inputs for vLLM generation.

    Uses official Qwen method from qwen_vl_utils.

    Args:
        messages: Chat messages with image content
        processor: AutoProcessor for the model

    Returns:
        dict: Input dict for vLLM generate()
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def make_classify_msg(image_path: str) -> list:
    """Create classification message for an image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": CLASSIFY_PROMPT},
            ],
        }
    ]


def make_diagram_msg(image_path: str) -> list:
    """Create diagram description message for an image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": DIAGRAM_PROMPT},
            ],
        }
    ]


# =============================================================================
# Classification
# =============================================================================

def parse_classification(response: str) -> dict:
    """Parse JSON response from classifier."""
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
                return {'type': 'mixed', 'confidence': 0.0, 'method': 'parse_failed'}
        else:
            return {'type': 'mixed', 'confidence': 0.0, 'method': 'parse_failed'}

    # Validate and normalize
    if 'type' not in result:
        result['type'] = 'mixed'

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
    else:
        result['confidence'] = 0.5

    result['method'] = 'qwen3-vl'
    return result


def classify_pages(llm: LLM, processor: AutoProcessor, image_paths: list,
                   verbose: bool = False) -> list:
    """
    Classify multiple pages in batch using vLLM.

    Args:
        llm: vLLM model instance
        processor: AutoProcessor
        image_paths: List of paths to page images
        verbose: Print progress

    Returns:
        list: Classification dicts for each page
    """
    if not image_paths:
        return []

    if verbose:
        print(f"Classifying {len(image_paths)} pages...")

    # Prepare all inputs
    inputs = [prepare_inputs(make_classify_msg(p), processor) for p in image_paths]

    # Batch generate
    outputs = llm.generate(inputs, CLASSIFY_PARAMS)

    # Parse results
    classifications = []
    for i, output in enumerate(outputs):
        result = parse_classification(output.outputs[0].text)
        result['page'] = i + 1
        classifications.append(result)

        if verbose:
            print(f"  Page {i+1}: {result['type']} ({result['confidence']:.0%})")

    return classifications


# =============================================================================
# Diagram Description
# =============================================================================

DIAGRAM_TYPES = {'diagram', 'flowchart'}


def describe_diagrams(llm: LLM, processor: AutoProcessor, image_paths: list,
                      classifications: list = None, verbose: bool = False) -> dict:
    """
    Describe multiple diagrams in batch using vLLM.

    Args:
        llm: vLLM model instance
        processor: AutoProcessor
        image_paths: List of ALL page image paths
        classifications: Optional list of classifications to filter diagram pages
        verbose: Print progress

    Returns:
        dict: {page_number: description} for diagram pages
    """
    if not image_paths:
        return {}

    # If classifications provided, filter to diagram pages only
    if classifications:
        diagram_indices = []
        for i, c in enumerate(classifications):
            page_type = c.get('type', '').lower()
            has_diagrams = c.get('has_diagrams', False)
            if page_type in DIAGRAM_TYPES or (page_type == 'mixed' and has_diagrams):
                diagram_indices.append(i)

        if not diagram_indices:
            if verbose:
                print("No diagram pages found to describe")
            return {}

        diagram_paths = [image_paths[i] for i in diagram_indices]
        page_numbers = [i + 1 for i in diagram_indices]
    else:
        # Describe all pages
        diagram_paths = image_paths
        page_numbers = list(range(1, len(image_paths) + 1))

    if verbose:
        print(f"Describing {len(diagram_paths)} diagram pages...")

    # Prepare all inputs
    inputs = [prepare_inputs(make_diagram_msg(p), processor) for p in diagram_paths]

    # Batch generate
    outputs = llm.generate(inputs, DIAGRAM_PARAMS)

    # Build results dict
    descriptions = {}
    for page_num, output in zip(page_numbers, outputs):
        text = output.outputs[0].text.strip()
        descriptions[page_num] = text

        if verbose:
            print(f"  Page {page_num}: {len(text)} chars")

    return descriptions


# =============================================================================
# Utility: Save pages as temp files for vLLM
# =============================================================================

def save_pages_to_temp(pages: list, temp_dir: str = None) -> list:
    """
    Save PIL images to temp files for vLLM processing.

    vLLM requires file:// URIs for images.

    Args:
        pages: List of dicts with 'image' (PIL Image)
        temp_dir: Optional temp directory (created if None)

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
