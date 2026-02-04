#!/bin/bash
# Setup script for OCR Pipeline
# Creates venv for Qwen3-VL models (Stage 1 classifier, Stage 1.5 diagram describer)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "OCR Pipeline Setup"
echo "=============================================="

# Check if we're in the container
if [ ! -d "/workspace" ]; then
    echo "WARNING: /workspace not found. Are you inside the container?"
fi

# ============================================
# Stage 1 & 1.5: Qwen venv (newer transformers)
# ============================================
echo ""
echo "=== Creating venv for Qwen3-VL models (Stage 1 + 1.5) ==="

# Remove old venv if exists
rm -rf venv_qwen

# Create venv with system site packages (keeps system PyTorch/CUDA)
python3 -m venv --system-site-packages venv_qwen

# Activate and install
source venv_qwen/bin/activate

echo "=== Checking system PyTorch ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "=== Installing transformers >=4.50 (override system version) ==="
pip install --ignore-installed "transformers>=4.50.0" huggingface-hub tokenizers

echo "=== Installing Qwen dependencies ==="
pip install qwen-vl-utils pillow accelerate

echo "=== Verifying Qwen venv ==="
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "from transformers import Qwen3VLForConditionalGeneration; print('Qwen3VLForConditionalGeneration: OK')"
python3 -c "from qwen_vl_utils import process_vision_info; print('qwen_vl_utils: OK')"

deactivate

# ============================================
# Stage 2: System Python (DeepSeek)
# ============================================
echo ""
echo "=== Checking Stage 2 dependencies (system Python) ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import fitz; print(f'PyMuPDF: OK')" 2>/dev/null || pip install pymupdf
python3 -c "from PIL import Image; print('Pillow: OK')"

# ============================================
# Models check
# ============================================
echo ""
echo "=== Checking models ==="

MODELS_DIR="/workspace/models"

if [ -d "$MODELS_DIR/DeepSeek-OCR-model" ]; then
    echo "DeepSeek-OCR: OK ($MODELS_DIR/DeepSeek-OCR-model)"
else
    echo "DeepSeek-OCR: NOT FOUND"
    echo "  Download: hf download deepseek-ai/DeepSeek-OCR --local-dir $MODELS_DIR/DeepSeek-OCR-model"
fi

if [ -d "$MODELS_DIR/Qwen3-VL-8B-model" ]; then
    echo "Qwen3-VL-8B (classifier): OK ($MODELS_DIR/Qwen3-VL-8B-model)"
else
    echo "Qwen3-VL-8B (classifier): NOT FOUND"
    echo "  Download: hf download Qwen/Qwen3-VL-8B-Instruct --local-dir $MODELS_DIR/Qwen3-VL-8B-model"
fi

if [ -d "$MODELS_DIR/Qwen3-VL-32B-model" ]; then
    echo "Qwen3-VL-32B (diagrams): OK ($MODELS_DIR/Qwen3-VL-32B-model)"
else
    echo "Qwen3-VL-32B (diagrams): NOT FOUND (optional, ~65GB)"
    echo "  Download: hf download Qwen/Qwen3-VL-32B-Instruct --local-dir $MODELS_DIR/Qwen3-VL-32B-model"
    echo "  Note: Only needed if using --describe-diagrams flag"
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Usage:"
echo "  # Full pipeline (uses venv automatically)"
echo "  python3 ocr_pipeline.py input.pdf --classifier qwen3-vl-8b"
echo ""
echo "  # With diagram description (best for flowcharts/diagrams)"
echo "  python3 ocr_pipeline.py input.pdf --classifier qwen3-vl-8b --describe-diagrams"
echo ""
echo "  # Heuristic only (no Qwen model needed)"
echo "  python3 ocr_pipeline.py input.pdf --classifier heuristic"
echo ""
echo "  # Stage 1 standalone (classification, in venv)"
echo "  source venv_qwen/bin/activate"
echo "  python3 stage1_classifier.py input.pdf -o classifications.json"
echo "  deactivate"
echo ""
echo "  # Stage 1.5 standalone (diagram description, in venv)"
echo "  source venv_qwen/bin/activate"
echo "  python3 stage1_5_diagram.py input.pdf -c classifications.json -o diagrams.json"
echo "  deactivate"
echo ""
echo "  # Stage 2 standalone (OCR, system Python)"
echo "  python3 stage2_ocr.py input.pdf -c classifications.json -o output.md"
