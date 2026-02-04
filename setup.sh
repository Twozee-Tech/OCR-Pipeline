#!/bin/bash
# OCR Pipeline - One Command Setup
# Run from host: ./setup.sh
# Downloads models, builds Docker, ready to use

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "=============================================="
echo "       OCR Pipeline - One Command Setup      "
echo "=============================================="
echo -e "${NC}"

# ============================================
# Step 1: Check Docker
# ============================================
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "  ${RED}ERROR: Docker not installed${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Docker found"

if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo -e "  ${YELLOW}WARNING: NVIDIA runtime not detected${NC}"
    echo "  Make sure nvidia-container-toolkit is installed"
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo -e "  ${GREEN}✓${NC} GPU: $GPU_INFO"
else
    echo -e "  ${YELLOW}WARNING: nvidia-smi not found on host${NC}"
fi

# ============================================
# Step 2: Configure Model Directory
# ============================================
echo ""
echo -e "${YELLOW}[2/5] Configure model storage${NC}"
echo ""
echo "  Where should models be stored on your host?"
echo "  (This directory will be mounted into Docker)"
echo ""

DEFAULT_MODELS_DIR="$HOME/.cache/ocr-models"
echo -e "  Default: ${BLUE}$DEFAULT_MODELS_DIR${NC}"
read -p "  Model directory [$DEFAULT_MODELS_DIR]: " MODELS_DIR
MODELS_DIR="${MODELS_DIR:-$DEFAULT_MODELS_DIR}"

# Expand ~ if used
MODELS_DIR="${MODELS_DIR/#\~/$HOME}"

# Create directory
mkdir -p "$MODELS_DIR"
echo -e "  ${GREEN}✓${NC} Using: $MODELS_DIR"

# ============================================
# Step 3: Select & Download Models
# ============================================
echo ""
echo -e "${YELLOW}[3/5] Select models to download${NC}"
echo ""
echo "  Available models:"
echo "    1) DeepSeek-OCR-2      (~8GB)  - Required for OCR"
echo "    2) Qwen3-VL-8B         (~16GB) - Page classification (recommended)"
echo "    3) Qwen3-VL-32B        (~65GB) - Diagram description (optional)"
echo ""
echo "  VRAM requirements (models load sequentially):"
echo "    - DeepSeek only:            ~10GB"
echo "    - DeepSeek + Qwen-8B:       ~24GB"
echo "    - All three:                ~64GB"
echo ""

# Check existing models
DEEPSEEK_PATH="$MODELS_DIR/DeepSeek-OCR-2-model"
QWEN8B_PATH="$MODELS_DIR/Qwen3-VL-8B-model"
QWEN32B_PATH="$MODELS_DIR/Qwen3-VL-32B-Thinking"

[ -d "$DEEPSEEK_PATH" ] && echo -e "  ${GREEN}✓${NC} DeepSeek-OCR-2 exists"
[ -d "$QWEN8B_PATH" ] && echo -e "  ${GREEN}✓${NC} Qwen3-VL-8B exists"
[ -d "$QWEN32B_PATH" ] && echo -e "  ${GREEN}✓${NC} Qwen3-VL-32B exists"
echo ""

# Ask what to download
DL_DEEPSEEK="n"
DL_QWEN8B="n"
DL_QWEN32B="n"

if [ ! -d "$DEEPSEEK_PATH" ]; then
    read -p "  Download DeepSeek-OCR-2? (required) [Y/n]: " DL_DEEPSEEK
    DL_DEEPSEEK="${DL_DEEPSEEK:-Y}"
fi

if [ ! -d "$QWEN8B_PATH" ]; then
    read -p "  Download Qwen3-VL-8B? (recommended) [Y/n]: " DL_QWEN8B
    DL_QWEN8B="${DL_QWEN8B:-Y}"
fi

if [ ! -d "$QWEN32B_PATH" ]; then
    read -p "  Download Qwen3-VL-32B? (65GB, optional) [y/N]: " DL_QWEN32B
    DL_QWEN32B="${DL_QWEN32B:-N}"
fi

# Download using a temporary container
if [[ "$DL_DEEPSEEK" =~ ^[Yy] ]] || [[ "$DL_QWEN8B" =~ ^[Yy] ]] || [[ "$DL_QWEN32B" =~ ^[Yy] ]]; then
    echo ""
    echo -e "  ${BLUE}Downloading models (using Docker container)...${NC}"
    echo ""

    # Build download commands
    DL_CMDS=""

    if [[ "$DL_DEEPSEEK" =~ ^[Yy] ]]; then
        DL_CMDS+="echo '>>> Downloading DeepSeek-OCR-2...' && "
        DL_CMDS+="huggingface-cli download deepseek-ai/DeepSeek-OCR-2 --local-dir /models/DeepSeek-OCR-2-model && "
    fi

    if [[ "$DL_QWEN8B" =~ ^[Yy] ]]; then
        DL_CMDS+="echo '>>> Downloading Qwen3-VL-8B...' && "
        DL_CMDS+="huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir /models/Qwen3-VL-8B-model && "
    fi

    if [[ "$DL_QWEN32B" =~ ^[Yy] ]]; then
        DL_CMDS+="echo '>>> Downloading Qwen3-VL-32B (this takes a while)...' && "
        DL_CMDS+="huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --local-dir /models/Qwen3-VL-32B-Thinking && "
    fi

    DL_CMDS+="echo '>>> Downloads complete!'"

    # Check if we already have the downloader image
    HAD_PYTHON_IMAGE=$(docker images -q python:3.11-slim 2>/dev/null)

    # Run downloads in container
    docker run --rm \
        -v "$MODELS_DIR:/models" \
        -e HF_HOME=/models/.cache \
        python:3.11-slim \
        bash -c "pip install -q huggingface_hub && $DL_CMDS"

    echo -e "  ${GREEN}✓${NC} Model downloads complete"

    # Cleanup: remove HuggingFace cache
    if [ -d "$MODELS_DIR/.cache" ]; then
        rm -rf "$MODELS_DIR/.cache"
        echo -e "  ${GREEN}✓${NC} Cleaned up download cache"
    fi

    # Cleanup: remove python image if we pulled it
    if [ -z "$HAD_PYTHON_IMAGE" ]; then
        docker rmi python:3.11-slim >/dev/null 2>&1 || true
        echo -e "  ${GREEN}✓${NC} Removed temporary Docker image"
    fi
fi

# ============================================
# Step 4: Build Docker Image
# ============================================
echo ""
echo -e "${YELLOW}[4/5] Building Docker image...${NC}"

docker build -t ocr-pipeline "$SCRIPT_DIR"
echo -e "  ${GREEN}✓${NC} Docker image 'ocr-pipeline' built"

# ============================================
# Step 5: Update Configuration
# ============================================
echo ""
echo -e "${YELLOW}[5/5] Updating configuration...${NC}"

# Update ocr_config.json (paths inside container)
cat > "$SCRIPT_DIR/ocr_config.json" << 'EOF'
{
    "deepseek_model_path": "/workspace/models/DeepSeek-OCR-2-model",
    "qwen_model_path": "/workspace/models/Qwen3-VL-8B-model",
    "qwen_describer_path": "/workspace/models/Qwen3-VL-32B-Thinking",
    "classifier": "qwen3-vl-8b",
    "precision": "fp16",
    "describe_diagrams": false
}
EOF

# Update docker-compose.yml with host model path
cat > "$SCRIPT_DIR/docker-compose.yml" << EOF
services:
  ocr:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - OCR_CLASSIFIER=qwen3-vl-8b
      - OCR_DESCRIBE_DIAGRAMS=false
    volumes:
      - $MODELS_DIR:/workspace/models:ro
      - ./input:/data/input:ro
      - ./output:/data/output
EOF

# Create convenience wrapper
cat > "$SCRIPT_DIR/ocr" << 'WRAPPER'
#!/bin/bash
# OCR Pipeline - Quick Run Script
# Usage: ./ocr document.pdf [--diagrams]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: ./ocr <input.pdf> [--diagrams]"
    echo ""
    echo "Options:"
    echo "  --diagrams    Enable detailed diagram description (needs Qwen-32B)"
    exit 1
fi

INPUT_FILE="$(realpath "$1")"
INPUT_DIR="$(dirname "$INPUT_FILE")"
INPUT_NAME="$(basename "$INPUT_FILE")"
OUTPUT_DIR="${SCRIPT_DIR}/output"

mkdir -p "$OUTPUT_DIR"

# Check for --diagrams flag
DESCRIBE_DIAGRAMS="false"
if [[ "$2" == "--diagrams" ]]; then
    DESCRIBE_DIAGRAMS="true"
fi

# Load model path from docker-compose
MODELS_DIR=$(grep -oP '^\s*-\s*\K[^:]+(?=:/workspace/models)' "$SCRIPT_DIR/docker-compose.yml" | head -1)

echo "OCR Pipeline"
echo "  Input:  $INPUT_FILE"
echo "  Output: $OUTPUT_DIR/"
echo "  Models: $MODELS_DIR"
echo ""

docker run --rm --gpus all \
    -v "$MODELS_DIR:/workspace/models:ro" \
    -v "$INPUT_DIR:/data/input:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    -e OCR_INPUT_PDF="/data/input/$INPUT_NAME" \
    -e OCR_DESCRIBE_DIAGRAMS="$DESCRIBE_DIAGRAMS" \
    ocr-pipeline

echo ""
echo "Done! Output in: $OUTPUT_DIR/"
WRAPPER

chmod +x "$SCRIPT_DIR/ocr"

echo -e "  ${GREEN}✓${NC} Configuration updated"

# ============================================
# Summary
# ============================================
echo ""
echo -e "${GREEN}=============================================="
echo "            Setup Complete!"
echo "==============================================${NC}"
echo ""
echo "Models stored in: $MODELS_DIR"
echo ""

echo "Available models:"
[ -d "$DEEPSEEK_PATH" ] && echo -e "  ${GREEN}✓${NC} DeepSeek-OCR-2"
[ -d "$QWEN8B_PATH" ] && echo -e "  ${GREEN}✓${NC} Qwen3-VL-8B"
[ -d "$QWEN32B_PATH" ] && echo -e "  ${GREEN}✓${NC} Qwen3-VL-32B"

echo ""
echo -e "${BLUE}Usage:${NC}"
echo ""
echo "  # Process a PDF"
echo "  ./ocr /path/to/document.pdf"
echo ""
echo "  # With diagram description (needs Qwen-32B)"
echo "  ./ocr /path/to/document.pdf --diagrams"
echo ""
echo "  # Output appears in ./output/"
echo ""
