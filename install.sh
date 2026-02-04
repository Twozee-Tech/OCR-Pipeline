#!/bin/bash
# OCR Pipeline - One Line Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/Twozee-Tech/OCR-Pipeline/main/install.sh | bash
#
# Installs a single 'ocr' command - no repo files left behind

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REPO_RAW="https://raw.githubusercontent.com/Twozee-Tech/OCR-Pipeline/main"
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo -e "${BLUE}"
echo "=============================================="
echo "       OCR Pipeline - Installer              "
echo "=============================================="
echo -e "${NC}"

# ============================================
# Step 1: Check Prerequisites
# ============================================
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "  ${RED}ERROR: Docker not installed${NC}"
    echo "  Install: https://docs.docker.com/engine/install/"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Docker found"

if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo -e "  ${YELLOW}WARNING: NVIDIA container runtime not detected${NC}"
    echo "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

if command -v nvidia-smi &> /dev/null; then
    GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    echo -e "  ${GREEN}✓${NC} GPU: $GPU"
else
    echo -e "  ${YELLOW}WARNING: nvidia-smi not found${NC}"
fi

# ============================================
# Step 2: Configure Model Directory
# ============================================
echo ""
echo -e "${YELLOW}[2/5] Configure model storage${NC}"
echo ""
echo "  Where should models be stored?"
echo ""

DEFAULT_MODELS_DIR="$HOME/.cache/ocr-models"
echo -e "  Default: ${BLUE}$DEFAULT_MODELS_DIR${NC}"
read -p "  Model directory [$DEFAULT_MODELS_DIR]: " MODELS_DIR
MODELS_DIR="${MODELS_DIR:-$DEFAULT_MODELS_DIR}"
MODELS_DIR="${MODELS_DIR/#\~/$HOME}"

mkdir -p "$MODELS_DIR"
echo -e "  ${GREEN}✓${NC} Using: $MODELS_DIR"

# ============================================
# Step 3: Download Models
# ============================================
echo ""
echo -e "${YELLOW}[3/5] Select models to download${NC}"
echo ""
echo "  Available models:"
echo "    1) DeepSeek-OCR-2      (~8GB)  - Required"
echo "    2) Qwen3-VL-8B         (~16GB) - Classification (recommended)"
echo "    3) Qwen3-VL-32B        (~65GB) - Diagrams (optional)"
echo ""

DEEPSEEK_PATH="$MODELS_DIR/DeepSeek-OCR-2-model"
QWEN8B_PATH="$MODELS_DIR/Qwen3-VL-8B-model"
QWEN32B_PATH="$MODELS_DIR/Qwen3-VL-32B-Thinking"

[ -d "$DEEPSEEK_PATH" ] && echo -e "  ${GREEN}✓${NC} DeepSeek-OCR-2 exists"
[ -d "$QWEN8B_PATH" ] && echo -e "  ${GREEN}✓${NC} Qwen3-VL-8B exists"
[ -d "$QWEN32B_PATH" ] && echo -e "  ${GREEN}✓${NC} Qwen3-VL-32B exists"
echo ""

DL_DEEPSEEK="n"
DL_QWEN8B="n"
DL_QWEN32B="n"

[ ! -d "$DEEPSEEK_PATH" ] && read -p "  Download DeepSeek-OCR-2? (required) [Y/n]: " DL_DEEPSEEK && DL_DEEPSEEK="${DL_DEEPSEEK:-Y}"
[ ! -d "$QWEN8B_PATH" ] && read -p "  Download Qwen3-VL-8B? (recommended) [Y/n]: " DL_QWEN8B && DL_QWEN8B="${DL_QWEN8B:-Y}"
[ ! -d "$QWEN32B_PATH" ] && read -p "  Download Qwen3-VL-32B? (65GB, optional) [y/N]: " DL_QWEN32B && DL_QWEN32B="${DL_QWEN32B:-N}"

if [[ "$DL_DEEPSEEK" =~ ^[Yy] ]] || [[ "$DL_QWEN8B" =~ ^[Yy] ]] || [[ "$DL_QWEN32B" =~ ^[Yy] ]]; then
    echo ""
    echo -e "  ${BLUE}Downloading models...${NC}"

    DL_CMDS=""
    [[ "$DL_DEEPSEEK" =~ ^[Yy] ]] && DL_CMDS+="echo '>>> DeepSeek-OCR-2...' && huggingface-cli download deepseek-ai/DeepSeek-OCR-2 --local-dir /models/DeepSeek-OCR-2-model && "
    [[ "$DL_QWEN8B" =~ ^[Yy] ]] && DL_CMDS+="echo '>>> Qwen3-VL-8B...' && huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir /models/Qwen3-VL-8B-model && "
    [[ "$DL_QWEN32B" =~ ^[Yy] ]] && DL_CMDS+="echo '>>> Qwen3-VL-32B...' && huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --local-dir /models/Qwen3-VL-32B-Thinking && "
    DL_CMDS+="echo '>>> Done!'"

    HAD_PYTHON_IMAGE=$(docker images -q python:3.11-slim 2>/dev/null)

    docker run --rm \
        -v "$MODELS_DIR:/models" \
        -e HF_HOME=/models/.cache \
        python:3.11-slim \
        bash -c "pip install -q huggingface_hub && $DL_CMDS"

    # Cleanup
    rm -rf "$MODELS_DIR/.cache" 2>/dev/null || true
    [ -z "$HAD_PYTHON_IMAGE" ] && docker rmi python:3.11-slim >/dev/null 2>&1 || true

    echo -e "  ${GREEN}✓${NC} Models downloaded"
fi

# ============================================
# Step 4: Build Docker Image
# ============================================
echo ""
echo -e "${YELLOW}[4/5] Building Docker image...${NC}"

# Download only the files needed for Docker build
curl -fsSL "$REPO_RAW/Dockerfile" -o "$TEMP_DIR/Dockerfile"
curl -fsSL "$REPO_RAW/ocr_config.json" -o "$TEMP_DIR/ocr_config.json"
curl -fsSL "$REPO_RAW/entrypoint.py" -o "$TEMP_DIR/entrypoint.py"
curl -fsSL "$REPO_RAW/ocr_pipeline.py" -o "$TEMP_DIR/ocr_pipeline.py"
curl -fsSL "$REPO_RAW/stage1_classifier.py" -o "$TEMP_DIR/stage1_classifier.py"
curl -fsSL "$REPO_RAW/stage1_5_diagram.py" -o "$TEMP_DIR/stage1_5_diagram.py"
curl -fsSL "$REPO_RAW/stage2_ocr.py" -o "$TEMP_DIR/stage2_ocr.py"

docker build -t ocr-pipeline "$TEMP_DIR"
echo -e "  ${GREEN}✓${NC} Docker image built"

# ============================================
# Step 5: Install 'ocr' Command
# ============================================
echo ""
echo -e "${YELLOW}[5/5] Installing 'ocr' command...${NC}"

# Determine install location
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"

# Create self-contained ocr script
cat > "$BIN_DIR/ocr" << WRAPPER
#!/bin/bash
# OCR Pipeline - Process PDFs with AI
# Generated by installer - models at: $MODELS_DIR

MODELS_DIR="$MODELS_DIR"

if [ -z "\$1" ]; then
    echo "Usage: ocr <input.pdf> [options]"
    echo ""
    echo "Options:"
    echo "  --diagrams    Enable detailed diagram description (needs Qwen-32B)"
    echo "  --output DIR  Output directory (default: ./output)"
    echo ""
    echo "Models: \$MODELS_DIR"
    exit 1
fi

INPUT_FILE="\$(realpath "\$1")"
INPUT_DIR="\$(dirname "\$INPUT_FILE")"
INPUT_NAME="\$(basename "\$INPUT_FILE")"

# Parse options
DESCRIBE_DIAGRAMS="false"
OUTPUT_DIR="./output"

shift
while [[ \$# -gt 0 ]]; do
    case \$1 in
        --diagrams) DESCRIBE_DIAGRAMS="true"; shift ;;
        --output) OUTPUT_DIR="\$2"; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p "\$OUTPUT_DIR"
OUTPUT_DIR="\$(realpath "\$OUTPUT_DIR")"

echo "OCR Pipeline"
echo "  Input:  \$INPUT_FILE"
echo "  Output: \$OUTPUT_DIR/"
echo ""

docker run --rm --gpus all \\
    -v "\$MODELS_DIR:/workspace/models:ro" \\
    -v "\$INPUT_DIR:/data/input:ro" \\
    -v "\$OUTPUT_DIR:/data/output" \\
    -e OCR_INPUT_PDF="/data/input/\$INPUT_NAME" \\
    -e OCR_DESCRIBE_DIAGRAMS="\$DESCRIBE_DIAGRAMS" \\
    ocr-pipeline

echo ""
echo "Done! Output: \$OUTPUT_DIR/"
WRAPPER

chmod +x "$BIN_DIR/ocr"
echo -e "  ${GREEN}✓${NC} Installed to $BIN_DIR/ocr"

# Add to PATH if needed
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    SHELL_RC=""
    [ -f "$HOME/.zshrc" ] && SHELL_RC="$HOME/.zshrc"
    [ -f "$HOME/.bashrc" ] && SHELL_RC="${SHELL_RC:-$HOME/.bashrc}"

    if [ -n "$SHELL_RC" ] && ! grep -q ".local/bin" "$SHELL_RC" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        echo -e "  ${GREEN}✓${NC} Added ~/.local/bin to PATH in $SHELL_RC"
        echo -e "  ${YELLOW}Run: source $SHELL_RC${NC}"
    fi
fi

# ============================================
# Done
# ============================================
echo ""
echo -e "${GREEN}=============================================="
echo "            Installation Complete!            "
echo "==============================================${NC}"
echo ""
echo "Usage:"
echo "  ocr document.pdf"
echo "  ocr document.pdf --diagrams"
echo "  ocr document.pdf --output /path/to/output"
echo ""
echo "Models stored in: $MODELS_DIR"
echo ""
