#!/bin/bash
# Advanced OCR for NVIDIA DGX Spark - Installer v2.0
# Usage: curl -fsSL https://raw.githubusercontent.com/Twozee-Tech/Advanced-OCR-Nvidia-DGX-SPARK/main/install.sh | bash
#
# Installs a single 'ocr' command - no repo files left behind
VERSION="2.2"

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REPO_RAW="https://raw.githubusercontent.com/Twozee-Tech/Advanced-OCR-Nvidia-DGX-SPARK/main"
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Function to read input - works even when script is piped
ask() {
    local prompt="$1"
    local default="$2"
    local reply

    # Print prompt to terminal
    printf "%s" "$prompt" > /dev/tty
    # Read from terminal
    read -r reply < /dev/tty

    if [ -z "$reply" ]; then
        echo "$default"
    else
        echo "$reply"
    fi
}

echo -e "${BLUE}"
echo "=============================================="
echo "       OCR Pipeline - Installer v${VERSION}      "
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
    GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "detected")
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
MODELS_DIR=$(ask "  Model directory [$DEFAULT_MODELS_DIR]: " "$DEFAULT_MODELS_DIR")

# Expand ~ if used
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

# Function to check if a directory contains a valid model (has config.json and weights)
is_valid_model() {
    local dir="$1"
    [ -d "$dir" ] && [ -f "$dir/config.json" ] && \
    (ls "$dir"/*.safetensors >/dev/null 2>&1 || ls "$dir"/*.bin >/dev/null 2>&1)
}

# Function to find model in common locations/names
find_model() {
    local base_dir="$1"
    local patterns="$2"  # comma-separated patterns

    IFS=',' read -ra PATTERNS <<< "$patterns"
    for pattern in "${PATTERNS[@]}"; do
        for dir in "$base_dir"/$pattern; do
            if is_valid_model "$dir"; then
                echo "$dir"
                return 0
            fi
        done
    done
    return 1
}

# Search for models with various naming conventions
DEEPSEEK_PATTERNS="DeepSeek-OCR*,deepseek-ocr*,deepseek*OCR*"
QWEN8B_PATTERNS="Qwen3-VL-8B*,Qwen*8B*,qwen*8b*"
QWEN32B_PATTERNS="Qwen3-VL-32B*,Qwen*32B*,qwen*32b*"

DEEPSEEK_PATH=$(find_model "$MODELS_DIR" "$DEEPSEEK_PATTERNS" 2>/dev/null || echo "")
QWEN8B_PATH=$(find_model "$MODELS_DIR" "$QWEN8B_PATTERNS" 2>/dev/null || echo "")
QWEN32B_PATH=$(find_model "$MODELS_DIR" "$QWEN32B_PATTERNS" 2>/dev/null || echo "")

# Show what's found
[ -n "$DEEPSEEK_PATH" ] && echo -e "  ${GREEN}✓${NC} DeepSeek-OCR-2 found: $(basename "$DEEPSEEK_PATH")"
[ -n "$QWEN8B_PATH" ] && echo -e "  ${GREEN}✓${NC} Qwen3-VL-8B found: $(basename "$QWEN8B_PATH")"
[ -n "$QWEN32B_PATH" ] && echo -e "  ${GREEN}✓${NC} Qwen3-VL-32B found: $(basename "$QWEN32B_PATH")"
echo ""

DL_DEEPSEEK="n"
DL_QWEN8B="n"
DL_QWEN32B="n"

if [ -z "$DEEPSEEK_PATH" ]; then
    DL_DEEPSEEK=$(ask "  Download DeepSeek-OCR-2? (required) [Y/n]: " "Y")
    DEEPSEEK_PATH="$MODELS_DIR/DeepSeek-OCR-2-model"
fi

if [ -z "$QWEN8B_PATH" ]; then
    DL_QWEN8B=$(ask "  Download Qwen3-VL-8B? (recommended) [Y/n]: " "Y")
    QWEN8B_PATH="$MODELS_DIR/Qwen3-VL-8B-model"
fi

if [ -z "$QWEN32B_PATH" ]; then
    DL_QWEN32B=$(ask "  Download Qwen3-VL-32B? (65GB, optional) [y/N]: " "N")
    QWEN32B_PATH="$MODELS_DIR/Qwen3-VL-32B-Thinking"
fi

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

# Download files needed for Docker build
mkdir -p "$TEMP_DIR/src" "$TEMP_DIR/config"
curl -fsSL "$REPO_RAW/docker/Dockerfile" -o "$TEMP_DIR/Dockerfile"
curl -fsSL "$REPO_RAW/src/entrypoint.py" -o "$TEMP_DIR/src/entrypoint.py"
curl -fsSL "$REPO_RAW/src/ocr_pipeline.py" -o "$TEMP_DIR/src/ocr_pipeline.py"
curl -fsSL "$REPO_RAW/src/stage1_classifier.py" -o "$TEMP_DIR/src/stage1_classifier.py"
curl -fsSL "$REPO_RAW/src/stage1_5_diagram.py" -o "$TEMP_DIR/src/stage1_5_diagram.py"
curl -fsSL "$REPO_RAW/src/stage2_ocr.py" -o "$TEMP_DIR/src/stage2_ocr.py"

# Generate config with actual model paths (relative to /workspace/models inside container)
DEEPSEEK_NAME=$(basename "$DEEPSEEK_PATH")
QWEN8B_NAME=$(basename "$QWEN8B_PATH")
QWEN32B_NAME=$(basename "$QWEN32B_PATH")

cat > "$TEMP_DIR/config/ocr_config.json" << CONFIGEOF
{
    "deepseek_model_path": "/workspace/models/$DEEPSEEK_NAME",
    "qwen_model_path": "/workspace/models/$QWEN8B_NAME",
    "qwen_describer_path": "/workspace/models/$QWEN32B_NAME",
    "classifier": "qwen3-vl-8b",
    "precision": "fp16",
    "describe_diagrams": false
}
CONFIGEOF

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

# Create self-contained ocr script with model path baked in
cat > "$BIN_DIR/ocr" << WRAPPER
#!/bin/bash
# OCR Pipeline - Process PDFs with AI
# Models: $MODELS_DIR

MODELS_DIR="$MODELS_DIR"

if [ -z "\$1" ]; then
    echo "Usage: ocr <input.pdf> [output.md] [options]"
    echo ""
    echo "Options:"
    echo "  --diagrams    Enable detailed diagram description (needs Qwen-32B)"
    echo ""
    echo "Examples:"
    echo "  ocr document.pdf                    # Output to ./output/document.md"
    echo "  ocr document.pdf result.md          # Output to ./result.md"
    echo "  ocr document.pdf --diagrams         # With diagram description"
    echo ""
    echo "Models: \$MODELS_DIR"
    exit 1
fi

INPUT_FILE="\$(realpath "\$1")"
INPUT_DIR="\$(dirname "\$INPUT_FILE")"
INPUT_BASENAME="\$(basename "\$INPUT_FILE")"
INPUT_NAME="\${INPUT_BASENAME%.pdf}"
shift

# Parse arguments
DESCRIBE_DIAGRAMS="false"
OUTPUT_FILE=""

while [[ \$# -gt 0 ]]; do
    case \$1 in
        --diagrams) DESCRIBE_DIAGRAMS="true"; shift ;;
        *.md) OUTPUT_FILE="\$1"; shift ;;
        *) shift ;;
    esac
done

# Determine output path
if [ -n "\$OUTPUT_FILE" ]; then
    # User specified output file
    OUTPUT_DIR="\$(dirname "\$OUTPUT_FILE")"
    OUTPUT_NAME="\$(basename "\$OUTPUT_FILE")"
    [ "\$OUTPUT_DIR" = "." ] && OUTPUT_DIR="\$(pwd)"
    mkdir -p "\$OUTPUT_DIR"
    OUTPUT_DIR="\$(realpath "\$OUTPUT_DIR")"
else
    # Default: ./output/<input>.md
    OUTPUT_DIR="\$(pwd)/output"
    OUTPUT_NAME="\${INPUT_NAME}.md"
    mkdir -p "\$OUTPUT_DIR"
fi

echo "OCR Pipeline"
echo "  Input:  \$INPUT_FILE"
echo "  Output: \$OUTPUT_DIR/\$OUTPUT_NAME"
echo ""

docker run --rm --gpus all \\
    --user \$(id -u):\$(id -g) \\
    -v "\$MODELS_DIR:/workspace/models:ro" \\
    -v "\$INPUT_DIR:/data/input:ro" \\
    -v "\$OUTPUT_DIR:/data/output" \\
    -e OCR_INPUT_PDF="/data/input/\$INPUT_BASENAME" \\
    -e OCR_OUTPUT_PATH="/data/output/\$OUTPUT_NAME" \\
    -e OCR_DESCRIBE_DIAGRAMS="\$DESCRIBE_DIAGRAMS" \\
    ocr-pipeline

echo ""
echo "Done! Output: \$OUTPUT_DIR/\$OUTPUT_NAME"
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
echo "Models: $MODELS_DIR"
echo ""
