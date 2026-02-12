#!/bin/bash
# Advanced OCR for NVIDIA DGX Spark - Installer v3.0
# Usage: curl -fsSL https://raw.githubusercontent.com/Twozee-Tech/Advanced-OCR-Nvidia-DGX-SPARK/main/install.sh | bash
#
# Installs a single 'ocr' command - no repo files left behind
VERSION="3.0"

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REPO_RAW="https://raw.githubusercontent.com/Twozee-Tech/Advanced-OCR-Nvidia-DGX-SPARK/featuresv2"
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
echo "     OCR Pipeline v${VERSION} - vLLM Edition     "
echo "=============================================="
echo -e "${NC}"

# ============================================
# Step 1: Check Prerequisites
# ============================================
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

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
echo -e "${YELLOW}[2/6] Configure model storage${NC}"
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
echo -e "${YELLOW}[3/6] Select models to download${NC}"
echo ""
echo "  Available models:"
echo "    1) DeepSeek-OCR-2      (~8GB)  - Required for text extraction"
echo "    2) Qwen3-VL-30B        (~60GB) - Classification + diagrams"
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
QWEN30B_PATTERNS="Qwen3-VL-30B*,Qwen*30B*,qwen*30b*"

DEEPSEEK_PATH=$(find_model "$MODELS_DIR" "$DEEPSEEK_PATTERNS" 2>/dev/null || echo "")
QWEN30B_PATH=$(find_model "$MODELS_DIR" "$QWEN30B_PATTERNS" 2>/dev/null || echo "")

# Show what's found
[ -n "$DEEPSEEK_PATH" ] && echo -e "  ${GREEN}✓${NC} DeepSeek-OCR-2 found: $(basename "$DEEPSEEK_PATH")"
[ -n "$QWEN30B_PATH" ] && echo -e "  ${GREEN}✓${NC} Qwen3-VL-30B found: $(basename "$QWEN30B_PATH")"
echo ""

DL_DEEPSEEK="n"
DL_QWEN30B="n"

if [ -z "$DEEPSEEK_PATH" ]; then
    DL_DEEPSEEK=$(ask "  Download DeepSeek-OCR-2? (required) [Y/n]: " "Y")
    DEEPSEEK_PATH="$MODELS_DIR/DeepSeek-OCR-2-model"
fi

if [ -z "$QWEN30B_PATH" ]; then
    DL_QWEN30B=$(ask "  Download Qwen3-VL-30B? (60GB, recommended) [Y/n]: " "Y")
    QWEN30B_PATH="$MODELS_DIR/Qwen3-VL-30B-model"
fi

if [[ "$DL_DEEPSEEK" =~ ^[Yy] ]] || [[ "$DL_QWEN30B" =~ ^[Yy] ]]; then
    echo ""
    echo -e "  ${BLUE}Downloading models...${NC}"

    DL_CMDS=""
    [[ "$DL_DEEPSEEK" =~ ^[Yy] ]] && DL_CMDS+="echo '>>> DeepSeek-OCR-2...' && hf download deepseek-ai/DeepSeek-OCR-2 --local-dir /models/DeepSeek-OCR-2-model && "
    [[ "$DL_QWEN30B" =~ ^[Yy] ]] && DL_CMDS+="echo '>>> Qwen3-VL-30B-FP8...' && hf download Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --local-dir /models/Qwen3-VL-30B-FP8-model && "
    DL_CMDS+="echo '>>> Done!'"

    HAD_PYTHON_IMAGE=$(docker images -q python:3.11-slim 2>/dev/null)

    docker run --rm \
        -v "$MODELS_DIR:/models" \
        -e HF_HOME=/models/.cache \
        -e HF_TOKEN="${HF_TOKEN:-}" \
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
echo -e "${YELLOW}[4/6] Building Docker image...${NC}"

# Download files needed for Docker build
mkdir -p "$TEMP_DIR/src/templates" "$TEMP_DIR/config"
curl -fsSL "$REPO_RAW/docker/Dockerfile" -o "$TEMP_DIR/Dockerfile"
curl -fsSL "$REPO_RAW/src/entrypoint.py" -o "$TEMP_DIR/src/entrypoint.py"
curl -fsSL "$REPO_RAW/src/ocr_pipeline.py" -o "$TEMP_DIR/src/ocr_pipeline.py"
curl -fsSL "$REPO_RAW/src/qwen_processor.py" -o "$TEMP_DIR/src/qwen_processor.py"
curl -fsSL "$REPO_RAW/src/stage2_ocr.py" -o "$TEMP_DIR/src/stage2_ocr.py"
curl -fsSL "$REPO_RAW/src/stage2_ocr_worker.py" -o "$TEMP_DIR/src/stage2_ocr_worker.py"
curl -fsSL "$REPO_RAW/src/web_app.py" -o "$TEMP_DIR/src/web_app.py"
curl -fsSL "$REPO_RAW/src/qwen_ocr.py" -o "$TEMP_DIR/src/qwen_ocr.py"
curl -fsSL "$REPO_RAW/src/templates/index.html" -o "$TEMP_DIR/src/templates/index.html"

# Generate config with actual model paths (relative to /workspace/models inside container)
DEEPSEEK_NAME=$(basename "$DEEPSEEK_PATH")
QWEN30B_NAME=$(basename "$QWEN30B_PATH")

cat > "$TEMP_DIR/config/ocr_config.json" << CONFIGEOF
{
    "deepseek_model_path": "/workspace/models/$DEEPSEEK_NAME",
    "qwen_model_path": "/workspace/models/$QWEN30B_NAME",
    "qwen_gpu_memory_utilization": 0.40,
    "qwen_max_model_len": 6144,
    "ocr_gpu_memory_utilization": 0.80,
    "ocr_max_model_len": 8192,
    "describe_diagrams": false
}
CONFIGEOF

docker build -t ocr-pipeline "$TEMP_DIR"
echo -e "  ${GREEN}✓${NC} Docker image built"

# ============================================
# Step 5: Install 'ocr' Command
# ============================================
echo ""
echo -e "${YELLOW}[5/6] Installing 'ocr' command...${NC}"

# Determine install location
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"

# Create self-contained ocr script with model path baked in
cat > "$BIN_DIR/ocr" << WRAPPER
#!/bin/bash
# OCR Pipeline v3.0 - Process PDFs with AI (vLLM)
# Models: $MODELS_DIR

MODELS_DIR="$MODELS_DIR"

if [ -z "\$1" ]; then
    echo "Usage: ocr <input.pdf> [output.md] [options]"
    echo ""
    echo "Options:"
    echo "  --diagrams    Enable detailed diagram description"
    echo "  -v, --verbose Verbose output (disables progress bar)"
    echo ""
    echo "Examples:"
    echo "  ocr document.pdf                    # Output to ./output/document.md"
    echo "  ocr document.pdf result.md          # Output to ./result.md"
    echo "  ocr document.pdf --diagrams         # With diagram description"
    echo "  ocr document.pdf -v                 # Verbose output"
    echo ""
    echo "Environment variables:"
    echo "  OCR_PIPELINE=qwen    Use single-phase Qwen OCR (experimental)"
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
VERBOSE="false"
OUTPUT_FILE=""

while [[ \$# -gt 0 ]]; do
    case \$1 in
        --diagrams) DESCRIBE_DIAGRAMS="true"; shift ;;
        -v|--verbose) VERBOSE="true"; shift ;;
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

echo "OCR Pipeline v3.0"
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
    -e OCR_VERBOSE="\$VERBOSE" \\
    -e OCR_PIPELINE="\${OCR_PIPELINE:-default}" \\
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
# Step 6: Install 'ocr-web' Command
# ============================================
echo ""
echo -e "${YELLOW}[6/6] Installing 'ocr-web' command...${NC}"

DATA_DIR="$HOME/.local/share/ocr-data"
mkdir -p "$DATA_DIR"

cat > "$BIN_DIR/ocr-web" << WEBWRAPPER
#!/bin/bash
# OCR Pipeline v3.0 - Web UI
# Models: $MODELS_DIR

MODELS_DIR="$MODELS_DIR"
DATA_DIR="$DATA_DIR"
IMAGE_NAME="ocr-pipeline"
CONTAINER_NAME="ocr-web"

case "\${1:-help}" in
  start)
    if docker ps -q -f name="\$CONTAINER_NAME" 2>/dev/null | grep -q .; then
        echo "ocr-web is already running. Use 'ocr-web stop' first."
        exit 1
    fi
    docker rm "\$CONTAINER_NAME" 2>/dev/null || true
    docker run --gpus all -d --name "\$CONTAINER_NAME" \\
      -p "\${OCR_WEB_PORT:-14000}:\${OCR_WEB_PORT:-14000}" \\
      -e OCR_WEB_MODE=true \\
      -e OCR_WEB_PORT="\${OCR_WEB_PORT:-14000}" \\
      -e OCR_WEB_USER="\${OCR_WEB_USER:-admin}" \\
      -e OCR_WEB_PASS="\${OCR_WEB_PASS:-changeme}" \\
      -e CODER_CONTAINER_NAME="\${CODER_CONTAINER_NAME:-}" \\
      -e OCR_PIPELINE="\${OCR_PIPELINE:-default}" \\
      -v "\$MODELS_DIR:/workspace/models:ro" \\
      -v "\$DATA_DIR:/data" \\
      -v /var/run/docker.sock:/var/run/docker.sock \\
      "\$IMAGE_NAME"
    echo "OCR Web UI started on http://localhost:\${OCR_WEB_PORT:-14000}"
    echo "  Username: \${OCR_WEB_USER:-admin}"
    echo "  Stop with: ocr-web stop"
    ;;
  stop)
    docker stop "\$CONTAINER_NAME" 2>/dev/null && docker rm "\$CONTAINER_NAME" 2>/dev/null
    echo "OCR Web UI stopped."
    ;;
  status)
    docker ps -f name="\$CONTAINER_NAME"
    ;;
  logs)
    docker logs -f "\$CONTAINER_NAME"
    ;;
  *)
    echo "Usage: ocr-web <command>"
    echo ""
    echo "Commands:"
    echo "  start   Start the web UI (default port 14000)"
    echo "  stop    Stop the web UI"
    echo "  status  Show container status"
    echo "  logs    Follow container logs"
    echo ""
    echo "Environment variables (set before 'start'):"
    echo "  OCR_WEB_USER            Username for basic auth (default: admin)"
    echo "  OCR_WEB_PASS            Password for basic auth (default: changeme)"
    echo "  OCR_WEB_PORT            Port to listen on (default: 14000)"
    echo "  CODER_CONTAINER_NAME    Auto-pause this container during OCR (optional)"
    echo "  OCR_PIPELINE            Pipeline to use: default or qwen (default: default)"
    echo ""
    echo "Examples:"
    echo "  OCR_WEB_USER=myuser OCR_WEB_PASS=secret ocr-web start"
    echo "  CODER_CONTAINER_NAME=vllm-Qwen3Next-Coder ocr-web start"
    ;;
esac
WEBWRAPPER

chmod +x "$BIN_DIR/ocr-web"
echo -e "  ${GREEN}✓${NC} Installed to $BIN_DIR/ocr-web"

# ============================================
# Done
# ============================================
echo ""
echo -e "${GREEN}=============================================="
echo "            Installation Complete!            "
echo "==============================================${NC}"
echo ""
echo "Usage:"
echo "  ocr document.pdf                  # CLI mode"
echo "  ocr document.pdf --diagrams       # CLI with diagrams"
echo ""
echo "  ocr-web start                     # Start web UI on :14000"
echo "  ocr-web stop                      # Stop web UI"
echo ""
echo "Models: $MODELS_DIR"
echo ""
