#!/bin/bash
# OCR Pipeline - One Line Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/<user>/ocr-pipeline/main/install.sh | bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REPO_URL="https://github.com/Twozee-Tech/OCR-Pipeline.git"
INSTALL_DIR="$HOME/ocr-pipeline"

echo -e "${BLUE}"
echo "=============================================="
echo "       OCR Pipeline - Installer              "
echo "=============================================="
echo -e "${NC}"

# ============================================
# Prerequisites
# ============================================
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check git
if ! command -v git &> /dev/null; then
    echo -e "${RED}ERROR: git not installed${NC}"
    exit 1
fi

# Check docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker not installed${NC}"
    echo "Install Docker: https://docs.docker.com/engine/install/"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Docker found"

# Check nvidia-container-toolkit
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo -e "${YELLOW}WARNING: NVIDIA container runtime not detected${NC}"
    echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "  ${GREEN}✓${NC} GPU: $GPU"
else
    echo -e "${YELLOW}WARNING: nvidia-smi not found${NC}"
fi

# ============================================
# Clone Repository
# ============================================
echo ""
echo -e "${YELLOW}Installing to $INSTALL_DIR...${NC}"

if [ -d "$INSTALL_DIR" ]; then
    echo -e "  Directory exists, updating..."
    cd "$INSTALL_DIR"
    git pull --quiet
else
    git clone --quiet "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

echo -e "  ${GREEN}✓${NC} Repository ready"

# ============================================
# Run Setup
# ============================================
echo ""
chmod +x setup.sh
./setup.sh

# ============================================
# Add to PATH (optional)
# ============================================
echo ""
echo -e "${YELLOW}Add to PATH?${NC}"
echo "  This lets you run 'ocr' from anywhere"
read -p "  Add $INSTALL_DIR to PATH? [Y/n]: " ADD_PATH
ADD_PATH="${ADD_PATH:-Y}"

if [[ "$ADD_PATH" =~ ^[Yy] ]]; then
    SHELL_RC=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    fi

    if [ -n "$SHELL_RC" ]; then
        if ! grep -q "ocr-pipeline" "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo "# OCR Pipeline" >> "$SHELL_RC"
            echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$SHELL_RC"
            echo -e "  ${GREEN}✓${NC} Added to $SHELL_RC"
            echo -e "  Run: ${BLUE}source $SHELL_RC${NC} or restart terminal"
        else
            echo -e "  ${GREEN}✓${NC} Already in PATH"
        fi
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
echo "  cd $INSTALL_DIR"
echo "  ./ocr /path/to/document.pdf"
echo ""
echo "Or if added to PATH:"
echo "  ocr /path/to/document.pdf"
echo ""
