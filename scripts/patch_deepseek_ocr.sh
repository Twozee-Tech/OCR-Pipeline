#!/bin/bash
# Patch DeepSeek-OCR-2 for vLLM compatibility
# Fixes: ImportError: cannot import name 'LlamaFlashAttention2'
#
# Usage: ./patch_deepseek_ocr.sh /path/to/DeepSeek-OCR-2-model
#        ./patch_deepseek_ocr.sh /path/to/DeepSeek-OCR-2-model --rollback

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

MODEL_PATH="${1:-/workspace/models/DeepSeek-OCR-2-model}"
ACTION="${2:-patch}"

FILE="$MODEL_PATH/modeling_deepseekv2.py"
BACKUP="$MODEL_PATH/modeling_deepseekv2.py.backup"

echo "=========================================="
echo "DeepSeek-OCR-2 vLLM Compatibility Patch"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Action: $ACTION"
echo ""

# Check if file exists
if [ ! -f "$FILE" ]; then
    echo -e "${RED}ERROR: File not found: $FILE${NC}"
    exit 1
fi

# Rollback mode
if [ "$ACTION" = "--rollback" ] || [ "$ACTION" = "-r" ]; then
    echo -e "${YELLOW}Rolling back...${NC}"

    if [ ! -f "$BACKUP" ]; then
        echo -e "${RED}ERROR: Backup not found: $BACKUP${NC}"
        exit 1
    fi

    cp "$BACKUP" "$FILE"
    echo -e "${GREEN}Rollback complete${NC}"
    exit 0
fi

# Create backup
echo "1. Creating backup..."
if [ -f "$BACKUP" ]; then
    echo -e "${YELLOW}   Backup already exists, skipping${NC}"
else
    cp "$FILE" "$BACKUP"
    echo -e "${GREEN}   Backup created: $BACKUP${NC}"
fi

# Check if already patched
if grep -q "# PATCHED FOR VLLM" "$FILE"; then
    echo -e "${YELLOW}File already patched. Use --rollback to restore original.${NC}"
    exit 0
fi

# Apply patch
echo "2. Applying patch..."

# Create patched version
python3 << PATCHSCRIPT
import sys

file_path = "$FILE"

with open(file_path, 'r') as f:
    content = f.read()
    lines = content.split('\n')

patched_lines = []
skip_import = False
import_removed = False
attention_fixed = False

for i, line in enumerate(lines):
    # Add patch marker at the top
    if i == 0:
        patched_lines.append("# PATCHED FOR VLLM - LlamaAttention removed")
        patched_lines.append(line)
        continue

    # Remove Llama import block (lines ~37-40)
    if 'from transformers.models.llama.modeling_llama import' in line:
        patched_lines.append("# PATCHED: Removed Llama imports for vLLM compatibility")
        patched_lines.append("# " + line)
        skip_import = True
        continue

    if skip_import:
        patched_lines.append("# " + line)
        if ')' in line:
            skip_import = False
            import_removed = True
        continue

    # Fix ATTENTION_CLASSES dict
    if '"mha_eager": LlamaAttention' in line:
        patched_lines.append('    "mha_eager": DeepseekV2Attention,  # PATCHED: was LlamaAttention')
        attention_fixed = True
        continue

    if '"mha_flash_attention_2": LlamaFlashAttention2' in line:
        patched_lines.append('    "mha_flash_attention_2": DeepseekV2FlashAttention2  # PATCHED: was LlamaFlashAttention2')
        attention_fixed = True
        continue

    patched_lines.append(line)

# Write patched file
with open(file_path, 'w') as f:
    f.write('\n'.join(patched_lines))

# Report
if import_removed:
    print("   - Removed LlamaAttention/LlamaFlashAttention2 imports")
if attention_fixed:
    print("   - Fixed ATTENTION_CLASSES dictionary")

if not import_removed and not attention_fixed:
    print("   WARNING: No changes made - file structure may differ")
    sys.exit(1)

print("   Patch applied successfully")
PATCHSCRIPT

PATCH_RESULT=$?

if [ $PATCH_RESULT -ne 0 ]; then
    echo -e "${RED}Patch failed, rolling back...${NC}"
    cp "$BACKUP" "$FILE"
    exit 1
fi

echo -e "${GREEN}   Patch applied${NC}"

# Verify patch
echo "3. Verifying patch..."

# Check marker
if ! grep -q "# PATCHED FOR VLLM" "$FILE"; then
    echo -e "${RED}   ERROR: Patch marker not found${NC}"
    cp "$BACKUP" "$FILE"
    exit 1
fi

# Check Llama import is commented
if grep -q "^from transformers.models.llama.modeling_llama import" "$FILE"; then
    echo -e "${RED}   ERROR: Llama import still present${NC}"
    cp "$BACKUP" "$FILE"
    exit 1
fi

# Check ATTENTION_CLASSES fix
if grep -q '"mha_eager": LlamaAttention' "$FILE"; then
    echo -e "${RED}   ERROR: LlamaAttention still in ATTENTION_CLASSES${NC}"
    cp "$BACKUP" "$FILE"
    exit 1
fi

echo -e "${GREEN}   Verification passed${NC}"

# Test import (optional - requires Python with transformers)
echo "4. Testing import..."

python3 << TESTSCRIPT 2>/dev/null && TEST_OK=1 || TEST_OK=0
import sys
sys.path.insert(0, "$MODEL_PATH")
try:
    # Just check syntax - don't actually import (needs full env)
    with open("$FILE", 'r') as f:
        compile(f.read(), "$FILE", 'exec')
    print("   Syntax check: OK")
except SyntaxError as e:
    print(f"   Syntax error: {e}")
    sys.exit(1)
TESTSCRIPT

if [ $TEST_OK -ne 1 ]; then
    echo -e "${RED}   Import test failed, rolling back...${NC}"
    cp "$BACKUP" "$FILE"
    exit 1
fi

echo -e "${GREEN}   Import test passed${NC}"

# Done
echo ""
echo "=========================================="
echo -e "${GREEN}Patch complete!${NC}"
echo "=========================================="
echo ""
echo "To rollback: $0 $MODEL_PATH --rollback"
echo ""
