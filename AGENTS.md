# OCR Pipeline - Development Guide for AGENTS

## Build & Deployment

### Docker (Primary Deployment)
```bash
# Build image (automatically from root)
docker-compose -f docker/docker-compose.yml build

# Run pipeline (CLI mode)
ocr document.pdf                           # Uses wrapper script
ocr document.pdf --diagrams               # With diagram descriptions
ocr document.pdf result.md                # Custom output

# Web UI mode
ocr-web start                             # Start FastAPI on port 14000
ocr-web stop                              # Stop web UI
```

### Environment Variables (Docker)
| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_INPUT_PDF` | auto-detect | Input PDF path |
| `OCR_OUTPUT_PATH` | `/data/output/{name}.md` | Output markdown path |
| `OCR_DESCRIBE_DIAGRAMS` | `false` | Stage 1.5 diagram descriptions |
| `OCR_DPI` | `200` | PDF rendering resolution |
| `OCR_VERBOSE` | `false` | Verbose output mode |
| `OCR_WEB_MODE` | `false` | Enable web UI server |
| `OCR_WEB_PORT` | `14000` | Web UI port |
| `OCR_WEB_USER` | `admin` | Web UI username |
| `OCR_WEB_PASS` | `changeme` | Web UI password |
| `OCR_PIPELINE` | `default` | Pipeline: `default` or `qwen` (experimental) |

### Models (Required)
- **DeepSeek-OCR-2** (~8GB) - Required for Stage 2 OCR
- **Qwen3-VL-30B** (~60GB) - Required for classification + diagrams
- Storage: `~/.cache/ocr-models` (default, configurable via installer)

## Code Style

### Python Conventions
- **Version**: Python 3.11+
- **Imports**: Explicit imports only. No wildcard imports (`from x import *`)
- **Line Length**: 100 characters maximum
- **Type Hints**: Required for all function signatures
- **Docstrings**: Google-style docstrings for public functions

### Naming Conventions
| Pattern | Example | Use For |
|---------|---------|---------|
| `snake_case` | `ocr_pipeline.py` | Modules, files, variables |
| `PascalCase` | `ProgressBar` | Classes |
| `UPPER_CASE` | `DIAGRAM_TYPES` | Constants |

### Error Handling
- Use `sys.exit(1)` for CLI errors (not exceptions)
- Raise exceptions for library code (don't exit)
- Log errors with `print()` in CLI, `logger.warning()` in web app
- Always cleanup temp files in `finally` blocks

### Code Organization
```
src/
├── entrypoint.py          # Docker entrypoint (CLI or web mode)
├── ocr_pipeline.py        # Main orchestrator (2-stage: classify → OCR)
├── qwen_processor.py      # Stage 1: Qwen3-VL classification/descriptions
├── stage2_ocr.py          # Stage 2: DeepSeek-OCR processing
├── stage2_ocr_worker.py   # DeepSeek worker (transformers subprocess)
├── qwen_ocr.py            # Experimental single-phase Qwen OCR
└── web_app.py             # FastAPI web UI (basic auth, job queue)
```

## Key Patterns

### 1. Two-Venv Architecture (Docker)
- `/opt/venv_vllm/` - vLLM + Qwen dependencies (Stage 1)
- `/opt/venv_ocr/` - transformers 4.46.3 + DeepSeek (Stage 2)
- Never mix dependencies between venvs

### 2. VRAM Management (Critical)
- Load model → process → unload → load next model
- Sequential loading prevents OOM on 48GB GPUs
- Always call `torch.cuda.empty_cache()` and `torch.cuda.synchronize()` after unload

### 3. Input/Output Patterns
- **Images**: vLLM requires `file://` URIs, use `qwen_processor.save_pages_to_temp()`
- **Classification**: JSON format for Stage 1 → Stage 2 IPC
- **PDF → Images**: PyMuPDF with configurable DPI (default 200)

### 4. Prompts
All prompts are string constants in their respective modules:
- `CLASSIFY_PROMPT` in `qwen_processor.py` - Page classification
- `DIAGRAM_PROMPT` in `qwen_processor.py` - Diagram descriptions
- `OCR_PROMPT` in `qwen_ocr.py` - Single-phase OCR
- `PROMPTS` dict in `stage2_ocr.py` - Content-aware prompts

## Web UI (FastAPI)
- Port: 14000 (configurable via `OCR_WEB_PORT`)
- Auth: Basic auth (`OCR_WEB_USER`/`OCR_WEB_PASS`)
- Job queue: Async processing with cancellation support
- Auto-pause feature: Stops sibling vLLM container during OCR to free GPU

## Testing
No formal test suite exists. Test manually:
1. Test Stage 1 standalone: `python3 src/qwen_processor.py`
2. Test Stage 2 standalone: `python3 src/stage2_ocr.py`
3. Full pipeline: `python3 src/ocr_pipeline.py input.pdf`

## Common Tasks

### Add New Pipeline Mode
1. Create new processor module in `src/`
2. Update `entrypoint.py` to handle new mode
3. Update `web_app.py` `_run_ocr_subprocess()` to include new mode
4. Update `ocr` wrapper script in `install.sh`

### Add New Content Type
1. Update classification prompt in `qwen_processor.py`
2. Add prompt variant in `stage2_ocr.py` `PROMPTS` dict
3. Update `stage2_ocr.py` `clean_result()` if special handling needed

### Debug VRAM Issues
1. Check `nvidia-smi` before/after model load/unload
2. Verify sequential loading (no simultaneous models)
3. Reduce `gpu_memory_utilization` in config if needed
