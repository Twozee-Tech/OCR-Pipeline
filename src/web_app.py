"""Web UI for OCR Pipeline — FastAPI app with basic auth and job queue."""
import asyncio
import os
import secrets
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEB_USER = os.environ.get("OCR_WEB_USER", "admin")
WEB_PASS = os.environ.get("OCR_WEB_PASS", "changeme")
UPLOAD_DIR = Path("/data/uploads")
OUTPUT_DIR = Path("/data/output")
TEMPLATE_DIR = Path(__file__).parent / "templates"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="OCR Pipeline Web UI")
security = HTTPBasic()
job_queue: asyncio.Queue = asyncio.Queue()
jobs: dict = {}  # job_id -> Job dict


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _verify(creds: HTTPBasicCredentials = Depends(security)) -> str:
    ok = secrets.compare_digest(creds.username, WEB_USER) and secrets.compare_digest(
        creds.password, WEB_PASS
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return creds.username


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
def _run_ocr_subprocess(job: dict) -> str:
    """Run the OCR pipeline as a subprocess so GPU memory is fully freed on exit."""
    cmd = [
        sys.executable, "/app/ocr_pipeline.py",
        job["pdf_path"], job["output_path"],
        "--dpi", str(job["dpi"]),
        "--verbose",
    ]
    if job["describe_diagrams"]:
        cmd.append("--describe-diagrams")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip().splitlines()
        last_lines = "\n".join(stderr[-5:]) if stderr else "unknown error"
        raise RuntimeError(f"Pipeline failed (exit {result.returncode}):\n{last_lines}")
    return job["output_path"]


async def worker_loop() -> None:
    """Pull jobs off the queue and process them one at a time."""
    loop = asyncio.get_event_loop()
    while True:
        job_id = await job_queue.get()
        job = jobs.get(job_id)
        if job is None or job["status"] == "cancelled":
            job_queue.task_done()
            continue

        job["status"] = "processing"
        job["started_at"] = _now()
        try:
            result_path = await loop.run_in_executor(
                None, _run_ocr_subprocess, job
            )
            job["status"] = "done"
            job["result_path"] = result_path
        except Exception as exc:
            job["status"] = "error"
            job["error_message"] = str(exc)
        finally:
            job["completed_at"] = _now()
            job_queue.task_done()


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(worker_loop())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(_user: str = Depends(_verify)):
    html_path = TEMPLATE_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/health")
async def health():
    processing = sum(1 for j in jobs.values() if j["status"] == "processing")
    queued = sum(1 for j in jobs.values() if j["status"] == "queued")
    return {"status": "ok", "processing": processing, "queued": queued}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    describe_diagrams: bool = Form(False),
    dpi: int = Form(200),
    _user: str = Depends(_verify),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    job_id = uuid.uuid4().hex[:12]
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_upload_dir / "original.pdf"
    file_size = 0
    with open(pdf_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
            file_size += len(chunk)

    job_output_dir = OUTPUT_DIR / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(job_output_dir / "original.md")

    job = {
        "id": job_id,
        "filename": file.filename,
        "status": "queued",
        "created_at": _now(),
        "started_at": None,
        "completed_at": None,
        "result_path": None,
        "error_message": None,
        "describe_diagrams": describe_diagrams,
        "dpi": dpi,
        "file_size_bytes": file_size,
        "pdf_path": str(pdf_path),
        "output_path": output_path,
    }
    jobs[job_id] = job
    await job_queue.put(job_id)

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs")
async def list_jobs(_user: str = Depends(_verify)):
    sorted_jobs = sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return [
        {
            "id": j["id"],
            "filename": j["filename"],
            "status": j["status"],
            "created_at": j["created_at"],
            "started_at": j["started_at"],
            "completed_at": j["completed_at"],
            "error_message": j["error_message"],
            "describe_diagrams": j["describe_diagrams"],
            "dpi": j["dpi"],
            "file_size_bytes": j["file_size_bytes"],
        }
        for j in sorted_jobs
    ]


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, _user: str = Depends(_verify)):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "error_message": job["error_message"],
        "describe_diagrams": job["describe_diagrams"],
        "dpi": job["dpi"],
        "file_size_bytes": job["file_size_bytes"],
    }


@app.get("/jobs/{job_id}/download")
async def download(job_id: str, _user: str = Depends(_verify)):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not finished yet")

    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    stem = Path(job["filename"]).stem
    return FileResponse(
        result_path,
        media_type="text/markdown",
        filename=f"{stem}.md",
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, _user: str = Depends(_verify)):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "queued":
        job["status"] = "cancelled"

    # Clean up files
    upload_path = UPLOAD_DIR / job_id
    output_path = OUTPUT_DIR / job_id
    shutil.rmtree(upload_path, ignore_errors=True)
    shutil.rmtree(output_path, ignore_errors=True)

    del jobs[job_id]
    return {"status": "deleted"}
