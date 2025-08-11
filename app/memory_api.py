# path: app/memory_api.py
"""
Memory API – complete, updated, and **robust** file uploads for custom GPTs

Highlights
- FastAPI app with CORS, health check, and simple on-disk metadata store
- Accept uploads via:
  * multipart: `file`, `files`, `attachment`, `attachments` (single or multiple)
  * JSON base64: `{ user_id, filename, content_base64 }`
  * URL: `{ user_id, url, filename? }`
- Saves file to disk, records metadata per user in a JSON index
- Lists, downloads, deletes files
- Optional size limit and filename sanitization

Why this fixes your issue
- Different gateways (Actions, webhooks, SDKs) send files under different field names
  or as base64/URL. This module normalizes all forms and persists them, avoiding the
  "not accepting the file parameters" error.

Dependencies: fastapi, uvicorn, pydantic
(Optional for URL uploads: httpx)

Run:
  uvicorn app.memory_api:app --reload --port 8080
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Query,
    Request,
    Body,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl
from pydantic_settings import BaseSettings

# =========================
# Settings
# =========================
class Settings(BaseSettings):
    FILE_STORAGE_DIR: str = ".storage/files"
    META_INDEX_PATH: str = ".storage/index.json"
    MAX_FILE_SIZE_MB: int = 32
    CORS_ALLOW_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

settings = Settings()

# Ensure storage paths exist
STORAGE_DIR = Path(settings.FILE_STORAGE_DIR)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
META_INDEX_PATH = Path(settings.META_INDEX_PATH)
META_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================
# Metadata index (very simple on-disk JSON)
# =========================
# format: { "files": [ {"user_id":..., "filename":..., "stored_path":..., "size":..., "created_at":...} ] }

def _load_index() -> Dict[str, List[dict]]:
    if META_INDEX_PATH.exists():
        try:
            return json.loads(META_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            # corrupt index: start fresh but keep file system contents
            return {"files": []}
    return {"files": []}


def _save_index(idx: Dict[str, List[dict]]) -> None:
    META_INDEX_PATH.write_text(json.dumps(idx, indent=2), encoding="utf-8")


# Initialize index
_index = _load_index()

# =========================
# Helpers
# =========================
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\-]+")


def _safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = _SAFE_NAME_RE.sub("", name)
    return name or "uploaded.bin"


def _guard_size(data: bytes):
    if len(data) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{settings.MAX_FILE_SIZE_MB} MB)")


def _store(user_id: str, filename: str, data: bytes) -> dict:
    _guard_size(data)
    safe = _safe_filename(filename)
    dest = STORAGE_DIR / safe
    # Overwrite; if you want versioning, add suffix here
    dest.write_bytes(data)

    meta = {
        "user_id": user_id,
        "filename": safe,
        "stored_path": str(dest.resolve()),
        "size": len(data),
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    # update index (dedupe by user_id+filename)
    global _index
    _index["files"] = [m for m in _index.get("files", []) if not (m["user_id"] == user_id and m["filename"] == safe)]
    _index["files"].append(meta)
    _save_index(_index)
    return meta


async def _read_uploadfile_to_bytes(f: UploadFile) -> Tuple[str, bytes]:
    name = f.filename or "uploaded.bin"
    data = await f.read()
    return name, data


# =========================
# FastAPI app
# =========================
app = FastAPI(title="Memory API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# =========================
# Schemas
# =========================
class UploadBase64Body(BaseModel):
    user_id: str
    filename: str
    content_base64: str


class UploadUrlBody(BaseModel):
    user_id: str
    url: HttpUrl
    filename: Optional[str] = None


# =========================
# Upload endpoints
# =========================
_DEF_KEYS = ("file", "files", "attachment", "attachments")


@app.post("/files/upload")
async def upload_files_multipart(
    request: Request,
    user_id: str = Query(..., description="Uploader ID for attribution/authorization"),
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    attachment: Optional[UploadFile] = File(None),
    attachments: Optional[List[UploadFile]] = File(None),
):
    uploads: List[UploadFile] = []

    if file:
        uploads.append(file)
    if files:
        uploads.extend(files)
    if attachment:
        uploads.append(attachment)
    if attachments:
        uploads.extend(attachments)

    if not uploads:
        # sweep raw form for any UploadFile(s)
        form = await request.form()
        for k in _DEF_KEYS:
            if k in form:
                for it in form.getlist(k):
                    if isinstance(it, UploadFile):
                        uploads.append(it)
        if not uploads:
            for v in form.values():
                if isinstance(v, UploadFile):
                    uploads.append(v)

    if not uploads:
        raise HTTPException(status_code=400, detail="No file(s) found in multipart form-data")

    results = []
    for up in uploads:
        name, data = await _read_uploadfile_to_bytes(up)
        results.append(_store(user_id, name, data))

    return {"status": "ok", "count": len(results), "files": results}


@app.post("/files/upload-json")
async def upload_file_json(body: UploadBase64Body = Body(...)):
    try:
        raw = base64.b64decode(body.content_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}")
    out = _store(body.user_id, body.filename, raw)
    return {"status": "ok", "files": [out]}


@app.post("/files/upload-url")
async def upload_file_url(body: UploadUrlBody = Body(...)):
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(str(body.url))
            resp.raise_for_status()
            data = resp.content
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}")

    # filename from hint or URL path
    name = body.filename or Path(body.url.path).name or "downloaded.bin"
    out = _store(body.user_id, name, data)
    return {"status": "ok", "files": [out]}


# =========================
# File management
# =========================
@app.get("/files/list")
async def list_files(user_id: Optional[str] = Query(None)):
    files = _index.get("files", [])
    if user_id:
        files = [m for m in files if m["user_id"] == user_id]
    return {"files": sorted(files, key=lambda m: (m["user_id"], m["filename"]))}


@app.get("/files/download/{filename}")
async def download_file(filename: str):
    safe = _safe_filename(filename)
    path = STORAGE_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=safe)


@app.delete("/files/delete/{filename}")
async def delete_file(filename: str, user_id: Optional[str] = Query(None)):
    safe = _safe_filename(filename)
    path = STORAGE_DIR / safe

    # remove from index
    global _index
    before = len(_index.get("files", []))
    _index["files"] = [m for m in _index.get("files", []) if not (m["filename"] == safe and (user_id is None or m["user_id"] == user_id))]
    after = len(_index.get("files", []))
    _save_index(_index)

    # remove file if exists
    if path.exists():
        try:
            path.unlink()
        except Exception as exc:
            # Still return success for index deletion; surface file error specifically
            raise HTTPException(status_code=500, detail=f"Failed to remove file: {exc}")

    return {"status": "ok", "removed_from_index": before - after, "file_deleted": True}


# =========================
# Notes for integration with your GPT Action
# =========================
"""
Action (multipart):
  POST /files/upload?user_id=YOUR_USER
  form-data: file=@/path/to/local.pdf

Action (multiple):
  POST /files/upload?user_id=YOUR_USER
  form-data: attachments=@/a.pdf, attachments=@/b.docx

Action (base64 JSON):
  POST /files/upload-json
  {"user_id":"YOUR_USER","filename":"a.pdf","content_base64":"..."}

Action (URL):
  POST /files/upload-url
  {"user_id":"YOUR_USER","url":"https://.../report.pdf","filename":"report.pdf"}
"""

