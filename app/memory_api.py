# app/memory_api.py
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    select,
    text as sql_text,
)
from sqlalchemy.engine import Engine
 the
from sqlalchemy.exc import SQLAlchemyError, NoResultFound
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# =========================
# Settings
# =========================

class Settings(BaseSettings):
    API_ACCESS_KEY: Optional[str] = None
    DB_PATH: str = "./data/memory.sqlite"
    FILE_STORAGE_DIR: str = "./data/files"
    CORS_ALLOW_ORIGINS: Optional[str] = "*"
    APP_VERSION: str = "3.1.0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
os.makedirs(settings.FILE_STORAGE_DIR, exist_ok=True)

# =========================
# Database (SQLite)
# =========================

Base = declarative_base()
engine: Engine = create_engine(
    f"sqlite:///{settings.DB_PATH}",
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class MemoryORM(Base):
    __tablename__ = "memories"
    id = Column(String(64), primary_key=True)
    user_id = Column(String(128), index=True, nullable=False)
    role = Column(String(128), index=True, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(Text, nullable=True)  # JSON list
    created_at = Column(DateTime, nullable=False, index=True)

class FileORM(Base):
    __tablename__ = "files"
    id = Column(String(64), primary_key=True)
    user_id = Column(String(128), index=True, nullable=False)
    filename = Column(String(512), nullable=False)
    stored_path = Column(String(1024), nullable=False)
    size = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, index=True)
    sha256 = Column(String(64), nullable=False, index=True)
    mime = Column(String(128), nullable=True)

Base.metadata.create_all(engine)

# =========================
# Models
# =========================

class Healthz(BaseModel):
    ok: bool = True
    version: str
    bm25: bool = True
    embeddings: str = "openai"

class MemoryItem(BaseModel):
    user_id: str
    role: str
    content: str
    tags: Optional[List[str]] = None

class SaveMemoryResponse(BaseModel):
    id: str
    stored: bool

class ScoreComponents(BaseModel):
    lexical: float = 0.0
    semantic: float = 0.0
    feedback: float = 0.0

class QueryDoc(BaseModel):
    id: str
    user_id: str
    role: str
    content: str
    tags: Optional[List[str]] = None
    created_at: str

class QueryMatch(BaseModel):
    doc: QueryDoc
    score: float
    score_components: ScoreComponents
    highlight: Optional[str] = None

class QueryItem(BaseModel):
    user_id: str
    role: str
    query: str
    top_k: int = Field(3, ge=1, le=50)
    mode: str = Field("lexical", pattern="^(lexical|semantic|hybrid)$")

class QueryResponse(BaseModel):
    matches: List[QueryMatch] = []

class IngestUrlRequest(BaseModel):
    user_id: str
    role: str
    url: str
    tags: Optional[List[str]] = None

class IngestJobResponse(BaseModel):
    job_id: str
    status: str

class JobCounts(BaseModel):
    total: int = 0
    done: int = 0
    skipped: int = 0

class JobStatus(BaseModel):
    job_id: str
    status: str  # queued|running|done|error
    progress: float
    counts: JobCounts
    created_at: str
    error: Optional[str] = None

class FileMeta(BaseModel):
    id: str
    user_id: str
    filename: str
    stored_path: str
    size: int
    created_at: str
    sha256: str
    mime: Optional[str] = None

class FilesUploadResponse(BaseModel):
    status: str
    count: int
    files: List[FileMeta]
    saved_memories: Optional[List[str]] = None

# =========================
# Auth
# =========================

def _parse_bearer(auth_header: Optional[str]) -> Optional[str]:
    if not auth_header:
        return None
    parts = auth_header.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None

def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    required = settings.API_ACCESS_KEY
    if not required:
        return
    supplied = x_api_key or _parse_bearer(authorization)
    if not supplied:
        raise HTTPException(status_code=401, detail="Missing API credentials")
    if supplied != required:
        raise HTTPException(status_code=403, detail="Invalid API credentials")

# =========================
# App + CORS
# =========================

app = FastAPI(title="Plan C GPT Memory API", version=settings.APP_VERSION)

if settings.CORS_ALLOW_ORIGINS:
    origins = (
        ["*"] if settings.CORS_ALLOW_ORIGINS.strip() == "*"
        else [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=600,
    )

# =========================
# Helpers
# =========================

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _now_iso() -> str:
    return _utcnow().replace(microsecond=0).isoformat()

def _mask(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    if len(token) <= 6:
        return "***"
    return f"{token[:3]}...{token[-3:]}"

def _highlight_snippet(text_: str, query: str, radius: int = 60) -> str:
    try:
        m = re.search(re.escape(query), text_, re.IGNORECASE)
        if not m:
            return text_[:radius] + ("..." if len(text_) > radius else "")
        start = max(0, m.start() - radius // 2)
        end = min(len(text_), m.end() + radius // 2)
        return text_[start:end]
    except re.error:
        return text_[:radius]

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _store_upload(user_id: str, upload: UploadFile) -> Tuple[FileORM, bytes]:
    suffix = Path(upload.filename or "file.bin").suffix
    file_id = str(uuid.uuid4())
    safe_name = Path(upload.filename or "file").name
    folder = Path(settings.FILE_STORAGE_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{file_id}{suffix}"

    data = upload.file.read()
    sha = _sha256_bytes(data)

    with open(dest, "wb") as f:
        f.write(data)

    meta = FileORM(
        id=file_id,
        user_id=user_id,
        filename=safe_name,
        stored_path=str(dest),
        size=len(data),
        created_at=_utcnow(),
        sha256=sha,
        mime=upload.content_type or None,
    )
    return meta, data

# =========================
# Debug / Meta
# =========================

@app.get("/", tags=["meta"])
def root() -> Dict[str, Any]:
    return {
        "name": "Plan C GPT Memory API",
        "version": settings.APP_VERSION,
        "auth": "X-API-Key or Authorization: Bearer",
        "docs": "/docs",
    }

@app.get("/healthz", response_model=Healthz, tags=["meta"])
def healthz() -> Healthz:
    return Healthz(ok=True, version=settings.APP_VERSION, bm25=True, embeddings="openai")

@app.get("/debug/auth", tags=["debug"])
def debug_auth(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    return {
        "api_key_required": bool(settings.API_ACCESS_KEY),
        "received": {
            "x_api_key": _mask(x_api_key),
            "authorization_bearer": _mask(_parse_bearer(authorization)),
        },
    }

@app.get("/debug/key-hash", tags=["debug"])
def debug_key_hash():
    if not settings.API_ACCESS_KEY:
        return {"configured": False, "sha256": None}
    return {"configured": True, "sha256": hashlib.sha256(settings.API_ACCESS_KEY.encode()).hexdigest()}

# =========================
# Memory: save / query
# =========================

@app.post("/memory/save", response_model=SaveMemoryResponse, dependencies=[Depends(require_api_key)], tags=["memory"])
def save_memory(item: MemoryItem) -> SaveMemoryResponse:
    mem_id = str(uuid.uuid4())
    created = _utcnow()
    try:
        with SessionLocal() as s:
            s.add(
                MemoryORM(
                    id=mem_id,
                    user_id=item.user_id,
                    role=item.role,
                    content=item.content,
                    tags=json.dumps(item.tags or []),
                    created_at=created,
                )
            )
            s.commit()
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}") from e
    return SaveMemoryResponse(id=mem_id, stored=True)

@app.post("/memory/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)], tags=["memory"])
def query_memory(q: QueryItem) -> QueryResponse:
    like = f"%{q.query}%"
    rows: List[MemoryORM] = []
    try:
        with SessionLocal() as s:
            stmt = sql_text(
                """
                SELECT id, user_id, role, content, tags, created_at
                FROM memories
                WHERE user_id = :user_id
                  AND role = :role
                  AND content LIKE :like
                ORDER BY created_at DESC
                LIMIT :limit
                """
            )
            res = s.execute(
                stmt,
                {"user_id": q.user_id, "role": q.role, "like": like, "limit": int(q.top_k)},
            )
            for r in res:
                rows.append(
                    MemoryORM(
                        id=r.id,
                        user_id=r.user_id,
                        role=r.role,
                        content=r.content,
                        tags=r.tags,
                        created_at=r.created_at,
                    )
                )
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}") from e

    matches: List[QueryMatch] = []
    for r in rows:
        try:
            tags_list = json.loads(r.tags) if r.tags else None
        except Exception:
            tags_list = None
        doc = QueryDoc(
            id=r.id,
            user_id=r.user_id,
            role=r.role,
            content=r.content,
            tags=tags_list,
            created_at=r.created_at.replace(microsecond=0).isoformat(),
        )
        matches.append(
            QueryMatch(
                doc=doc,
                score=1.0,
                score_components=ScoreComponents(lexical=1.0, semantic=0.0, feedback=0.0),
                highlight=_highlight_snippet(r.content, q.query),
            )
        )
    return QueryResponse(matches=matches)

# =========================
# Files: upload / download
# =========================

@app.post(
    "/files/upload",
    response_model=FilesUploadResponse,
    dependencies=[Depends(require_api_key)],
    tags=["files"],
)
async def upload_files(
    background: BackgroundTasks,
    user_id: str = Form(...),
    role: Optional[str] = Form(None),
    save_to_memory: bool = Form(False),
    tags: Optional[str] = Form(None),  # comma-separated
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    attachment: Optional[UploadFile] = File(None),
    attachments: Optional[List[UploadFile]] = File(None),
):
    uploads: List[UploadFile] = []
    for x in (file, attachment):
        if x:
            uploads.append(x)
    for lst in (files or [], attachments or []):
        uploads.extend(lst)
    if not uploads:
        raise HTTPException(status_code=400, detail="No files provided")

    tag_list = [t.strip() for t in (tags.split(",") if tags else []) if t.strip()] or None
    saved_files: List[FileMeta] = []
    saved_memory_ids: List[str] = []

    with SessionLocal() as s:
        try:
            for up in uploads:
                meta, data = _store_upload(user_id, up)
                s.add(meta)
                s.commit()

                saved_files.append(
                    FileMeta(
                        id=meta.id,
                        user_id=meta.user_id,
                        filename=meta.filename,
                        stored_path=meta.stored_path,
                        size=meta.size,
                        created_at=meta.created_at.replace(microsecond=0).isoformat(),
                        sha256=meta.sha256,
                        mime=meta.mime,
                    )
                )

                if save_to_memory:
                    mem_id = str(uuid.uuid4())
                    content_text = data.decode("utf-8", errors="ignore")
                    s.add(
                        MemoryORM(
                            id=mem_id,
                            user_id=user_id,
                            role=role or "files",
                            content=content_text[:20000],  # guard against huge payload
                            tags=json.dumps(tag_list or ["uploaded_file", up.filename or "file"]),
                            created_at=_utcnow(),
                        )
                    )
                    s.commit()
                    saved_memory_ids.append(mem_id)
        except SQLAlchemyError as e:
            s.rollback()
            raise HTTPException(status_code=500, detail=f"DB error: {e}") from e

    return FilesUploadResponse(
        status="ok",
        count=len(saved_files),
        files=saved_files,
        saved_memories=(saved_memory_ids or None),
    )

@app.get(
    "/files/{id}",
    dependencies=[Depends(require_api_key)],
    tags=["files"],
    responses={200: {"content": {"application/octet-stream": {}}}},
)
def download_file(id: str):
    with SessionLocal() as s:
        fo = s.get(FileORM, id)
        if not fo:
            raise HTTPException(status_code=404, detail="Not found")
        path = Path(fo.stored_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Stored file missing")
        return FileResponse(path, filename=fo.filename, media_type=fo.mime or "application/octet-stream")

# =========================
# Ingest from URL + Jobs
# =========================

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = asyncio.Lock()

async def _run_ingest_job(job_id: str, file_id: str):
    try:
        async with _JOBS_LOCK:
            job = _JOBS[job_id]
            job["status"] = "running"
            job["progress"] = 0.1

        # Simulate embedding/indexing work
        for i in range(1, 6):
            await asyncio.sleep(0.4)
            async with _JOBS_LOCK:
                job = _JOBS[job_id]
                job["progress"] = min(0.1 + i * 0.15, 0.95)
                job["counts"]["done"] = i

        async with _JOBS_LOCK:
            job = _JOBS[job_id]
            job["status"] = "done"
            job["progress"] = 1.0
    except Exception as e:
        async with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job:
                job["status"] = "error"
                job["error"] = str(e)

@app.post(
    "/memory/ingest",
    response_model=IngestJobResponse,
    dependencies=[Depends(require_api_key)],
    tags=["files", "memory"],
)
async def ingest_url(body: IngestUrlRequest, background: BackgroundTasks):
    # fetch the URL, store as a file record, then kick off a background job
    try:
        async with aiohttp.ClientSession(raise_for_status=True) as sess:
            async with sess.get(body.url) as resp:
                data = await resp.read()
                mime = resp.headers.get("Content-Type")
                filename = body.url.split("/")[-1] or "downloaded"
                # wrap into UploadFile-like interface for reuse
                class _Buf:
                    def __init__(self, b: bytes): self._b = b
                    def read(self) -> bytes: return self._b
                upload_like = UploadFile(filename=filename, file=_Buf(data), content_type=mime)

        with SessionLocal() as s:
            meta, _ = _store_upload(body.user_id, upload_like)
            s.add(meta)
            s.commit()

        job_id = str(uuid.uuid4())
        async with _JOBS_LOCK:
            _JOBS[job_id] = {
                "job_id": job_id,
                "file_id": meta.id,
                "status": "queued",
                "progress": 0.0,
                "counts": {"total": 5, "done": 0, "skipped": 0},
                "created_at": _now_iso(),
                "error": None,
            }
        background.add_task(_run_ingest_job, job_id, meta.id)
        return IngestJobResponse(job_id=job_id, status="queued")
    except aiohttp.ClientResponseError as e:
        raise HTTPException(status_code=400, detail=f"Fetch failed: {e.status} {e.message}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fetch failed: {e}")

@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
async def get_job_status(job_id: str):
    async with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Not found")
        return JobStatus(
            job_id=job["job_id"],
            status=job["status"],
            progress=float(job["progress"]),
            counts=JobCounts(**job["counts"]),
            created_at=job["created_at"],
            error=job.get("error"),
        )


