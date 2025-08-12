# path: app/memory_api.py
from __future__ import annotations

from fastapi import (
    FastAPI, File, UploadFile, HTTPException, Query, Body, Depends, Header,
    Form
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any, Tuple
from sqlmodel import SQLModel, Field, Session, create_engine, select
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import base64
import uuid
import aiohttp
import asyncio
import hashlib
import mimetypes
import json
import math

# For PDF and doc processing (optional, best-effort imports)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx  # python-docx
except Exception:
    docx = None

# Lightweight text splitter
try:
    import tiktoken
except Exception:
    tiktoken = None

# In-memory job tracker for async ingestion
_JOBS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Plan C GPT Memory API", version="3.0.0")
# Optional CORS (tighten allow_origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================ Settings ============================
class Settings(BaseSettings):
    API_ACCESS_KEY: Optional[str] = os.getenv("API_ACCESS_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Storage
    FILE_STORAGE_DIR: str = os.getenv("FILE_STORAGE_DIR", "/tmp/memory_files")
    DB_PATH: str = os.getenv("DB_PATH", "/tmp/memory.sqlite")

    # Embeddings
    EMBED_PROVIDER: str = os.getenv("EMBED_PROVIDER", "openai")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    # Simple lexical search toggle
    BM25_ENABLED: bool = os.getenv("BM25_ENABLED", "true").lower() == "true"

    # PDF and fetch limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    URL_FETCH_TIMEOUT_S: int = int(os.getenv("URL_FETCH_TIMEOUT_S", "25"))

    # Chunking defaults
    CHUNK_TOKENS: int = int(os.getenv("CHUNK_TOKENS", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "120"))
    MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "200"))

settings = Settings()
Path(settings.FILE_STORAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(Path(settings.DB_PATH).parent).mkdir(parents=True, exist_ok=True)

# ============================ Auth ============================
from fastapi.middleware.cors import CORSMiddleware

def _parse_bearer(auth_header: Optional[str]) -> Optional[str]:
    # why: support common "Authorization: Bearer <token>" usage from connectors
    if not auth_header:
        return None
    parts = auth_header.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None

def _require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    # why: accept either X-API-Key or Authorization: Bearer
    required = settings.API_ACCESS_KEY
    if not required:
        return  # open in dev when no key configured
    supplied = x_api_key or _parse_bearer(authorization)
    if not supplied:
        raise HTTPException(status_code=401, detail="Missing API credentials")
    if supplied != required:
        raise HTTPException(status_code=403, detail="Invalid API credentials")

# ============================ DB Models ============================
class MemoryRow(SQLModel, table=True):
    mem_id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True, index=True)
    user_id: str = Field(index=True)
    role: str = Field(index=True)
    content: str
    tags_json: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z", index=True)
    source_json: Optional[str] = None  # file+chunk traceability

class FileRow(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True, index=True)
    user_id: str = Field(index=True)
    filename: str
    mime: Optional[str] = None
    size: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z", index=True)
    path: str

class EmbeddingRow(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    mem_id: str = Field(index=True)
    provider: str
    model: str
    dim: int
    vector: bytes  # float32 array bytes
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

engine = create_engine(f"sqlite:///{settings.DB_PATH}")
SQLModel.metadata.create_all(engine)

# ============================ Schemas ============================
class SaveMemoryRequest(BaseModel):
    user_id: str
    role: str
    content: str
    tags: Optional[List[str]] = []
    source: Optional[Dict[str, Any]] = None

class SaveMemoryResponse(BaseModel):
    id: str
    stored: bool

class QueryMemoryRequest(BaseModel):
    user_id: str
    role: str
    query: str
    top_k: int = 5
    mode: str = "hybrid"  # "lexical" | "vector" | "hybrid"

class QueryHit(BaseModel):
    id: str
    content: str
    score: float
    tags: List[str] = []

class QueryResponse(BaseModel):
    answer: str
    context_used: List[Dict[str, Any]]
    tokens: int
    mode: str

class IngestUrlRequest(BaseModel):
    user_id: str
    role: str
    url: HttpUrl
    tags: Optional[List[str]] = []

class IngestJobResponse(BaseModel):
    job_id: str
    status: str

# ============================ Helpers ============================

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _safe_json(v: Any, default: str = "{}") -> str:
    try:
        return json.dumps(v or json.loads(default))
    except Exception:
        return default

def _tokenize_len(text: str) -> int:
    if not text:
        return 0
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)

# very small BM25-ish scorer
from collections import Counter

def _bm25_scores(query: str, docs: List[str]) -> List[float]:
    q_terms = [w.lower() for w in query.split() if w.strip()]
    q_counts = Counter(q_terms)
    scores = []
    avg_len = (sum(len(d.split()) for d in docs) / max(1, len(docs)))
    k1, b = 1.5, 0.75
    for d in docs:
        d_terms = d.lower().split()
        d_counts = Counter(d_terms)
        score = 0.0
        for t, qf in q_counts.items():
            df = sum(1 for doc in docs if t in doc.lower()) or 1
            idf = math.log((len(docs) - df + 0.5) / (df + 0.5) + 1.0)
            tf = d_counts.get(t, 0)
            denom = tf + k1 * (1 - b + b * (len(d_terms) / (avg_len or 1)))
            score += idf * ((tf * (k1 + 1)) / (denom or 1))
        scores.append(score)
    return scores

# embeddings
async def _openai_embed(texts: List[str], model: str) -> Tuple[List[List[float]], int]:
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": texts}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status != 200:
                txt = await r.text()
                raise HTTPException(status_code=502, detail=f"Embedding error {r.status}: {txt[:200]}")
            data = await r.json()
    vecs = [d["embedding"] for d in data.get("data", [])]
    dims = len(vecs[0]) if vecs else 0
    return vecs, dims

def _to_bytes_f32(vec: List[float]) -> bytes:
    import array
    a = array.array("f", vec)
    return a.tobytes()

# ============================ Routes: root & health ============================
@app.get("/")
def root_info():
    return {
        "name": "Plan C GPT Memory API",
        "version": "3.0.0",
        "health": "/healthz",
        "auth": {
            "required": bool(settings.API_ACCESS_KEY),
            "header_options": ["X-API-Key: <key>", "Authorization: Bearer <key>"],
        },
        "endpoints": [
            "POST /memory/save",
            "POST /memory/query",
            "POST /memory/ingest",
            "GET  /files/{id}",
            "GET  /jobs/{job_id}",
        ],
        "docs": "/docs"
    }

@app.get("/debug/auth")
def debug_auth(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    # why: help troubleshoot header formats without leaking the secret
    def mask(v):
        if not v:
            return None
        return v[:4] + "..." + v[-4:] if len(v) >= 8 else "***"
    bearer = _parse_bearer(authorization)
    return {
        "api_key_required": bool(settings.API_ACCESS_KEY),
        "received": {
            "x_api_key": mask(x_api_key),
            "authorization_bearer": mask(bearer),
        }
    }

@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "version": app.version,
        "bm25": settings.BM25_ENABLED,
        "embeddings": settings.EMBED_PROVIDER,
    }

# ============================ Files ============================
@app.get("/files/{file_id}")
async def get_file(file_id: str, _=Depends(_require_api_key)):
    with Session(engine) as s:
        row = s.get(FileRow, file_id)
        if not row:
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(path=row.path, filename=row.filename, media_type=row.mime or "application/octet-stream")

# ============================ Memory Save ============================
@app.post("/memory/save", response_model=SaveMemoryResponse)
async def memory_save(req: SaveMemoryRequest, _=Depends(_require_api_key)):
    if not req.content:
        raise HTTPException(status_code=400, detail="content required")
    with Session(engine) as s:
        row = MemoryRow(
            user_id=req.user_id,
            role=req.role,
            content=req.content,
            tags_json=json.dumps(req.tags or []),
            source_json=_safe_json(req.source, "{}"),
        )
        s.add(row)
        s.commit()
        s.refresh(row)
    return {"id": row.mem_id, "stored": True}

# ============================ Memory Query ============================
@app.post("/memory/query", response_model=QueryResponse)
async def memory_query(req: QueryMemoryRequest, _=Depends(_require_api_key)):
    # Fetch all docs for user/role
    with Session(engine) as s:
        docs = list(s.exec(select(MemoryRow).where((MemoryRow.user_id == req.user_id) & (MemoryRow.role == req.role))).all())

    if not docs:
        return {"answer": "", "context_used": [], "tokens": 0, "mode": req.mode}

    contents = [d.content for d in docs]

    # lexical scores
    lex_scores = _bm25_scores(req.query, contents) if settings.BM25_ENABLED else [0.0] * len(contents)

    # vector scores (optional)
    vec_scores = [0.0] * len(contents)
    dims = 0
    if req.mode in ("vector", "hybrid") and settings.OPENAI_API_KEY:
        try:
            q_vecs, dims = await _openai_embed([req.query], settings.EMBED_MODEL)
            d_vecs, _ = await _openai_embed(contents, settings.EMBED_MODEL)
            # cosine similarity
            import numpy as np
            q = np.array(q_vecs[0], dtype=np.float32)
            for i, dv in enumerate(d_vecs):
                v = np.array(dv, dtype=np.float32)
                denom = (np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
                vec_scores[i] = float(np.dot(q, v) / denom)
        except HTTPException:
            pass
        except Exception:
            pass

    # combine
    scores = []
    for i in range(len(contents)):
        if req.mode == "lexical":
            s = lex_scores[i]
        elif req.mode == "vector":
            s = vec_scores[i]
        else:
            s = 0.5 * lex_scores[i] + 0.5 * vec_scores[i]
        scores.append((s, i))

    scores.sort(reverse=True)
    top_idx = [i for _, i in scores[: req.top_k]]
    top_docs = [docs[i] for i in top_idx]

    answer = "\n\n".join(d.content for d in top_docs)
    ctx = [
        {
            "id": d.mem_id,
            "user_id": d.user_id,
            "role": d.role,
            "content": d.content,
            "tags": json.loads(d.tags_json or "[]"),
            "created_at": d.created_at,
            "source": json.loads(d.source_json or "{}"),
        }
        for d in top_docs
    ]
    return {"answer": answer, "context_used": ctx, "tokens": sum(len(d.content or "") for d in top_docs), "mode": req.mode}

# ============================ Ingest (URL & File) ============================
async def _fetch_url(url: str) -> Tuple[bytes, str]:
    timeout = aiohttp.ClientTimeout(total=settings.URL_FETCH_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.get(url) as r:
            if r.status != 200:
                raise HTTPException(status_code=502, detail=f"fetch {r.status}")
            data = await r.read()
            ctype = r.headers.get("content-type", "application/octet-stream").split(";")[0]
            return data, ctype

def _ensure_under_limit(size_bytes: int):
    if size_bytes > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"file too large (>{settings.MAX_FILE_SIZE_MB}MB)")

async def _save_file(user_id: str, filename: str, data: bytes, mime: Optional[str]) -> FileRow:
    _ensure_under_limit(len(data))
    safe_name = f"{uuid.uuid4().hex}_{filename}"
    path = str(Path(settings.FILE_STORAGE_DIR) / safe_name)
    with open(path, "wb") as f:
        f.write(data)
    row = FileRow(user_id=user_id, filename=filename, mime=mime, size=len(data), path=path)
    with Session(engine) as s:
        s.add(row)
        s.commit()
        s.refresh(row)
    return row

async def _text_from_pdf(data: bytes) -> str:
    if not fitz:
        return ""
    doc = fitz.open(stream=data, filetype="pdf")
    if doc.page_count > settings.MAX_PDF_PAGES:
        raise HTTPException(status_code=400, detail=f"pdf too long (>{settings.MAX_PDF_PAGES} pages)")
    parts = []
    for i in range(doc.page_count):
        parts.append(doc.load_page(i).get_text("text"))
    return "\n".join(parts)

async def _text_from_docx(data: bytes) -> str:
    if not docx:
        return ""
    from io import BytesIO
    d = docx.Document(BytesIO(data))
    return "\n".join(p.text for p in d.paragraphs)

async def _guess_text_from_bytes(data: bytes, mime: Optional[str], filename: str) -> str:
    if mime == "application/pdf" or filename.lower().endswith(".pdf"):
        return await _text_from_pdf(data)
    if filename.lower().endswith(".docx"):
        return await _text_from_docx(data)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

async def _split_text(text: str) -> List[str]:
    if not text:
        return []
    max_tokens = settings.CHUNK_TOKENS
    overlap = settings.CHUNK_OVERLAP
    if tiktoken:
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        chunks = []
        for i in range(0, len(toks), max(1, max_tokens - overlap)):
            chunk = toks[i : i + max_tokens]
            chunks.append(enc.decode(chunk))
        return chunks
    # Fallback by chars
    step = max_tokens * 4
    chunks = []
    for i in range(0, len(text), max(1, step - overlap * 4)):
        chunks.append(text[i : i + step])
    return chunks

@app.post("/memory/ingest", response_model=IngestJobResponse)
async def ingest_url(req: IngestUrlRequest, _=Depends(_require_api_key)):
    # fetch
    data, ctype = await _fetch_url(str(req.url))
    # save file
    filename = Path(req.url.path).name or "downloaded"
    row = await _save_file(req.user_id, filename, data, ctype)
    # extract text
    text = await _guess_text_from_bytes(data, ctype, filename)
    chunks = await _split_text(text)
    # create mems
    mem_ids: List[str] = []
    with Session(engine) as s:
        for chunk in chunks:
            mr = MemoryRow(
                user_id=req.user_id,
                role=req.role,
                content=chunk,
                tags_json=json.dumps((req.tags or []) + ["source:url", str(req.url)]),
                source_json=json.dumps({"file_id": row.id, "filename": row.filename}),
            )
            s.add(mr)
            s.commit()
            s.refresh(mr)
            mem_ids.append(mr.mem_id)

    # async job stub (for future embedding indexing)
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {"status": "queued", "created_at": datetime.utcnow().isoformat() + "Z", "counts": {"total": len(mem_ids), "done": 0, "skipped": 0}, "progress": 0.0}
    asyncio.create_task(_background_embed_job(job_id, mem_ids))
    return {"job_id": job_id, "status": "queued"}

async def _background_embed_job(job_id: str, mem_ids: List[str]):
    try:
        with Session(engine) as s:
            docs = list(s.exec(select(MemoryRow).where(MemoryRow.mem_id.in_(mem_ids))).all())
        if not docs:
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["progress"] = 1.0
            return

        if not settings.OPENAI_API_KEY:
            _JOBS[job_id]["status"] = "skipped"
            _JOBS[job_id]["progress"] = 1.0
            return

        texts = [d.content for d in docs]
        batch = 96
        done = 0
        skipped = 0
        now = datetime.utcnow().isoformat() + "Z"
        # ensure EmbeddingRow table exists
        SQLModel.metadata.create_all(engine)
        for i in range(0, len(texts), batch):
            part = docs[i : i + batch]
            vecs, dim = await _openai_embed([d.content for d in part], settings.EMBED_MODEL)
            with Session(engine) as s:
                for drow, v in zip(part, vecs):
                    s.add(EmbeddingRow(
                        mem_id=drow.mem_id,
                        provider=settings.EMBED_PROVIDER,
                        model=settings.EMBED_MODEL,
                        dim=dim,
                        vector=_to_bytes_f32(v),
                        created_at=now
                    ))
                s.commit()
            done += len(part)
            _JOBS[job_id]["counts"].update(done=done, skipped=skipped)
            _JOBS[job_id]["progress"] = (done + skipped) / max(1, len(docs))

        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["progress"] = 1.0
    except Exception as e:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"] = str(e)

@app.get("/jobs/{job_id}")
async def get_job(job_id: str, _=Depends(_require_api_key)):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

