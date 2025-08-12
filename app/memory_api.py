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
from datetime import datetime
import os
import base64
import httpx
import uuid
import hashlib
import mimetypes
import asyncio
import math
import json

# external libs used lazily: pypdf, python-docx, tiktoken
# embeddings: openai>=1.30.0
from openai import AsyncOpenAI

app = FastAPI(title="Plan C GPT Memory API", version="3.0.0")

# ============================ Settings ============================
class Settings(BaseSettings):
    API_ACCESS_KEY: Optional[str] = os.getenv("API_ACCESS_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GPT_MODEL: str = os.getenv("GPT_MODEL", "gpt-5-thinking")
    EMBED_PROVIDER: str = os.getenv("EMBED_PROVIDER", "openai")
    EMBED_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    FILE_STORAGE_DIR: str = os.getenv("FILE_STORAGE_DIR", "./data/files")
    DB_PATH: str = os.getenv("DB_PATH", "./data/memory.sqlite")

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
def _require_api_key(x_api_key: Optional[str] = Header(None)):
    # why: avoid silent exposure if API_ACCESS_KEY unset
    if settings.API_ACCESS_KEY and x_api_key != settings.API_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

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
    filename: str = Field(index=True)
    stored_path: str
    size: int
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z", index=True)
    sha256: str = Field(index=True)
    mime: Optional[str] = None

class EmbeddingRow(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True, index=True)
    mem_id: str = Field(index=True)
    provider: str = Field(index=True)
    model: str = Field(index=True)
    dim: int
    # why: pack float32 to bytes to reduce sqlite bloat
    vector: bytes
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z", index=True)

engine = create_engine(f"sqlite:///{settings.DB_PATH}", echo=False, connect_args={"check_same_thread": False})
SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    return Session(engine)

# ============================ In-memory aux stores ============================
FEEDBACK: List[Dict[str, Any]] = []
GOALS: Dict[str, str] = {}
_JOBS: Dict[str, Dict[str, Any]] = {}  # simple job status

# ============================ Models (API) ============================
class MemoryItem(BaseModel):
    user_id: str
    role: str
    content: str
    tags: Optional[List[str]] = []

class QueryItem(BaseModel):
    user_id: str
    role: str
    query: str
    top_k: int = 3
    mode: str = "lexical"  # lexical|semantic|hybrid

class UpdateItem(BaseModel):
    memory_id: str
    role: str
    new_content: str

class FeedbackItem(BaseModel):
    memory_id: str
    role: str
    user_id: str
    feedback_type: str
    feedback_text: str

class GoalItem(BaseModel):
    user_id: str
    content: str

class UploadJsonItem(BaseModel):
    filename: str
    content_base64: str

class UploadJsonBody(BaseModel):
    user_id: str
    items: List[UploadJsonItem]
    role: Optional[str] = None
    save_to_memory: bool = False
    tags: Optional[List[str]] = None
    chunk_tokens: Optional[int] = None
    chunk_overlap: Optional[int] = None

class UploadUrlItem(BaseModel):
    url: HttpUrl
    filename: Optional[str] = None

class UploadUrlBody(BaseModel):
    user_id: str
    items: List[UploadUrlItem]
    role: Optional[str] = None
    save_to_memory: bool = False
    tags: Optional[List[str]] = None
    chunk_tokens: Optional[int] = None
    chunk_overlap: Optional[int] = None

class FileMeta(BaseModel):
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

class IngestBody(BaseModel):
    user_id: str
    filenames: List[str]
    role: str
    tags: Optional[List[str]] = None
    chunk_tokens: Optional[int] = None
    chunk_overlap: Optional[int] = None

class ContextBuildRequest(BaseModel):
    user_id: str
    role: str
    query: str
    top_k: int = 3
    ask_gpt: bool = False
    mode: str = "general"

class ReindexRequest(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None
    model: Optional[str] = None
    force: bool = False

# ============================ Utils ============================
def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _detect_mime(filename: str) -> Optional[str]:
    m, _ = mimetypes.guess_type(filename)
    return m

def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def _safe_user_dir(user_id: str) -> Path:
    p = Path(settings.FILE_STORAGE_DIR) / user_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def _store_file(user_id: str, filename: str, content: bytes) -> FileMeta:
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Payload too large")
    sha = _hash_bytes(content)
    dest = _safe_user_dir(user_id) / filename
    dest.write_bytes(content)
    meta = FileMeta(
        user_id=user_id,
        filename=filename,
        stored_path=str(dest),
        size=len(content),
        created_at=_now_iso(),
        sha256=sha,
        mime=_detect_mime(filename),
    )
    with get_session() as s:
        s.add(FileRow(
            user_id=user_id, filename=filename, stored_path=str(dest),
            size=len(content), sha256=sha, mime=meta.mime
        ))
        s.commit()
    return meta

def _extract_text_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".markdown", ".csv", ".json"}:
            return path.read_text(errors="ignore")
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = reader.pages[: settings.MAX_PDF_PAGES]
            return "\n".join((p.extract_text() or "") for p in pages)
        if suffix == ".docx":
            import docx  # python-docx
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""

def _get_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")

def chunk_text_by_tokens(text: str, chunk_tokens: int, overlap: int) -> List[str]:
    if chunk_tokens <= 0:
        return [text]
    enc = _get_tokenizer()
    toks = enc.encode(text)
    if not toks:
        return []
    chunks = []
    step = max(1, chunk_tokens - max(0, overlap))
    for i in range(0, len(toks), step):
        window = toks[i : i + chunk_tokens]
        chunks.append(enc.decode(window))
    return chunks

def _save_memory(user_id: str, role: str, content: str, tags: Optional[List[str]], meta: Dict[str, Any]) -> str:
    tags_json = json.dumps(tags or [])
    source_json = json.dumps(meta or {})
    with get_session() as s:
        row = MemoryRow(user_id=user_id, role=role, content=content, tags_json=tags_json, source_json=source_json)
        s.add(row)
        s.commit()
        s.refresh(row)
        return row.mem_id

def _save_memory_from_file(user_id: str, role: str, path: Path, tags: Optional[List[str]],
                           chunk_tokens: int, chunk_overlap: int) -> List[str]:
    text = _extract_text_from_path(path)
    if not text.strip():
        return []
    chunks = chunk_text_by_tokens(text, chunk_tokens, chunk_overlap) if chunk_tokens > 0 else [text]
    mem_ids: List[str] = []
    for idx, ch in enumerate(chunks):
        meta = {"filename": path.name, "stored_path": str(path), "chunk_index": idx, "total_chunks": len(chunks)}
        mem_ids.append(_save_memory(user_id, role, ch, tags, meta))
    return mem_ids

def _parse_tags_csv(tags_csv: Optional[str], tags_list: Optional[List[str]]) -> Optional[List[str]]:
    if tags_list:
        return tags_list
    if tags_csv:
        return [t.strip() for t in tags_csv.split(",") if t.strip()]
    return None

# ============================ Embeddings helpers ============================
_client_async: Optional[AsyncOpenAI] = None

def _embeddings_enabled() -> bool:
    return settings.EMBED_PROVIDER == "openai" and bool(settings.OPENAI_API_KEY)

def _ensure_client():
    global _client_async
    if _client_async is None and settings.OPENAI_API_KEY:
        _client_async = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def _embed_async(texts: List[str]) -> Tuple[List[List[float]], int, str]:
    _ensure_client()
    if not _client_async:
        raise RuntimeError("OpenAI client not configured")
    # why: OpenAI supports batching; keep simple (single call)
    resp = await _client_async.embeddings.create(model=settings.EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    dim = len(vecs[0]) if vecs else 0
    return vecs, dim, settings.EMBED_MODEL

def _to_bytes_f32(vec: List[float]) -> bytes:
    import array
    arr = array.array("f", vec)
    return arr.tobytes()

def _from_bytes_f32(blob: bytes) -> List[float]:
    import array
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)

def _cosine(a: List[float], b: List[float]) -> float:
    # guard: mismatched dims
    if not a or not b or len(a) != len(b):
        return 0.0
    s_ab = 0.0
    s_a = 0.0
    s_b = 0.0
    for x, y in zip(a, b):
        s_ab += x * y
        s_a += x * x
        s_b += y * y
    if s_a == 0.0 or s_b == 0.0:
        return 0.0
    return s_ab / (math.sqrt(s_a) * math.sqrt(s_b))

# ============================ Health ============================
@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "gpt_model": settings.GPT_MODEL,
        "bm25": True,
        "embeddings_enabled": _embeddings_enabled(),
        "embed_provider": settings.EMBED_PROVIDER,
        "embed_model": settings.EMBED_MODEL,
        "db": settings.DB_PATH,
    }

# ============================ Memory ============================
@app.post("/memory/save", dependencies=[Depends(_require_api_key)])
async def save_memory(item: MemoryItem):
    mem_id = _save_memory(item.user_id, item.role, item.content, item.tags, meta={})
    return {"status": "ok", "memory_id": mem_id}

@app.post("/memory/query", dependencies=[Depends(_require_api_key)])
async def query_memory(item: QueryItem):
    # basic lexical (substring) + optional semantic/hybrid
    with get_session() as s:
        docs = s.exec(
            select(MemoryRow).where(
                (MemoryRow.role == item.role) & ((MemoryRow.user_id == item.user_id) | (MemoryRow.user_id == "shared"))
            )
        ).all()

    def lexical_score(text: str, needle: str) -> float:
        return 1.0 if needle in text.lower() else 0.0

    results = []
    q = item.query.lower()

    # lexical pass
    for m in docs:
        score_lex = lexical_score(m.content.lower(), q)
        if score_lex > 0 or item.mode in ("semantic", "hybrid"):
            results.append({
                "row": m,
                "lex": score_lex,
                "sem": 0.0
            })

    # semantic pass
    if item.mode in ("semantic", "hybrid") and _embeddings_enabled() and results:
        q_vecs, dim, _ = await _embed_async([item.query])
        qv = q_vecs[0]
        # load vectors for rows present
        mem_ids = [r["row"].mem_id for r in results]
        with get_session() as s:
            vec_rows = s.exec(
                select(EmbeddingRow).where((EmbeddingRow.mem_id.in_(mem_ids)) & (EmbeddingRow.model == settings.EMBED_MODEL))
            ).all()
        vec_map = {vr.mem_id: _from_bytes_f32(vr.vector) for vr in vec_rows}
        for r in results:
            mv = vec_map.get(r["row"].mem_id)
            r["sem"] = _cosine(qv, mv) if mv else 0.0

    # blend
    blended = []
    for r in results:
        # simple weighted sum; lexical dominates when exact term present
        score = 0.6 * r["lex"] + 0.4 * r["sem"] if item.mode == "hybrid" else (r["sem"] if item.mode == "semantic" else r["lex"])
        blended.append((score, r))

    blended.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, r in blended[: max(item.top_k, 1)]:
        m = r["row"]
        out.append({
            "doc": {
                "id": m.mem_id,
                "user_id": m.user_id,
                "role": m.role,
                "content": m.content,
                "tags": json.loads(m.tags_json or "[]"),
                "created_at": m.created_at,
                "source": json.loads(m.source_json or "{}"),
            },
            "score": score,
            "score_components": {"lexical": r["lex"], "semantic": r["sem"], "feedback": 1.0},
            "highlight": m.content[:240]
        })
    return {"matches": out}

@app.post("/memory/update", dependencies=[Depends(_require_api_key)])
async def update_memory(item: UpdateItem):
    with get_session() as s:
        row = s.get(MemoryRow, item.memory_id)
        if not row or row.role != item.role:
            raise HTTPException(status_code=404, detail="Memory not found")
        row.content = item.new_content
        s.add(row)
        s.commit()
    return {"status": "ok"}

@app.delete("/memory/delete/{role}/{memory_id}", dependencies=[Depends(_require_api_key)])
async def delete_memory(role: str, memory_id: str):
    with get_session() as s:
        row = s.get(MemoryRow, memory_id)
        if not row or row.role != role:
            raise HTTPException(status_code=404, detail="Memory not found")
        s.delete(row)
        # cascade: delete embeddings
        emb_rows = s.exec(select(EmbeddingRow).where(EmbeddingRow.mem_id == memory_id)).all()
        for e in emb_rows:
            s.delete(e)
        s.commit()
    return {"status": "ok"}

@app.post("/memory/auto-query", dependencies=[Depends(_require_api_key)])
async def auto_query_memory(item: QueryItem):
    keys = [k for k in item.query.lower().split() if k]
    with get_session() as s:
        docs = s.exec(
            select(MemoryRow).where(
                (MemoryRow.role.in_([item.role, "shared"])) &
                ((MemoryRow.user_id == item.user_id) | (MemoryRow.user_id == "shared"))
            )
        ).all()
    hits = []
    for m in docs:
        text = m.content.lower()
        if any(k in text for k in keys):
            hits.append(m.content)
    return {"matches": hits[: max(item.top_k, 1)]}

@app.post("/memory/feedback", dependencies=[Depends(_require_api_key)])
async def feedback_memory(item: FeedbackItem):
    FEEDBACK.append({**item.model_dump(), "created_at": _now_iso()})
    return {"status": "ok"}

@app.post("/memory/tag-search", dependencies=[Depends(_require_api_key)])
async def tag_search_memory(item: Dict[str, Any] = Body(...)):
    role = item.get("role")
    tags = item.get("tags", [])
    hits = {}
    with get_session() as s:
        docs = s.exec(select(MemoryRow)).all()
    for m in docs:
        if role and m.role != role:
            continue
        mt = set(json.loads(m.tags_json or "[]"))
        if tags and not set(tags).issubset(mt):
            continue
        hits[m.mem_id] = {
            "id": m.mem_id,
            "user_id": m.user_id,
            "role": m.role,
            "content": m.content,
            "tags": list(mt),
            "created_at": m.created_at,
            "source": json.loads(m.source_json or "{}"),
        }
    return {"matches": hits}

@app.get("/memory/self-review", dependencies=[Depends(_require_api_key)])
async def self_review():
    pos = sum(1 for f in FEEDBACK if f.get("feedback_type") == "positive")
    neg = sum(1 for f in FEEDBACK if f.get("feedback_type") == "negative")
    with get_session() as s:
        total = s.exec(select(MemoryRow)).count()
    return {"feedback_summary": {"positive": pos, "negative": neg, "memories": total}}

@app.post("/memory/goals", dependencies=[Depends(_require_api_key)])
async def save_goal(item: GoalItem):
    gid = uuid.uuid4().hex
    GOALS[gid] = item.content
    return {"status": "ok"}

@app.get("/memory/goals", dependencies=[Depends(_require_api_key)])
async def get_goals():
    return {"documents": list(GOALS.values())}

@app.delete("/memory/goals/{goal_id}", dependencies=[Depends(_require_api_key)])
async def delete_goal(goal_id: str):
    if goal_id not in GOALS:
        raise HTTPException(status_code=404, detail="Goal not found")
    del GOALS[goal_id]
    return {"status": "ok"}

# ============================ Files: multipart (tolerant) ============================
@app.post("/files/upload", response_model=FilesUploadResponse, dependencies=[Depends(_require_api_key)])
async def upload_files(
    # Query params (preferred)
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    save_to_memory: Optional[bool] = Query(None),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    chunk_tokens: Optional[int] = Query(None, ge=0),
    chunk_overlap: Optional[int] = Query(None, ge=0),
    # Fallback form fields (many clients omit query params in multipart)
    user_id_form: Optional[str] = Form(None),
    role_form: Optional[str] = Form(None),
    save_to_memory_form: Optional[bool] = Form(None),
    tags_form: Optional[str] = Form(None),
    chunk_tokens_form: Optional[int] = Form(None),
    chunk_overlap_form: Optional[int] = Form(None),
    # Files under common keys
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    attachment: Optional[UploadFile] = File(None),
    attachments: Optional[List[UploadFile]] = File(None),
    file_multi: Optional[List[UploadFile]] = File(None, alias="file"),
    files_brackets: Optional[List[UploadFile]] = File(None, alias="files[]"),
):
    # resolve params from form if query missing
    user_id = user_id or user_id_form
    role = role or role_form
    if save_to_memory is None:
        save_to_memory = save_to_memory_form if save_to_memory_form is not None else False
    tags = tags if tags is not None else tags_form
    if chunk_tokens is None:
        chunk_tokens = chunk_tokens_form
    if chunk_overlap is None:
        chunk_overlap = chunk_overlap_form

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    tag_list = _parse_tags_csv(tags, None)
    ct = chunk_tokens if chunk_tokens is not None else settings.CHUNK_TOKENS
    co = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    # gather files from any supported fields
    bucket: List[UploadFile] = []
    for f in (file, attachment):
        if f:
            bucket.append(f)
    for group in (files or []), (attachments or []), (file_multi or []), (files_brackets or []):
        bucket.extend(group or [])
    if not bucket:
        raise HTTPException(status_code=400, detail="No files provided")

    uploaded: List[FileMeta] = []
    saved_ids: List[str] = []

    for f in bucket:
        content = await f.read()
        meta = _store_file(user_id, f.filename, content)
        uploaded.append(meta)

        if save_to_memory and role:
            saved_ids.extend(
                _save_memory_from_file(
                    user_id=user_id,
                    role=role,
                    path=Path(meta.stored_path),
                    tags=tag_list,
                    chunk_tokens=ct,
                    chunk_overlap=co,
                )
            )

    return FilesUploadResponse(status="ok", count=len(uploaded), files=uploaded, saved_memories=(saved_ids or None))

# ============================ Files: JSON base64 ============================
@app.post("/files/upload-json", response_model=FilesUploadResponse, dependencies=[Depends(_require_api_key)])
async def upload_file_json(body: UploadJsonBody):
    ct = body.chunk_tokens if body.chunk_tokens is not None else settings.CHUNK_TOKENS
    co = body.chunk_overlap if body.chunk_overlap is not None else settings.CHUNK_OVERLAP

    uploaded: List[FileMeta] = []
    saved_ids: List[str] = []

    for item in body.items:
        try:
            content = base64.b64decode(item.content_base64)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid base64 for {item.filename}")
        meta = _store_file(body.user_id, item.filename, content)
        uploaded.append(meta)
        if body.save_to_memory and body.role:
            saved_ids.extend(
                _save_memory_from_file(
                    user_id=body.user_id,
                    role=body.role,
                    path=Path(meta.stored_path),
                    tags=body.tags,
                    chunk_tokens=ct,
                    chunk_overlap=co,
                )
            )
    return FilesUploadResponse(status="ok", count=len(uploaded), files=uploaded, saved_memories=(saved_ids or None))

# ============================ Files: URL fetch ============================
@app.post("/files/upload-url", response_model=FilesUploadResponse, dependencies=[Depends(_require_api_key)])
async def upload_file_url(body: UploadUrlBody):
    ct = body.chunk_tokens if body.chunk_tokens is not None else settings.CHUNK_TOKENS
    co = body.chunk_overlap if body.chunk_overlap is not None else settings.CHUNK_OVERLAP

    uploaded: List[FileMeta] = []
    saved_ids: List[str] = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=settings.URL_FETCH_TIMEOUT_S) as client:
        for item in body.items:
            r = await client.get(str(item.url))
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Fetch failed: {r.status_code} {item.url}")
            fname = item.filename or (Path(str(item.url)).name or f"file_{uuid.uuid4().hex}")
            meta = _store_file(body.user_id, fname, r.content)
            uploaded.append(meta)
            if body.save_to_memory and body.role:
                saved_ids.extend(
                    _save_memory_from_file(
                        user_id=body.user_id,
                        role=body.role,
                        path=Path(meta.stored_path),
                        tags=body.tags,
                        chunk_tokens=ct,
                        chunk_overlap=co,
                    )
                )
    return FilesUploadResponse(status="ok", count=len(uploaded), files=uploaded, saved_memories=(saved_ids or None))

# ============================ Files: list/download/delete ============================
@app.get("/files/list", dependencies=[Depends(_require_api_key)])
async def list_files(user_id: Optional[str] = Query(None)):
    with get_session() as s:
        q = select(FileRow)
        if user_id:
            q = q.where(FileRow.user_id == user_id)
        rows = s.exec(q).all()
    return {"files": [FileMeta(user_id=r.user_id, filename=r.filename, stored_path=r.stored_path,
                               size=r.size, created_at=r.created_at, sha256=r.sha256, mime=r.mime).model_dump()
                      for r in rows]}

def _resolve_file_key(filename: str, user_id: Optional[str]) -> Optional[FileRow]:
    with get_session() as s:
        if user_id:
            r = s.exec(select(FileRow).where((FileRow.user_id == user_id) & (FileRow.filename == filename))).first()
            if r:
                return r
        # fallback: any user
        r = s.exec(select(FileRow).where(FileRow.filename == filename)).first()
        return r

@app.get("/files/download/{filename}", dependencies=[Depends(_require_api_key)])
async def download_file(filename: str, user_id: Optional[str] = Query(None)):
    row = _resolve_file_key(filename, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(row.stored_path, filename=filename)

@app.delete("/files/delete/{filename}", dependencies=[Depends(_require_api_key)])
async def delete_file(filename: str, user_id: Optional[str] = Query(None)):
    row = _resolve_file_key(filename, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    try:
        Path(row.stored_path).unlink(missing_ok=True)
    except Exception:
        pass
    with get_session() as s:
        s.delete(row)
        s.commit()
    return {"status": "ok"}

# ============================ Files: Ingest existing ============================
@app.post("/files/ingest", dependencies=[Depends(_require_api_key)])
async def ingest_files(body: IngestBody):
    ct = body.chunk_tokens if body.chunk_tokens is not None else settings.CHUNK_TOKENS
    co = body.chunk_overlap if body.chunk_overlap is not None else settings.CHUNK_OVERLAP

    saved_ids: List[str] = []
    with get_session() as s:
        for fname in body.filenames:
            row = s.exec(select(FileRow).where((FileRow.filename == fname) & (FileRow.user_id == body.user_id))).first()
            if not row:
                raise HTTPException(status_code=404, detail=f"File not found: {fname}")
            saved_ids.extend(
                _save_memory_from_file(
                    user_id=body.user_id,
                    role=body.role,
                    path=Path(row.stored_path),
                    tags=body.tags,
                    chunk_tokens=ct,
                    chunk_overlap=co,
                )
            )
    return {"status": "ok", "saved_memories": saved_ids, "count": len(saved_ids)}

# ============================ Embeddings reindex (async job) ============================
@app.post("/embed/reindex", dependencies=[Depends(_require_api_key)])
async def reindex_embeddings(req: ReindexRequest = Body(default=ReindexRequest())):
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "counts": {"total": 0, "done": 0, "skipped": 0},
        "error": None,
    }
    asyncio.create_task(_run_reindex_job(job_id, req))
    return {"job_id": job_id}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    j = _JOBS.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Not found")
    return j

async def _run_reindex_job(job_id: str, req: ReindexRequest):
    try:
        if not _embeddings_enabled():
            _JOBS[job_id].update(status="error", error="Embeddings disabled")
            return
        model = req.model or settings.EMBED_MODEL

        # load candidate docs
        with get_session() as s:
            stmt = select(MemoryRow)
            if req.user_id:
                stmt = stmt.where(MemoryRow.user_id == req.user_id)
            if req.role:
                stmt = stmt.where(MemoryRow.role == req.role)
            docs: List[MemoryRow] = s.exec(stmt).all()

        _JOBS[job_id]["counts"]["total"] = len(docs)
        _JOBS[job_id]["status"] = "running"

        # which to embed
        with get_session() as s:
            existing_ids = list(
                s.exec(select(EmbeddingRow.mem_id).where(EmbeddingRow.model == model)).all()
            )
        have = set(existing_ids) if existing_ids else set()
        to_embed = docs if req.force else [d for d in docs if d.mem_id not in have]
        skipped = 0 if req.force else (len(docs) - len(to_embed))

        # force: delete old vectors for this model
        if req.force:
            with get_session() as s:
                olds = s.exec(select(EmbeddingRow).where(EmbeddingRow.model == model)).all()
                for r in olds:
                    s.delete(r)
                s.commit()

        # embed in batches
        done = 0
        batch = 100
        for i in range(0, len(to_embed), batch):
            part = to_embed[i : i + batch]
            texts = [d.content for d in part]
            if not texts:
                continue
            vecs, dim, mname = await _embed_async(texts)
            now = _now_iso()
            with get_session() as s:
                for drow, v in zip(part, vecs):
                    s.add(EmbeddingRow(
                        mem_id=drow.mem_id,
                        provider=settings.EMBED_PROVIDER,
                        model=mname,
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

# ============================ Context build ============================
@app.post("/context/build", dependencies=[Depends(_require_api_key)])
async def build_context(req: ContextBuildRequest):
    with get_session() as s:
        docs = s.exec(
            select(MemoryRow).where(
                (MemoryRow.role == req.role) &
                ((MemoryRow.user_id == req.user_id) | (MemoryRow.user_id == "shared"))
            )
        ).all()
    top_docs = docs[: max(req.top_k, 1)]
    if req.mode == "summarise":
        answer = "\n\n".join((d.content or "") for d in top_docs)
    elif req.mode in {"draft", "review", "brainstorm"}:
        answer = f"Mode: {req.mode}. Context count: {len(top_docs)}. Query: {req.query}"
    else:
        answer = f"Context results: {len(top_docs)}. Top match: {(top_docs[0].content[:160] + '...') if top_docs else ''}"
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

