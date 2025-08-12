# path: app/memory_api.py
from __future__ import annotations

# ==== bootstrap env early ====
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Header, Depends, Body, UploadFile, File, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import os, uuid, hashlib, mimetypes, base64, json, math, asyncio

import httpx
from sqlmodel import SQLModel, Field, Session, select, create_engine

# third-party helpers
from rank_bm25 import BM25Okapi
import tiktoken

# OpenAI embeddings (async)
try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

# ======================================================================================
# Settings
# ======================================================================================

class Settings(BaseSettings):
    # Auth: prefer API_KEY, else API_ACCESS_KEY (Render)
    API_KEY: Optional[str] = os.getenv("API_KEY") or os.getenv("API_ACCESS_KEY")

    # Storage
    STORAGE_DIR: str = os.getenv("STORAGE_DIR") or os.getenv("FILE_STORAGE_DIR", "./data/files")
    DB_PATH: str = os.getenv("DB_PATH", "./data/memory.sqlite")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")  # not used, kept for compat

    # Limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    URL_FETCH_TIMEOUT_S: int = int(os.getenv("URL_FETCH_TIMEOUT_S", "25"))

    # Chunking defaults
    CHUNK_TOKENS: int = int(os.getenv("CHUNK_TOKENS", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", os.getenv("CHUNK_OVERLAP_TOKENS", "120")))
    MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "200"))

    # Embeddings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    EMBED_PROVIDER: str = os.getenv("EMBED_PROVIDER", "openai")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL") or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Hybrid weights
    W_SEM: float = float(os.getenv("W_SEM", "0.65"))
    W_LEX: float = float(os.getenv("W_LEX", "0.35"))

settings = Settings()

# Ensure directories exist (why: avoid startup crashes on fresh envs)
for path in [Path(settings.DB_PATH).parent,
             Path(settings.STORAGE_DIR),
             Path(settings.CHROMA_PERSIST_DIR)]:
    path.mkdir(parents=True, exist_ok=True)

# ======================================================================================
# DB Models
# ======================================================================================

class MemoryRow(SQLModel, table=True):
    mem_id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    user_id: str
    role: str
    content: str
    tags_json: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat().replace("+00:00","Z"))
    source_json: Optional[str] = None  # filename, chunk index, etc.

class EmbeddingRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    mem_id: str = Field(index=True, foreign_key="memoryrow.mem_id")
    provider: str
    model: str
    dim: int
    vector: bytes  # float32 array bytes
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat().replace("+00:00","Z"))

class FeedbackRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    mem_id: str = Field(index=True)
    user_id: str
    role: str
    feedback_type: str
    feedback_text: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat().replace("+00:00","Z"))

class GoalRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str
    content: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat().replace("+00:00","Z"))

class FileRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str
    filename: str
    stored_path: str
    size: int
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat().replace("+00:00","Z"))
    sha256: str
    mime: Optional[str] = None

engine = create_engine(f"sqlite:///{settings.DB_PATH}", connect_args={"check_same_thread": False})
SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    return Session(engine)

# ======================================================================================
# Schemas (requests)
# ======================================================================================

class MemoryItem(BaseModel):
    user_id: str
    role: str
    content: str
    tags: Optional[List[str]] = None
    source: Optional[Dict[str, Any]] = None

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
    feedback_text: str = ""

class GoalItem(BaseModel):
    user_id: str
    content: str

class ContextBuildRequest(BaseModel):
    user_id: str
    role: str
    query: str
    top_k: int = 3
    ask_gpt: bool = False
    mode: str = "general"  # general|summarise|draft|review|brainstorm

class ReindexRequest(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None
    model: Optional[str] = None
    force: bool = False

# ======================================================================================
# Utils
# ======================================================================================

def _now_iso() -> str:
    return datetime.utcnow().isoformat().replace("+00:00", "Z")

def _safe_user_dir(user_id: str) -> Path:
    p = Path(settings.STORAGE_DIR) / user_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def _hash_bytes(content: bytes) -> str:
    import hashlib
    return hashlib.sha256(content).hexdigest()

def _detect_mime(filename: str) -> Optional[str]:
    m, _ = mimetypes.guess_type(filename)
    return m

def _extract_text_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".markdown", ".csv", ".json"}:
            return path.read_text(errors="ignore")
        if suffix == ".pdf":
            from pypdf import PdfReader
            r = PdfReader(str(path))
            pages = r.pages[: settings.MAX_PDF_PAGES]
            return "\n".join((p.extract_text() or "") for p in pages)
        if suffix == ".docx":
            import docx  # python-docx
            d = docx.Document(str(path))
            return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""

def _get_tokenizer():
    # why: load once on demand
    return tiktoken.get_encoding("cl100k_base")

def chunk_text_by_tokens(text: str, chunk_tokens: int, overlap: int) -> List[str]:
    if chunk_tokens <= 0:
        return [text]
    enc = _get_tokenizer()
    toks = enc.encode(text)
    if not toks:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_tokens - max(0, overlap))
    while i < len(toks):
        window = toks[i:i+chunk_tokens]
        chunks.append(enc.decode(window))
        i += step
    return chunks

# ---- vectors

def _norm(v: List[float]) -> List[float]:
    s = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/s for x in v]

def _cos(a: List[float], b: List[float]) -> float:
    return sum(x*y for x,y in zip(a,b))

def _to_bytes_f32(v: List[float]) -> bytes:
    import array
    return array.array("f", v).tobytes()

def _from_bytes_f32(b: bytes) -> List[float]:
    import array
    arr = array.array("f")
    arr.frombytes(b)
    return list(arr)

# ======================================================================================
# Auth
# ======================================================================================

def _require_api_key(x_api_key: Optional[str] = Header(None)):
    if settings.API_KEY and x_api_key != settings.API_KEY:
        # compatibility: support API_ACCESS_KEY if API_KEY was None during Settings load
        if os.getenv("API_ACCESS_KEY") and x_api_key == os.getenv("API_ACCESS_KEY"):
            return
        raise HTTPException(status_code=401, detail="Unauthorized")

# ======================================================================================
# Embedding provider
# ======================================================================================

def _embeddings_enabled() -> bool:
    return settings.EMBED_PROVIDER == "openai" and bool(settings.OPENAI_API_KEY) and AsyncOpenAI is not None

async def _embed_async(texts: List[str]) -> Tuple[List[List[float]], int, str]:
    if not _embeddings_enabled():
        raise RuntimeError("Embeddings disabled or provider missing")
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    # OpenAI returns floats already normalized for cosine in most cases, but we normalize for safety.
    resp = await client.embeddings.create(model=settings.EMBED_MODEL, input=texts)
    vecs = [ _norm(e.embedding) for e in resp.data ]
    dim = len(vecs[0]) if vecs else 0
    return vecs, dim, settings.EMBED_MODEL

# ======================================================================================
# Jobs (in-proc)
# ======================================================================================

_JOBS: Dict[str, Dict[str, Any]] = {}

# ======================================================================================
# FastAPI
# ======================================================================================

app = FastAPI(title="Plan C GPT Memory API", version="3.0.0")

# --------------------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------------------

@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "gpt_model": "gpt-5-thinking",
        "bm25": True,
        "embeddings_enabled": _embeddings_enabled(),
        "embed_provider": settings.EMBED_PROVIDER,
        "embed_model": settings.EMBED_MODEL,
        "db": settings.DB_PATH,
    }

# --------------------------------------------------------------------------------------
# Memory CRUD
# --------------------------------------------------------------------------------------

@app.post("/memory/save", dependencies=[Depends(_require_api_key)])
async def save_memory(item: MemoryItem):
    with get_session() as s:
        row = MemoryRow(
            user_id=item.user_id,
            role=item.role,
            content=item.content,
            tags_json=json.dumps(item.tags or []),
            source_json=json.dumps(item.source or None),
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return {"status": "ok", "memory_id": row.mem_id}

@app.post("/memory/update", dependencies=[Depends(_require_api_key)])
async def update_memory(item: UpdateItem):
    with get_session() as s:
        row = s.get(MemoryRow, item.memory_id)
        if not row or row.role != item.role:
            raise HTTPException(status_code=404, detail="Memory not found")
        row.content = item.new_content
        row.created_at = _now_iso()  # simple bump
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
        s.commit()
    return {"status": "ok"}

@app.post("/memory/feedback", dependencies=[Depends(_require_api_key)])
async def feedback_memory(item: FeedbackItem):
    with get_session() as s:
        s.add(FeedbackRow(
            mem_id=item.memory_id, user_id=item.user_id, role=item.role,
            feedback_type=item.feedback_type, feedback_text=item.feedback_text
        ))
        s.commit()
    return {"status": "ok"}

@app.get("/memory/self-review", dependencies=[Depends(_require_api_key)])
async def self_review():
    with get_session() as s:
        pos = s.exec(select(FeedbackRow).where(FeedbackRow.feedback_type=="positive")).all()
        neg = s.exec(select(FeedbackRow).where(FeedbackRow.feedback_type=="negative")).all()
        mem_count = s.exec(select(MemoryRow)).all()
        return {"feedback_summary": {"positive": len(pos), "negative": len(neg), "memories": len(mem_count)}}

@app.post("/memory/tag-search", dependencies=[Depends(_require_api_key)])
async def tag_search_memory(payload: Dict[str, Any] = Body(...)):
    role = payload.get("role")
    tags = set(payload.get("tags", []) or [])
    results: Dict[str, Any] = {}
    with get_session() as s:
        stmt = select(MemoryRow)
        if role:
            stmt = stmt.where(MemoryRow.role == role)
        rows = s.exec(stmt).all()
        for r in rows:
            rtags = set(json.loads(r.tags_json or "[]"))
            if tags and not tags.issubset(rtags):
                continue
            results[r.mem_id] = {
                "id": r.mem_id, "user_id": r.user_id, "role": r.role, "content": r.content,
                "tags": list(rtags), "created_at": r.created_at, "source": json.loads(r.source_json or "null")
            }
    return {"matches": results}

# --------------------------------------------------------------------------------------
# Query (lexical / semantic / hybrid)
# --------------------------------------------------------------------------------------

def _lexical_scores(rows: List[MemoryRow], query: str) -> Dict[str, float]:
    # BM25Okapi expects tokens; naive split is fine here
    corpus = [r.content.split() for r in rows]
    if not corpus:
        return {}
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.split())
    # normalize to [0,1] for nicer merging
    lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
    rng = (hi - lo) or 1.0
    return {r.mem_id: (scores[i] - lo) / rng for i, r in enumerate(rows)}

def _semantic_scores(rows: List[MemoryRow], qvec: List[float]) -> Dict[str, float]:
    if not rows:
        return {}
    # fetch vectors
    result: Dict[str, float] = {}
    with get_session() as s:
        ids = [r.mem_id for r in rows]
        vec_rows = s.exec(select(EmbeddingRow).where(EmbeddingRow.mem_id.in_(ids)).where(EmbeddingRow.model==settings.EMBED_MODEL)).all()
        vec_map = {vr.mem_id: _from_bytes_f32(vr.vector) for vr in vec_rows}
    for r in rows:
        v = vec_map.get(r.mem_id)
        if v is None:
            continue
        result[r.mem_id] = max(0.0, min(1.0, ( _cos(qvec, v) + 1.0 ) / 2.0))  # clamp to [0,1]
    return result

def _highlights(text: str, query: str) -> str:
    q = query.strip()
    if not q:
        return text[:160]
    idx = text.lower().find(q.lower())
    if idx < 0:
        return text[:160]
    start = max(0, idx - 40)
    end = min(len(text), idx + len(q) + 40)
    return text[start:end]

@app.post("/memory/query", dependencies=[Depends(_require_api_key)])
async def query_memory(item: QueryItem):
    # gather candidate docs (same role; user-owned or shared)
    with get_session() as s:
        stmt = select(MemoryRow).where(MemoryRow.role == item.role).where(
            (MemoryRow.user_id == item.user_id) | (MemoryRow.user_id == "shared")
        )
        rows: List[MemoryRow] = s.exec(stmt).all()

    if not rows:
        return {"matches": []}

    mode = item.mode.lower()
    top_k = max(1, item.top_k)

    # lexical shortlist
    lex = _lexical_scores(rows, item.query)
    # semantic
    sem: Dict[str, float] = {}
    if mode in ("semantic", "hybrid") and _embeddings_enabled():
        qvecs, dim, mdl = await _embed_async([item.query])
        sem = _semantic_scores(rows, qvecs[0])

    # scoring
    matches: List[Tuple[str, float, float, float]] = []
    for r in rows:
        lex_s = lex.get(r.mem_id, 0.0)
        sem_s = sem.get(r.mem_id, 0.0)
        if mode == "lexical":
            total = lex_s
        elif mode == "semantic":
            total = sem_s
        else:
            total = settings.W_LEX * lex_s + settings.W_SEM * sem_s
        if total > 0:
            matches.append((r.mem_id, total, lex_s, sem_s))

    matches.sort(key=lambda x: x[1], reverse=True)
    out = []
    with get_session() as s:
        for mem_id, total, lex_s, sem_s in matches[:top_k]:
            r = s.get(MemoryRow, mem_id)
            out.append({
                "doc": {
                    "id": r.mem_id, "user_id": r.user_id, "role": r.role, "content": r.content,
                    "tags": json.loads(r.tags_json or "[]"), "created_at": r.created_at,
                    "source": json.loads(r.source_json or "null"),
                },
                "score": total,
                "score_components": {
                    "lexical": lex_s, "semantic": sem_s, "feedback": 1.0  # placeholder for future learning-to-rank
                },
                "highlight": _highlights(r.content, item.query)
            })
    return {"matches": out}

# --------------------------------------------------------------------------------------
# Context build (demo)
# --------------------------------------------------------------------------------------

@app.post("/context/build", dependencies=[Depends(_require_api_key)])
async def build_context(req: ContextBuildRequest):
    with get_session() as s:
        stmt = select(MemoryRow).where(MemoryRow.role == req.role).where(
            (MemoryRow.user_id == req.user_id) | (MemoryRow.user_id == "shared")
        )
        docs = s.exec(stmt).all()
    top_docs = docs[: max(req.top_k, 1)]

    if req.mode == "summarise":
        answer = "\n\n".join(d.content for d in top_docs)
    elif req.mode in {"draft", "review", "brainstorm"}:
        answer = f"Mode: {req.mode}. Context count: {len(top_docs)}. Query: {req.query}"
    else:
        first = (top_docs[0].content[:160] + "...") if top_docs else ""
        answer = f"Context results: {len(top_docs)}. Top match: {first}"

    tokens = sum(len(d.content) for d in top_docs)
    payload_docs = [{
        "id": d.mem_id, "user_id": d.user_id, "role": d.role, "content": d.content,
        "tags": json.loads(d.tags_json or "[]"), "created_at": d.created_at,
        "source": json.loads(d.source_json or "null")
    } for d in top_docs]

    return {"answer": answer, "context_used": payload_docs, "tokens": tokens, "mode": req.mode}

# --------------------------------------------------------------------------------------
# Embedding reindex & jobs
# --------------------------------------------------------------------------------------

@app.post("/embed/reindex", dependencies=[Depends(_require_api_key)])
async def embed_reindex(req: ReindexRequest = Body(default=ReindexRequest())):
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {"job_id": job_id, "status": "queued", "progress": 0.0,
                     "counts": {"total": 0, "done": 0, "skipped": 0}, "error": None}
    asyncio.create_task(_run_reindex_job(job_id, req))
    return {"job_id": job_id}

@app.get("/jobs/{job_id}", dependencies=[Depends(_require_api_key)])
async def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    return job

async def _run_reindex_job(job_id: str, req: ReindexRequest):
    try:
        if not _embeddings_enabled():
            _JOBS[job_id].update(status="error", error="Embeddings disabled")
            return
        model = req.model or settings.EMBED_MODEL

        # load docs
        with get_session() as s:
            stmt = select(MemoryRow)
            if req.user_id:
                stmt = stmt.where(MemoryRow.user_id == req.user_id)
            if req.role:
                stmt = stmt.where(MemoryRow.role == req.role)
            docs: List[MemoryRow] = s.exec(stmt).all()

        _JOBS[job_id]["counts"]["total"] = len(docs)
        _JOBS[job_id]["status"] = "running"

        # determine which to embed
        with get_session() as s:
            if not req.force:
                # ✅ SQLModel: .exec(select(...)) already returns ScalarResult of values
                existing_ids = list(
                    s.exec(
                        select(EmbeddingRow.mem_id).where(EmbeddingRow.model == model)
                    )
                )
                have = set(existing_ids)
                to_embed = [d for d in docs if d.mem_id not in have]
                skipped = len(docs) - len(to_embed)
            else:
                # force re-embed: delete old vectors for this model
                olds = s.exec(select(EmbeddingRow).where(EmbeddingRow.model == model)).all()
                for r in olds:
                    s.delete(r)
                s.commit()
                to_embed = docs
                skipped = 0

        # embed in batches
        done = 0
        batch = 100
        for i in range(0, len(to_embed), batch):
            part = to_embed[i:i+batch]
            vecs, dim, mname = await _embed_async([d.content for d in part])
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

    except Exception as e:
        _JOBS[job_id].update(status="error", error=str(e))

# --------------------------------------------------------------------------------------
# Minimal Files API (kept for parity; ingest-on-save via chunker)
# --------------------------------------------------------------------------------------

class FilesUploadResponse(BaseModel):
    status: str
    count: int
    files: List[Dict[str, Any]]
    saved_memories: Optional[List[str]] = None

def _store_file(user_id: str, filename: str, content: bytes) -> Dict[str, Any]:
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Payload too large")
    sha = _hash_bytes(content)
    dest = _safe_user_dir(user_id) / filename
    dest.write_bytes(content)
    meta = {
        "user_id": user_id, "filename": filename, "stored_path": str(dest),
        "size": len(content), "created_at": _now_iso(), "sha256": sha,
        "mime": _detect_mime(filename),
    }
    # Optional: persist FileRow if you want listing
    with get_session() as s:
        s.add(FileRow(**meta))
        s.commit()
    return meta

@app.post("/files/upload", dependencies=[Depends(_require_api_key)], response_model=FilesUploadResponse)
async def upload_files(
    user_id: str = Query(...),
    role: Optional[str] = Query(None),
    save_to_memory: bool = Query(False),
    tags: Optional[str] = Query(None),
    chunk_tokens: Optional[int] = Query(None),
    chunk_overlap: Optional[int] = Query(None),
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
):
    bucket: List[UploadFile] = []
    if file: bucket.append(file)
    bucket.extend(files or [])
    if not bucket:
        raise HTTPException(status_code=400, detail="No files provided")

    tag_list = [t.strip() for t in (tags or "").split(",") if t and t.strip()]
    ct = chunk_tokens if chunk_tokens is not None else settings.CHUNK_TOKENS
    co = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    uploaded, saved_ids = [], []
    for f in bucket:
        content = await f.read()
        meta = _store_file(user_id, f.filename, content)
        uploaded.append(meta)
        if save_to_memory and role:
            text = _extract_text_from_path(Path(meta["stored_path"]))
            if not text.strip():
                continue
            chunks = chunk_text_by_tokens(text, ct, co) if ct > 0 else [text]
            with get_session() as s:
                for idx, ch in enumerate(chunks):
                    row = MemoryRow(
                        user_id=user_id, role=role, content=ch,
                        tags_json=json.dumps(tag_list),
                        source_json=json.dumps({"filename": f.filename, "chunk_index": idx, "total_chunks": len(chunks)})
                    )
                    s.add(row); s.commit(); s.refresh(row)
                    saved_ids.append(row.mem_id)

    return FilesUploadResponse(status="ok", count=len(uploaded), files=uploaded, saved_memories=(saved_ids or None))

# --------------------------------------------------------------------------------------
# Goals
# --------------------------------------------------------------------------------------

@app.post("/memory/goals", dependencies=[Depends(_require_api_key)])
async def save_goal(item: GoalItem):
    with get_session() as s:
        s.add(GoalRow(user_id=item.user_id, content=item.content))
        s.commit()
    return {"status": "ok"}

@app.get("/memory/goals", dependencies=[Depends(_require_api_key)])
async def get_goals():
    with get_session() as s:
        rows = s.exec(select(GoalRow)).all()
        return {"documents": [r.content for r in rows]}

@app.delete("/memory/goals/{goal_id}", dependencies=[Depends(_require_api_key)])
async def delete_goal(goal_id: int):
    with get_session() as s:
        row = s.get(GoalRow, goal_id)
        if not row:
            raise HTTPException(status_code=404, detail="Goal not found")
        s.delete(row); s.commit()
    return {"status": "ok"}
