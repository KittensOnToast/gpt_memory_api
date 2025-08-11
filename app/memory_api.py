# path: app/memory_api.py
import base64
import hashlib
import hmac
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document
from openai import OpenAI

# External deps for upgrades
from rank_bm25 import BM25Okapi  # keyword retrieval
from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # AES-GCM encryption

# =========================
# Settings
# =========================
class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str
    GPT_MODEL: str = "gpt-5"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Storage
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    FILE_STORAGE_DIR: str = "./data/files"
    MAX_FILE_SIZE_MB: int = 20

    # Security / CORS
    API_ACCESS_KEY: Optional[str] = None
    ALLOWED_ORIGINS: str = ""  # CSV

    # Rate limits (global)
    RATE_LIMIT_DEFAULT_PER_MIN: int = 60
    RATE_LIMIT_CONTEXT_PER_MIN: int = 20

    # Roles & per-user quotas
    DEFAULT_ROLE: str = "standard"
    ADMIN_USER_IDS: str = ""  # CSV of admin IDs
    QUOTA_STANDARD_PER_MIN: int = 60
    QUOTA_ADMIN_PER_MIN: int = 600

    # Audit & Retrieval
    ENABLE_AUDIT_TRAIL: bool = True
    ENABLE_BM25: bool = True
    CONTEXT_TOKEN_BUDGET: int = 3000
    CHUNK_TOKENS: int = 500
    CHUNK_OVERLAP_TOKENS: int = 50

    # Optional crypto / presigned downloads
    CRYPTO_ENC_KEY: Optional[str] = None  # base64 32 bytes
    DOWNLOAD_TOKEN_SECRET: Optional[str] = None  # HMAC secret

    class Config:
        env_file = ".env"

settings = Settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# =========================
# Tokenizer (tiktoken)
# =========================
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None  # fallback to naive split

def count_tokens(text: str) -> int:
    if not _enc:
        return max(1, len(text.split()) // 0.75)  # crude fallback
    return len(_enc.encode(text))

def token_chunks(text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    # Why: token-aware chunks reduce retrieval drift vs naive words
    if not _enc:
        words = text.split()
        step = max(1, chunk_tokens - overlap_tokens)
        return [" ".join(words[i:i+chunk_tokens]) for i in range(0, len(words), step)]
    toks = _enc.encode(text)
    chunks = []
    step = max(1, chunk_tokens - overlap_tokens)
    for i in range(0, len(toks), step):
        seg = toks[i:i+chunk_tokens]
        if not seg:
            break
        chunks.append(_enc.decode(seg))
        if len(seg) < chunk_tokens:
            break
    return chunks

# =========================
# Init Chroma
# =========================
embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=settings.OPENAI_API_KEY,
    model_name=settings.EMBEDDING_MODEL
)
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

collections = {
    "persona": chroma_client.get_or_create_collection("persona", embedding_function=embedder),
    "shared": chroma_client.get_or_create_collection("shared", embedding_function=embedder),
    "personal_assistant": chroma_client.get_or_create_collection("personal_assistant", embedding_function=embedder),
    "writing_review": chroma_client.get_or_create_collection("writing_review", embedding_function=embedder),
    "goals": chroma_client.get_or_create_collection("goals", embedding_function=embedder),
    "feedback": chroma_client.get_or_create_collection("feedback", embedding_function=embedder),
    "files": chroma_client.get_or_create_collection("files", embedding_function=embedder),
    "files_index": chroma_client.get_or_create_collection("files_index", embedding_function=embedder),
    "audit": chroma_client.get_or_create_collection("audit", embedding_function=embedder),
}

# =========================
# Models
# =========================
class MemoryItem(BaseModel):
    user_id: str
    role: str
    content: str
    tags: List[str] = []

class QueryItem(BaseModel):
    user_id: str
    role: str
    query: str
    top_k: int = 3

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

class TagSearchItem(BaseModel):
    role: str
    tags: List[str]
    top_k: int = 3

class GoalItem(BaseModel):
    user_id: str
    content: str

class ContextBuildRequest(BaseModel):
    user_id: str
    role: str
    query: str
    top_k: int = 3
    ask_gpt: bool = False
    mode: str = "general"  # general, summarise, draft, review, brainstorm

# =========================
# Persona
# =========================
PERSONA_TEXT = """
You are Plan C Assistant — a knowledgeable, proactive, culturally aware virtual team member for Plan C.
Mission: Empower communities through inclusive, innovative planning and strategy, turning aspirations into measurable outcomes.
Core values: Reciprocity, excellence, respect for difference, agents of change, creativity, collaboration, integrity, industriousness.
Tone: Warm, respectful, plain-English, culturally aware. In casual conversation, you may use light Aussie expressions and humour.
When tasks are formal or client-facing, maintain precise, professional, and culturally respectful language.
Always align suggestions with Plan C's mission, values, and community-first approach.
"""

if collections["persona"].count() == 0:
    collections["persona"].add(
        documents=[PERSONA_TEXT],
        metadatas=[{"created_at": datetime.utcnow().isoformat()}],
        ids=[str(uuid.uuid4())]
    )

# =========================
# App + Middleware
# =========================
app = FastAPI(title="Plan C GPT Memory API")

# CORS (allow-list)
_allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "x-api-key", "x-user-id"],
        max_age=600,
    )

# Metrics (simple in-memory counters)
_metrics = {
    "requests": 0,
    "errors_4xx": 0,
    "errors_5xx": 0,
    "rate_limit_hits": 0,
    "avg_ms": 0.0,
}

# Helpers for roles/quotas
_admin_set = set([x.strip() for x in settings.ADMIN_USER_IDS.split(",") if x.strip()])

def _user_role(user_id: Optional[str]) -> str:
    if user_id and user_id in _admin_set:
        return "admin"
    return settings.DEFAULT_ROLE

def _user_quota(role: str) -> int:
    return settings.QUOTA_ADMIN_PER_MIN if role == "admin" else settings.QUOTA_STANDARD_PER_MIN

# Log scrubber
@app.middleware("http")
async def log_scrubber(request: Request, call_next):
    t0 = time.perf_counter()
    _metrics["requests"] += 1
    try:
        response = await call_next(request)
        return response
    except HTTPException as he:
        if 400 <= he.status_code < 500:
            _metrics["errors_4xx"] += 1
        else:
            _metrics["errors_5xx"] += 1
        raise
    except Exception:
        _metrics["errors_5xx"] += 1
        raise
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _metrics["avg_ms"] = (_metrics["avg_ms"] * 0.95) + (dt_ms * 0.05)

# API key guard
@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    open_paths = {"/healthz", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    if request.url.path in open_paths or not settings.API_ACCESS_KEY:
        return await call_next(request)
    key = request.headers.get("x-api-key")
    if key != settings.API_ACCESS_KEY:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: missing or invalid x-api-key"})
    return await call_next(request)

# Rate limiting (per API key/IP + per user)
WINDOW_SECONDS = 60
_global_rates: Dict[str, Dict[str, Any]] = {}
_user_rates: Dict[str, Dict[str, Any]] = {}

def _rate_key(request: Request) -> str:
    return request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")

def _bucket(rates: Dict[str, Dict[str, Any]], key: str, path: str, limit: int) -> Optional[int]:
    now = int(time.time())
    window_start = now - (now % WINDOW_SECONDS)
    bucket = rates.setdefault(key, {}).get(path)
    if not bucket or bucket["window_start"] != window_start:
        rates.setdefault(key, {})[path] = {"window_start": window_start, "count": 0}
        bucket = rates[key][path]
    if bucket["count"] >= limit:
        return WINDOW_SECONDS - (now - bucket["window_start"])
    bucket["count"] += 1
    return None

@app.middleware("http")
async def rate_limit_mw(request: Request, call_next):
    open_paths = {"/healthz", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    if request.url.path in open_paths:
        return await call_next(request)

    # Global limit per key/IP (retain previous behavior)
    global_limit = settings.RATE_LIMIT_CONTEXT_PER_MIN if request.url.path == "/context/build" else settings.RATE_LIMIT_DEFAULT_PER_MIN
    retry_in = _bucket(_global_rates, _rate_key(request), request.url.path, max(1, global_limit))
    if retry_in is not None:
        _metrics["rate_limit_hits"] += 1
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded", "retry_in_seconds": retry_in})

    # Per-user quota by role
    user_id = request.headers.get("x-user-id")
    role = _user_role(user_id)
    user_limit = _user_quota(role)
    retry_in2 = _bucket(_user_rates, user_id or "anon", request.url.path, max(1, user_limit))
    if retry_in2 is not None:
        _metrics["rate_limit_hits"] += 1
        return JSONResponse(status_code=429, content={"detail": "User quota exceeded", "retry_in_seconds": retry_in2, "role": role})

    return await call_next(request)

# Audit trail (append-only to Chroma)
def _audit_log(request: Request, status: int, latency_ms: int):
    if not settings.ENABLE_AUDIT_TRAIL:
        return
    try:
        user_id = request.headers.get("x-user-id") or "unknown"
        aid = str(uuid.uuid4())
        collections["audit"].add(
            documents=[f"{request.method} {request.url.path} -> {status} ({latency_ms}ms)"],
            metadatas=[{
                "user_id": user_id,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_ms": latency_ms,
                "created_at": datetime.utcnow().isoformat(),
            }],
            ids=[aid],
        )
    except Exception:
        # avoid breaking requests on audit errors
        pass

@app.middleware("http")
async def audit_mw(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    _audit_log(request, resp.status_code, latency_ms)
    return resp

# =========================
# Helpers (file I/O, crypto)
# =========================
def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)

def _extract_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file_path.suffix.lower() in (".docx", ".doc"):
        doc = Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)
    elif file_path.suffix.lower() in (".txt", ".md", ".csv", ".json"):
        return file_path.read_text(encoding="utf-8", errors="ignore")
    else:
        return ""

def _encrypt_bytes(plaintext: bytes) -> Tuple[bytes, bytes]:
    # Why: AES-GCM provides confidentiality + integrity; we keep nonce separate
    if not settings.CRYPTO_ENC_KEY:
        return b"", plaintext
    key = base64.b64decode(settings.CRYPTO_ENC_KEY)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return nonce, ct

def _decrypt_bytes(nonce: bytes, ciphertext: bytes) -> bytes:
    if not settings.CRYPTO_ENC_KEY:
        return ciphertext
    key = base64.b64decode(settings.CRYPTO_ENC_KEY)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)

def _write_file_encrypted(path: Path, data: bytes):
    nonce, ct = _encrypt_bytes(data)
    with path.open("wb") as f:
        if nonce:
            f.write(b"ENC1" + nonce + ct)  # simple header + nonce + ciphertext
        else:
            f.write(data)

def _read_file_encrypted(path: Path) -> bytes:
    data = path.read_bytes()
    if data.startswith(b"ENC1"):
        nonce = data[4:16]
        ct = data[16:]
        return _decrypt_bytes(nonce, ct)
    return data

def _presign(filename: str, expires_in: int) -> Dict[str, Any]:
    if not settings.DOWNLOAD_TOKEN_SECRET:
        raise HTTPException(status_code=400, detail="Presign disabled")
    exp = int(time.time()) + max(10, min(expires_in, 3600))
    msg = f"{filename}:{exp}".encode()
    sig = hmac.new(settings.DOWNLOAD_TOKEN_SECRET.encode(), msg, hashlib.sha256).hexdigest()
    return {"filename": filename, "token": sig, "exp": exp}

def _verify_presign(filename: str, token: str, exp: int) -> bool:
    if int(time.time()) > int(exp):
        return False
    msg = f"{filename}:{exp}".encode()
    sig = hmac.new(settings.DOWNLOAD_TOKEN_SECRET.encode(), msg, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, token)

# =========================
# Endpoints: health & metrics
# =========================
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "embedding_model": settings.EMBEDDING_MODEL,
        "gpt_model": settings.GPT_MODEL,
        "bm25": settings.ENABLE_BM25,
        "persona_seeded": collections["persona"].count() > 0,
    }

@app.get("/metrics-lite")
def metrics_lite():
    return _metrics

# =========================
# Memory endpoints
# =========================
class MemoryItemId(BaseModel):
    id: str

class _BadRole(HTTPException):
    def __init__(self): super().__init__(status_code=400, detail="Invalid role")

@app.post("/memory/save")
def save_memory(item: MemoryItem):
    if item.role not in collections: raise _BadRole()
    mem_id = str(uuid.uuid4())
    collections[item.role].add(
        documents=[item.content],
        metadatas=[{
            "user_id": item.user_id,
            "tags": ",".join(item.tags),
            "created_at": datetime.utcnow().isoformat()
        }],
        ids=[mem_id]
    )
    return {"status": "success", "memory_id": mem_id}

@app.post("/memory/query")
def query_memory(item: QueryItem):
    if item.role not in collections: raise _BadRole()
    results = collections[item.role].query(
        query_texts=[item.query], n_results=item.top_k, include=["documents", "metadatas", "ids", "distances"]
    )
    return {"matches": results}

@app.post("/memory/update")
def update_memory(item: UpdateItem):
    if item.role not in collections: raise _BadRole()
    collections[item.role].update(
        ids=[item.memory_id],
        documents=[item.new_content],
        metadatas=[{"updated_at": datetime.utcnow().isoformat()}]
    )
    return {"status": "success"}

@app.delete("/memory/delete/{role}/{memory_id}")
def delete_memory(role: str, memory_id: str):
    if role not in collections: raise _BadRole()
    collections[role].delete(ids=[memory_id])
    return {"status": "success"}

@app.post("/memory/auto-query")
def auto_query(item: QueryItem):
    all_results = []
    for role_key in ["shared", item.role]:
        if role_key in collections:
            result = collections[role_key].query(
                query_texts=[item.query], n_results=item.top_k, include=["documents"]
            )
            if result and result.get("documents") and result["documents"][0]:
                all_results.extend(result["documents"][0])
    return {"matches": all_results}

@app.post("/memory/feedback")
def save_feedback(item: FeedbackItem):
    fb_id = str(uuid.uuid4())
    collections["feedback"].add(
        documents=[item.feedback_text],
        metadatas=[{
            "user_id": item.user_id,
            "memory_id": item.memory_id,
            "feedback": item.feedback_type,
            "created_at": datetime.utcnow().isoformat()
        }],
        ids=[fb_id]
    )
    return {"status": "success", "feedback_id": fb_id}

@app.post("/memory/tag-search")
def tag_search(item: TagSearchItem):
    if item.role not in collections: raise _BadRole()
    result = collections[item.role].get(where={"tags": ",".join(item.tags)})
    return {"matches": result}

@app.get("/memory/self-review")
def self_review():
    return {"feedback_summary": collections["feedback"].get()}

@app.post("/memory/goals")
def save_goal(goal: GoalItem):
    goal_id = str(uuid.uuid4())
    collections["goals"].add(
        documents=[goal.content],
        metadatas=[{"user_id": goal.user_id, "created_at": datetime.utcnow().isoformat()}],
        ids=[goal_id]
    )
    return {"status": "success", "goal_id": goal_id}

@app.get("/memory/goals")
def list_goals():
    return collections["goals"].get()

@app.delete("/memory/goals/{goal_id}")
def delete_goal(goal_id: str):
    collections["goals"].delete(ids=[goal_id])
    return {"status": "success"}

# =========================
# Files: upload/list/presign/download/delete
# =========================
@app.post("/files/upload")
def upload_file(user_id: str = Query(...), file: UploadFile = File(...)):
    if file.filename == "": raise HTTPException(status_code=400, detail="Empty filename")
    data = file.file.read()
    if len(data) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{settings.MAX_FILE_SIZE_MB} MB)")

    safe_name = _safe_filename(file.filename)
    dest_path = Path(settings.FILE_STORAGE_DIR) / safe_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_file_encrypted(dest_path, data)

    # Index content with token-aware chunking
    text = _extract_text(dest_path if not settings.CRYPTO_ENC_KEY else Path(dest_path))  # decrypted read handled in _extract_text? We wrote encrypted bytes; so extract_text must read decrypted.
    if settings.CRYPTO_ENC_KEY:
        # re-read decrypted for extraction
        try:
            text = _read_file_encrypted(dest_path).decode("utf-8", errors="ignore")
        except Exception:
            text = ""

    chunks = token_chunks(text, settings.CHUNK_TOKENS, settings.CHUNK_OVERLAP_TOKENS)
    token_counts = [count_tokens(c) for c in chunks]
    for chunk, tokc in zip(chunks, token_counts):
        collections["files_index"].add(
            documents=[chunk],
            metadatas=[{
                "user_id": user_id,
                "source_file": safe_name,
                "tokens": tokc,
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[str(uuid.uuid4())]
        )

    collections["files"].add(
        documents=[safe_name],
        metadatas=[{
            "user_id": user_id,
            "stored_path": str(dest_path),
            "encrypted": bool(settings.CRYPTO_ENC_KEY),
            "created_at": datetime.utcnow().isoformat()
        }],
        ids=[str(uuid.uuid4())]
    )
    return {"status": "success", "filename": safe_name, "chunks_indexed": len(chunks)}

@app.get("/files/list")
def list_files():
    return collections["files"].get()

@app.get("/files/presign")
def presign_download(filename: str, expires_in: int = 300):
    return _presign(filename, expires_in)

@app.get("/files/download/{filename}")
def download_file(filename: str, token: Optional[str] = None, exp: Optional[int] = None):
    safe_name = _safe_filename(filename)
    file_path = Path(settings.FILE_STORAGE_DIR) / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if settings.DOWNLOAD_TOKEN_SECRET:
        if not token or not exp or not _verify_presign(safe_name, token, int(exp)):
            raise HTTPException(status_code=403, detail="Invalid or expired download token")

    # If encrypted, decrypt to temp bytes and return as file response stream
    data = _read_file_encrypted(file_path)
    tmp = Path(settings.FILE_STORAGE_DIR) / f"._tmp_{uuid.uuid4().hex}"
    tmp.write_bytes(data)
    try:
        return FileResponse(str(tmp), filename=safe_name)
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass

@app.delete("/files/delete/{filename}")
def delete_file(filename: str):
    safe_name = _safe_filename(filename)
    file_path = Path(settings.FILE_STORAGE_DIR) / safe_name
    if file_path.exists():
        file_path.unlink()
    return {"status": "success"}
# --- New: upload by URL (server fetches) ---
import requests  # top-level import already ok if added

@app.post("/files/upload-by-url")
def upload_file_by_url(user_id: str = Query(...), url: str = Query(...), filename: Optional[str] = None):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download: {e}")

    if len(data) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{settings.MAX_FILE_SIZE_MB} MB)")

    # derive name
    safe_name = _safe_filename(filename or url.split("?")[0].split("/")[-1] or f"remote_{uuid.uuid4().hex}.bin")
    dest_path = Path(settings.FILE_STORAGE_DIR) / safe_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_file_encrypted(dest_path, data)

    # text extraction (supports encryption)
    try:
        text = _read_file_encrypted(dest_path).decode("utf-8", errors="ignore")
    except Exception:
        text = _extract_text(dest_path)

    chunks = token_chunks(text, settings.CHUNK_TOKENS, settings.CHUNK_OVERLAP_TOKENS)
    for chunk in chunks:
        collections["files_index"].add(
            documents=[chunk],
            metadatas=[{
                "user_id": user_id,
                "source_file": safe_name,
                "tokens": count_tokens(chunk),
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[str(uuid.uuid4())]
        )

    collections["files"].add(
        documents=[safe_name],
        metadatas=[{
            "user_id": user_id,
            "stored_path": str(dest_path),
            "encrypted": bool(settings.CRYPTO_ENC_KEY),
            "created_at": datetime.utcnow().isoformat()
        }],
        ids=[str(uuid.uuid4())]
    )
    return {"status": "success", "filename": safe_name, "chunks_indexed": len(chunks)}
# --- ADD: helpers for safe HTTP fetch ---
import io
try:
    import requests  # allowed in Docker build; used only for server-side fetching by URL
except Exception:
    requests = None

def _download_http(url: str, max_mb: int) -> bytes:
    if not requests:
        raise HTTPException(status_code=500, detail="requests not available on server")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = 0
        buf = io.BytesIO()
        for chunk in r.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"File too large (>{max_mb} MB)")
            buf.write(chunk)
        return buf.getvalue()

# --- ADD: models for base64 upload + save-file ---
class UploadBase64Body(BaseModel):
    user_id: str
    filename: str
    content_base64: str

class SaveFileToMemoryBody(BaseModel):
    user_id: str
    role: str
    filename: str
    tags: List[str] = []

# --- ADD: upload-by-url ---
@app.post("/files/upload-by-url")
def upload_file_by_url(user_id: str = Query(...), url: str = Query(...), filename: Optional[str] = Query(None)):
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")
    raw = _download_http(url, settings.MAX_FILE_SIZE_MB)
    name = _safe_filename(filename or url.split("/")[-1] or f"download-{uuid.uuid4().hex}")
    dest_path = Path(settings.FILE_STORAGE_DIR) / name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_file_encrypted(dest_path, raw)

    # extract + index
    text = ""
    if settings.CRYPTO_ENC_KEY:
        try:
            text = _read_file_encrypted(dest_path).decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    else:
        text = _extract_text(dest_path)

    chunks = token_chunks(text, settings.CHUNK_TOKENS, settings.CHUNK_OVERLAP_TOKENS)
    for chunk in chunks:
        collections["files_index"].add(
            documents=[chunk],
            metadatas=[{
                "user_id": user_id,
                "source_file": name,
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[str(uuid.uuid4())]
        )

    collections["files"].add(
        documents=[name],
        metadatas=[{
            "user_id": user_id,
            "stored_path": str(dest_path),
            "encrypted": bool(settings.CRYPTO_ENC_KEY),
            "created_at": datetime.utcnow().isoformat()
        }],
        ids=[str(uuid.uuid4())]
    )
    return {"status": "success", "filename": name, "chunks_indexed": len(chunks)}

# --- ADD: upload-base64 ---
@app.post("/files/upload-base64")
def upload_file_base64(body: UploadBase64Body):
    data: bytes
    try:
        data = base64.b64decode(body.content_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    if len(data) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{settings.MAX_FILE_SIZE_MB} MB)")

    safe_name = _safe_filename(body.filename or f"upload-{uuid.uuid4().hex}")
    dest_path = Path(settings.FILE_STORAGE_DIR) / safe_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_file_encrypted(dest_path, data)

    text = ""
    if settings.CRYPTO_ENC_KEY:
        try:
            text = _read_file_encrypted(dest_path).decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    else:
        text = _extract_text(dest_path)

    chunks = token_chunks(text, settings.CHUNK_TOKENS, settings.CHUNK_OVERLAP_TOKENS)
    for chunk in chunks:
        collections["files_index"].add(
            documents=[chunk],
            metadatas=[{
                "user_id": body.user_id,
                "source_file": safe_name,
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[str(uuid.uuid4())]
        )

    collections["files"].add(
        documents=[safe_name],
        metadatas=[{
            "user_id": body.user_id,
            "stored_path": str(dest_path),
            "encrypted": bool(settings.CRYPTO_ENC_KEY),
            "created_at": datetime.utcnow().isoformat()
        }],
        ids=[str(uuid.uuid4())]
    )
    return {"status": "success", "filename": safe_name, "chunks_indexed": len(chunks)}

# --- ADD: save-file to memory (uses already-uploaded file) ---
@app.post("/memory/save-file")
def save_file_to_memory(body: SaveFileToMemoryBody):
    safe_name = _safe_filename(body.filename)
    file_path = Path(settings.FILE_STORAGE_DIR) / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found. Upload first.")

    # re-extract full text (handles encryption)
    text = ""
    if settings.CRYPTO_ENC_KEY:
        try:
            text = _read_file_encrypted(file_path).decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    else:
        text = _extract_text(file_path)

    if body.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")

    mem_id = str(uuid.uuid4())
    collections[body.role].add(
        documents=[text],
        metadatas=[{
            "user_id": body.user_id,
            "source_file": safe_name,
            "tags": ",".join(body.tags or []),
            "created_at": datetime.utcnow().isoformat()
        }],
        ids=[mem_id]
    )
    return {"status": "success", "memory_id": mem_id, "saved_from": safe_name}

# --- New: upload base64 (Actions-friendly) ---
@app.post("/files/upload-base64")
def upload_file_base64(user_id: str, filename: str, content_base64: str):
    try:
        data = base64.b64decode(content_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 content")

    if len(data) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{settings.MAX_FILE_SIZE_MB} MB)")

    safe_name = _safe_filename(filename)
    dest_path = Path(settings.FILE_STORAGE_DIR) / safe_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_file_encrypted(dest_path, data)

    # extract text (handle encryption)
    try:
        text = _read_file_encrypted(dest_path).decode("utf-8", errors="ignore")
    except Exception:
        text = _extract_text(dest_path)

    chunks = token_chunks(text, settings.CHUNK_TOKENS, settings.CHUNK_OVERLAP_TOKENS)
    for chunk in chunks:
        collections["files_index"].add(
            documents=[chunk],
            metadatas=[{
                "user_id": user_id,
                "source_file": safe_name,
                "tokens": count_tokens(chunk),
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[str(uuid.uuid4())]
        )

    collections["files"].add(
        documents=[safe_name],
        metadatas=[{
            "user_id": user_id,
            "stored_path": str(dest_path),
            "encrypted": bool(settings.CRYPTO_ENC_KEY),
            "created_at": datetime.utcnow().isoformat()
        }],
        ids=[str(uuid.uuid4())]
    )
    return {"status": "success", "filename": safe_name, "chunks_indexed": len(chunks)}

# --- New: copy indexed file text into a role memory item ---
class SaveFileToMemoryReq(BaseModel):
    user_id: str
    role: str
    filename: str
    tags: List[str] = []

@app.post("/memory/save-file")
def save_file_to_memory(body: SaveFileToMemoryReq):
    if body.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")

    safe_name = _safe_filename(body.filename)
    file_path = Path(settings.FILE_STORAGE_DIR) / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # prefer decrypted read (works for encrypted/plain)
    try:
        text = _read_file_encrypted(file_path).decode("utf-8", errors="ignore")
    except Exception:
        text = _extract_text(file_path)

    # cap the memory content to a sensible token budget to keep memory lean
    max_tokens = 1500
    out, used = [], 0
    for ch in token_chunks(text, settings.CHUNK_TOKENS, settings.CHUNK_OVERLAP_TOKENS):
        t = count_tokens(ch)
        if used + t > max_tokens: break
        out.append(ch)
        used += t
    content = f"[source_file: {safe_name}]\n\n" + "\n\n".join(out) if out else f"[source_file: {safe_name}]"

    mem_id = str(uuid.uuid4())
    collections[body.role].add(
        documents=[content],
        metadatas=[{
            "user_id": body.user_id,
            "tags": ",".join(["from_file"] + list(body.tags)),
            "source_file": safe_name,
            "created_at": datetime.utcnow().isoformat()
        }],
        ids=[mem_id]
    )
    return {"status": "success", "memory_id": mem_id, "tokens_stored": used}

# =========================
# Context Builder (Hybrid + Feedback rerank + Token budget)
# =========================
def _feedback_map() -> Dict[str, float]:
    # Why: re-rank: +0.05 per positive, -0.05 per negative
    fb = collections["feedback"].get()
    score: Dict[str, float] = {}
    for md in (fb.get("metadatas") or []):
        if not md:
            continue
        for m in md:
            mid = m.get("memory_id")
            typ = m.get("feedback")
            if not mid:
                continue
            score.setdefault(mid, 0.0)
            score[mid] += 0.05 if typ == "positive" else (-0.05 if typ == "negative" else 0.0)
    return score

def _bm25_scores(query: str, docs: List[str]) -> List[float]:
    if not settings.ENABLE_BM25 or not docs:
        return [0.0] * len(docs)
    # quick whitespace tokenization is OK for BM25
    corpus = [d.split() for d in docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.split())
    # normalize 0..1
    if not scores.any():  # type: ignore
        return [0.0] * len(docs)
    mn, mx = float(min(scores)), float(max(scores))
    if mx - mn < 1e-9:
        return [0.0] * len(docs)
    return [(float(s) - mn) / (mx - mn) for s in scores]  # type: ignore

@app.post("/context/build")
def build_context(item: ContextBuildRequest):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")

    # Pull candidates via embeddings (wider net)
    def search_coll(cname: str) -> List[Dict[str, Any]]:
        if cname not in collections:
            return []
        res = collections[cname].query(
            query_texts=[item.query],
            n_results=max(item.top_k * 4, 8),
            include=["documents", "metadatas", "ids", "distances"]
        )
        docs = res.get("documents", [[]])[0] or []
        mds = res.get("metadatas", [[]])[0] or []
        ids = res.get("ids", [[]])[0] or []
        dists = res.get("distances", [[]])[0] or []
        out: List[Dict[str, Any]] = []
        for d, md, idv, dist in zip(docs, mds, ids, dists):
            if not d:
                continue
            out.append({
                "id": idv,
                "source": cname,
                "content": d,
                "metadata": md or {},
                "emb_sim": 1 - float(dist)
            })
        return out

    candidates: List[Dict[str, Any]] = []
    for src in ["persona", item.role, "shared", "goals", "files_index"]:
        candidates.extend(search_coll(src))

    # BM25 scores over candidate docs
    bm25 = _bm25_scores(item.query, [c["content"] for c in candidates])

    # Feedback re-ranking
    fb_map = _feedback_map()

    # Merge scores: 0.65*embedding + 0.35*bm25 + feedback
    for idx, c in enumerate(candidates):
        c["bm25"] = bm25[idx] if idx < len(bm25) else 0.0
        c["fb"] = fb_map.get(c.get("id") or "", 0.0)
        c["score"] = 0.65 * c["emb_sim"] + 0.35 * c["bm25"] + c["fb"]

    # Rank and take top
    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    top = ranked[:item.top_k * 2]  # take a bit more before token budgeting

    # Build context under token budget
    context_chunks = []
    used_tokens = 0
    budget = max(512, settings.CONTEXT_TOKEN_BUDGET)
    for c in top:
        t = count_tokens(c["content"])
        if used_tokens + t > budget:
            continue
        context_chunks.append(c)
        used_tokens += t
        if used_tokens >= budget:
            break

    context_text = "\n\n".join([c["content"] for c in context_chunks])

    if not item.ask_gpt:
        return {
            "context_used": [{"source": c["source"], "id": c["id"], "score": round(c["score"], 4)} for c in context_chunks],
            "context_text": context_text,
            "tokens": used_tokens,
        }

    mode_map = {
        "general": {"temp": 0.3, "prefix": ""},
        "summarise": {"temp": 0.2, "prefix": "Please summarise the following context clearly and concisely."},
        "draft": {"temp": 0.3, "prefix": "Please draft a professional update based on the following context."},
        "review": {"temp": 0.2, "prefix": "Please review the following context and highlight key terms, risks, or important details."},
        "brainstorm": {"temp": 0.8, "prefix": "Please brainstorm creative ideas based on the following context."}
    }
    mode_cfg = mode_map.get(item.mode, mode_map["general"])

    try:
        resp = client.chat.completions.create(
            model=settings.GPT_MODEL,
            messages=[
                {"role": "system", "content": PERSONA_TEXT},
                {"role": "user", "content": f"{mode_cfg['prefix']}\n\nQuery: {item.query}\n\nContext:\n{context_text}"}
            ],
            max_tokens=700,
            temperature=mode_cfg["temp"]
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling GPT: {e}")

    return {
        "answer": answer,
        "context_used": [{"source": c["source"], "id": c["id"], "score": round(c["score"], 4)} for c in context_chunks],
        "tokens": used_tokens,
        "mode": item.mode
    }

