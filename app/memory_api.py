# path: app/memory_api.py
"""
Plan C GPT Memory API – complete FastAPI app (v2.6.2)

Includes
- Files: upload (multipart/JSON/URL), list, download, delete
- Memory: save, query (BM25-like), update, delete, auto-query, feedback, tag-search, self-review
- Goals: save/list/delete
- Context: build hybrid context with simple answer modes
- Storage: portable JSON indices on disk
- Auth: optional x-api-key enforcement via env var SETTINGS__API_KEY

Run:
  uvicorn app.memory_api:app --reload --port 8080

Env (examples):
  export SETTINGS__FILE_STORAGE_DIR=.storage/files
  export SETTINGS__API_KEY=your-secret  # optional; if set, requests must send x-api-key
"""
from __future__ import annotations

import base64
import json
import math
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime
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
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# =========================
# Settings
# =========================
class Settings(BaseSettings):
    FILE_STORAGE_DIR: str = ".storage/files"
    FILE_INDEX_PATH: str = ".storage/files_index.json"
    MEMORY_INDEX_PATH: str = ".storage/memory.json"
    GOALS_INDEX_PATH: str = ".storage/goals.json"
    FEEDBACK_INDEX_PATH: str = ".storage/feedback.json"
    MAX_FILE_SIZE_MB: int = 32
    CORS_ALLOW_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])
    API_KEY: Optional[str] = None  # if set, require matching x-api-key header


settings = Settings()

# Ensure storage paths exist
STORAGE_DIR = Path(settings.FILE_STORAGE_DIR)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

FILE_INDEX_PATH = Path(settings.FILE_INDEX_PATH)
FILE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
MEMORY_INDEX_PATH = Path(settings.MEMORY_INDEX_PATH)
GOALS_INDEX_PATH = Path(settings.GOALS_INDEX_PATH)
FEEDBACK_INDEX_PATH = Path(settings.FEEDBACK_INDEX_PATH)


# =========================
# JSON file helpers
# =========================

def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# In-memory copies (write-through on change)
_files_index: Dict[str, List[dict]] = _load_json(FILE_INDEX_PATH, {"files": []})
_memory_index: Dict[str, List[dict]] = _load_json(MEMORY_INDEX_PATH, {"memories": []})
_goals_index: Dict[str, List[dict]] = _load_json(GOALS_INDEX_PATH, {"documents": []})
_feedback_index: Dict[str, List[dict]] = _load_json(FEEDBACK_INDEX_PATH, {"feedback": []})


# =========================
# Utilities
# =========================
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\-]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _safe_filename(name: str) -> str:
    name = (name or "uploaded.bin").strip().replace(" ", "_")
    name = _SAFE_NAME_RE.sub("", name)
    return name or "uploaded.bin"


def _guard_size(data: bytes):
    if len(data) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{settings.MAX_FILE_SIZE_MB} MB)")


def _store_file_meta(meta: dict) -> dict:
    # dedupe by user_id+filename
    global _files_index
    _files_index["files"] = [m for m in _files_index.get("files", []) if not (m["user_id"] == meta["user_id"] and m["filename"] == meta["filename"])]
    _files_index["files"].append(meta)
    _save_json(FILE_INDEX_PATH, _files_index)
    return meta


def _store(user_id: str, filename: str, data: bytes) -> dict:
    _guard_size(data)
    safe = _safe_filename(filename)
    dest = STORAGE_DIR / safe
    dest.write_bytes(data)
    meta = {
        "user_id": user_id,
        "filename": safe,
        "stored_path": str(dest.resolve()),
        "size": len(data),
        "created_at": _now_iso(),
    }
    return _store_file_meta(meta)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")] 


def _bm25_like(query: str, documents: List[dict], field: str = "content", k1: float = 1.5, b: float = 0.75):
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    N = len(documents)
    doc_tokens = []
    doc_lengths = []
    df = defaultdict(int)
    for d in documents:
        toks = _tokenize(d.get(field, ""))
        doc_tokens.append(toks)
        doc_lengths.append(len(toks))
        for t in set(toks):
            df[t] += 1
    avgdl = (sum(doc_lengths) / N) if N else 0.0

    def idf(term):
        n = df.get(term, 0) + 0.5
        return math.log((N - df.get(term, 0) + 0.5) / n + 1)

    scores = []
    for i, _ in enumerate(documents):
        toks = doc_tokens[i]
        if not toks:
            scores.append((i, 0.0))
            continue
        tf = Counter(toks)
        dl = doc_lengths[i]
        s = 0.0
        for qt in q_tokens:
            if qt not in tf:
                continue
            term_freq = tf[qt]
            denom = term_freq + k1 * (1 - b + b * (dl / (avgdl or 1)))
            s += idf(qt) * (term_freq * (k1 + 1)) / (denom or 1)
        scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# =========================
# Auth (optional)
# =========================
async def _require_api_key(x_api_key: Optional[str] = Query(None, alias="x-api-key")):
    # why: allow easy local testing (no key) but secure prod if API_KEY is set
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


# =========================
# FastAPI app
# =========================
app = FastAPI(title="Plan C GPT Memory API", version="2.6.2")

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
# Schemas (request bodies)
# =========================
class UploadBase64Body(BaseModel):
    user_id: str
    filename: str
    content_base64: str


class UploadUrlBody(BaseModel):
    user_id: str
    url: str
    filename: Optional[str] = None


class MemoryItem(BaseModel):
    user_id: str
    role: str
    content: str
    tags: Optional[List[str]] = None


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
    feedback_type: str  # "positive" | "negative"
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
    mode: str = "general"  # general|summarise|draft|review|brainstorm


# =========================
# File endpoints
# =========================
_DEF_KEYS = ("file", "files", "attachment", "attachments")


@app.post("/files/upload")
async def upload_files_multipart(
    request: Request,
    user_id: str = Query(..., description="Uploader ID"),
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    attachment: Optional[UploadFile] = File(None),
    attachments: Optional[List[UploadFile]] = File(None),
    _=Depends(_require_api_key),
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
        name = up.filename or "uploaded.bin"
        data = await up.read()
        results.append(_store(user_id, name, data))

    return {"status": "ok", "count": len(results), "files": results}


@app.post("/files/upload-json")
async def upload_file_json(body: UploadBase64Body = Body(...), _=Depends(_require_api_key)):
    try:
        raw = base64.b64decode(body.content_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}")
    out = _store(body.user_id, body.filename, raw)
    return {"status": "ok", "files": [out]}


@app.post("/files/upload-url")
async def upload_file_url(body: UploadUrlBody = Body(...), _=Depends(_require_api_key)):
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(str(body.url))
            resp.raise_for_status()
            data = resp.content
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}")

    name = body.filename or Path(str(body.url)).name or "downloaded.bin"
    out = _store(body.user_id, name, data)
    return {"status": "ok", "files": [out]}


@app.get("/files/list")
async def list_files(user_id: Optional[str] = Query(None), _=Depends(_require_api_key)):
    files = _files_index.get("files", [])
    if user_id:
        files = [m for m in files if m["user_id"] == user_id]
    return {"files": sorted(files, key=lambda m: (m["user_id"], m["filename"]))}


@app.get("/files/download/{filename}")
async def download_file(filename: str, _=Depends(_require_api_key)):
    safe = _safe_filename(filename)
    path = STORAGE_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=safe)


@app.delete("/files/delete/{filename}")
async def delete_file(filename: str, user_id: Optional[str] = Query(None), _=Depends(_require_api_key)):
    safe = _safe_filename(filename)
    path = STORAGE_DIR / safe

    global _files_index
    before = len(_files_index.get("files", []))
    _files_index["files"] = [m for m in _files_index.get("files", []) if not (m["filename"] == safe and (user_id is None or m["user_id"] == user_id))]
    after = len(_files_index.get("files", []))
    _save_json(FILE_INDEX_PATH, _files_index)

    if path.exists():
        try:
            path.unlink()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to remove file: {exc}")

    return {"status": "ok", "removed_from_index": before - after, "file_deleted": True}


# =========================
# Memory endpoints
# =========================
@app.post("/memory/save")
async def save_memory(item: MemoryItem, _=Depends(_require_api_key)):
    mem = {
        "memory_id": uuid.uuid4().hex,
        "user_id": item.user_id,
        "role": item.role,
        "content": item.content,
        "tags": item.tags or [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    _memory_index.setdefault("memories", []).append(mem)
    _save_json(MEMORY_INDEX_PATH, _memory_index)
    return {"status": "ok", "memory_id": mem["memory_id"]}


@app.post("/memory/query")
async def query_memory(q: QueryItem, _=Depends(_require_api_key)):
    docs = [m for m in _memory_index.get("memories", []) if m.get("role") == q.role and (m.get("user_id") == q.user_id or m.get("user_id") == "shared")]
    ranked = _bm25_like(q.query, docs)
    top = [docs[i] for i, s in ranked[: max(q.top_k, 1)]]
    return {"matches": {str(i): d for i, d in enumerate(top)}}


@app.post("/memory/update")
async def update_memory(up: UpdateItem, _=Depends(_require_api_key)):
    for m in _memory_index.get("memories", []):
        if m["memory_id"] == up.memory_id and m["role"] == up.role:
            m["content"] = up.new_content
            m["updated_at"] = _now_iso()
            _save_json(MEMORY_INDEX_PATH, _memory_index)
            return {"status": "ok"}
    raise HTTPException(status_code=404, detail="memory not found")


@app.delete("/memory/delete/{role}/{memory_id}")
async def delete_memory(role: str, memory_id: str, _=Depends(_require_api_key)):
    before = len(_memory_index.get("memories", []))
    _memory_index["memories"] = [m for m in _memory_index.get("memories", []) if not (m["memory_id"] == memory_id and m["role"] == role)]
    after = len(_memory_index.get("memories", []))
    _save_json(MEMORY_INDEX_PATH, _memory_index)
    if before == after:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"status": "ok"}


@app.post("/memory/auto-query")
async def auto_query_memory(q: QueryItem, _=Depends(_require_api_key)):
    return await query_memory(q)


@app.post("/memory/feedback")
async def feedback_memory(item: FeedbackItem, _=Depends(_require_api_key)):
    if item.feedback_type not in {"positive", "negative"}:
        raise HTTPException(status_code=400, detail="feedback_type must be 'positive' or 'negative'")
    fb = item.model_dump()
    fb["created_at"] = _now_iso()
    _feedback_index.setdefault("feedback", []).append(fb)
    _save_json(FEEDBACK_INDEX_PATH, _feedback_index)
    return {"status": "ok"}


@app.post("/memory/tag-search")
async def tag_search_memory(ts: TagSearchItem, _=Depends(_require_api_key)):
    tags_lc = {t.lower() for t in ts.tags}
    docs = [m for m in _memory_index.get("memories", []) if m.get("role") == ts.role and any(t.lower() in tags_lc for t in (m.get("tags") or []))]
    def score(m):  # prefer tag overlap, then recency
        overlap = len(tags_lc.intersection({t.lower() for t in (m.get("tags") or [])}))
        return (overlap, m.get("updated_at", m.get("created_at", "")))
    docs.sort(key=score, reverse=True)
    top = docs[: max(ts.top_k, 1)]
    return {"matches": {str(i): d for i, d in enumerate(top)}}


@app.get("/memory/self-review")
async def self_review_memory(_=Depends(_require_api_key)):
    fbs = _feedback_index.get("feedback", [])
    pos = sum(1 for f in fbs if f.get("feedback_type") == "positive")
    neg = sum(1 for f in fbs if f.get("feedback_type") == "negative")
    return {"feedback_summary": {"count": len(fbs), "positive": pos, "negative": neg}}


# =========================
# Goals endpoints
# =========================
@app.post("/memory/goals")
async def save_goal(g: GoalItem, _=Depends(_require_api_key)):
    goal = {"goal_id": uuid.uuid4().hex, "user_id": g.user_id, "content": g.content, "created_at": _now_iso()}
    _goals_index.setdefault("documents", []).append(goal)
    _save_json(GOALS_INDEX_PATH, _goals_index)
    return {"status": "ok"}


@app.get("/memory/goals")
async def get_goals(_=Depends(_require_api_key)):
    return {"documents": [g["content"] for g in _goals_index.get("documents", [])]}


@app.delete("/memory/goals/{goal_id}")
async def delete_goal(goal_id: str, _=Depends(_require_api_key)):
    before = len(_goals_index.get("documents", []))
    _goals_index["documents"] = [g for g in _goals_index.get("documents", []) if g.get("goal_id") != goal_id]
    after = len(_goals_index.get("documents", []))
    _save_json(GOALS_INDEX_PATH, _goals_index)
    if before == after:
        raise HTTPException(status_code=404, detail="goal not found")
    return {"status": "ok"}


# =========================
# Context builder
# =========================
@app.post("/context/build")
async def build_context(req: ContextBuildRequest, _=Depends(_require_api_key)):
    docs = [m for m in _memory_index.get("memories", []) if m.get("role") == req.role and (m.get("user_id") == req.user_id or m.get("user_id") == "shared")]
    ranked = _bm25_like(req.query, docs)
    top_docs = [docs[i] for i, _ in ranked[: max(req.top_k, 1)]]

    if req.mode == "summarise":
        # keep paragraphs separated for readability
        answer = "\n\n".join(d.get("content", "") for d in top_docs)
    elif req.mode in {"draft", "review", "brainstorm"}:
        answer = f"Mode: {req.mode}. Context count: {len(top_docs)}. Query: {req.query}"
    else:
        answer = (
            f"Context results: {len(top_docs)}. "
            f"Top match: {(top_docs[0]['content'][:160] + '...') if top_docs else ''}"
        )

    return {
        "answer": answer,
        "context_used": top_docs,
        "tokens": sum(len(_tokenize(d.get("content", ""))) for d in top_docs),
        "mode": req.mode,
    }

