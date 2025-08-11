# path: app/memory_api.py
from __future__ import annotations

import os
import base64
import string
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================ Settings ============================

class Settings(BaseSettings):
    """Env-driven config. Set SETTINGS__API_KEY in your env/Render."""
    model_config = SettingsConfigDict(env_prefix="SETTINGS__", case_sensitive=False)
    API_KEY: Optional[str] = None  # keep secrets out of code and git
    FILE_STORAGE_DIR: str = os.getenv("STORAGE_DIR", "/data")  # persistent on Render
    MAX_FILE_SIZE_MB: int = 20
    URL_FETCH_TIMEOUT_S: int = 20

settings = Settings()
Path(settings.FILE_STORAGE_DIR).mkdir(parents=True, exist_ok=True)

# ============================ App ============================

app = FastAPI(
    title="Plan C GPT Memory API",
    version="2.6.0",
    description=(
        "Full API (memory + files). Robust uploads (multipart/JSON/URL). "
        "Auth via x-api-key header (query fallback for local dev)."
    ),
)


# ============================ Auth ============================

def _require_api_key(
    x_api_key_header: Optional[str] = Header(None, alias="x-api-key"),
    x_api_key_query: Optional[str] = Query(None, alias="x-api-key"),
):
    # why: GPT Actions and your smoke tests send header; keep query for local curl
    provided = x_api_key_header or x_api_key_query
    if settings.API_KEY and provided != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ============================ In-memory stores ============================

_store_lock = RLock()
_memory_index: Dict[str, List[Dict[str, Any]]] = {
    "memories": [],
    "goals": [],
    "feedback": [],
}
_file_index: Dict[str, Dict[str, Any]] = {}  # key: filename -> meta


# ============================ Schemas ============================

class MemoryItem(BaseModel):
    user_id: str
    role: str
    content: str
    tags: List[str] = Field(default_factory=list)


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
    feedback_type: str = Field(pattern="^(positive|negative)$")
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
    mode: str = Field(default="general", pattern="^(general|summarise|draft|review|brainstorm)$")


class UploadJsonBody(BaseModel):
    user_id: str
    filename: str
    content_base64: str


class UploadUrlBody(BaseModel):
    user_id: str
    url: str
    filename: Optional[str] = None


# ============================ Helpers ============================

def _safe_filename(name: str) -> str:
    # why: avoid traversal & odd chars, keep extension
    s = (name or "").strip().replace("\\", "/").split("/")[-1]
    s = s.replace(" ", "_")
    s = "".join(c for c in s if c.isalnum() or c in "._-")
    return s[:200] or f"file_{uuid.uuid4().hex}"


def _tokenize(text: str) -> List[str]:
    table = str.maketrans({c: " " for c in string.punctuation})
    return [t for t in text.lower().translate(table).split() if t]


def _bm25_like(query: str, docs: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    q = set(_tokenize(query))
    scores: List[Tuple[int, float]] = []
    for i, d in enumerate(docs):
        tokens = set(_tokenize(d.get("content", "")))
        overlap = len(tokens & q)
        if overlap:
            score = overlap / (1 + abs(len(tokens) - len(q)))
            scores.append((i, float(score)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _guard_size(data: bytes) -> None:
    if len(data) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{settings.MAX_FILE_SIZE_MB} MB)")


def _store_file_bytes(user_id: str, filename: str, data: bytes) -> Dict[str, Any]:
    _guard_size(data)
    safe = _safe_filename(filename)
    user_dir = Path(settings.FILE_STORAGE_DIR) / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / safe
    dest.write_bytes(data)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta = {
        "user_id": user_id,
        "filename": safe,
        "stored_path": str(dest.resolve()),
        "size": dest.stat().st_size,
        "created_at": created,
    }
    with _store_lock:
        _file_index[safe] = meta
    return meta


# ============================ Health ============================

@app.get("/healthz")
def healthz():
    # left unauthenticated so Actions/monitors can probe readiness
    return {
        "status": "ok",
        "embedding_model": "mock-embeddings",
        "gpt_model": "gpt-5-thinking",
        "bm25": True,
        "persona_seeded": True,
    }


# ============================ Files ============================

@app.post("/files/upload", dependencies=[Depends(_require_api_key)])
async def upload_files_multipart(
    request: Request,
    user_id: str = Query(...),
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    attachment: Optional[UploadFile] = File(None),
    attachments: Optional[List[UploadFile]] = File(None),
):
    uploads: List[UploadFile] = []
    if file: uploads.append(file)
    if files: uploads.extend(files)
    if attachment: uploads.append(attachment)
    if attachments: uploads.extend(attachments)

    if not uploads:
        # sweep form parts to be resilient to name differences
        form = await request.form()
        for v in form.values():
            if isinstance(v, UploadFile):
                uploads.append(v)
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, UploadFile):
                        uploads.append(it)

    if not uploads:
        raise HTTPException(status_code=400, detail="No file(s) found in multipart form-data")

    results = []
    for up in uploads:
        data = await up.read()
        results.append(_store_file_bytes(user_id, up.filename or "uploaded.bin", data))
    return {"status": "ok", "count": len(results), "files": results}


@app.post("/files/upload-json", dependencies=[Depends(_require_api_key)])
async def upload_file_json(body: UploadJsonBody):
    try:
        raw = base64.b64decode(body.content_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}") from exc
    meta = _store_file_bytes(body.user_id, body.filename, raw)
    return {"status": "ok", "count": 1, "files": [meta]}


@app.post("/files/upload-url", dependencies=[Depends(_require_api_key)])
async def upload_file_url(body: UploadUrlBody):
    name = body.filename or _safe_filename(body.url.split("/")[-1] or f"fetch_{uuid.uuid4().hex}")
    try:
        # why: follow_redirects avoids 301/302 failures on common CDNs
        async with httpx.AsyncClient(timeout=settings.URL_FETCH_TIMEOUT_S, follow_redirects=True) as client:
            resp = await client.get(body.url)
            resp.raise_for_status()
            content = resp.content
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Fetch failed: {exc}") from exc
    meta = _store_file_bytes(body.user_id, name, content)
    return {"status": "ok", "count": 1, "files": [meta]}


@app.get("/files/list", dependencies=[Depends(_require_api_key)])
def list_files(user_id: Optional[str] = Query(None)):
    with _store_lock:
        metas = list(_file_index.values())
    if user_id:
        metas = [m for m in metas if m["user_id"] == user_id]
    return {"files": metas}


@app.get("/files/download/{filename}", dependencies=[Depends(_require_api_key)])
def download_file(filename: str):
    with _store_lock:
        meta = _file_index.get(filename)
    if not meta:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=meta["stored_path"], filename=meta["filename"])


@app.delete("/files/delete/{filename}", dependencies=[Depends(_require_api_key)])
def delete_file(filename: str, user_id: Optional[str] = Query(None)):
    with _store_lock:
        meta = _file_index.get(filename)
    if not meta:
        return {"status": "ok"}  # idempotent
    if user_id and meta["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Cannot delete other user's file")
    Path(meta["stored_path"]).unlink(missing_ok=True)
    with _store_lock:
        _file_index.pop(filename, None)
    return {"status": "ok"}


# ============================ Memory ============================

@app.post("/memory/save", dependencies=[Depends(_require_api_key)])
def save_memory(item: MemoryItem):
    doc = {
        "id": uuid.uuid4().hex,
        "user_id": item.user_id,
        "role": item.role,
        "content": item.content,
        "tags": item.tags or [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with _store_lock:
        _memory_index["memories"].append(doc)
    return {"status": "ok", "memory_id": doc["id"]}


@app.post("/memory/query", dependencies=[Depends(_require_api_key)])
def query_memory(q: QueryItem):
    with _store_lock:
        docs = [m for m in _memory_index["memories"] if m["role"] == q.role and m["user_id"] == q.user_id]
    ranked = _bm25_like(q.query, docs)
    results = [{"doc": docs[i], "score": score} for i, score in ranked[: max(q.top_k, 1)]]
    return {"matches": results}


@app.post("/memory/update", dependencies=[Depends(_require_api_key)])
def update_memory(u: UpdateItem):
    with _store_lock:
        for m in _memory_index["memories"]:
            if m["id"] == u.memory_id and m["role"] == u.role:
                m["content"] = u.new_content
                m["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Memory not found")


@app.delete("/memory/delete/{role}/{memory_id}", dependencies=[Depends(_require_api_key)])
def delete_memory(role: str, memory_id: str):
    with _store_lock:
        before = len(_memory_index["memories"])
        _memory_index["memories"] = [m for m in _memory_index["memories"] if not (m["id"] == memory_id and m["role"] == role)]
        after = len(_memory_index["memories"])
    if before == after:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "ok"}


@app.post("/memory/auto-query", dependencies=[Depends(_require_api_key)])
def auto_query_memory(q: QueryItem):
    with _store_lock:
        docs = [
            m for m in _memory_index["memories"]
            if m["role"] == q.role and (m["user_id"] == q.user_id or m["user_id"] == "shared")
        ]
    ranked = _bm25_like(q.query, docs)
    texts = [docs[i]["content"] for i, _ in ranked[: max(q.top_k, 1)]]
    return {"matches": texts}


@app.post("/memory/feedback", dependencies=[Depends(_require_api_key)])
def feedback_memory(fb: FeedbackItem):
    item = fb.model_dump()
    item["id"] = uuid.uuid4().hex
    item["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _store_lock:
        _memory_index["feedback"].append(item)
    return {"status": "ok"}


@app.post("/memory/tag-search", dependencies=[Depends(_require_api_key)])
def tag_search_memory(t: TagSearchItem):
    with _store_lock:
        pool = [m for m in _memory_index["memories"] if m["role"] == t.role]
    matches = [m for m in pool if set(t.tags).intersection(set(m.get("tags", [])))]
    ranked = sorted(matches, key=lambda m: len(set(t.tags).intersection(m.get("tags", []))), reverse=True)
    out = [{"doc": m, "score": len(set(t.tags).intersection(m.get("tags", [])))} for m in ranked[: max(t.top_k, 1)]]
    return {"matches": out}


@app.get("/memory/self-review", dependencies=[Depends(_require_api_key)])
def self_review_memory():
    with _store_lock:
        pos = sum(1 for f in _memory_index["feedback"] if f["feedback_type"] == "positive")
        neg = sum(1 for f in _memory_index["feedback"] if f["feedback_type"] == "negative")
        mcount = len(_memory_index["memories"])
    return {"feedback_summary": {"positive": pos, "negative": neg, "memories": mcount}}


# ============================ Goals ============================

@app.post("/memory/goals", dependencies=[Depends(_require_api_key)])
def save_goal(goal: GoalItem):
    g = {
        "id": uuid.uuid4().hex,
        "user_id": goal.user_id,
        "content": goal.content,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with _store_lock:
        _memory_index["goals"].append(g)
    return {"status": "ok"}


@app.get("/memory/goals", dependencies=[Depends(_require_api_key)])
def get_goals():
    with _store_lock:
        docs = [g["content"] for g in _memory_index["goals"]]
    return {"documents": docs}


@app.delete("/memory/goals/{goal_id}", dependencies=[Depends(_require_api_key)])
def delete_goal(goal_id: str):
    with _store_lock:
        before = len(_memory_index["goals"])
        _memory_index["goals"] = [g for g in _memory_index["goals"] if g["id"] != goal_id]
        after = len(_memory_index["goals"])
    if before == after:
        raise HTTPException(status_code=404, detail="Goal not found")
    return {"status": "ok"}


# ============================ Context ============================

@app.post("/context/build", dependencies=[Depends(_require_api_key)])
def build_context(req: ContextBuildRequest):
    with _store_lock:
        docs = [
            m for m in _memory_index["memories"]
            if m.get("role") == req.role and (m.get("user_id") == req.user_id or m.get("user_id") == "shared")
        ]
    ranked = _bm25_like(req.query, docs)
    top_docs = [docs[i] for i, _ in ranked[: max(req.top_k, 1)]]

    if req.mode == "summarise":
        answer = "\n\n".join(d.get("content", "") for d in top_docs)
    elif req.mode in {"draft", "review", "brainstorm"}:
        answer = f"Mode: {req.mode}. Context count: {len(top_docs)}. Query: {req.query}"
    else:
        snippet = (top_docs[0]["content"][:160] + "...") if top_docs else ""
        answer = f"Context results: {len(top_docs)}. Top match: {snippet}"

    tokens = sum(len(_tokenize(d.get("content", ""))) for d in top_docs)
    return {"answer": answer, "context_used": top_docs, "tokens": tokens, "mode": req.mode}

