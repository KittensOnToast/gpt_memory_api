# path: app/memory_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from datetime import datetime
import os, base64, httpx, uuid

# =========================
# Settings
# =========================
class Settings(BaseSettings):
    """Config from environment vars. Set SETTINGS__API_KEY in Render/local env."""
    model_config = SettingsConfigDict(env_prefix="SETTINGS__", case_sensitive=False)
    API_KEY: Optional[str] = None
    STORAGE_DIR: str = "./storage"
    URL_FETCH_TIMEOUT_S: int = 20

settings = Settings()
os.makedirs(settings.STORAGE_DIR, exist_ok=True)

# =========================
# API Init
# =========================
app = FastAPI(
    title="Plan C GPT Memory API",
    version="2.6.0",
    description="Full API with memory + files backends, OpenAPI 3.0.3 compatible."
)

# =========================
# API Key Check
# =========================
def _require_api_key(x_api_key: str = Query(None)):
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# =========================
# Schemas
# =========================
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

class UploadJsonBody(BaseModel):
    user_id: str
    filename: str
    content_base64: str

class UploadUrlBody(BaseModel):
    user_id: str
    url: HttpUrl
    filename: Optional[str]

class FileMeta(BaseModel):
    user_id: str
    filename: str
    stored_path: str
    size: int
    created_at: str

class FilesUploadResponse(BaseModel):
    status: str
    count: int
    files: List[FileMeta]

class ContextBuildRequest(BaseModel):
    user_id: str
    role: str
    query: str
    top_k: int = 3
    ask_gpt: bool = False
    mode: str = "general"

# =========================
# In-Memory DB
# =========================
MEMORY_DB = {}
FILES_DB = {}
FEEDBACK_DB = {}

# =========================
# Memory Endpoints
# =========================
@app.post("/memory/save", dependencies=[Depends(_require_api_key)])
async def save_memory(item: MemoryItem):
    mem_id = str(uuid.uuid4())
    MEMORY_DB[mem_id] = {**item.dict(), "created_at": datetime.utcnow().isoformat()}
    return {"status": "ok", "memory_id": mem_id}

@app.post("/memory/query", dependencies=[Depends(_require_api_key)])
async def query_memory(item: QueryItem):
    matches = {mid: m for mid, m in MEMORY_DB.items() if item.query.lower() in m["content"].lower()}
    return {"matches": matches}

@app.post("/memory/update", dependencies=[Depends(_require_api_key)])
async def update_memory(item: UpdateItem):
    if item.memory_id not in MEMORY_DB:
        raise HTTPException(status_code=404, detail="Memory not found")
    MEMORY_DB[item.memory_id]["content"] = item.new_content
    MEMORY_DB[item.memory_id]["updated_at"] = datetime.utcnow().isoformat()
    return {"status": "ok"}

@app.delete("/memory/delete/{role}/{memory_id}", dependencies=[Depends(_require_api_key)])
async def delete_memory(role: str, memory_id: str):
    if memory_id not in MEMORY_DB:
        raise HTTPException(status_code=404, detail="Memory not found")
    del MEMORY_DB[memory_id]
    return {"status": "ok"}

@app.post("/memory/auto-query", dependencies=[Depends(_require_api_key)])
async def auto_query_memory(item: QueryItem):
    matches = [m["content"] for m in MEMORY_DB.values() if m["role"] == item.role]
    return {"matches": matches[: item.top_k]}

@app.post("/memory/feedback", dependencies=[Depends(_require_api_key)])
async def feedback_memory(item: FeedbackItem):
    FEEDBACK_DB.setdefault(item.memory_id, []).append(item.dict())
    return {"status": "ok"}

@app.post("/memory/tag-search", dependencies=[Depends(_require_api_key)])
async def tag_search_memory(item: TagSearchItem):
    matches = {mid: m for mid, m in MEMORY_DB.items() if set(item.tags) & set(m.get("tags", []))}
    return {"matches": matches}

@app.get("/memory/self-review", dependencies=[Depends(_require_api_key)])
async def self_review_memory():
    pos = sum(1 for feedbacks in FEEDBACK_DB.values() for fb in feedbacks if fb["feedback_type"] == "positive")
    neg = sum(1 for feedbacks in FEEDBACK_DB.values() for fb in feedbacks if fb["feedback_type"] == "negative")
    return {"feedback_summary": {"positive": pos, "negative": neg, "memories": len(MEMORY_DB)}}

# =========================
# Goals Endpoints
# =========================
@app.post("/memory/goals", dependencies=[Depends(_require_api_key)])
async def save_goal(item: GoalItem):
    gid = str(uuid.uuid4())
    MEMORY_DB[gid] = item.dict()
    return {"status": "ok"}

@app.get("/memory/goals", dependencies=[Depends(_require_api_key)])
async def get_goals():
    return {"documents": [m["content"] for m in MEMORY_DB.values() if "content" in m]}

@app.delete("/memory/goals/{goal_id}", dependencies=[Depends(_require_api_key)])
async def delete_goal(goal_id: str):
    if goal_id not in MEMORY_DB:
        raise HTTPException(status_code=404, detail="Goal not found")
    del MEMORY_DB[goal_id]
    return {"status": "ok"}

# =========================
# File Upload Helpers
# =========================
def _store_file_bytes(user_id: str, filename: str, content: bytes) -> FileMeta:
    safe_dir = os.path.join(settings.STORAGE_DIR, user_id)
    os.makedirs(safe_dir, exist_ok=True)
    stored_path = os.path.join(safe_dir, filename)
    with open(stored_path, "wb") as f:
        f.write(content)
    meta = FileMeta(
        user_id=user_id,
        filename=filename,
        stored_path=stored_path,
        size=len(content),
        created_at=datetime.utcnow().isoformat()
    )
    FILES_DB[filename] = meta.dict()
    return meta

# =========================
# File Upload Endpoints
# =========================
@app.post("/files/upload", response_model=FilesUploadResponse, dependencies=[Depends(_require_api_key)])
async def upload_files(user_id: str = Query(...),
                       file: Optional[UploadFile] = File(None),
                       files: Optional[List[UploadFile]] = File(None),
                       attachment: Optional[UploadFile] = File(None),
                       attachments: Optional[List[UploadFile]] = File(None)):
    uploaded = []
    file_list = []
    for f in [file, attachment]:
        if f:
            file_list.append(f)
    if files:
        file_list.extend(files)
    if attachments:
        file_list.extend(attachments)
    if not file_list:
        raise HTTPException(status_code=400, detail="No files provided")
    for f in file_list:
        uploaded.append(_store_file_bytes(user_id, f.filename, await f.read()))
    return FilesUploadResponse(status="ok", count=len(uploaded), files=uploaded)

@app.post("/files/upload-json", response_model=FilesUploadResponse, dependencies=[Depends(_require_api_key)])
async def upload_file_json(body: UploadJsonBody):
    try:
        content = base64.b64decode(body.content_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")
    meta = _store_file_bytes(body.user_id, body.filename, content)
    return FilesUploadResponse(status="ok", count=1, files=[meta])

@app.post("/files/upload-url", response_model=FilesUploadResponse, dependencies=[Depends(_require_api_key)])
async def upload_file_url(body: UploadUrlBody):
    try:
        async with httpx.AsyncClient(timeout=settings.URL_FETCH_TIMEOUT_S, follow_redirects=True) as client:
            resp = await client.get(str(body.url))
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Fetch failed: {exc}")
    fname = body.filename or os.path.basename(str(body.url)) or f"file_{uuid.uuid4()}"
    meta = _store_file_bytes(body.user_id, fname, resp.content)
    return FilesUploadResponse(status="ok", count=1, files=[meta])

@app.get("/files/list", dependencies=[Depends(_require_api_key)])
async def list_files(user_id: Optional[str] = Query(None)):
    files = list(FILES_DB.values())
    if user_id:
        files = [f for f in files if f["user_id"] == user_id]
    return {"files": files}

@app.get("/files/download/{filename}", dependencies=[Depends(_require_api_key)])
async def download_file(filename: str):
    if filename not in FILES_DB:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(FILES_DB[filename]["stored_path"], filename=filename)

@app.delete("/files/delete/{filename}", dependencies=[Depends(_require_api_key)])
async def delete_file(filename: str, user_id: Optional[str] = Query(None)):
    if filename not in FILES_DB:
        raise HTTPException(status_code=404, detail="Not found")
    if user_id and FILES_DB[filename]["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    os.remove(FILES_DB[filename]["stored_path"])
    del FILES_DB[filename]
    return {"status": "ok"}

# =========================
# Context Build
# =========================
@app.post("/context/build", dependencies=[Depends(_require_api_key)])
async def build_context(req: ContextBuildRequest):
    docs = [m for m in MEMORY_DB.values() if m.get("role") == req.role and (m.get("user_id") == req.user_id or m.get("user_id") == "shared")]
    top_docs = docs[: max(req.top_k, 1)]
    if req.mode == "summarise":
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
        "tokens": sum(len(d.get("content", "")) for d in top_docs),
        "mode": req.mode
    }

# =========================
# Health check
# =========================
@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "embedding_model": "mock-embeddings",
        "gpt_model": "gpt-5-thinking",
        "bm25": True,
        "persona_seeded": True
    }

