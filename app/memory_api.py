# path: app/memory_api.py
import os
import re
import uuid
from datetime import datetime
from typing import List, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document
from openai import OpenAI

# =========================
# Settings
# =========================
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    GPT_MODEL: str = "gpt-5"
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    FILE_STORAGE_DIR: str = "./data/files"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    MAX_FILE_SIZE_MB: int = 20
    API_ACCESS_KEY: str | None = None  # API key for auth; if None, auth is disabled

    class Config:
        env_file = ".env"

settings = Settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# =========================
# Init Chroma
# =========================
embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=settings.OPENAI_API_KEY,
    model_name=settings.EMBEDDING_MODEL,
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
    mode: str = "general"  # general | summarise | draft | review | brainstorm

# =========================
# Helpers
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
    elif file_path.suffix.lower() in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8", errors="ignore")
    else:
        return ""


def _chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

# =========================
# Persona Seed (from Plan C Company Brain)
# =========================
PERSONA_TEXT = (
    "\n".join(
        [
            "You are Plan C Assistant — a knowledgeable, proactive, culturally aware virtual team member for Plan C.",
            "Mission: Empower communities through inclusive, innovative planning and strategy, turning aspirations into measurable outcomes.",
            "Core values: Reciprocity, excellence, respect for difference, agents of change, creativity, collaboration, integrity, industriousness.",
            "Tone: Warm, respectful, plain-English, culturally aware. In casual conversation, you may use light Aussie expressions and humour.",
            "When tasks are formal or client-facing, maintain precise, professional, and culturally respectful language.",
            "Always align suggestions with Plan C's mission, values, and community-first approach.",
        ]
    )
)

if collections["persona"].count() == 0:
    collections["persona"].add(
        documents=[PERSONA_TEXT],
        metadatas=[{"created_at": datetime.utcnow().isoformat()}],
        ids=[str(uuid.uuid4())],
    )

# =========================
# App Init + Security Middleware
# =========================
app = FastAPI(title="Plan C GPT Memory API")

@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    open_paths = {
        "/healthz",
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }
    # Allow everything if API key not configured (dev mode) or for open paths
    if not settings.API_ACCESS_KEY or request.url.path in open_paths:
        return await call_next(request)

    key = request.headers.get("x-api-key")
    if key != settings.API_ACCESS_KEY:
        return JSONResponse(
            status_code=401, content={"detail": "Unauthorized: missing or invalid x-api-key"}
        )
    return await call_next(request)

# =========================
# Health
# =========================
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "embedding_model": settings.EMBEDDING_MODEL,
        "gpt_model": settings.GPT_MODEL,
        "auth": bool(settings.API_ACCESS_KEY),
    }

# =========================
# Memory Endpoints
# =========================
@app.post("/memory/save")
def save_memory(item: MemoryItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    mem_id = str(uuid.uuid4())
    collections[item.role].add(
        documents=[item.content],
        metadatas=[
            {
                "user_id": item.user_id,
                "tags": ",".join(item.tags),
                "created_at": datetime.utcnow().isoformat(),
            }
        ],
        ids=[mem_id],
    )
    return {"status": "success", "memory_id": mem_id}


@app.post("/memory/query")
def query_memory(item: QueryItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    results = collections[item.role].query(query_texts=[item.query], n_results=item.top_k)
    return {"matches": results}


@app.post("/memory/update")
def update_memory(item: UpdateItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    collections[item.role].update(
        ids=[item.memory_id],
        documents=[item.new_content],
        metadatas=[{"updated_at": datetime.utcnow().isoformat()}],
    )
    return {"status": "success"}


@app.delete("/memory/delete/{role}/{memory_id}")
def delete_memory(role: str, memory_id: str):
    if role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    collections[role].delete(ids=[memory_id])
    return {"status": "success"}


@app.post("/memory/auto-query")
def auto_query(item: QueryItem):
    all_results: List[str] = []
    for role_key in ["shared", item.role]:
        if role_key in collections:
            result = collections[role_key].query(query_texts=[item.query], n_results=item.top_k)
            if result and result.get("documents") and result["documents"][0]:
                all_results.extend(result["documents"][0])
    return {"matches": all_results}


@app.post("/memory/feedback")
def save_feedback(item: FeedbackItem):
    fb_id = str(uuid.uuid4())
    collections["feedback"].add(
        documents=[item.feedback_text],
        metadatas=[
            {
                "user_id": item.user_id,
                "memory_id": item.memory_id,
                "feedback": item.feedback_type,
                "created_at": datetime.utcnow().isoformat(),
            }
        ],
        ids=[fb_id],
    )
    return {"status": "success", "feedback_id": fb_id}


@app.post("/memory/tag-search")
def tag_search(item: TagSearchItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    result = collections[item.role].get(where={"tags": ",".join(item.tags)})
    return {"matches": result}


@app.get("/memory/self-review")
def self_review():
    feedback_data = collections["feedback"].get()
    return {"feedback_summary": feedback_data}


@app.post("/memory/goals")
def save_goal(goal: GoalItem):
    goal_id = str(uuid.uuid4())
    collections["goals"].add(
        documents=[goal.content],
        metadatas=[{"user_id": goal.user_id, "created_at": datetime.utcnow().isoformat()}],
        ids=[goal_id],
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
# File Handling
# =========================
@app.post("/files/upload")
def upload_file(user_id: str = Query(...), file: UploadFile = File(...)):
    if file.filename == "":
        raise HTTPException(status_code=400, detail="Empty filename")

    safe_name = _safe_filename(file.filename)
    dest_path = Path(settings.FILE_STORAGE_DIR) / safe_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # size check (approx; UploadFile doesn't stream length)
    content = file.file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    with dest_path.open("wb") as buffer:
        buffer.write(content)

    # index content
    text = _extract_text(dest_path)
    chunks = _chunk_text(text)
    for chunk in chunks:
        collections["files_index"].add(
            documents=[chunk],
            metadatas=[
                {
                    "user_id": user_id,
                    "source_file": safe_name,
                    "created_at": datetime.utcnow().isoformat(),
                }
            ],
            ids=[str(uuid.uuid4())],
        )

    collections["files"].add(
        documents=[safe_name],
        metadatas=[
            {
                "user_id": user_id,
                "stored_path": str(dest_path),
                "created_at": datetime.utcnow().isoformat(),
            }
        ],
        ids=[str(uuid.uuid4())],
    )
    return {"status": "success", "filename": safe_name}


@app.get("/files/list")
def list_files():
    return collections["files"].get()


@app.get("/files/download/{filename}")
def download_file(filename: str):
    safe_name = _safe_filename(filename)
    file_path = Path(settings.FILE_STORAGE_DIR) / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))


@app.delete("/files/delete/{filename}")
def delete_file(filename: str):
    safe_name = _safe_filename(filename)
    file_path = Path(settings.FILE_STORAGE_DIR) / safe_name
    if file_path.exists():
        file_path.unlink()
    return {"status": "success"}

# =========================
# Context Builder with GPT
# =========================
@app.post("/context/build")
def build_context(item: ContextBuildRequest):
    def vector_search(role_key: str) -> List[dict[str, Any]]:
        if role_key not in collections:
            return []
        results = collections[role_key].query(
            query_texts=[item.query],
            n_results=item.top_k,
            include=["documents", "metadatas", "distances"],
        )
        matches: List[dict[str, Any]] = []
        docs = results.get("documents", [[]])[0]
        mds = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, md, dist in zip(docs, mds, dists):
            if not doc:
                continue
            matches.append(
                {
                    "source": role_key,
                    "content": doc,
                    "metadata": md or {},
                    "score": 1 - float(dist),  # convert distance to similarity
                }
            )
        return matches

    # aggregate across sources
    all_results: List[dict[str, Any]] = []
    for src in ["persona", item.role, "shared", "goals", "files_index"]:
        all_results.extend(vector_search(src))

    ranked = sorted(all_results, key=lambda x: x["score"], reverse=True)
    top_ranked = ranked[: item.top_k]
    context_text = "\n\n".join(m["content"] for m in top_ranked)

    if not item.ask_gpt:
        return {"context": top_ranked, "context_text": context_text}

    mode_map = {
        "general": {"temp": 0.3, "prefix": ""},
        "summarise": {"temp": 0.2, "prefix": "Please summarise the following context clearly and concisely."},
        "draft": {"temp": 0.3, "prefix": "Please draft a professional update based on the following context."},
        "review": {"temp": 0.2, "prefix": "Please review the following context and highlight key terms, risks, or important details."},
        "brainstorm": {"temp": 0.8, "prefix": "Please brainstorm creative ideas based on the following context."},
    }
    mode_cfg = mode_map.get(item.mode, mode_map["general"])

    try:
        response = client.chat.completions.create(
            model=settings.GPT_MODEL,
            messages=[
                {"role": "system", "content": PERSONA_TEXT},
                {
                    "role": "user",
                    "content": f"{mode_cfg['prefix']}\n\nQuery: {item.query}\n\nContext:\n{context_text}",
                },
            ],
            max_tokens=500,
            temperature=mode_cfg["temp"],
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling GPT: {e}")

    return {"answer": answer, "context_used": top_ranked}

