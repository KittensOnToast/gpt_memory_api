# path: app/memory_api.py
"""
Plan C GPT Memory API — v1

Why these choices:
- Per-user scoping on queries/updates/deletes to prevent data leakage.
- Persistent Chroma (configurable path) so data survives restarts.
- File pipeline: upload->persist->extract->chunk->embed->query->delete, with ownership checks.
- Pydantic v2 models + response models for clear FastAPI docs and validation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import os
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions


# -----------------------------
# Settings & Constants
# -----------------------------

class Settings(BaseSettings):
    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")
    chroma_persist_dir: str = Field(default=".chroma", validation_alias="CHROMA_PERSIST_DIR")
    embedding_model: str = Field(default="text-embedding-3-small", validation_alias="EMBEDDING_MODEL")
    file_storage_dir: str = Field(default=".files", validation_alias="FILE_STORAGE_DIR")
    max_file_size_mb: int = Field(default=20, validation_alias="MAX_FILE_SIZE_MB")

    class Config:
        env_file = ".env"
        case_sensitive = False


class RoleEnum(str, Enum):
    shared = "shared"
    personal_assistant = "personal_assistant"
    writing_review = "writing_review"
    goals = "goals"
    feedback = "feedback"


ALLOWED_EXTS = {".txt", ".md", ".pdf", ".docx", ".csv", ".json"}


# -----------------------------
# App & Dependencies
# -----------------------------

def get_settings() -> Settings:
    return Settings()  # raises if OPENAI_API_KEY missing


def get_embedder(settings: Settings = Depends(get_settings)):
    # Centralized for DI; easier to swap models/vendors later.
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )


def get_chroma_client(settings: Settings = Depends(get_settings)) -> chromadb.Client:
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collections(
    client: chromadb.Client = Depends(get_chroma_client),
    embedder: Any = Depends(get_embedder),
) -> Dict[str, Collection]:
    names = [e.value for e in RoleEnum]
    cols: Dict[str, Collection] = {}
    for name in names:
        cols[name] = client.get_or_create_collection(name, embedding_function=embedder)
    # File registry and chunk index
    cols["files"] = client.get_or_create_collection("files", embedding_function=embedder)
    cols["files_index"] = client.get_or_create_collection("files_index", embedding_function=embedder)
    return cols


# -----------------------------
# Schemas
# -----------------------------

class MemoryItem(BaseModel):
    user_id: str = Field(..., min_length=1)
    role: RoleEnum
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: List[str]) -> List[str]:
        return sorted({t.strip().lower() for t in v if t and t.strip()})


class MemorySaved(BaseModel):
    status: str
    memory_id: str


class QueryItem(BaseModel):
    user_id: str
    role: RoleEnum
    query: str
    top_k: int = Field(3, ge=1, le=20)


class QueryMatches(BaseModel):
    matches: List[Dict[str, Any]]


class UpdateItem(BaseModel):
    memory_id: str
    role: RoleEnum
    user_id: str
    new_content: str


class DeleteResult(BaseModel):
    status: str


class FeedbackItem(BaseModel):
    memory_id: str
    role: RoleEnum
    user_id: str
    feedback_type: str = Field(..., pattern=r"^(positive|negative)$")
    feedback_text: str


class TagSearchItem(BaseModel):
    user_id: str
    role: RoleEnum
    tags: List[str]
    top_k: int = Field(3, ge=1, le=20)


class GoalItem(BaseModel):
    user_id: str
    content: str


class GoalSaved(BaseModel):
    status: str
    goal_id: str


class GoalsList(BaseModel):
    ids: List[str]
    documents: List[str]
    metadatas: List[Dict[str, Any]]


class FileUploadResponse(BaseModel):
    status: str
    file_id: str
    filename: str
    chunks_indexed: int


class FileRecord(BaseModel):
    file_id: str
    filename: str
    role: RoleEnum
    size_bytes: int
    created_at: str
    tags: List[str] = Field(default_factory=list)


class FileListResponse(BaseModel):
    files: List[FileRecord]


class FileQueryItem(BaseModel):
    user_id: str
    query: str
    top_k: int = Field(3, ge=1, le=20)


# -----------------------------
# App Factory
# -----------------------------

def create_app() -> FastAPI:
    app = FastAPI(title="Plan C GPT Memory API", version="1.0.0")

    @app.middleware("http")
    async def add_version_header(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-PlanC-Version"] = app.version
        return response

    @app.exception_handler(HTTPException)
    async def http_exc_handler(_: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.get("/healthz")
    def healthz(settings: Settings = Depends(get_settings)):
        return {"status": "ok", "embedding_model": settings.embedding_model}

    # -------- Memory --------

    @app.post("/memory/save", response_model=MemorySaved)
    def save_memory(
        item: MemoryItem,
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        if item.role.value not in collections:
            raise HTTPException(status_code=400, detail="Invalid role")

        mem_id = str(uuid.uuid4())
        meta = {
            "user_id": item.user_id,
            "tags": item.tags,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": None,
        }
        collections[item.role.value].add(
            documents=[item.content],
            metadatas=[meta],
            ids=[mem_id],
        )
        return MemorySaved(status="success", memory_id=mem_id)

    @app.post("/memory/query", response_model=QueryMatches)
    def query_memory(
        item: QueryItem,
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        if item.role.value not in collections:
            raise HTTPException(status_code=400, detail="Invalid role")

        col = collections[item.role.value]
        results = col.query(query_texts=[item.query], n_results=item.top_k)
        return QueryMatches(matches=_pack_matches_for_user(results, user_id=item.user_id))

    @app.post("/memory/update", response_model=DeleteResult)
    def update_memory(
        item: UpdateItem,
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        if item.role.value not in collections:
            raise HTTPException(status_code=400, detail="Invalid role")

        col = collections[item.role.value]
        existing = col.get(ids=[item.memory_id])
        if not existing.get("ids") or not existing["ids"]:
            raise HTTPException(status_code=404, detail="Memory not found")
        meta = existing.get("metadatas", [{}])[0] or {}
        if meta.get("user_id") != item.user_id:
            raise HTTPException(status_code=403, detail="Forbidden: not your memory")

        col.update(
            ids=[item.memory_id],
            documents=[item.new_content],
            metadatas=[{**meta, "updated_at": datetime.utcnow().isoformat()}],
        )
        return DeleteResult(status="success")

    @app.delete("/memory/delete/{role}/{memory_id}", response_model=DeleteResult)
    def delete_memory(
        role: RoleEnum = Path(...),
        memory_id: str = Path(...),
        user_id: str = Query(..., min_length=1),
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        if role.value not in collections:
            raise HTTPException(status_code=400, detail="Invalid role")

        col = collections[role.value]
        existing = col.get(ids=[memory_id])
        if not existing.get("ids") or not existing["ids"]:
            raise HTTPException(status_code=404, detail="Memory not found")
        meta = existing.get("metadatas", [{}])[0] or {}
        if meta.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Forbidden: not your memory")

        col.delete(ids=[memory_id])
        return DeleteResult(status="success")

    @app.post("/memory/auto-query", response_model=QueryMatches)
    def auto_query(
        item: QueryItem,
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        all_docs: List[Dict[str, Any]] = []
        for role_key in [RoleEnum.shared.value, item.role.value]:
            if role_key in collections:
                result = collections[role_key].query(query_texts=[item.query], n_results=item.top_k)
                all_docs.extend(_pack_matches_for_user(result, user_id=item.user_id))
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for m in all_docs:
            mid = m.get("id")
            if mid and mid not in seen:
                deduped.append(m)
                seen.add(mid)
        return QueryMatches(matches=deduped)

    @app.post("/memory/feedback", response_model=MemorySaved)
    def save_feedback(
        item: FeedbackItem,
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        fb_id = str(uuid.uuid4())
        meta = {
            "user_id": item.user_id,
            "memory_id": item.memory_id,
            "role": item.role.value,
            "feedback": item.feedback_type,
            "created_at": datetime.utcnow().isoformat(),
        }
        collections[RoleEnum.feedback.value].add(
            documents=[item.feedback_text],
            metadatas=[meta],
            ids=[fb_id],
        )
        return MemorySaved(status="success", memory_id=fb_id)

    @app.post("/memory/tag-search", response_model=QueryMatches)
    def tag_search(
        item: TagSearchItem,
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        if item.role.value not in collections:
            raise HTTPException(status_code=400, detail="Invalid role")

        query_text = ", ".join(sorted({t.strip().lower() for t in item.tags if t.strip()}))
        res = collections[item.role.value].query(query_texts=[query_text or "tags"], n_results=item.top_k)
        return QueryMatches(matches=_pack_matches_for_user(res, user_id=item.user_id, tags_filter=set(item.tags)))

    @app.get("/memory/self-review")
    def self_review(collections: Dict[str, Collection] = Depends(get_collections)):
        feedback_data = collections[RoleEnum.feedback.value].get()
        return {"feedback_summary": feedback_data}

    # -------- Goals --------

    @app.post("/memory/goals", response_model=GoalSaved)
    def save_goal(
        goal: GoalItem,
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        try:
            goal_id = str(uuid.uuid4())
            meta = {
                "user_id": goal.user_id,
                "created_at": datetime.utcnow().isoformat(),
            }
            collections[RoleEnum.goals.value].add(
                documents=[goal.content],
                metadatas=[meta],
                ids=[goal_id],
            )
            return GoalSaved(status="success", goal_id=goal_id)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Error saving goal: {e}")

    @app.get("/memory/goals", response_model=GoalsList)
    def list_goals(
        user_id: str = Query(..., min_length=1),
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        data = collections[RoleEnum.goals.value].get()
        idxs = [i for i, md in enumerate(data.get("metadatas", [])) if (md or {}).get("user_id") == user_id]
        return GoalsList(
            ids=[data.get("ids", [])[i] for i in idxs],
            documents=[data.get("documents", [])[i] for i in idxs],
            metadatas=[(data.get("metadatas", [])[i] or {}) for i in idxs],
        )

    @app.delete("/memory/goals/{goal_id}", response_model=DeleteResult)
    def delete_goal(
        goal_id: str,
        user_id: str = Query(..., min_length=1),
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        data = collections[RoleEnum.goals.value].get(ids=[goal_id])
        if not data.get("ids"):
            raise HTTPException(status_code=404, detail="Goal not found")
        md = (data.get("metadatas", [{}])[0] or {})
        if md.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Forbidden: not your goal")
        collections[RoleEnum.goals.value].delete(ids=[goal_id])
        return DeleteResult(status="success")

    # -------- Files --------

    @app.post("/files/upload", response_model=FileUploadResponse)
    async def upload_file(
        request: Request,
        user_id: str = Form(...),
        role: RoleEnum = Form(...),
        tags_csv: Optional[str] = Form(None),
        file: UploadFile = File(...),
        settings: Settings = Depends(get_settings),
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")

        max_bytes = settings.max_file_size_mb * 1024 * 1024
        contents = await file.read()
        if len(contents) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large (>{settings.max_file_size_mb} MB)")

        os.makedirs(settings.file_storage_dir, exist_ok=True)
        user_dir = os.path.join(settings.file_storage_dir, _safe_name(user_id))
        os.makedirs(user_dir, exist_ok=True)

        file_id = str(uuid.uuid4())
        safe_name = f"{file_id}_{_safe_name(file.filename or 'upload')}"
        path = os.path.join(user_dir, safe_name)
        with open(path, "wb") as f:
            f.write(contents)

        text = _extract_text(contents, ext)
        chunks = _chunk_text(text)
        now = datetime.utcnow().isoformat()
        tags = _parse_tags(tags_csv)

        if chunks:
            ids = [f"{file_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "user_id": user_id,
                    "role": role.value,
                    "file_id": file_id,
                    "filename": file.filename,
                    "path": path,
                    "chunk_index": i,
                    "tags": tags,
                    "created_at": now,
                }
                for i in range(len(chunks))
            ]
            collections["files_index"].add(documents=chunks, metadatas=metadatas, ids=ids)

        collections["files"].add(
            documents=[f"FILE::{file.filename}"],
            metadatas=[{
                "user_id": user_id,
                "role": role.value,
                "file_id": file_id,
                "filename": file.filename,
                "path": path,
                "size_bytes": len(contents),
                "tags": tags,
                "created_at": now,
            }],
            ids=[file_id],
        )

        return FileUploadResponse(status="success", file_id=file_id, filename=file.filename or "", chunks_indexed=len(chunks))

    @app.get("/files", response_model=FileListResponse)
    def list_files(
        user_id: str = Query(..., min_length=1),
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        data = collections["files"].get()
        files: List[FileRecord] = []
        for i, md in enumerate(data.get("metadatas", [])):
            md = md or {}
            if md.get("user_id") != user_id:
                continue
            files.append(
                FileRecord(
                    file_id=md.get("file_id"),
                    filename=md.get("filename"),
                    role=RoleEnum(md.get("role", RoleEnum.shared.value)),
                    size_bytes=int(md.get("size_bytes", 0)),
                    created_at=md.get("created_at", ""),
                    tags=md.get("tags", []) or [],
                )
            )
        return FileListResponse(files=files)

    @app.get("/files/{file_id}")
    def download_file(
        file_id: str,
        user_id: str = Query(..., min_length=1),
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        rec = collections["files"].get(ids=[file_id])
        if not rec.get("ids"):
            raise HTTPException(status_code=404, detail="File not found")
        md = (rec.get("metadatas", [{}])[0] or {})
        if md.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Forbidden: not your file")
        path = md.get("path")
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=410, detail="File missing on disk")
        return FileResponse(path, filename=md.get("filename") or "download")

    @app.delete("/files/{file_id}", response_model=DeleteResult)
    def delete_file(
        file_id: str,
        user_id: str = Query(..., min_length=1),
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        rec = collections["files"].get(ids=[file_id])
        if not rec.get("ids"):
            raise HTTPException(status_code=404, detail="File not found")
        md = (rec.get("metadatas", [{}])[0] or {})
        if md.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Forbidden: not your file")

        path = md.get("path")
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:  # noqa: BLE001
            pass  # best-effort removal

        collections["files"].delete(ids=[file_id])

        idx = collections["files_index"].get()
        to_delete: List[str] = []
        for i, md2 in enumerate(idx.get("metadatas", [])):
            md2 = md2 or {}
            if md2.get("file_id") == file_id and md2.get("user_id") == user_id:
                to_delete.append(idx.get("ids", [])[i])
        if to_delete:
            collections["files_index"].delete(ids=to_delete)

        return DeleteResult(status="success")

    @app.post("/files/query", response_model=QueryMatches)
    def query_files_index(
        item: FileQueryItem,
        collections: Dict[str, Collection] = Depends(get_collections),
    ):
        res = collections["files_index"].query(query_texts=[item.query], n_results=item.top_k)
        return QueryMatches(matches=_pack_matches_for_user(res, user_id=item.user_id))

    return app


# -----------------------------
# Helpers
# -----------------------------

class ChromaQueryResult(TypedDict, total=False):
    ids: List[List[str]]
    documents: List[List[str]]
    metadatas: List[List[Dict[str, Any]]]
    distances: List[List[float]]


def _pack_matches_for_user(
    res: Optional[ChromaQueryResult],
    *,
    user_id: str,
    tags_filter: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    if not res:
        return []

    ids_nested = res.get("ids") or [[]]
    docs_nested = res.get("documents") or [[]]
    metas_nested = res.get("metadatas") or [[]]
    dists_nested = res.get("distances") or [[]]

    out: List[Dict[str, Any]] = []
    for i, _id in enumerate(ids_nested[0] if ids_nested else []):
        meta = (metas_nested[0][i] if metas_nested and metas_nested[0] else {}) or {}
        if meta.get("user_id") not in (user_id, None):
            continue
        if tags_filter:
            tags = set(t.lower() for t in (meta.get("tags") or []))
            if not (tags & set(t.lower() for t in tags_filter)):
                continue
        out.append(
            {
                "id": _id,
                "document": (docs_nested[0][i] if docs_nested and docs_nested[0] else None),
                "metadata": meta,
                "distance": (dists_nested[0][i] if dists_nested and dists_nested[0] else None),
            }
        )
    return out


def _safe_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".")).strip(".") or "file"


def _parse_tags(csv_str: Optional[str]) -> List[str]:
    if not csv_str:
        return []
    return sorted({t.strip().lower() for t in csv_str.split(",") if t.strip()})


def _chunk_text(text: str, *, size: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def _extract_text(contents: bytes, ext: str) -> str:
    ext = ext.lower()
    if ext in {".txt", ".md", ".csv", ".json"}:
        try:
            return contents.decode("utf-8")
        except UnicodeDecodeError:
            return contents.decode("latin-1", errors="ignore")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"PDF support missing: {e}")
        try:
            import io
            reader = PdfReader(io.BytesIO(contents))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}")
    if ext == ".docx":
        try:
            import io
            from docx import Document
            doc = Document(io.BytesIO(contents))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Failed to parse DOCX: {e}")
    raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")


# Entrypoint for uvicorn
app = create_app()

