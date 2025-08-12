# app/memory_api.py
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware  # why: required for browser/Actions callers
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sqlalchemy import Column, DateTime, String, Text, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

# ----------------------------
# Settings
# ----------------------------

class Settings(BaseSettings):
    API_ACCESS_KEY: Optional[str] = None
    DB_PATH: str = "./data/memory.sqlite"
    CORS_ALLOW_ORIGINS: Optional[str] = "*"  # comma-separated or "*" for all
    APP_VERSION: str = "3.0.0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Ensure data dir exists
os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)

# ----------------------------
# Database (SQLite + SQLAlchemy)
# ----------------------------

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
    tags = Column(Text, nullable=True)  # JSON list as text
    created_at = Column(DateTime, nullable=False, index=True)

Base.metadata.create_all(engine)

# ----------------------------
# Models
# ----------------------------

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
    mode: str = Field("lexical", pattern="^(lexical|semantic|hybrid)$")  # semantic stubbed

class QueryResponse(BaseModel):
    matches: List[QueryMatch] = []

# ----------------------------
# Auth
# ----------------------------

def _parse_bearer(auth_header: Optional[str]) -> Optional[str]:
    if not auth_header:
        return None
    # tolerate extra spaces/case
    parts = auth_header.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None

def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    # No key configured: open in dev; keep closed in prod by setting API_ACCESS_KEY
    required = settings.API_ACCESS_KEY
    if not required:
        return
    supplied = x_api_key or _parse_bearer(authorization)
    if not supplied:
        raise HTTPException(status_code=401, detail="Missing API credentials")
    if supplied != required:
        # why: differentiate wrong creds vs no creds for easier debugging
        raise HTTPException(status_code=403, detail="Invalid API credentials")

# ----------------------------
# App (CORS + routes)
# ----------------------------

app = FastAPI(title="Plan C GPT Memory API", version=settings.APP_VERSION)

# CORS (safe default "*" in dev; restrict in prod)
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

# ----------------------------
# Helpers
# ----------------------------

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _mask(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    if len(token) <= 6:
        return "***"
    return f"{token[:3]}...{token[-3:]}"

def _highlight_snippet(text_: str, query: str, radius: int = 60) -> str:
    # why: quick UX for inspection; avoids over-returning large docs
    try:
        m = re.search(re.escape(query), text_, re.IGNORECASE)
        if not m:
            return text_[:radius] + ("..." if len(text_) > radius else "")
        start = max(0, m.start() - radius // 2)
        end = min(len(text_), m.end() + radius // 2)
        return text_[start:end]
    except re.error:
        return text_[:radius]

# ----------------------------
# Routes
# ----------------------------

@app.get("/", tags=["meta"])
def root() -> dict:
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
    # never return real secret; only masked headers received and whether a key is enforced
    return {
        "api_key_required": bool(settings.API_ACCESS_KEY),
        "received": {
            "x_api_key": _mask(x_api_key),
            "authorization_bearer": _mask(_parse_bearer(authorization)),
        },
    }

@app.get("/debug/key-hash", tags=["debug"])
def debug_key_hash():
    # why: safe way to compare without revealing the key
    if not settings.API_ACCESS_KEY:
        return {"configured": False, "sha256": None}
    h = hashlib.sha256(settings.API_ACCESS_KEY.encode("utf-8")).hexdigest()
    return {"configured": True, "sha256": h}

@app.post("/memory/save", response_model=SaveMemoryResponse, dependencies=[Depends(require_api_key)], tags=["memory"])
def save_memory(item: MemoryItem) -> SaveMemoryResponse:
    mem_id = str(uuid.uuid4())
    created = datetime.utcnow()
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
        # why: surface DB issues quickly during setup
        raise HTTPException(status_code=500, detail=f"DB error: {e}") from e
    return SaveMemoryResponse(id=mem_id, stored=True)

@app.post("/memory/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)], tags=["memory"])
def query_memory(q: QueryItem) -> QueryResponse:
    # lexical-only baseline (LIKE); semantic/hybrid can be layered later
    like = f"%{q.query}%"
    rows: List[MemoryORM] = []
    try:
        with SessionLocal() as s:
            stmt = text(
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
        tags_list = None
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
            created_at=r.created_at.replace(microsecond=0).isoformat() + "Z",
        )
        matches.append(
            QueryMatch(
                doc=doc,
                score=1.0,  # lexical stub
                score_components=ScoreComponents(lexical=1.0, semantic=0.0, feedback=0.0),
                highlight=_highlight_snippet(r.content, q.query),
            )
        )
    return QueryResponse(matches=matches)

