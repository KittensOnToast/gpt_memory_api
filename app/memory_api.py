from __future__ import annotations

import os
import json
import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Literal
from secrets import compare_digest

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Depends,
    Query,
    Path as FPath,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Text,
    select,
    func,
    delete as sqldelete,
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

API_KEYS = [k.strip() for k in (
    os.getenv("API_ACCESS_KEY") or
    os.getenv("API_KEYS") or
    os.getenv("API_KEY") or
    "dev-key"
).split(",") if k.strip()]
DB_PATH = os.getenv("DB_PATH", "./memory.db")
FILES_DIR = Path(os.getenv("FILES_DIR", "./files")).resolve()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
ALLOWED_MIME_PREFIXES = os.getenv("ALLOWED_MIME_PREFIXES", "text/,application/json").split(",")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")

FILES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Virtual Employee Memory API", version="0.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

en gine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


class MemoryORM(Base):
    __tablename__ = "memories"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    role = Column(String, index=True, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FileORM(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    original_name = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)
    mime_type = Column(String, nullable=True)
    sha256 = Column(String, index=True)
    size_bytes = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class JobORM(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    kind = Column(String, index=True)
    status = Column(String, index=True, default="queued")
    input = Column(Text)
    result = Column(Text)
    progress = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MemoryItem(BaseModel):
    user_id: str
    role: str
    content: str
    tags: List[str] = Field(default_factory=list)


class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    tags: Optional[List[str]] = None


class QueryItem(BaseModel):
    user_id: str
    role: str
    query: str
    mode: Literal["lexical", "semantic", "hybrid"] = "lexical"
    limit: int = 20
    offset: int = 0
    since: Optional[datetime] = None
    until: Optional[datetime] = None


class QueryHit(BaseModel):
    id: int
    content: str
    created_at: datetime
    score: float
    snippet: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class JobCreate(BaseModel):
    kind: str
    payload: dict = Field(default_factory=dict)


class JobView(BaseModel):
    id: int
    kind: str
    status: str
    progress: int
    result: Optional[dict] = None


def require_api_key(request: Request):
    supplied = (
        request.headers.get("X-API-Key")
        or (request.headers.get("Authorization") or "").removeprefix("Bearer ")
    )
    for valid in API_KEYS:
        if supplied and compare_digest(supplied, valid):
            return True
    raise HTTPException(status_code=403, detail="Invalid API credentials")


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return []


Base.metadata.create_all(engine)

with engine.begin() as conn:
    conn.exec_driver_sql(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            user_id,
            role,
            created_at UNINDEXED,
            tags,
            content='memories',
            content_rowid='id'
        );
        """
    )
    conn.exec_driver_sql(
        """
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, user_id, role, created_at, tags)
            VALUES (new.id, new.content, new.user_id, new.role, new.created_at, COALESCE(new.tags,''));
        END;
        """
    )
    conn.exec_driver_sql(
        """
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
        END;
        """
    )
    conn.exec_driver_sql(
        """
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
            INSERT INTO memories_fts(rowid, content, user_id, role, created_at, tags)
            VALUES (new.id, new.content, new.user_id, new.role, new.created_at, COALESCE(new.tags,''));
        END;
        """
    )


@app.get("/healthz")
def healthz(db: Session = Depends(get_db)):
    db.scalar(select(func.count(MemoryORM.id)))
    return {
        "ok": True,
        "fts": True,
        "db": str(Path(DB_PATH).resolve()),
        "cors": CORS_ALLOW_ORIGINS,
        "version": app.version,
    }


@app.post("/memory/save", dependencies=[Depends(require_api_key)])
def save_memory(item: MemoryItem, db: Session = Depends(get_db)):
    m = MemoryORM(
        user_id=item.user_id,
        role=item.role,
        content=item.content,
        tags=json_dumps(item.tags),
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return {"id": m.id, "created_at": m.created_at}


@app.get("/memory/{mem_id}", dependencies=[Depends(require_api_key)])
def get_memory(mem_id: int = FPath(..., ge=1), db: Session = Depends(get_db)):
    m = db.get(MemoryORM, mem_id)
    if not m:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {
        "id": m.id,
        "user_id": m.user_id,
        "role": m.role,
        "content": m.content,
        "tags": json_loads(m.tags),
        "created_at": m.created_at,
        "updated_at": m.updated_at,
    }


@app.put("/memory/{mem_id}", dependencies=[Depends(require_api_key)])
def update_memory(mem_id: int, patch: MemoryUpdate, db: Session = Depends(get_db)):
    m = db.get(MemoryORM, mem_id)
    if not m:
        raise HTTPException(status_code=404, detail="Memory not found")
    if patch.content is not None:
        m.content = patch.content
    if patch.tags is not None:
        m.tags = json_dumps(patch.tags)
    db.commit()
    return {"ok": True}


@app.delete("/memory/{mem_id}", dependencies=[Depends(require_api_key)])
def delete_memory(mem_id: int, db: Session = Depends(get_db)):
    result = db.execute(sqldelete(MemoryORM).where(MemoryORM.id == mem_id))
    db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"ok": True}


@app.post("/memory/query", response_model=List[QueryHit], dependencies=[Depends(require_api_key)])
def query_memory(q: QueryItem, db: Session = Depends(get_db)):
    return _fts_search(db, q)


def _fts_search(db: Session, q: QueryItem) -> List[QueryHit]:
    match = q.query.replace("\"", " ")
    params = {"user_id": q.user_id, "role": q.role, "limit": q.limit, "offset": q.offset, "match": match}
    date_clause = ""
    if q.since:
        params["since"] = q.since
        date_clause += " AND m.created_at >= :since"
    if q.until:
        params["until"] = q.until
        date_clause += " AND m.created_at <= :until"
    sql = f"""
        SELECT m.id, m.content, m.created_at, m.tags,
               bm25(memories_fts) AS score,
               snippet(memories_fts, 0, '[', ']', ' … ', 8) AS snip
        FROM memories_fts
        JOIN memories m ON m.id = memories_fts.rowid
        WHERE memories_fts.user_id=:user_id AND memories_fts.role=:role AND memories_fts MATCH :match{date_clause}
        ORDER BY score LIMIT :limit OFFSET :offset
    """
    rows = db.execute(sql, params).fetchall()
    hits: List[QueryHit] = []
    for r in rows:
        hits.append(
            QueryHit(
                id=r[0],
                content=r[1],
                created_at=r[2],
                tags=json_loads(r[3] or "[]"),
                score=float(r[4]) if r[4] is not None else 0.0,
                snippet=r[5],
            )
        )
    return hits


@app.post("/files/upload", dependencies=[Depends(require_api_key)])
async def upload_files(
    user_id: str = Query(...),
    uploads: List[UploadFile] = File(...),
    ingest: bool = Query(False),
    db: Session = Depends(get_db),
):
    saved = []
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    for upl in uploads:
        data = await upl.read()
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large (>{MAX_UPLOAD_MB}MB)")
        mime = upl.content_type or mimetypes.guess_type(upl.filename or "")[0] or "application/octet-stream"
        if not any(mime.startswith(pfx) for pfx in ALLOWED_MIME_PREFIXES):
            raise HTTPException(status_code=415, detail=f"MIME not allowed: {mime}")
        digest = sha256_bytes(data)
        safe_name = Path(upl.filename or f"upload-{digest[:8]}").name
        dest = FILES_DIR / f"{digest[:8]}-{safe_name}"
        with open(dest, "wb") as f:
            f.write(data)
        rec = FileORM(
            user_id=user_id,
            original_name=safe_name,
            stored_path=str(dest),
            mime_type=mime,
            sha256=digest,
            size_bytes=len(data),
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        saved.append({"id": rec.id, "name": safe_name, "bytes": rec.size_bytes})
        if ingest and mime.startswith("text/"):
            job_payload = {"file_id": rec.id, "user_id": user_id}
            _enqueue_job(db, kind="ingest", payload=job_payload)
    return {"saved": saved}


@app.get("/files/{file_id}", dependencies=[Depends(require_api_key)])
def download_file(file_id: int, db: Session = Depends(get_db)):
    rec = db.get(FileORM, file_id)
    if not rec:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(rec.stored_path, filename=rec.original_name, media_type=rec.mime_type)


@app.post("/jobs", response_model=JobView, dependencies=[Depends(require_api_key)])
def create_job(req: JobCreate, db: Session = Depends(get_db)):
    job = _enqueue_job(db, req.kind, req.payload)
    return _job_to_view(job)


@app.get("/jobs/{job_id}", response_model=JobView, dependencies=[Depends(require_api_key)])
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(JobORM, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_view(job)


def _enqueue_job(db: Session, kind: str, payload: dict) -> JobORM:
    job = JobORM(kind=kind, status="queued", progress=0, input=json_dumps(payload))
    db.add(job)
    db.commit()
    db.refresh(job)
    _run_job(job.id)
    return job


def _run_job(job_id: int):
    with engine.begin() as conn:
        job = conn.exec_driver_sql("SELECT id, kind, input FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not job:
            return
        kind = job[1]
        payload = json.loads(job[2]) if job[2] else {}
        conn.exec_driver_sql("UPDATE jobs SET status='running', progress=5, updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
        try:
            if kind == "ingest":
                file_id = int(payload["file_id"])  # type: ignore
                user_id = str(payload["user_id"])  # type: ignore
                _job_ingest_text(conn, job_id, file_id, user_id)
            conn.exec_driver_sql("UPDATE jobs SET status='done', progress=100, updated_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
        except Exception as e:
            conn.exec_driver_sql(
                "UPDATE jobs SET status='error', result=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (json_dumps({"error": str(e)}), job_id),
            )


def _job_ingest_text(conn, job_id: int, file_id: int, user_id: str):
    rec = conn.exec_driver_sql("SELECT stored_path FROM files WHERE id=?", (file_id,)).fetchone()
    if not rec:
        raise RuntimeError("file not found for ingest")
    path = rec[0]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    total = max(1, len(lines))
    tag = f"file:{file_id}"
    for idx, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        conn.exec_driver_sql(
            "INSERT INTO memories(user_id, role, content, tags, created_at, updated_at) VALUES(?,?,?,?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
            (user_id, "system", text, json_dumps([tag])),
        )
        if idx % 20 == 0:
            pct = int(5 + (idx / total) * 90)
            conn.exec_driver_sql("UPDATE jobs SET progress=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (pct, job_id))


def _job_to_view(job: JobORM) -> JobView:
    return JobView(
        id=job.id,
        kind=job.kind,
        status=job.status,
        progress=job.progress,
        result=json.loads(job.result) if job.result else None,
    )


def compute_embedding(text: str) -> List[float]:
    return []


@app.get("/memory", dependencies=[Depends(require_api_key)])
def list_memories(
    user_id: str = Query(...),
    role: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    since: Optional[datetime] = Query(None),
    until: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
):
    q = select(MemoryORM).where(MemoryORM.user_id == user_id)
    if role:
        q = q.where(MemoryORM.role == role)
    if since:
        q = q.where(MemoryORM.created_at >= since)
    if until:
        q = q.where(MemoryORM.created_at <= until)
    q = q.order_by(MemoryORM.created_at.desc()).limit(limit).offset(offset)
    rows = db.execute(q).scalars().all()
    return [
        {
            "id": m.id,
            "user_id": m.user_id,
            "role": m.role,
            "content": m.content,
            "tags": json_loads(m.tags),
            "created_at": m.created_at,
            "updated_at": m.updated_at,
        }
        for m in rows
    ]


@app.get("/debug/key-hash")
def debug_key_hash():
    return {"keys_configured": len(API_KEYS)}

