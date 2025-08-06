# memory_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uuid
import chromadb
from chromadb.utils import embedding_functions

# --- Initialize Chroma DB ---
chroma_client = chromadb.Client()
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create or get collections for shared + role-specific memory
collections = {
    "shared": chroma_client.get_or_create_collection(name="shared", embedding_function=embedder),
    "personal_assistant": chroma_client.get_or_create_collection(name="personal_assistant", embedding_function=embedder),
    "writing_review": chroma_client.get_or_create_collection(name="writing_review", embedding_function=embedder)
}

# --- API Setup ---
app = FastAPI(title="Custom GPT Memory API")

# --- Data Models ---
class MemoryItem(BaseModel):
    user_id: str
    role: str  # "shared", "personal_assistant", or "writing_review"
    content: str
    tags: list[str] = []

class QueryItem(BaseModel):
    user_id: str
    role: str
    query: str
    top_k: int = 3

class UpdateItem(BaseModel):
    memory_id: str
    role: str
    new_content: str

# --- API Endpoints ---
@app.post("/memory/save")
def save_memory(item: MemoryItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")

    mem_id = str(uuid.uuid4())
    try:
        collections[item.role].add(
            documents=[item.content],
            metadatas=[{
                "user_id": item.user_id,
                "tags": ",".join(item.tags) if item.tags else "",
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[mem_id]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving memory: {e}")
    return {"status": "success", "memory_id": mem_id}

@app.post("/memory/query")
def query_memory(item: QueryItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")

    try:
        results = collections[item.role].query(
            query_texts=[item.query],
            n_results=item.top_k
        )
        return {"matches": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying memory: {e}")

@app.post("/memory/update")
def update_memory(item: UpdateItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")

    try:
        collections[item.role].update(
            ids=[item.memory_id],
            documents=[item.new_content],
            metadatas=[{
                "updated_at": datetime.utcnow().isoformat()
            }]
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating memory: {e}")

@app.delete("/memory/delete/{role}/{memory_id}")
def delete_memory(role: str, memory_id: str):
    if role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")

    try:
        collections[role].delete(ids=[memory_id])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting memory: {e}")

