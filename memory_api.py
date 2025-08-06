# memory_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uuid
import os
import chromadb
from chromadb.utils import embedding_functions

# --- Load OpenAI API key ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

# --- Initialize Chroma DB with OpenAI Embeddings ---
embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"  # Fast & low-cost
)
chroma_client = chromadb.Client()

# Create collections for shared + role-specific memory
collections = {
    "shared": chroma_client.create_collection(
        name="shared", embedding_function=embedder, get_or_create=True
    ),
    "personal_assistant": chroma_client.create_collection(
        name="personal_assistant", embedding_function=embedder, get_or_create=True
    ),
    "writing_review": chroma_client.create_collection(
        name="writing_review", embedding_function=embedder, get_or_create=True
    )
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
    """Save a memory item to ChromaDB."""
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    try:
        mem_id = str(uuid.uuid4())
        collections[item.role].add(
            documents=[item.content],
            metadatas=[{
                "user_id": item.user_id,
                "tags": ",".join(item.tags),  # store tags as comma-separated string
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[mem_id]
        )
        return {"status": "success", "memory_id": mem_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving memory: {e}")

@app.post("/memory/query")
def query_memory(item: QueryItem):
    """Query stored memories for relevant information."""
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
    """Update an existing memory entry."""
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    try:
        collections[item.role].update(
            ids=[item.memory_id],
            documents=[item.new_content],
            metadatas=[{"updated_at": datetime.utcnow().isoformat()}]
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating memory: {e}")

@app.delete("/memory/delete/{role}/{memory_id}")
def delete_memory(role: str, memory_id: str):
    """Delete a memory entry."""
    if role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    try:
        collections[role].delete(ids=[memory_id])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting memory: {e}")

@app.post("/memory/auto-query")
def auto_query(item: QueryItem):
    """
    Automatically search both 'shared' and role-specific memory for relevant context.
    """
    try:
        all_results = []
        for role_key in ["shared", item.role]:
            if role_key in collections:
                result = collections[role_key].query(
                    query_texts=[item.query],
                    n_results=item.top_k
                )
                if result and result.get("documents") and result["documents"][0]:
                    all_results.extend(result["documents"][0])
        return {"matches": all_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in auto-query: {e}")

