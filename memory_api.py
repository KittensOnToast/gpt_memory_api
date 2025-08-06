# memory_api.py
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from datetime import datetime
import uuid
import chromadb
from chromadb.utils import embedding_functions
import os

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# --- Initialize Chroma DB ---
chroma_client = chromadb.Client()
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

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
    role: str  # "shared", "personal_assistant", "writing_review"
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

    try:
        mem_id = str(uuid.uuid4())
        collections[item.role].add(
            documents=[item.content],
            metadatas=[{
                "user_id": item.user_id,
                "tags": ",".join(item.tags),  # Store tags as a comma-separated string
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[mem_id]
        )
        return {"status": "success", "memory_id": mem_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving memory: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Error querying memory: {str(e)}")


@app.post("/memory/update")
def update_memory(item: UpdateItem):
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
        raise HTTPException(status_code=500, detail=f"Error updating memory: {str(e)}")


@app.delete("/memory/delete/{role}/{memory_id}")
def delete_memory(role: str, memory_id: str):
    if role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")

    try:
        collections[role].delete(ids=[memory_id])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting memory: {str(e)}")


# --- NEW: Auto Query Endpoint ---
@app.post("/memory/auto-query")
def auto_query_memory(
    user_id: str = Body(...),
    query: str = Body(...)
):
    try:
        all_results = []
        for role in ["shared", "personal_assistant", "writing_review"]:
            results = collections[role].query(
                query_texts=[query],
                n_results=1
            )
            if results["documents"] and results["documents"][0]:
                all_results.append({
                    "role": role,
                    "document": results["documents"][0][0],
                    "metadata": results["metadatas"][0][0]
                })

        if all_results:
            best_result = sorted(
                all_results,
                key=lambda x: ["shared", "personal_assistant", "writing_review"].index(x["role"])
            )[0]
            return {
                "match_found": True,
                "best_role": best_result["role"],
                "content": best_result["document"]
            }

        return {"match_found": False, "content": None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error auto-querying memory: {str(e)}")

