from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uuid
import os
import openai
import chromadb

# --- Setup OpenAI ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to get embeddings from OpenAI
def get_embedding(text: str) -> list[float]:
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

# --- Initialize Chroma DB ---
chroma_client = chromadb.Client()

# Create collections for shared + role-specific memory
collections = {
    "shared": chroma_client.get_or_create_collection(name="shared"),
    "personal_assistant": chroma_client.get_or_create_collection(name="personal_assistant"),
    "writing_review": chroma_client.get_or_create_collection(name="writing_review")
}

# --- API Setup ---
app = FastAPI(title="Custom GPT Memory API")

# --- Data Models ---
class MemoryItem(BaseModel):
    user_id: str
    role: str
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

# --- Endpoints ---
@app.post("/memory/save")
def save_memory(item: MemoryItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")

    mem_id = str(uuid.uuid4())
    try:
        embedding = get_embedding(item.content)
        collections[item.role].add(
            documents=[item.content],
            embeddings=[embedding],
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
        embedding = get_embedding(item.query)
        results = collections[item.role].query(
            query_embeddings=[embedding],
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
            metadatas=[{"updated_at": datetime.utcnow().isoformat()}]
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


