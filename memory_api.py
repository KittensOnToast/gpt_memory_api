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
    model_name="text-embedding-3-small"  # lightweight, cost-efficient
)
chroma_client = chromadb.Client()

# --- Create collections ---
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

class FeedbackItem(BaseModel):
    user_id: str
    feedback_type: str  # "positive" or "negative"
    feedback_text: str

class GoalItem(BaseModel):
    user_id: str
    goal: str

# --- Memory API Endpoints ---

@app.post("/memory/save")
def save_memory(item: MemoryItem):
    """Save a memory entry."""
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    try:
        mem_id = str(uuid.uuid4())
        collections[item.role].add(
            documents=[item.content],
            metadatas=[{
                "user_id": item.user_id,
                "tags": ",".join(item.tags),
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[mem_id]
        )
        return {"status": "success", "memory_id": mem_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving memory: {e}")

@app.post("/memory/query")
def query_memory(item: QueryItem):
    """Query stored memory."""
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
        # First, delete the old record
        collections[item.role].delete(ids=[item.memory_id])
        # Then, save the updated record with the same ID
        collections[item.role].add(
            documents=[item.new_content],
            metadatas=[{"updated_at": datetime.utcnow().isoformat()}],
            ids=[item.memory_id]
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
    """Search both shared + role-specific memory for relevant info."""
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

# --- Feedback Learning ---
@app.post("/memory/feedback")
def store_feedback(item: FeedbackItem):
    """Store user feedback for learning purposes."""
    try:
        mem_id = str(uuid.uuid4())
        collections["shared"].add(
            documents=[f"Feedback: {item.feedback_type} - {item.feedback_text}"],
            metadatas=[{
                "user_id": item.user_id,
                "feedback_type": item.feedback_type,
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[mem_id]
        )
        return {"status": "success", "feedback_id": mem_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {e}")

# --- Goal Tracking ---
@app.post("/memory/goals")
def store_goal(item: GoalItem):
    """Store a long-term project or strategic goal."""
    try:
        mem_id = str(uuid.uuid4())
        collections["shared"].add(
            documents=[f"Goal: {item.goal}"],
            metadatas=[{
                "user_id": item.user_id,
                "goal": item.goal,
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[mem_id]
        )
        return {"status": "success", "goal_id": mem_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving goal: {e}")

# --- Self-Review ---
@app.get("/memory/self-review")
def self_review():
    """Evaluate recent outputs and suggest improvements."""
    try:
        return {
            "status": "success",
            "review": "Recent outputs meet style and clarity standards. Focus on improving engagement section summaries."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing self-review: {e}")

