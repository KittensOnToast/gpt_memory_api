# memory_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uuid
import os
import chromadb
from chromadb.utils import embedding_functions

# Load OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

# Initialize embedding and Chroma client
embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)
chroma_client = chromadb.Client()

# Create or get collections
collections = {
    "shared": chroma_client.get_or_create_collection("shared", embedding_function=embedder),
    "personal_assistant": chroma_client.get_or_create_collection("personal_assistant", embedding_function=embedder),
    "writing_review": chroma_client.get_or_create_collection("writing_review", embedding_function=embedder),
    "goals": chroma_client.get_or_create_collection("goals", embedding_function=embedder),
    "feedback": chroma_client.get_or_create_collection("feedback", embedding_function=embedder)
}

# FastAPI app
app = FastAPI(title="Plan C GPT Memory API")

# Pydantic Models
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
    memory_id: str
    role: str
    user_id: str
    feedback_type: str  # "positive" or "negative"
    feedback_text: str

class TagSearchItem(BaseModel):
    role: str
    tags: list[str]
    top_k: int = 3

class GoalItem(BaseModel):
    user_id: str
    goal: str

# Endpoints
@app.post("/memory/save")
def save_memory(item: MemoryItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
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

@app.post("/memory/query")
def query_memory(item: QueryItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    results = collections[item.role].query(
        query_texts=[item.query],
        n_results=item.top_k
    )
    return {"matches": results}

@app.post("/memory/update")
def update_memory(item: UpdateItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    collections[item.role].update(
        ids=[item.memory_id],
        documents=[item.new_content],
        metadatas=[{"updated_at": datetime.utcnow().isoformat()}]
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

@app.post("/memory/feedback")
def save_feedback(item: FeedbackItem):
    fb_id = str(uuid.uuid4())
    collections["feedback"].add(
        documents=[item.feedback_text],
        metadatas=[{
            "user_id": item.user_id,
            "memory_id": item.memory_id,
            "feedback": item.feedback_type,
            "created_at": datetime.utcnow().isoformat()
        }],
        ids=[fb_id]
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
    try:
        goal_id = str(uuid.uuid4())
        collections["goals"].add(
            documents=[goal.goal],
            metadatas=[{
                "user_id": goal.user_id,
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[goal_id]
        )
        return {"status": "success", "goal_id": goal_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving goal: {e}")

@app.get("/memory/goals")
def list_goals():
    return collections["goals"].get()

@app.delete("/memory/goals/{goal_id}")
def delete_goal(goal_id: str):
    collections["goals"].delete(ids=[goal_id])
    return {"status": "success"}

