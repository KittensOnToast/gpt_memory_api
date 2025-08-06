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
    model_name="text-embedding-3-small"  # Lightweight, cost-efficient
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
app = FastAPI(title="Custom GPT Memory API with Feedback Learning")

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
    memory_id: str
    role: str
    feedback: str  # "positive" or "negative"

class TagSearchItem(BaseModel):
    role: str
    tags: list[str]
    top_k: int = 3

class GoalItem(BaseModel):
    user_id: str
    content: str

# --- Helper: Get metadata with defaults ---
def default_metadata(user_id, tags):
    return {
        "user_id": user_id,
        "tags": ",".join(tags),
        "feedback_score": 0,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }

# --- Save Memory ---
@app.post("/memory/save")
def save_memory(item: MemoryItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    try:
        mem_id = str(uuid.uuid4())
        collections[item.role].add(
            documents=[item.content],
            metadatas=[default_metadata(item.user_id, item.tags)],
            ids=[mem_id]
        )
        return {"status": "success", "memory_id": mem_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving memory: {e}")

# --- Query Memory ---
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

# --- Update Memory ---
@app.post("/memory/update")
def update_memory(item: UpdateItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    try:
        updated_meta = {"updated_at": datetime.utcnow().isoformat()}
        collections[item.role].update(
            ids=[item.memory_id],
            documents=[item.new_content],
            metadatas=[updated_meta]
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating memory: {e}")

# --- Delete Memory ---
@app.delete("/memory/delete/{role}/{memory_id}")
def delete_memory(role: str, memory_id: str):
    if role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    try:
        collections[role].delete(ids=[memory_id])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting memory: {e}")

# --- Feedback Learning ---
@app.post("/memory/feedback")
def memory_feedback(item: FeedbackItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    try:
        existing = collections[item.role].get(ids=[item.memory_id])
        if not existing["ids"]:
            raise HTTPException(status_code=404, detail="Memory not found")
        meta = existing["metadatas"][0]
        score = int(meta.get("feedback_score", 0))
        if item.feedback == "positive":
            score = min(score + 1, 5)
        elif item.feedback == "negative":
            score = max(score - 1, -5)
        meta["feedback_score"] = score
        collections[item.role].update(
            ids=[item.memory_id],
            metadatas=[meta]
        )
        return {"status": "success", "new_feedback_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating feedback: {e}")

# --- Tag Search ---
@app.post("/memory/tag-search")
def tag_search(item: TagSearchItem):
    if item.role not in collections:
        raise HTTPException(status_code=400, detail="Invalid role")
    try:
        all_memories = collections[item.role].get()
        filtered = []
        for doc, meta, mid in zip(all_memories["documents"], all_memories["metadatas"], all_memories["ids"]):
            if all(tag in meta.get("tags", "") for tag in item.tags):
                filtered.append((doc, meta, mid))
        filtered.sort(key=lambda x: int(x[1].get("feedback_score", 0)), reverse=True)
        return {"matches": filtered[:item.top_k]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in tag search: {e}")

# --- Auto Query (Upgraded) ---
@app.post("/memory/auto-query")
def auto_query(item: QueryItem):
    try:
        combined_results = []
        for role_key in ["shared", item.role]:
            if role_key in collections:
                result = collections[role_key].query(
                    query_texts=[item.query],
                    n_results=item.top_k
                )
                for doc, meta, mid in zip(result["documents"][0], result["metadatas"][0], result["ids"][0]):
                    combined_results.append((doc, meta, mid))
        combined_results.sort(key=lambda x: int(x[1].get("feedback_score", 0)), reverse=True)
        return {"matches": combined_results[:item.top_k]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in auto-query: {e}")

# --- Self Review ---
@app.get("/memory/self-review")
def self_review():
    try:
        review_data = []
        for role_key, coll in collections.items():
            all_memories = coll.get()
            if not all_memories["ids"]:
                continue
            total = len(all_memories["ids"])
            avg_score = sum(int(m.get("feedback_score", 0)) for m in all_memories["metadatas"]) / total
            best = sorted(
                zip(all_memories["documents"], all_memories["metadatas"]),
                key=lambda x: int(x[1].get("feedback_score", 0)),
                reverse=True
            )[:3]
            worst = sorted(
                zip(all_memories["documents"], all_memories["metadatas"]),
                key=lambda x: int(x[1].get("feedback_score", 0))
            )[:3]
            review_data.append({
                "role": role_key,
                "total_memories": total,
                "average_feedback_score": avg_score,
                "best_rated": best,
                "worst_rated": worst
            })
        return {"self_review": review_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating self-review: {e}")

# --- Goals ---
@app.post("/memory/goals")
def save_goal(goal: GoalItem):
    try:
        mem_id = str(uuid.uuid4())
        meta = default_metadata(goal.user_id, ["goal"])
        collections["shared"].add(
            documents=[goal.content],
            metadatas=[meta],
            ids=[mem_id]
        )
        return {"status": "success", "goal_id": mem_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving goal: {e}")

@app.get("/memory/goals")
def get_goals():
    try:
        all_goals = collections["shared"].get()
        goals = []
        for doc, meta, mid in zip(all_goals["documents"], all_goals["metadatas"], all_goals["ids"]):
            if "goal" in meta.get("tags", ""):
                goals.append((doc, meta, mid))
        return {"goals": goals}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving goals: {e}")

@app.delete("/memory/goals/{goal_id}")
def delete_goal(goal_id: str):
    try:
        collections["shared"].delete(ids=[goal_id])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting goal: {e}")

