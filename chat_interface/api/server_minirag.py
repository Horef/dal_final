from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json, asyncio, os

app = FastAPI(title="MiniRAG Chat API")

# ----------------------------
# Wire THESE TWO functions to your MiniRAG
# ----------------------------
def retrieve_top_k(query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Return sources: [{title, snippet, score, url?, path?}, ...]
    TODO: Replace with your MiniRAG retrieval call (BM25/embeddings/hybrid).
    """
    # Example skeleton:
    # hits = my_minirag_retriever.search(query, top_k=top_k)
    # return [{"title": h.id, "snippet": h.text, "score": float(h.score)} for h in hits]
    return [{"title": "DEMO_CHUNK_1", "snippet": "<retrieved text>", "score": 0.87}][:top_k]

async def generate_stream(prompt: str):
    """
    Yield dicts like {"delta": "..."} as tokens arrive.
    TODO: Replace with your MiniRAG LLM stream (transformers, vLLM, Ollama, etc.)
    """
    for tok in ["This ", "is ", "a ", "placeholder."]:
        yield {"delta": tok}
        await asyncio.sleep(0.01)

# ----------------------------
# Minimal chat schemas
# ----------------------------
class Turn(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    query: str
    history: List[Turn] = []
    top_k: int = 4

def build_prompt(query: str, sources: List[Dict[str, Any]], history: List[Dict[str, str]]) -> str:
    ctx = "\n\n".join([f"[{i+1}] {s['snippet']}" for i, s in enumerate(sources)])
    hist = "\n".join([f"{t['role']}: {t['content']}" for t in history[-6:]])
    return (
        "You are MiniRAG. Answer concisely and cite sources as [#] when relevant.\n"
        f"History:\n{hist}\n\n"
        f"Context:\n{ctx}\n\n"
        f"User question:\n{query}\n"
    )

@app.post("/api/chat/stream")
async def chat_stream(req: ChatReq):
    sources = retrieve_top_k(req.query, req.top_k)
    prompt = build_prompt(req.query, sources, [t.dict() for t in req.history])

    async def gen():
        async for piece in generate_stream(prompt):
            if "delta" in piece:
                yield json.dumps({"delta": piece["delta"]}) + "\n"
        yield json.dumps({"done": True, "sources": sources}) + "\n"

    return StreamingResponse(gen(), media_type="text/plain")

@app.post("/api/chat")
async def chat(req: ChatReq):
    sources = retrieve_top_k(req.query, req.top_k)
    prompt = build_prompt(req.query, sources, [t.dict() for t in req.history])
    text = ""
    async for piece in generate_stream(prompt):
        text += piece.get("delta", "")
    return JSONResponse({"answer": text, "sources": sources})
