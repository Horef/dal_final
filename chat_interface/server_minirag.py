from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import json, asyncio

# TODO: import & init your MiniRAG object here (index path, model name, etc.)
# from minirag import MiniRAG
# rag = MiniRAG(index_dir="path/to/your/index", model_name="Qwen2.5-3B-Instruct")

app = FastAPI()

class Turn(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    query: str
    history: list[Turn] = []
    top_k: int = 4

def minirag_query(query: str, history: list[dict], top_k: int):
    # TODO: replace with the actual MiniRAG call
    # answer, sources = rag.query(query, history=history, top_k=top_k, stream=True)
    # yield {"delta": "..."} pieces then final {"done": True, "sources": [...]}
    # Demo fallback:
    yield {"delta": "Answer placeholder from MiniRAG. "}
    yield {"done": True, "sources": [{"title": "chunk_1", "snippet": "…"}]}

@app.post("/api/chat/stream")
async def chat_stream(req: ChatReq):
    async def gen():
        for piece in minirag_query(req.query, [t.dict() for t in req.history], req.top_k):
            await asyncio.sleep(0)  # keep responsive
            yield json.dumps(piece) + "\n"
    return StreamingResponse(gen(), media_type="text/plain")

@app.post("/api/chat")
async def chat(req: ChatReq):
    # Non-stream fallback (compose full text then return once)
    text = ""
    last_sources = []
    for piece in minirag_query(req.query, [t.dict() for t in req.history], req.top_k):
        if "delta" in piece:
            text += piece["delta"]
        if "sources" in piece:
            last_sources = piece["sources"]
    return JSONResponse({"answer": text, "sources": last_sources})
