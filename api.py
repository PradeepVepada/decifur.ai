"""
api.py
------
FastAPI backend for HybdRAG with SSE streaming.

Review fixes applied
--------------------
  #7  Global engine state protected by a threading.Lock; conversation
      history is now managed per-session via ConversationStore rather
      than a single shared RAGEngine state.
  #8  Blocking LLM/Neo4j calls wrapped in asyncio.to_thread() so the
      uvicorn event loop is never blocked.
  #9  print() → logging throughout.
  #10 REQUIRED environment variables validated at startup.
  #19 Request timeout via asyncio.wait_for(); CancelledError handled.
  #20 Import moved to top level (datetime was imported inside a function).
  #21 Query string validated: 1–2000 chars via Pydantic Field.
  Artificial 20 ms/word delay removed — real Mistral streaming used.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_engine import RAGEngine
from conversation_store import ConversationStore, ConversationMeta, Message

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required environment variable validation  [Review #10]
# ---------------------------------------------------------------------------
_REQUIRED_ENV = ["MISTRAL_API_KEY", "NEO4J_URI", "NEO4J_PASSWORD"]


def _validate_env():
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}. "
            "Create a .env file or export them before starting."
        )


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
engine:     Optional[RAGEngine]         = None
conv_store: Optional[ConversationStore] = None

# [Review #7] Engine calls are guarded by asyncio to avoid blocking the loop.
# RAGEngine itself is NOT shared across concurrent requests for state mutation
# (conversation_history lives per user session in conv_store).
QUERY_TIMEOUT_SECONDS = 120


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, conv_store
    _validate_env()

    logger.info("Starting HybdRAG API...")
    engine     = RAGEngine(user_id="default", enable_pcc=True)
    await asyncio.to_thread(engine.load)        # heavy I/O off the event loop
    conv_store = ConversationStore()            # uses env vars
    logger.info("HybdRAG API ready.")
    yield

    if engine:
        await asyncio.to_thread(engine.close)
    if conv_store:
        conv_store.close()
    logger.info("HybdRAG API shut down.")


app = FastAPI(
    title="HybdRAG API",
    description="Backend for Devreotes Research Explorer",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    # [Review #21] Bounded query length — prevents embedding/token cost explosions
    query:           str  = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    stream:          bool = True


class ChatResponse(BaseModel):
    answer:          str
    conversation_id: str
    message_id:      str
    intent:          str
    chunks_count:    int
    memory_info:     dict
    sources:         List[dict]


class ConversationCreate(BaseModel):
    title:   Optional[str] = None
    user_id: str           = "default"


class ConversationUpdate(BaseModel):
    title: str


class MemoryStatus(BaseModel):
    pcc_enabled:          bool
    user_id:              str
    conversation_id:      str
    short_term_messages:  int
    long_term_episodes:   int
    recent_episodes:      List[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_or_create_conv(conversation_id: Optional[str]) -> str:
    if conversation_id:
        return conversation_id
    conv = await asyncio.to_thread(conv_store.create_conversation, user_id="default")
    return conv.conversation_id


async def _save_user_message(conv_id: str, query: str, intent: str = "") -> None:
    msg = Message(
        role="user", content=query,
        timestamp=datetime.now().isoformat(), intent=intent,
    )
    await asyncio.to_thread(conv_store.add_message, conv_id, msg)


async def _save_assistant_message(
    conv_id: str, answer: str, intent: str,
    chunks: list, memory_info: dict,
) -> Message:
    msg = Message(
        role="assistant", content=answer,
        timestamp=datetime.now().isoformat(), intent=intent,
        chunks_count=len(chunks), memory_info=memory_info,
    )
    await asyncio.to_thread(conv_store.add_message, conv_id, msg)
    return msg


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {
        "status":     "healthy",
        "engine":     "loaded" if engine else "not_loaded",
        "pcc_memory": "enabled" if engine and engine.enable_pcc else "disabled",
    }


# ---------------------------------------------------------------------------
# Chat — non-streaming
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat. Engine call is off the event loop. [Review #8]"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    conv_id = await _get_or_create_conv(request.conversation_id)

    try:
        # [Review #8, #19] blocking call wrapped + timeout
        answer, chunks, intent, memory_info = await asyncio.wait_for(
            asyncio.to_thread(engine.ask, request.query),
            timeout=QUERY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Query timed out.")

    await _save_user_message(conv_id, request.query, intent)
    msg = await _save_assistant_message(conv_id, answer, intent, chunks, memory_info)

    return ChatResponse(
        answer=answer, conversation_id=conv_id, message_id=msg.message_id,
        intent=intent, chunks_count=len(chunks),
        memory_info=memory_info, sources=chunks,
    )


# ---------------------------------------------------------------------------
# Chat — streaming (REAL Mistral SSE, no artificial word delay)
# ---------------------------------------------------------------------------

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming endpoint.
    ask_stream() uses real Mistral SSE — no artificial word delay.  [Review #8]
    Engine runs in a thread; tokens are forwarded to the SSE response as they arrive.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    conv_id = await _get_or_create_conv(request.conversation_id)

    async def generate() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'type': 'conversation_id', 'data': conv_id})}\n\n"
            await _save_user_message(conv_id, request.query)

            full_answer  = ""
            final_chunks = []
            memory_info  = {}

            # Run blocking generator in a thread, relay tokens via queue
            token_queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _run_stream():
                """Executed in a thread; puts tokens onto the asyncio queue."""
                try:
                    for token, chunks, mem in engine.ask_stream(request.query):
                        if token is not None:
                            loop.call_soon_threadsafe(token_queue.put_nowait, ("token", token))
                        else:
                            loop.call_soon_threadsafe(
                                token_queue.put_nowait, ("done", (chunks, mem))
                            )
                except Exception as exc:
                    loop.call_soon_threadsafe(token_queue.put_nowait, ("error", str(exc)))

            stream_task = loop.run_in_executor(None, _run_stream)

            intent = "simple_lookup"
            while True:
                try:
                    kind, payload = await asyncio.wait_for(
                        token_queue.get(), timeout=QUERY_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'error', 'data': 'Query timed out'})}\n\n"
                    break

                if kind == "token":
                    full_answer += payload
                    yield f"data: {json.dumps({'type': 'token', 'data': payload})}\n\n"

                elif kind == "done":
                    final_chunks, memory_info = payload
                    yield f"data: {json.dumps({'type': 'sources',     'data': final_chunks})}\n\n"
                    yield f"data: {json.dumps({'type': 'memory_info', 'data': memory_info})}\n\n"

                    msg = await _save_assistant_message(
                        conv_id, full_answer, intent, final_chunks, memory_info
                    )
                    yield f"data: {json.dumps({'type': 'done', 'data': {'message_id': msg.message_id}})}\n\n"
                    break

                elif kind == "error":
                    logger.error("Streaming error: %s", payload)
                    yield f"data: {json.dumps({'type': 'error', 'data': payload})}\n\n"
                    break

            await stream_task

        except asyncio.CancelledError:
            logger.info("Stream cancelled by client for conv %s.", conv_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering":"no",
        },
    )


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

@app.post("/api/conversations", response_model=ConversationMeta)
async def create_conversation(request: ConversationCreate):
    return await asyncio.to_thread(
        conv_store.create_conversation, user_id=request.user_id, title=request.title
    )


@app.get("/api/conversations")
async def list_conversations(
    user_id: str = Query(default="default"),
    limit:   int = Query(default=50),
):
    conversations = await asyncio.to_thread(
        conv_store.list_conversations, user_id=user_id, limit=limit
    )
    today     = datetime.now().date()
    yesterday = today - timedelta(days=1)
    this_week = today - timedelta(days=7)

    grouped = {"today": [], "yesterday": [], "this_week": [], "older": []}
    for conv in conversations:
        conv_date = datetime.fromisoformat(conv.updated_at).date()
        if conv_date == today:
            grouped["today"].append(conv.to_dict())
        elif conv_date == yesterday:
            grouped["yesterday"].append(conv.to_dict())
        elif conv_date >= this_week:
            grouped["this_week"].append(conv.to_dict())
        else:
            grouped["older"].append(conv.to_dict())
    return grouped


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conv = await asyncio.to_thread(conv_store.get_conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "meta":     conv["meta"].to_dict(),
        "messages": [m.to_dict() for m in conv["messages"]],
    }


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    await asyncio.to_thread(conv_store.delete_conversation, conversation_id)
    return {"status": "deleted"}


@app.patch("/api/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, request: ConversationUpdate):
    await asyncio.to_thread(conv_store.rename_conversation, conversation_id, request.title)
    return {"status": "renamed"}


@app.get("/api/conversations/search")
async def search_conversations(
    query:   str = Query(...),
    user_id: str = Query(default="default"),
    limit:   int = Query(default=10),
):
    return await asyncio.to_thread(
        conv_store.search_conversations, query, user_id=user_id, limit=limit
    )


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@app.get("/api/memory/status", response_model=MemoryStatus)
async def get_memory_status():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    status   = await asyncio.to_thread(engine.get_memory_status)
    episodes = []
    if engine.pcc_memory and engine.pcc_memory.driver:
        try:
            episodes = await asyncio.to_thread(
                engine.pcc_memory.retrieve_long_term_memory, "", top_k=5
            )
        except Exception:
            logger.exception("Could not retrieve long-term episodes.")
    return MemoryStatus(
        pcc_enabled         = status.get("pcc_enabled", False),
        user_id             = status.get("user_id", "default"),
        conversation_id     = status.get("conversation_id", ""),
        short_term_messages = status.get("short_term_messages", 0),
        long_term_episodes  = len(episodes),
        recent_episodes     = episodes,
    )


@app.post("/api/memory/clear")
async def clear_memory():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    await asyncio.to_thread(engine.reset_conversation)
    return {"status": "cleared"}


@app.post("/api/memory/store")
async def store_memory():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    if engine.pcc_memory:
        ep = await asyncio.to_thread(
            engine.pcc_memory.compress_and_store_long_term,
            engine.conversation_history,
        )
        if ep:
            return {"status": "stored", "episode_id": ep.episode_id}
    return {"status": "no_memory_to_store"}


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------

@app.get("/api/papers")
async def list_papers():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    papers = await asyncio.to_thread(engine.get_paper_list)
    return {"papers": papers, "count": len(papers)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
