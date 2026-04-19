"""
api.py
------
FastAPI backend for HybdRAG with session-aware memory hydration.
"""

import os
import re
import json
import time
import logging
import asyncio
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Load .env before rag_engine — model_config reads OPENAI_API_KEY at import time.
load_dotenv(Path(__file__).resolve().parent / ".env")

from rag_engine import RAGEngine
from conversation_store import ConversationStore, ConversationMeta, Message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_REQUIRED_ENV = ["OLLAMA_BASE_URL", "NEO4J_URI", "NEO4J_PASSWORD"]


def _validate_env():
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}. "
            "Create a .env file or export them before starting."
        )


engine: Optional[RAGEngine] = None
conv_store: Optional[ConversationStore] = None
engine_lock: asyncio.Lock | None = None

QUERY_TIMEOUT_SECONDS = 120

ingest_mutex = threading.Lock()
ingest_status: dict = {"state": "idle", "message": "", "detail": None}


def _ingest_worker(tmp_path: str) -> None:
    global ingest_status, engine
    try:
        with ingest_mutex:
            ingest_status = {
                "state": "running",
                "message": "Extracting text and updating Neo4j…",
                "detail": None,
            }
        if not engine:
            raise RuntimeError("Engine not initialized")
        res = engine.ingest_pdf_path(tmp_path, copy_to_pdf_dir=True)
        with ingest_mutex:
            if res.get("ok"):
                ingest_status = {
                    "state": "done",
                    "message": f"Indexed {res.get('paper', 'paper')} ({res.get('chunks', 0)} chunks).",
                    "detail": res,
                }
            else:
                ingest_status = {
                    "state": "error",
                    "message": res.get("error") or "Ingest failed",
                    "detail": res,
                }
    except Exception as e:
        logger.exception("Background ingest failed.")
        with ingest_mutex:
            ingest_status = {"state": "error", "message": str(e), "detail": None}
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, conv_store, engine_lock

    _validate_env()

    logger.info("Starting HybdRAG API...")
    engine_lock = asyncio.Lock()
    _eng = RAGEngine(user_id="default", enable_pcc=True)
    try:
        await asyncio.to_thread(_eng.load)
        engine = _eng
    except Exception:
        logger.exception(
            "RAG engine failed to load — is Neo4j running at NEO4J_URI? "
            "Stub auth still works; corpus/chat return 503 until the graph loads."
        )
        try:
            await asyncio.to_thread(_eng.close)
        except Exception:
            logger.exception("Engine cleanup after failed load failed.")
        engine = None

    conv_store = ConversationStore()
    from model_config import (
        DEEP_REASONING_BACKEND,
        OPENAI_API_KEY,
        OLLAMA_MODEL,
        get_ollama_base_url,
    )

    _deep_openai = (
        DEEP_REASONING_BACKEND == "openai" and bool((OPENAI_API_KEY or "").strip())
    )
    logger.info(
        "RAG LLM routing: rewrite/retrieve/embeddings are local+Neo4j; chat completions use "
        "Ollama at %s (model=%s). Deep intents (compare across papers / topic evolution) use %s.",
        get_ollama_base_url(),
        OLLAMA_MODEL,
        f"OpenAI ({os.environ.get('OPENAI_DEEP_MODEL', 'gpt-4o')})"
        if _deep_openai
        else "the same Ollama endpoint",
    )
    if _deep_openai:
        logger.info(
            "To force those deep queries onto RunPod GPU as well, set DEEP_REASONING_BACKEND=ollama in .env and restart."
        )
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
    version="4.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    # Next dev often binds :3001+ when :3000 is taken; LAN testing uses host IPs.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d{1,5})?$|https?://\d{1,3}(\.\d{1,3}){3}(:\d{1,5})?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent


class SignInRequest(BaseModel):
    email: str = Field(default="", max_length=320)
    password: str = Field(default="", max_length=500)


class SignUpRequest(BaseModel):
    name: str = Field(default="", max_length=200)
    email: str = Field(default="", max_length=320)
    password: str = Field(default="", max_length=500)


@app.post("/api/auth/signin")
async def auth_signin(req: SignInRequest):
    """
    Minimal stub for the Next.js shell. Any email/password returns a session payload.
    Replace with real authentication before any production deployment.
    """
    email = (req.email or "").strip() or "researcher@local.dev"
    label = (email.split("@")[0] if "@" in email else email) or "Researcher"
    return {"token": "novaai-local-dev", "user": {"name": label, "email": email}}


@app.post("/api/auth/signup")
async def auth_signup(req: SignUpRequest):
    """Stub signup — same behavior as signin for local development."""
    email = (req.email or "").strip() or "researcher@local.dev"
    name = (req.name or "").strip() or (email.split("@")[0] if "@" in email else "Researcher")
    return {"token": "novaai-local-dev", "user": {"name": name, "email": email}}


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    stream: bool = True
    mode: str = Field(default="corpus", description='Use "web" for DuckDuckGo + OpenAI (requires OPENAI_API_KEY).')
    corpus_generation_model: Optional[str] = Field(
        default=None,
        description='Optional corpus answer LLM. Use "gpt-5-nano" or "gpt-4o-mini" (requires OPENAI_API_KEY); omit for default Ollama.',
    )


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    message_id: str
    intent: str
    chunks_count: int
    memory_info: dict
    sources: List[dict]


class ConversationCreate(BaseModel):
    title: Optional[str] = None
    user_id: str = "default"


class ConversationUpdate(BaseModel):
    title: str


class ArchiveSave(BaseModel):
    """Persist a transcript to the same SQLite store the Streamlit UI uses."""

    messages: list
    source_history: list = Field(default_factory=list)


class MemoryStatus(BaseModel):
    pcc_enabled: bool
    user_id: str
    conversation_id: str
    short_term_messages: int
    long_term_episodes: int
    recent_episodes: List[dict]


async def _get_or_create_conv(conversation_id: Optional[str]) -> str:
    if conversation_id:
        return conversation_id
    conv = await asyncio.to_thread(conv_store.create_conversation, user_id="default")
    return conv.conversation_id


async def _save_user_message(conv_id: str, query: str, intent: str = "") -> None:
    msg = Message(
        role="user",
        content=query,
        timestamp=datetime.now().isoformat(),
        intent=intent,
    )
    await asyncio.to_thread(conv_store.add_message, conv_id, msg)


async def _save_assistant_message(
    conv_id: str,
    answer: str,
    intent: str,
    chunks: list,
    memory_info: dict,
) -> Message:
    msg = Message(
        role="assistant",
        content=answer,
        timestamp=datetime.now().isoformat(),
        intent=intent,
        chunks_count=len(chunks),
        memory_info=memory_info,
    )
    try:
        await asyncio.to_thread(conv_store.add_message, conv_id, msg)
    except Exception as e:
        logger.warning(f"Failed to persist assistant message to Neo4j: {e}")
    return msg


async def _prepare_engine_for_conversation(conv_id: str) -> None:
    """
    Align the shared runtime engine state with the requested conversation.
    """
    engine.set_conversation_id(conv_id)

    conv = await asyncio.to_thread(conv_store.get_conversation, conv_id)
    if not conv:
        engine.hydrate_memory_from_messages([])
        return

    messages = [m.to_dict() for m in conv["messages"]]
    engine.hydrate_memory_from_messages(messages)


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy" if engine else "degraded",
        "engine": "loaded" if engine else "not_loaded",
        "pcc_memory": "enabled" if engine and engine.enable_pcc else "disabled",
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    conv_id = await _get_or_create_conv(request.conversation_id)

    try:
        async with engine_lock:
            await _prepare_engine_for_conversation(conv_id)

            answer, chunks, intent, memory_info = await asyncio.wait_for(
                asyncio.to_thread(
                    engine.ask,
                    request.query,
                    True,
                    request.corpus_generation_model,
                ),
                timeout=QUERY_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Query timed out.")

    await _save_user_message(conv_id, request.query, intent)
    msg = await _save_assistant_message(conv_id, answer, intent, chunks, memory_info)

    return ChatResponse(
        answer=answer,
        conversation_id=conv_id,
        message_id=msg.message_id,
        intent=intent,
        chunks_count=len(chunks),
        memory_info=memory_info,
        sources=chunks,
    )


async def _sse_consume_token_queue(
    token_queue: asyncio.Queue,
    conv_id: str,
    user_query: str,
    intent: str,
    stream_task,
) -> AsyncGenerator[str, None]:
    """Shared SSE loop for corpus RAG and web search streams."""
    full_answer = ""
    final_chunks: list = []
    memory_info: dict = {}
    while True:
        try:
            kind, payload = await asyncio.wait_for(
                token_queue.get(),
                timeout=QUERY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'data': 'Query timed out'})}\n\n"
            break

        if kind == "token":
            tok, meta = payload
            if isinstance(meta, dict) and meta.get("replace_full"):
                full_answer = tok
            else:
                full_answer += tok
            yield f"data: {json.dumps({'type': 'token', 'data': tok, 'replace_full': bool(isinstance(meta, dict) and meta.get('replace_full'))})}\n\n"

        elif kind == "done":
            final_chunks, memory_info = payload
            yield f"data: {json.dumps({'type': 'sources', 'data': final_chunks or []})}\n\n"
            yield f"data: {json.dumps({'type': 'memory_info', 'data': memory_info or {}})}\n\n"

            await _save_user_message(conv_id, user_query, intent)
            msg = await _save_assistant_message(
                conv_id, full_answer, intent, final_chunks, memory_info
            )

            yield f"data: {json.dumps({'type': 'done', 'data': {'message_id': msg.message_id}})}\n\n"
            break

        elif kind == "error":
            logger.error("Streaming error: %s", payload)
            yield f"data: {json.dumps({'type': 'error', 'data': payload})}\n\n"
            break

        elif kind == "ping":
            # ask_stream heartbeats during Neo4j/embed prep — keeps SSE alive and resets wait timeouts.
            yield f"data: {json.dumps({'type': 'status', 'data': 'retrieving'})}\n\n"

    await stream_task


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    conv_id = await _get_or_create_conv(request.conversation_id)

    async def generate() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'type': 'conversation_id', 'data': conv_id})}\n\n"
            mode = (request.mode or "corpus").strip().lower()

            if mode == "web":
                oa_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
                if not oa_key:
                    yield f"data: {json.dumps({'type': 'error', 'data': 'Web mode requires OPENAI_API_KEY in .env'})}\n\n"
                    return
                async with engine_lock:
                    token_queue: asyncio.Queue = asyncio.Queue()
                    loop = asyncio.get_event_loop()

                    def _run_web_stream():
                        try:
                            from openai import OpenAI
                            from web_search import web_search_answer_stream

                            client = OpenAI(api_key=oa_key)
                            for token, chunks, mem in web_search_answer_stream(
                                request.query, client
                            ):
                                if token is not None:
                                    loop.call_soon_threadsafe(
                                        token_queue.put_nowait,
                                        ("token", (token, mem or {})),
                                    )
                                elif isinstance(chunks, list):
                                    loop.call_soon_threadsafe(
                                        token_queue.put_nowait, ("done", (chunks, mem))
                                    )
                        except Exception as exc:
                            loop.call_soon_threadsafe(token_queue.put_nowait, ("error", str(exc)))

                    stream_task = loop.run_in_executor(None, _run_web_stream)
                    async for chunk in _sse_consume_token_queue(
                        token_queue,
                        conv_id,
                        request.query,
                        "web_search",
                        stream_task,
                    ):
                        yield chunk
                return

            async with engine_lock:
                await _prepare_engine_for_conversation(conv_id)

                cm = (request.corpus_generation_model or "").strip().lower()
                if cm in ("gpt-4o-mini", "gpt-5-nano") and not (os.environ.get("OPENAI_API_KEY") or "").strip():
                    yield f"data: {json.dumps({'type': 'error', 'data': 'OpenAI corpus mode requires OPENAI_API_KEY in .env'})}\n\n"
                    return

                intent = "simple_lookup"

                token_queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_event_loop()

                def _run_stream():
                    try:
                        for token, chunks, mem in engine.ask_stream(
                            request.query,
                            corpus_generation_model=request.corpus_generation_model,
                        ):
                            if token is not None:
                                loop.call_soon_threadsafe(
                                    token_queue.put_nowait, ("token", (token, mem))
                                )
                            elif isinstance(chunks, list):
                                # ask_stream terminal is (None, chunks, memory_info).
                                loop.call_soon_threadsafe(
                                    token_queue.put_nowait, ("done", (chunks, mem))
                                )
                            else:
                                # Heartbeat (None, None, None) while corpus prep or slow Ollama stream pump.
                                loop.call_soon_threadsafe(token_queue.put_nowait, ("ping", None))
                    except Exception as exc:
                        loop.call_soon_threadsafe(token_queue.put_nowait, ("error", str(exc)))

                stream_task = loop.run_in_executor(None, _run_stream)
                async for chunk in _sse_consume_token_queue(
                    token_queue,
                    conv_id,
                    request.query,
                    intent,
                    stream_task,
                ):
                    yield chunk

        except asyncio.CancelledError:
            logger.info("Stream cancelled by client for conv %s.", conv_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/conversations", response_model=ConversationMeta)
async def create_conversation(request: ConversationCreate):
    return await asyncio.to_thread(
        conv_store.create_conversation,
        user_id=request.user_id,
        title=request.title,
    )


@app.get("/api/conversations")
async def list_conversations(
    user_id: str = Query(default="default"),
    limit: int = Query(default=50),
):
    conversations = await asyncio.to_thread(
        conv_store.list_conversations, user_id=user_id, limit=limit
    )

    today = datetime.now().date()
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
        "meta": conv["meta"].to_dict(),
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
    query: str = Query(...),
    user_id: str = Query(default="default"),
    limit: int = Query(default=10),
):
    return await asyncio.to_thread(
        conv_store.search_conversations, query, user_id=user_id, limit=limit
    )


@app.get("/api/memory/status", response_model=MemoryStatus)
async def get_memory_status():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    async with engine_lock:
        status = await asyncio.to_thread(engine.get_memory_status)

        episodes = []
        if engine.pcc_memory and engine.pcc_memory.driver:
            try:
                episodes = await asyncio.to_thread(
                    engine.pcc_memory.retrieve_long_term_memory, "", top_k=5
                )
            except Exception:
                logger.exception("Could not retrieve long-term episodes.")

    return MemoryStatus(
        pcc_enabled=status.get("pcc_enabled", False),
        user_id=status.get("user_id", "default"),
        conversation_id=status.get("conversation_id", ""),
        short_term_messages=status.get("short_term_messages", 0),
        long_term_episodes=len(episodes),
        recent_episodes=episodes,
    )


@app.post("/api/memory/clear")
async def clear_memory():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    async with engine_lock:
        await asyncio.to_thread(engine.reset_conversation)

    return {"status": "cleared"}


@app.post("/api/memory/store")
async def store_memory():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    async with engine_lock:
        if engine.pcc_memory:
            ep = await asyncio.to_thread(
                engine.pcc_memory.compress_and_store_long_term,
                engine.conversation_history,
            )
            if ep:
                return {"status": "stored", "episode_id": ep.episode_id}

    return {"status": "no_memory_to_store"}


@app.get("/api/papers")
async def list_papers():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    papers = await asyncio.to_thread(engine.get_paper_list)
    return {"papers": papers, "count": len(papers)}


@app.get("/api/ingest/status")
async def get_ingest_status():
    with ingest_mutex:
        return dict(ingest_status)


@app.post("/api/papers/upload")
async def upload_paper(request: Request):
    """
    Accept multipart PDF via `file` field. Uses `request.form()` instead of `UploadFile = File(...)`
    so failures surface as JSON `detail` instead of Starlette's plain-text 500 during DI parsing.
    """
    global ingest_status

    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        form = await request.form()
        file = form.get("file")
        if file is None:
            raise HTTPException(status_code=400, detail="Missing multipart field 'file'.")
        if isinstance(file, str):
            raise HTTPException(status_code=400, detail="Field 'file' must be a PDF file upload.")

        name = (getattr(file, "filename", None) or "").strip()
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Upload a single PDF file.")

        max_mb = float(os.environ.get("MAX_UPLOAD_MB", "50"))
        max_bytes = int(max_mb * 1024 * 1024)

        with ingest_mutex:
            if ingest_status.get("state") == "running":
                raise HTTPException(
                    status_code=409,
                    detail="Another paper is still being processed.",
                )

        content = await file.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size ({max_mb} MB).",
            )

        safe = re.sub(r"[^\w.\- ]", "_", Path(name).name, flags=re.UNICODE).strip() or "upload.pdf"
        if not safe.lower().endswith(".pdf"):
            safe += ".pdf"

        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(content)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        with ingest_mutex:
            ingest_status = {
                "state": "running",
                "message": "Queued for processing…",
                "detail": {"filename": safe},
            }

        threading.Thread(target=_ingest_worker, args=(tmp_path,), daemon=True).start()
        return {"ok": True, "filename": safe, "state": "running"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("upload_paper failed.")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/analytics/gene-authors")
async def analytics_gene_authors(gene: str = Query(..., min_length=1, max_length=200)):
    """
    Corpus graph rollup: distinct authors linked to papers whose chunks STUDIES_MOLECULE the gene.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    async with engine_lock:
        data = await asyncio.to_thread(engine.store.get_authors_for_molecule, gene.strip())
    return data


@app.get("/api/ui/config")
async def ui_config():
    """Lightweight labels for the React shell (no secrets)."""
    from urllib.parse import urlparse

    from model_config import (
        DEEP_REASONING_BACKEND,
        OPENAI_API_KEY,
        get_ollama_base_url,
    )

    url = get_ollama_base_url()
    host = urlparse(url).hostname or urlparse(url).netloc or ""
    oa = bool((OPENAI_API_KEY or "").strip())
    deep_on_openai = DEEP_REASONING_BACKEND == "openai" and oa
    return {
        "ollama_model": os.environ.get("OLLAMA_MODEL", "qwen35-biomedical"),
        "ollama_remote": "proxy.runpod.net" in url or url.startswith("https://"),
        "ollama_host": host,
        "neo4j_uri": os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687"),
        "deep_reasoning_backend": DEEP_REASONING_BACKEND,
        "deep_intents_use_openai": deep_on_openai,
        "retrieval_is_local": True,
        "runpod_note": (
            "Embeddings + vector search run on this machine (Neo4j); RunPod GPU is used only "
            "when Ollama chat is called. If deep questions never spike GPU, set "
            "DEEP_REASONING_BACKEND=ollama so cross-paper / topic-evolution also use RunPod."
            if deep_on_openai
            else "Embeddings + vector search are local; RunPod handles Ollama chat for corpus answers."
        ),
    }


def _ollama_health_sync() -> dict:
    """One minimal chat completion against OLLAMA_BASE_URL (proves traffic reaches RunPod)."""
    from urllib.parse import urlparse

    from openai import OpenAI

    from model_config import OLLAMA_MODEL, get_ollama_base_url

    base = get_ollama_base_url()
    host = urlparse(base).hostname or urlparse(base).netloc or ""
    model = (os.environ.get("OLLAMA_MODEL") or "").strip() or OLLAMA_MODEL
    t0 = time.perf_counter()
    client = OpenAI(base_url=f"{base}/v1", api_key="ollama", timeout=25.0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=4,
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()
    ms = int((time.perf_counter() - t0) * 1000)
    return {"ok": True, "host": host, "model": model, "reply": text[:120], "latency_ms": ms}


@app.get("/api/ui/ollama-health")
async def ui_ollama_health():
    """
    Ping the configured Ollama-compatible endpoint with a 1-token style completion.
    Use this to confirm the RunPod proxy receives HTTP (GPU may still idle until a real decode runs).
    """
    try:
        return await asyncio.to_thread(_ollama_health_sync)
    except Exception as exc:
        from urllib.parse import urlparse

        from model_config import OLLAMA_MODEL, get_ollama_base_url

        base = get_ollama_base_url()
        host = urlparse(base).hostname or urlparse(base).netloc or ""
        model = (os.environ.get("OLLAMA_MODEL") or "").strip() or OLLAMA_MODEL
        return {
            "ok": False,
            "host": host,
            "model": model,
            "error": str(exc)[:800],
            "latency_ms": None,
        }


@app.get("/api/ui/archive")
async def ui_archive_list():
    """Sidebar \"Past conversations\" — same buckets as `streamlit_archive`."""
    from streamlit_archive import BUCKET_ORDER, list_bucketed

    buckets = await asyncio.to_thread(list_bucketed, PROJECT_ROOT)
    return {"buckets": [{"name": b, "items": buckets.get(b, [])} for b in BUCKET_ORDER]}


@app.get("/api/ui/archive/{archive_id}")
async def ui_archive_get(archive_id: str):
    from streamlit_archive import fetch_conversation

    row = await asyncio.to_thread(fetch_conversation, PROJECT_ROOT, archive_id)
    if not row:
        raise HTTPException(status_code=404, detail="Archive not found")
    messages, source_history = row
    return {"messages": messages, "source_history": source_history}


@app.post("/api/ui/archive")
async def ui_archive_save(body: ArchiveSave):
    from streamlit_archive import archive_conversation

    cid = await asyncio.to_thread(
        archive_conversation, PROJECT_ROOT, body.messages, body.source_history
    )
    return {"id": cid or None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")