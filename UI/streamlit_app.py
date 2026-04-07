from __future__ import annotations

import os
import sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import time
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from rag_engine import RAGEngine, REFUSAL_MESSAGE
from streamlit_archive import (
    BUCKET_ORDER,
    archive_conversation,
    fetch_conversation,
    list_bucketed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="decifur.ai",
    page_icon=" ",
    layout="wide",
)

st.title("decifur.ai")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PREVIEW_LENGTH = 500
RATE_LIMIT_SECONDS = 2  # Minimum seconds between queries (spam guard)
QUERY_TIMEOUT_SECONDS = 60  # Timeout for streaming responses
MAX_QUERIES_PER_SESSION = 100  # Prevent excessive usage in single session

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "source_history" not in st.session_state:
    st.session_state.source_history = []

if "engine" not in st.session_state:
    st.session_state.engine = None

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if not (os.environ.get("MISTRAL_API_KEY") or "").strip():
    st.error("MISTRAL_API_KEY is required for chat and RAG (Mistral). Set it in your `.env`.")
    st.stop()

# ---------------------------------------------------------------------------
# Engine init — cached so it only loads once per server session
# ---------------------------------------------------------------------------
@st.cache_resource
def init_engine() -> RAGEngine:
    """Initialize the RAG engine with PCC memory."""
    engine = RAGEngine(user_id="default", enable_pcc=True)
    engine.load()
    return engine

# ---------------------------------------------------------------------------
# Engine initialization
# ---------------------------------------------------------------------------
try:
    # Use cached resource but always ensure it's in session state
    if not st.session_state.initialized:
        st.session_state.engine = init_engine()
        st.session_state.initialized = True
except Exception as e:
    st.error(f"Failed to initialize engine: {e}")
    st.info("Make sure Neo4j is running and accessible.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")

    st.header("Settings")
    show_sources = st.checkbox("Show sources", value=True)

    st.success("Mistral API configured (chat / RAG)")
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if openai_key:
        st.success("OpenAI API configured (for `evaluation/llm_judge.py` only)")
    else:
        st.info(
            "Optional: set OPENAI_API_KEY in `.env` if you run LLM-as-judge evaluation "
            "(`python evaluation/llm_judge.py`). Not required for this UI."
        )

    neo4j_uri = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
    st.info(f"Neo4j: {neo4j_uri}")

    st.markdown("---")

    if st.session_state.engine:
        papers = st.session_state.engine.get_paper_list()
        # MEDIUM PRIORITY: Enhanced sidebar metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Papers", len(papers))
        with col2:
            st.metric("Messages", len(st.session_state.messages) // 2)
        with col3:
            st.metric("Sources", len(st.session_state.source_history))
        
        # LOW PRIORITY: Add refresh button for corpus
        if st.button("🔄 Refresh Papers", help="Reload the papers corpus"):
            st.cache_resource.clear()
            st.session_state.initialized = False
            st.session_state.engine = None
            st.rerun()
    
    st.markdown("---")

    st.subheader("Past conversations")
    st.caption(
        "Saved locally when you start a new chat. PCC memory resets when you "
        "switch chats so follow-ups apply to the thread you have open."
    )
    _buckets = list_bucketed(PROJECT_ROOT)
    _any_archived = any(_buckets[b] for b in BUCKET_ORDER)
    if not _any_archived:
        st.caption("No archived chats yet — use **New conversation** to save the current one.")
    else:
        for _bucket in BUCKET_ORDER:
            _items = _buckets.get(_bucket) or []
            if not _items:
                continue
            with st.expander(f"{_bucket} ({len(_items)})", expanded=_bucket == "Today"):
                for _it in _items:
                    _when = time.strftime(
                        "%Y-%m-%d %H:%M",
                        time.localtime(_it["saved_at"]),
                    )
                    if st.button(
                        _it["title"],
                        key=f"load_arch_{_it['id']}",
                        help=f"Saved {_when}",
                    ):
                        _loaded = fetch_conversation(PROJECT_ROOT, _it["id"])
                        if _loaded:
                            _msgs, _srcs = _loaded
                            st.session_state.messages = _msgs
                            st.session_state.source_history = _srcs
                            st.session_state.engine.reset_conversation()
                            st.toast("Conversation loaded — PCC starts fresh for new messages.")
                            st.rerun()

    st.markdown("---")

    # Session usage stats
    st.subheader("Usage")
    st.write(f"**Queries this session:** {st.session_state.query_count}/{MAX_QUERIES_PER_SESSION}")
    if st.session_state.query_count > 0:
        st.write(f"**Last query:** {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_query_time))}")

    if st.button(
        "➕ New conversation",
        help="Archive this chat locally, reset PCC, and start an empty thread",
    ):
        if st.session_state.messages:
            archive_conversation(
                PROJECT_ROOT,
                st.session_state.messages,
                st.session_state.source_history,
            )
        st.session_state.engine.reset_conversation()
        st.session_state.messages = []
        st.session_state.source_history = []
        st.rerun()

    if st.button(
        "🗑️ Clear conversation",
        help="Discard the current thread without archiving; resets PCC",
    ):
        st.session_state.engine.reset_conversation()
        st.session_state.messages = []
        st.session_state.source_history = []
        st.session_state.query_count = 0
        st.session_state.last_query_time = 0
        st.toast("Conversation cleared.")
        st.rerun()


# ---------------------------------------------------------------------------
# Helper: render source expander for a set of chunks
# ---------------------------------------------------------------------------
def render_sources(chunks: list[dict], key_prefix: str) -> None:
    """
    Render sources in an expandable section.
    
    Args:
        chunks: List of chunk dictionaries with metadata
        key_prefix: Unique prefix for widget keys
    """
    if not chunks:
        return
    with st.expander(f"Sources ({len(chunks)} results)", expanded=False):
        for idx, chunk in enumerate(chunks, start=1):
            st.markdown(f"#### [{idx}] {chunk.get('title', 'Untitled')}")
            st.write(f"**Paper:** {chunk.get('source', '')}")
            st.write(f"**Year:** {chunk.get('year', '?')}")
            st.write(f"**Match Score:** {float(chunk.get('score', 0.0)):.4f}")
            text_preview = chunk.get("text", "")
            truncated = (
                text_preview[:PREVIEW_LENGTH] + 
                ("..." if len(text_preview) > PREVIEW_LENGTH else "")
            )
            st.text_area(
                f"Chunk {key_prefix}-{idx}",
                value=truncated,
                height=100,
                disabled=True,
                key=f"chunk_{key_prefix}_{idx}",
            )


# ---------------------------------------------------------------------------
# Helper: validate rate limit
# ---------------------------------------------------------------------------
def check_rate_limit() -> tuple[bool, Optional[str]]:
    """
    Check if user has exceeded rate limits.
    
    Returns:
        (is_allowed, error_message)
    """
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_query_time
    
    # Check time-based rate limit
    if time_since_last < RATE_LIMIT_SECONDS:
        remaining = int(RATE_LIMIT_SECONDS - time_since_last)
        return False, f"⏱️ Please wait {remaining} second(s) before submitting another query"
    
    # Check session query limit
    if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
        return False, f"🚫 Query limit ({MAX_QUERIES_PER_SESSION}) reached for this session. Refresh the page to reset."
    
    return True, None


# ---------------------------------------------------------------------------
# Display previous chat history
# ---------------------------------------------------------------------------
assistant_turn_idx = 0
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # FIX 2: Only increment counter for assistant messages to prevent index mismatch
        if (show_sources and 
            msg["role"] == "assistant" and 
            msg["content"] != REFUSAL_MESSAGE and
            assistant_turn_idx < len(st.session_state.source_history)):
            
            render_sources(
                st.session_state.source_history[assistant_turn_idx],
                key_prefix=str(assistant_turn_idx)
            )
        
        if msg["role"] == "assistant":
            assistant_turn_idx += 1

# ---------------------------------------------------------------------------
# Chat input + Streaming response
# ---------------------------------------------------------------------------
user_prompt = st.chat_input("Ask a question about Dr. Peter Devreotes research")

if user_prompt and user_prompt.strip():
    # FIX 3: Input validation — strip whitespace and validate non-empty
    user_prompt = user_prompt.strip()
    
    # FIX 4: Rate limiting — prevent spam queries and session abuse
    is_allowed, error_message = check_rate_limit()
    if not is_allowed:
        st.warning(error_message)
        st.stop()
    
    # Update query tracking
    current_time = time.time()
    st.session_state.last_query_time = current_time
    st.session_state.query_count += 1

    # Show user message immediately
    with st.chat_message("user"):
        st.write(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("Searching research papers and reasoning.....")
        full_response = ""
        final_chunks = []
        final_memory_info = {}
        start_time = time.time()

        try:
            # Stream tokens — UI updates as each token arrives.
            # ask_stream() yields (token, None, None) during generation,
            # then (None, chunks, memory_info) once done.
            for token, chunks, memory_info in st.session_state.engine.ask_stream(user_prompt):
                # MEDIUM PRIORITY: Timeout protection
                elapsed_time = time.time() - start_time
                if elapsed_time > QUERY_TIMEOUT_SECONDS:
                    raise TimeoutError(
                        f"Query exceeded {QUERY_TIMEOUT_SECONDS}s timeout. "
                        "Please try a simpler question."
                    )
                
                if token is not None:
                    # Accumulate and re-render with a typing cursor
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                else:
                    # Streaming complete — capture metadata
                    final_chunks = chunks or []
                    final_memory_info = memory_info or {}

            # Remove typing cursor, show final clean response
            response_placeholder.markdown(full_response)

            # Sources
            if show_sources and full_response != REFUSAL_MESSAGE:
                render_sources(
                    final_chunks,
                    key_prefix=f"live_{len(st.session_state.source_history)}"
                )

            # Persist to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
            })
            st.session_state.source_history.append(final_chunks)

        except TimeoutError as e:
            error_msg = f"⏱️ {str(e)}"
            response_placeholder.error(error_msg)
            st.session_state.source_history.append([])
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            response_placeholder.error(error_msg)
            st.session_state.source_history.append([])
