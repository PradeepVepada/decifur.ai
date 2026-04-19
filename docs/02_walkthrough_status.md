# HybdRAG / DAI — Walkthrough & Status

Operational guide aligned with the current codebase (`rag_engine.py`, `model_config.py`, `api.py`, `UI/streamlit_app.py`, `graph_store.py`, `pcc_memory.py`).

---

## 1) Project status snapshot

| Area | Status | Notes |
| --- | --- | --- |
| Corpus RAG pipeline | Complete | Ollama generation + Neo4j hybrid retrieval + optional OpenAI deep path |
| PCC memory | Complete | Short + long-term; hydrated from `ConversationStore` when using API |
| Streamlit UI | Complete | Corpus vs web mode, streaming, rate limits, archive helpers |
| FastAPI | Complete | Sync + SSE; shared engine + per-conversation hydration |
| Incremental PDF ingest | Complete | `POST /api/papers/upload` → `engine.ingest_pdf_path` |
| Web search mode | Complete | OpenAI summarisation (`web_search.py`), not Ollama |
| Evaluation scripts | Complete | Under `evaluation/`; manual / offline runs |

---

## 2) Startup sequence

1. **Environment:** Load `.env`. API requires `OLLAMA_BASE_URL`, `NEO4J_URI`, `NEO4J_PASSWORD`. Streamlit blocks without `OLLAMA_BASE_URL`.
2. **`RAGEngine.load()`:** `GraphStore.load()` → SciSpaCy → optional biomedical normalizer → `create_pcc_memory` sharing the graph embedder and OpenAI-compatible client pointed at Ollama.
3. **API lifespan:** Instantiates `ConversationStore`, async lock around the single shared engine (avoids cross-talk between concurrent requests).

**Why a single shared engine:** One Neo4j driver pool and one heavy embedder; conversations are isolated by `conversation_id` + hydration, not by separate engine instances.

---

## 3) Per-query walkthrough (corpus mode)

**Linear order (matches code):**

1. **Graph analytics shortcut** — If the question is author/gene style, `try_author_gene_graph_answer` may return early from `store.get_authors_for_molecule` (no chunk RAG).
2. **`_prepare_query`** — PCC `add_message(user)`; rewrite follow-ups and paper-discovery queries (`REWRITE_MODEL` on Ollama); SciSpaCy NER + optional normalisation; **parallel** `store.search(retrieval_query, k, entities)` and `_get_pcc_context`.
3. **Relevance** — Empty chunks → refusal (`REFUSAL_MESSAGE`). Low score may still proceed when paper-discovery, rewritten follow-up, or author×gene relax applies.
4. **Branches** — **Paper discovery:** `_answer_paper_discovery_query`. **Standard:** `_resolve_generation(intent)` then `chat.completions.create` with `_rag_generation_kwargs` (Ollama `extra_body` when not using OpenAI deep).
5. **Post** — `_strip_thinking` / citation post-processing; `_post_process` updates PCC and sliding `conversation_history`.

**Streaming:** `ask_stream` yields a heartbeat first, then runs `_prepare_query` in a thread pool so the UI does not freeze; tokens stream from the chat completion API; final yield carries chunks and `memory_info`.

---

## 4) Retrieval walkthrough (`graph_store.py`)

1. Module-level LRU cache for query embeddings (reduces repeat cost).
2. **Parallel** dense (`chunk_embeddings`) and BM25 (`chunk_text`).
3. Merge candidate records; **RRF**; optional **entity boost** from chunk–entity edges.
4. Return top-k with fused `score` (see `rag_engine.TOP_K_RETRIEVAL` / `TOP_K_PAPER_DISCOVERY`).

**Why parallel dense + BM25:** Latency ≈ `max(dense, bm25) + boost` instead of summing sequential round-trips.

---

## 5) Memory walkthrough (`pcc_memory.py`)

- **Short-term:** Rolling window; compression on a fixed interval (`LLM_COMPRESS_EVERY_N`).
- **Long-term:** Episodes with embeddings; retrieval via Neo4j vector index; pruning by age.
- **API hydration:** `hydrate_memory_from_messages` replays stored messages when switching `conversation_id` so PCC matches server-side history.

**Why separate from paper context:** Prevents memory text from being mistaken for cited excerpts.

---

## 6) API + persistence

- **`_prepare_engine_for_conversation`:** Sets PCC conversation id and rebuilds in-memory history from Neo4j messages.
- **Assistant save:** `Message` includes `memory_info`; if Neo4j rejects nested types, persistence logs a warning (non-fatal).

Endpoints summary: health, chat, chat stream, conversations CRUD/search, memory status/clear/store, papers list, ingest status, PDF upload, gene–authors analytics.

---

## 7) Web mode (Streamlit)

User toggles **web** → `web_search` pipeline (DuckDuckGo → scrape → OpenAI stream). Does not use corpus retrieval; `memory_info` typically includes `mode: web`.

**Why OpenAI for web:** Keeps RunPod/Ollama for private corpus workloads and uses a small cloud model for live web summarisation.

---

## 8) Operational checklist

- Neo4j reachable; vector + fulltext indexes built (`build_index` / ingest).
- Ollama endpoint responds at `OLLAMA_BASE_URL` (no stray `/v1` on the env value — `model_config` strips it).
- Test: one corpus query with sources; optional `/api/health`; optional memory status after a few turns.

---

## 9) Open follow-ups

- Harden assistant message serialization if Neo4j property types still warn on `memory_info`.
- Expand automated tests for stream vs non-stream parity and ingest idempotency.
- CI check that `TOP_K_*` and env defaults stay in sync with docs.

---

*Last updated: April 2026.*
