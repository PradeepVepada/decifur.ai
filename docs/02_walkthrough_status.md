# HybdRAG Walkthrough & Status
## Codebase-Aligned Operational Guide

> Last updated: April 2026  
> Source of truth: current repository code (`api.py`, `rag_engine.py`, `graph_store.py`, `pcc_memory.py`, `UI/streamlit_app.py`)

---

## 1) Current Project State

| Area | Status | Notes |
|---|---|---|
| Core query pipeline | Complete | `rag_engine.py` orchestrates retrieval + generation |
| Hybrid Neo4j retrieval | Complete | Dense + BM25 + entity boost + RRF |
| PCC memory | Complete | Short + long-term memory wired into prompt context |
| Streamlit UI | Complete | Streaming answers, source rendering, session controls |
| FastAPI backend | Complete | Non-stream + SSE stream endpoints |
| Conversation persistence | Complete | Neo4j-backed via `conversation_store.py` |
| Evaluation scripts | Complete | `evaluation/llm_judge.py`, `evaluation/ragas_eval.py` |
| Cloud deployment hardening | In progress | Docs/planning present, infra not fully codified |

---

## 2) End-to-End Walkthrough (Runtime)

### Step A: Startup

1. Load env vars (`MISTRAL_API_KEY`, `NEO4J_URI`, `NEO4J_PASSWORD` required for API path).
2. Initialize `RAGEngine`.
3. `RAGEngine.load()`:
   - Connects to Neo4j via `GraphStore.load()`.
   - Loads SciSpaCy model.
   - Optionally initializes biomedical normalizer.
   - Initializes PCC memory with shared embedder/Mistral client.

### Step B: User query handling

1. Query enters from one of:
   - `UI/streamlit_app.py` (`ask_stream`)
   - `chatbot.py` (`ask_stream`)
   - `api.py` (`ask` or `ask_stream`)
2. `RAGEngine._prepare_query()`:
   - Adds user turn to short-term memory.
   - Extracts entities locally.
   - Optionally normalizes entities through BioPortal/UMLS.
   - Runs retrieval + PCC context fetch in parallel.
3. If no relevant chunks pass threshold, returns refusal message.
4. Otherwise calls Mistral generation with grounded prompt and source tags.
5. Stores assistant turn in memory and returns answer + source chunks + memory info.

### Step C: Streaming behavior

- Streaming is true token streaming from Mistral (`self.client.chat.stream(...)`).
- Streamlit updates token-by-token.
- FastAPI `/api/chat/stream` relays SSE events (`token`, `sources`, `memory_info`, `done`).

---

## 3) Retrieval Walkthrough

`GraphStore.search()` performs:

1. Query embedding with `NeuML/pubmedbert-base-embeddings` (cached).
2. Parallel calls:
   - Neo4j vector search on `chunk_embeddings`.
   - Neo4j fulltext BM25 search on `chunk_text`.
3. Candidate merge.
4. Reciprocal Rank Fusion scoring.
5. Optional entity-match boost from graph edges.
6. Top-k return to `rag_engine.py`.

---

## 4) Memory Walkthrough

### Short-term memory
- Sliding message window (`SHORT_TERM_WINDOW = 10`).
- Compression is lazy and triggered on retrieval, not on every write.
- Can use extractive fallback or Mistral compression.

### Long-term memory
- Stored as `MemoryEpisode` nodes with embeddings.
- Retrieved via Neo4j vector index (`memory_episode_embeddings`).
- Expired episodes are pruned by retention policy (`LONG_TERM_EXPIRY_DAYS = 30`).

### Injection strategy
- Memory context is inserted as dedicated background section, separate from paper context to avoid false citations.

---

## 5) Runtime Call Pattern (Per Query)

Typical chat query:

- Mistral: `1` generation call (plus optional `+1` compression call).
- Neo4j: multiple retrieval/memory queries (database ops, no LLM token billing).
- BioPortal/UMLS: variable, only if normalizer is enabled and entities are found.
- OpenAI judge: not used in normal chat path.

---

## 6) API and Interface Surface

### API (`api.py`)
- `POST /api/chat` (non-streaming)
- `POST /api/chat/stream` (SSE streaming)
- conversation CRUD/search endpoints
- memory status/clear/store endpoints
- papers list and health endpoints

### UI (`UI/streamlit_app.py`)
- streaming chat UX
- source display toggle
- rate limiting/session query caps
- local archive/load conversation controls

### CLI (`chatbot.py`)
- commands: `/papers`, `/sources`, `/memory`, `/reset`, `/help`, `/quit`

---

## 7) Validation Checklist (Operational)

Use this quick list after pulling latest code:

- Neo4j is up and reachable.
- Required env vars are present.
- `RAGEngine.load()` succeeds.
- Query returns source-backed answer in UI.
- `/api/health` returns healthy state.
- Memory status endpoint returns PCC fields.

---

## 8) Open Work / Recommended Next Pass

1. Formalize cloud deployment manifests and environment profiles.
2. Add automated integration tests for stream + non-stream parity.
3. Add budget guardrails and telemetry for external API usage.
4. Add CI quality checks for docs drift against runtime constants.

---

This document is intentionally implementation-focused and should be updated whenever retrieval, memory wiring, or endpoint behavior changes.
