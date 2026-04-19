Feature 1
# Blueprint for implementing a path to answer questions such as “who co-authored work on gene X?

The app is behaving as designed: refusal = retrieval/gating failed or produced no context, and there is no implemented path that answers “who co-authored work on gene X?” as a graph query. To support that reliably you’d add something like a Cypher-backed (or API) “analytics” mode for author–gene aggregation, and/or relax or special-case retrieval for entity+author questions.

Restructured (short)
What the message means — The app only refuses when no chunks survive retrieval + the relevance gate. It is not running a “list all co-authors for gene X” graph query.

How answers are built — Vector + BM25 on chunks → top‑K passages → LLM must cite those passages. Author–gene rollups need Cypher aggregation, which this chat path doesn’t do.

Why PTEN can fail — Top fused score can fall below RELEVANCE_THRESHOLD (0.015) so chunks are dropped; or PTEN doesn’t match graph entity names for boost; or retrieved chunks don’t carry the author signal you need.

Simple prompt (for users, to improve chunk retrieval)
Use a literal gene token, paper-like wording, and ask for evidence from passages (not a full lab roster):

In the indexed papers, find passages that discuss PTEN. Quote or paraphrase what they say about PTEN, and list the paper title and year for each. If the passage names authors, include them; if not, say the author list is not in the retrieved excerpts.
Even shorter variant
What do our papers say about PTEN? Cite paper title and year. Include author names only if they appear in the cited excerpts.
Optional “debug” prompt (see if retrieval is empty)
Search the corpus for the term PTEN and summarize only what appears in the retrieved text. If nothing is retrieved, say so explicitly.
If you want a single system/developer prompt to add in code (router / intent), say so and I’ll phrase one for Agent mode implementation.



Feature 2:
# Blueprint for adding papers
_Exported on 4/9/2026 at 17:39:31 EDT from Cursor (3.0.16)_

---

**User**
how  does "+ add papers " feature in this codebase works? I want to adopt this same feture into mine, give a blueprint. Addition of new paper should restart the pipeline and the new nodes should sit well with existing KG and build realtionships seamlessly

---
**Cursor**
Tracing the "add papers" flow in the codebase and mapping it to the pipeline and knowledge graph.

In this repo the sidebar control is labeled **“Add paper”** (one PDF at a time), not “add papers.” Here is how it works end-to-end and a blueprint you can reuse.

## How it works in this codebase

### 1. UI → upload → background worker

- Hidden file input + **Add paper** opens the OS file picker (`accept` PDF only).
- On file selection, the client `POST`s `multipart/form-data` to `/api/papers/upload` with field name `file`, then polls `/api/ingest/status` every 2s until `done` or `error`, and refreshes the corpus list.

```518:559:c:\Users\santosh Arsid\Desktop\Man Cave\Papa\ChatBot\frontend\index.html
document.getElementById('btn-add-paper').addEventListener('click', () => filePdf.click());
filePdf.addEventListener('change', async () => {
  const f = filePdf.files && filePdf.files[0];
  filePdf.value = '';
  if (!f) return;
  const fd = new FormData();
  fd.append('file', f);
  try {
    const r = await fetch(`${API}/api/papers/upload`, { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Upload failed');
    showToast('Processing PDF in the background…', true);
    startIngestPoll();
  } catch (e) {
    showToast(e.message, false);
  }
});

function startIngestPoll() {
  if (ingestPoll) clearInterval(ingestPoll);
  ingestPoll = setInterval(async () => {
    try {
      const s = await fetchJSON(`${API}/api/ingest/status`);
      if (s.state === 'running' || s.state === 'queued') {
        setStatus(null, s.message || 'Indexing…');
      } else if (s.state === 'done') {
        clearInterval(ingestPoll);
        ingestPoll = null;
        setStatus(true, 'Ready');
        showToast(s.message || 'Done', true);
        await loadPapers();
      } else if (s.state === 'error') {
        // ...
```
### 2. API: queue, save PDF, thread

- Rejects concurrent ingests (`ingest_status.state == "running"` → 409).
- Saves under `PAPERS_DIR` (env `GRAPH_RAG_PDF_DIR` or `./papers`) with `secure_filename`; duplicate filename on disk → 409.
- Starts a **daemon thread** `_ingest_worker` that calls `incremental_ingest_pdf` with the live `RAGEngine`’s `HybridVectorStore` and the shared `_engine_lock`.

```226:297:c:\Users\santosh Arsid\Desktop\Man Cave\Papa\ChatBot\app.py
def _ingest_worker(saved: Path):
    global ingest_status
    try:
        with ingest_lock:
            ingest_status = {
                "state": "running",
                "message": "Extracting text and updating indexes…",
                "detail": None,
            }
        engine = get_engine()
        result = incremental_ingest_pdf(
            saved, engine.store, engine_lock=_engine_lock, log=True
        )
        # ... sets done / error on ingest_status ...

@app.route("/api/papers/upload", methods=["POST"])
def upload_paper():
    # ...
    with ingest_lock:
        if ingest_status.get("state") == "running":
            return jsonify({"error": "Another paper is still being processed."}), 409
    # ... save file, start Thread(target=_ingest_worker, ...) ...
```

**Important:** The server process does **not** “restart the pipeline” in the sense of exiting and reloading. The **same** `RAGEngine` singleton stays loaded; ingest **extends** indexes and the graph in place. That is what keeps chat and retrieval consistent without a reboot.

### 3. Incremental ingest pipeline (`paper_ingest.py`)

Under `engine_lock` where needed:

1. **Dedup** by `source` (PDF filename) against chunks already in the store.
2. **Next chunk IDs** — monotonic: `max(existing id) + 1`.
3. **Extract + chunk** — `build_chunks_for_pdf` (metadata cache updated).
4. **Biology triples** — LLM (+ ontology/scispaCy) only for **new** chunks.
5. **Triples file** — `merge_triples_after_ingest`: rewrites `data/triples.json` as *full author graph for entire corpus* + *old bio triples minus anything tied to replaced chunk ids/sources* + *new bio triples*.
6. **Neo4j** — `load_triples(driver, author_delta + new_bio)` (new paper’s author triples + new biology triples only for this ingest).
7. **Vector index** — `store.append_chunks(new_chunks)` (Chroma `add` + **BM25 rebuilt** over full corpus) and persist `data/chunks.json`.

```41:139:c:\Users\santosh Arsid\Desktop\Man Cave\Papa\ChatBot\paper_ingest.py
def incremental_ingest_pdf(
    pdf_path: Path,
    store: "HybridVectorStore",
    engine_lock: Optional[threading.Lock] = None,
    *,
    log: bool = True,
) -> dict:
    # ...
    existing = locked(read_existing)
    if any(c.get("source") == pdf_path.name for c in existing):
        return {"ok": False, "error": f"The corpus already includes a file named “{pdf_path.name}”."}

    start_id = _next_chunk_id(existing)
    # ...
    new_chunks, meta_cache = build_chunks_for_pdf(pdf_path, start_id, meta_cache)
    # ...
    new_bio = extract_biology_triples_from_chunks(new_chunks, ...)
    merge_triples_after_ingest(merged, new_chunks, new_bio, TRIPLES_PATH)

    author_delta = generate_author_triples_for_sources(merged, {pdf_path.name})
    # load_triples(driver, author_delta + new_bio)
    locked(append_and_save)  # store.append_chunks + write chunks.json
```

### 4. Why new graph nodes “fit” the existing KG

Neo4j writes use **MERGE**, not blind CREATE:

- **Entities**: `MERGE (e:Entity {name, node_type})` then **append** `chunk_ids`, `papers`, `years` when the node already exists.
- **Relations**: `MERGE (s)-[r:REL {type}]->(o)` then merge evidence/chunk_ids and increment `count`.

So if the new paper mentions the **same** entity string and type as an existing node, you get **one** node with richer provenance, and new edges attach to that node. Cross-paper “relationships” appear when extraction emits triples between entities that already exist or that connect via traversal.

```124:163:c:\Users\santosh Arsid\Desktop\Man Cave\Papa\ChatBot\graph_loader.py
UPSERT_ENTITY = """
MERGE (e:Entity {name: $name, node_type: $node_type})
ON CREATE SET
    e.chunk_ids = [$chunk_id],
    e.papers    = [$paper_title],
    e.years     = [$year]
ON MATCH SET
    e.chunk_ids = CASE
        WHEN $chunk_id IN e.chunk_ids THEN e.chunk_ids
        ELSE e.chunk_ids + [$chunk_id]
    END,
    ...
"""

UPSERT_RELATION = """
MATCH (s:Entity {name: $subj_name, node_type: $subj_type})
MATCH (o:Entity {name: $obj_name,  node_type: $obj_type})
MERGE (s)-[r:REL {type: $relation}]->(o)
...
"""
```

Retrieval already uses **all** chunks in memory (`GraphRetriever` maps chunk IDs to the live `store.chunks`), so once `append_chunks` finishes, the new paper is searchable without reloading the engine.

---

## Blueprint to adopt in your project

| Layer | What to implement |
|--------|-------------------|
| **API** | `POST .../papers/upload` (multipart `file`), optional max size; **single-flight** or job queue so two ingests do not corrupt indexes; `GET .../ingest/status` for UX. |
| **Storage** | Stable **document key** (filename, DOI, or hash); reject duplicates; persist raw files in a dedicated directory. |
| **Chunking** | **Globally unique, monotonic chunk IDs** so vector DB IDs and graph `chunk_id` never collide with old data. |
| **Vector store** | **Append** embeddings to the vector DB; if you use sparse/BM25, **rebuild** over the full corpus (this repo does) or maintain an incremental sparse index if you need scale. |
| **Graph file (optional)** | Keep a canonical JSON (or DB) of triples; on add, **merge** policy: drop stale triples for replaced docs only; regenerate deterministic subgraphs (here: full author network) + append new extracted triples. |
| **Graph DB** | Use **MERGE** on a clear **entity identity** `(canonical_name_or_id, type)` (or better: internal entity UUID with display name property) so new docs **attach** to existing nodes; merge relationship properties for evidence and chunk provenance. |
| **Concurrency** | One lock (or queue) around: read-modify-write of chunk list, triple merge, vector append, and any “reload BM25” — aligned with how this app uses `_engine_lock` + `ingest_lock`. |
| **Failure** | On hard failure after PDF save, either delete the file, mark a tombstone, or support re-run idempotency (same source → skip or replace). This repo deletes the uploaded file if ingest fails with “already includes” after a race. |

### “Restart pipeline” vs this design

- **Full restart** (re-embed everything, rebuild graph from scratch): simplest mentally, expensive at scale.
- **This repo’s approach** = **incremental continuation** of the same pipeline stages: extract → chunk → embed append → extract triples → **MERGE** into KG. Functionally it “runs the pipeline again” for the **new document only**, while keeping stores consistent.

### Making relationships “seamless” in your KG

1. **Normalize entities** before MERGE (lowercasing, synonym map, ontology IDs) so “PI3K” and “phosphoinositide 3-kinase” collapse if you want them to.
2. **Prefer stable IDs** in the graph (`entity_id` from ontology or your own hash of normalized name+type) and store `name` as a label — reduces accidental duplicates from string drift.
3. **Provenance on nodes/edges** (chunk IDs, paper IDs) so you can debug and optionally prune per paper.

If you tell me your stack (e.g. only vectors vs Neo4j vs another graph, and your chunk ID scheme), the same blueprint can be narrowed to a minimal set of endpoints and locks for your codebase.

---

**User**

Following are the total scripts in my othercodebase and func, what is it missing to integrate the above feature:api.py — FastAPI app: chat (stream + non-stream), conversations, memory, paper list. Loads RAGEngine, enforces env (OPENAI_API_KEY, Neo4j), and persists chats via conversation_store. rag_engine.py — Core RAG: rewrites follow-ups, extracts entities, hybrid search via GraphStore, PCC memory context, OpenAI generation (incl. streaming). Handles “paper discovery” style questions with ranked paper lists. graph_store.py — Neo4j + embeddings: vector + BM25 hybrid search, entity boost, index build from chunks, paper/author/chunk/entity graph ingest, relationship discovery via LLM during build. pcc_memory.py — Session memory: short-term messages, LLM/extractive compression, long-term episodic nodes in Neo4j, vector recall of past episodes; factory create_pcc_memory. extract.py — PDF → text (PyMuPDF + optional Surya OCR), scispaCy chunking/NER, OpenAI metadata JSON, writes data/chunks.json when run as main. build_index.py — Orchestrator: runs build_chunks() then GraphStore().build(chunks) to populate Neo4j and save data/chunks.json. Refresh_embeddings.py — Re-embeds all Chunk texts in Neo4j with the configured SentenceTransformer and rebuilds the vector index (no full PDF re-extract).openai_models.py — Central OpenAI model IDs from env (OPENAI_FAST_MODEL, OPENAI_DEEP_MODEL, PCC/rewrite/relationship/metadata overrides).
biomedical_normalizer.py — Maps mentions to ontologies via BioPortal (primary) and UMLS (fallback); used by rag_engine for richer query entities.conversation_store.py — Local persistence for chat threads (meta + messages) used by the API.streamlit_archive.py — Helpers to archive/list bucketed conversations for the Streamlit UI (S3-style or local buckets—whatever the file implements).chatbot.py — CLI chat: loads RAGEngine, streams answers, commands like /papers, /sources, /memory, /quit.  UI/streamlit_app.py — Streamlit “decifur.ai” UI: chat, sources, rate limits, uses RAGEngine and archive helpers.

evaluation/evaluate.py — Batch or scripted evaluation harness against the RAG/chat pipeline (metrics runs).

evaluation/llm_judge.py — Uses OpenAI as an LLM judge for scoring answers (separate from main chat model).

evaluation/ragas_eval.py — RAGAS-style retrieval/generation metrics evaluation script.



---

**Cursor**


Your other codebase already has the **batch** spine (`extract.py` → `build_index.py` → `GraphStore.build`, plus `refresh_embeddings.py` for a **full** re-embed). What’s missing for parity with this repo’s **“Add paper”** flow is everything that implements **one-document ingest without wiping or fully rebuilding** the world, plus **API/UI wiring** and **safe concurrency**.

Below is a concise gap list mapped to **your** filenames.

---

### 1. `api.py` — missing surface area

- **`POST` upload route** (multipart `file`), save to a stable papers directory, optional max body size.
- **Job model**: background task / queue thread / worker (not blocking the event loop for minutes).
- **Status endpoint** (`GET /ingest/status` or job id) so UIs can poll.
- **Single-flight / lock**: reject or queue a second upload while one ingest is running (this repo uses `ingest_lock` + “already processing” 409).
- **Error contract**: duplicate file / duplicate logical paper id, missing keys, partial failure after save.

---

### 2. `graph_store.py` — missing **incremental** path

Today you likely have **`build(chunks)`** tuned for **full** corpus. You need a second path, e.g. **`ingest_new_chunks(chunks_delta)`** or **`append_paper(pdf_path)`** that:

| Concern | Why it’s missing in a batch-only design |
|--------|----------------------------------------|
| **Chunk IDs** | New chunks must get **globally new IDs** (max existing + 1 or UUIDs), never collide with Neo4j `Chunk` nodes / vector ids. |
| **Neo4j writes** | **MERGE** papers/authors/entities/chunks and **attach** only new relationships; avoid `DELETE`/rebuild of the whole graph for one PDF. |
| **Embeddings** | Either embed **only new** chunk texts and upsert into your vector index, or document that you call a **scoped** reindex (your `refresh_embeddings.py` is **all** chunks — wrong shape unless you add “only these node ids”). |
| **BM25 / hybrid index** | If BM25 lives **in memory**, you must **extend or rebuild** the sparse index over the **full** chunk list after append (same idea as `HybridVectorStore.append_chunks` in this repo). |
| **Relationship discovery (“LLM during build”)** | Full build probably runs discovery across everything; incremental needs **“run discovery only for new chunks”** (or a bounded second pass), with deduping of proposed edges. |

Without this, your only option is “run `build_index.py` again” — which is **not** the integrated feature, it’s a **pipeline restart**.

---

### 3. `extract.py` — missing **single-PDF** API

You have “run as main → full `chunks.json`”. You need something like **`chunks_for_pdf(pdf_path, start_id, metadata_cache)`** (this repo’s `build_chunks_for_pdf`) so the API does not re-chunk the entire corpus on every upload.

---

### 4. `rag_engine.py` — missing **post-ingest consistency**

After ingest completes, either:

- **`RAGEngine` / `GraphStore` holds live indexes** → expose **`reload()`** or **`refresh_corpus()`** (reload chunks from disk + rebuild BM25 + refresh any caches), **or**
- Ensure every query path reads **authoritative state from Neo4j** so in-memory caches cannot be stale.

Batch-only projects often assume process start = one build; long-running FastAPI breaks that assumption.

---

### 5. `conversation_store.py` / memory — optional but worth deciding

- **PCC / episodic memory**: usually **unchanged** by a new paper; no hard requirement unless you embed “corpus version” in memory keys.
- **System prompts**: if they mention fixed paper sets, bump a **corpus revision** after ingest so behavior is explicit.

---

### 6. `UI/streamlit_app.py` (and any other client) — missing UX

- **`st.file_uploader`** (or equivalent) → call your new upload API → **poll status** → refresh paper list / clear caches.
- **Rate limiting / size limits** aligned with `api.py`.

---

### 7. `refresh_embeddings.py` — role clarification

- As described, it’s a **maintenance** tool (full re-embed), **not** incremental ingest.
- Either add **`--chunk-ids`** / **`--since`** or leave it for repairs; **do not** make every new paper trigger a full refresh if you care about latency and cost.

---

### 8. Ops / idempotency (easy to forget)

- **Dedup** by filename, hash, or DOI before heavy work.
- **Partial failure**: PDF on disk but graph half-written — define **retry** or **tombstone** rules.
- **Migrations**: Neo4j constraints/indexes must support **MERGE** keys you use for entities (same spirit as this repo’s `(name, node_type)` uniqueness).

---

## Minimal “definition of done” for your stack

1. **API**: upload + background ingest + status + mutex.  
2. **Extract**: one-PDF → list of chunk dicts with new ids.  
3. **GraphStore**: `append` path: Neo4j MERGE + new embeddings + BM25/hybrid index updated.  
4. **RAG**: reload or zero-stale-cache guarantee after ingest.  
5. **UI**: upload + poll + refresh paper list.

If you paste (or open in workspace) your **`graph_store.build()`** signature and how it stores chunks/embeddings/BM25, the same checklist can be turned into a concrete function split (`build_full` vs `ingest_delta`) without guesswork.

