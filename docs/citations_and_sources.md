# Citations, sources, and grounding

## How `[S#]` works

1. **Retrieval** returns up to `TOP_K_RETRIEVAL` chunks from hybrid search (default **12**). The same default applies to `GraphStore.search(..., k=...)` so the UI and the model context stay aligned.
2. **`build_context`** labels excerpts in order: `[S1]`, `[S2]`, … `[SN]` — one tag per retrieved chunk, in **rank order**.
3. The answer is instructed to cite only those tags. **`_strip_invalid_citation_tags`** removes any `[S#]` where `#` is outside `1…N` (hallucinated range).
4. **`_inject_citations_post`** can add citations by sentence–chunk lexical overlap; tags still refer to the same chunk list.

The API returns the full ranked chunk list for the turn; the **chat UI** shows a **narrowed, deduplicated** view (see below).

### UI behavior (chat)

- If the answer contains `[S1]`, `[S2]`, … only those indices are listed (plus **deduplication by PDF** so the same file is not shown twice; multiple tags like `[S1] [S3]` appear on one row when they refer to the same `source` file).
- If the model omits all `[S#]` tags, the UI shows **deduplicated** retrieval (one row per file) with a short hint to add citations.

### OpenAI corpus model

Set `OPENAI_CORPUS_MODEL` (default **`gpt-5-nano`**) in `.env` next to `OPENAI_API_KEY` when using the **GPT-5 nano** option in the chat UI. Override if OpenAI assigns a different snapshot id for your account.

### Retrieval: entity needles

When the query includes extractable gene/protein symbols (including mixed-case names like **YakA**), chunks whose **text + title** do not mention any of those needles are dropped **unless** that would remove every chunk—in which case the full ranked list is kept so the system still returns an answer.

## Author / “who worked on X?” questions

Phrases like “what authors”, “who worked on”, “which authors” route to **`try_author_gene_graph_answer`**, which reads **Neo4j** (`STUDIES_MOLECULE` → authors) via `get_authors_for_molecule`.

Gene symbols are detected with:

- All-caps tokens (e.g. `PI3K`)
- Mixed-case symbols (e.g. `YakA`) via `_all_gene_symbol_candidates`

If the graph has no `Molecule` link for that string, the pipeline falls back to normal RAG.

## Conversation titles

The first **user** message in a conversation replaces the default `New Conversation YYYY-MM-DD` title with a **truncated copy of that message** (see `conversation_store._title_from_first_message`).

## Stopping generation

The UI aborts the fetch to `/api/chat/stream` (AbortSignal). Partial text is kept and marked as stopped. The API process may still finish work server-side until the worker exits; this is normal for streaming HTTP.

## Running tests

```bash
pytest tests/test_citation_alignment.py -q
```
