# Graph Report - C:\Users\santosh Arsid\Desktop\Man Cave\Gen AI\Cursor\NovaAI_v2  (2026-04-17)

## Corpus Check
- 37 files · ~1,914,034 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 489 nodes · 1000 edges · 23 communities detected
- Extraction: 72% EXTRACTED · 28% INFERRED · 0% AMBIGUOUS · INFERRED: 279 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]

## God Nodes (most connected - your core abstractions)
1. `RAGEngine` - 58 edges
2. `GraphStore` - 58 edges
3. `PCCMemory` - 52 edges
4. `ConversationStore` - 36 edges
5. `RAGEngine` - 35 edges
6. `Message` - 25 edges
7. `ConversationMeta` - 25 edges
8. `get_ollama_base_url()` - 15 edges
9. `main()` - 13 edges
10. `build_chunks_for_pdf()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `chatbot.py ---------- Interactive command-line chatbot for interrogating Prof.` --uses--> `RAGEngine`  [INFERRED]
  C:\Users\santosh Arsid\Desktop\Man Cave\Gen AI\Cursor\NovaAI_v2\chatbot.py → C:\Users\santosh Arsid\Desktop\Man Cave\Gen AI\Cursor\NovaAI_v2\rag_engine.py
- `evaluate.py ----------- Runs the chatbot against a set of benchmark questions` --uses--> `RAGEngine`  [INFERRED]
  C:\Users\santosh Arsid\Desktop\Man Cave\Gen AI\Cursor\NovaAI_v2\evaluation\evaluate.py → C:\Users\santosh Arsid\Desktop\Man Cave\Gen AI\Cursor\NovaAI_v2\rag_engine.py
- `llm_judge.py ------------ Evaluates the chatbot using LLM-as-a-Judge pattern.` --uses--> `RAGEngine`  [INFERRED]
  C:\Users\santosh Arsid\Desktop\Man Cave\Gen AI\Cursor\NovaAI_v2\evaluation\llm_judge.py → C:\Users\santosh Arsid\Desktop\Man Cave\Gen AI\Cursor\NovaAI_v2\rag_engine.py
- `Parse a float score from LLM output with range validation and logging.` --uses--> `RAGEngine`  [INFERRED]
  C:\Users\santosh Arsid\Desktop\Man Cave\Gen AI\Cursor\NovaAI_v2\evaluation\llm_judge.py → C:\Users\santosh Arsid\Desktop\Man Cave\Gen AI\Cursor\NovaAI_v2\rag_engine.py
- `ragas_eval.py ------------- Evaluates the RAG pipeline using the Ragas framewo` --uses--> `RAGEngine`  [INFERRED]
  C:\Users\santosh Arsid\Desktop\Man Cave\Gen AI\Cursor\NovaAI_v2\evaluation\ragas_eval.py → C:\Users\santosh Arsid\Desktop\Man Cave\Gen AI\Cursor\NovaAI_v2\rag_engine.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (47): analytics_gene_authors(), ArchiveSave, auth_signin(), auth_signup(), chat(), chat_stream(), ChatRequest, ChatResponse (+39 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (47): Store a compressed episode when enough new interaction has accumulated., _all_gene_symbol_candidates(), build_context(), _collect_protein_needles_from_query(), _filter_chunks_matching_protein_needles(), _format_recent_history_for_rewrite(), _gene_symbol_candidates(), _inject_citations_post() (+39 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (34): _ingest_worker(), main(), build_index.py -------------- One-time script: extracts text from all PDFs and, _cache_get(), _cache_put(), _distinct_paper_row(), GraphStore, normalize_doi() (+26 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (26): evaluate.py ----------- Runs the chatbot against a set of benchmark questions, run_evaluation(), evaluate_faithfulness(), evaluate_relevance(), _parse_score(), llm_judge.py ------------ Evaluates the chatbot using LLM-as-a-Judge pattern., Parse a float score from LLM output with range validation and logging., run_judge_evaluation() (+18 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (9): create_pcc_memory(), MemoryEpisode, PCCMemory, pcc_memory.py ------------- Session-aware Personal Context Compression (PCC) M, Rebuild short-term memory from an existing transcript., Session-aware Personal Context Compression Memory Manager., ShortTermMemory, Reload store state from Neo4j after external ingest. (+1 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (33): build_chunks(), build_chunks_for_pdf(), chunk_text_spacy(), extract_entities_from_chunk(), extract_paper_metadata(), extract_pdf_surya(), extract_pdf_text(), get_encoder() (+25 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (18): get_conversation(), list_conversations(), BiomedicalNormalizer, BioPortalService, create_biomedical_normalizer(), load_scispacy_model(), _make_session(), NormalizedEntity (+10 more)

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (25): ask_chatbot(), compute_overall_score(), contains_any(), evaluate_one(), extract_answer(), extract_context(), load_questions(), main() (+17 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (12): apiFetch(), getAuthHeaders(), publicApiBase(), sendChatMessage(), buildAssistantSourceRows(), dedupeRetrievalRowsByFile(), extractCitationIndicesFromAnswer(), handleClearMemory() (+4 more)

### Community 9 - "Community 9"
Cohesion: 0.14
Nodes (11): _ollama_health_sync(), ui_config(), ui_ollama_health(), _env(), get_ollama_base_url(), _normalize_ollama_base_url(), ollama_api_extra_body(), Ollama-compatible API + optional OpenAI config for the RAG stack.  OLLAMA_BASE_U (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.2
Nodes (13): _build_web_context(), _cache_get(), _cache_put(), _ddg_search(), web_search.py ------------- Low-latency web search pipeline:   1. DuckDuckGo sea, Scrape urls concurrently. Returns {url: text | None}., Merge search results with scraped content.     Prefers scraped text; falls back, Full pipeline: search → scrape → LLM stream.      Yields (token, None, None) per (+5 more)

### Community 11 - "Community 11"
Cohesion: 0.29
Nodes (8): hr(), main(), print_header(), print_help(), print_memory_status(), print_papers(), print_sources(), chatbot.py ---------- Interactive command-line chatbot for interrogating Prof.

### Community 12 - "Community 12"
Cohesion: 0.2
Nodes (5): AssistantMessageBody(), MarkdownErrorBoundary, balanceDisplayMathDelimiters(), sanitizeAssistantMarkdown(), stripPrivateUseChars()

### Community 13 - "Community 13"
Cohesion: 0.67
Nodes (1): Minimal env so `rag_engine` and related modules import in CI without a live Neo4

### Community 14 - "Community 14"
Cohesion: 0.67
Nodes (1): onSubmit()

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (0): 

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (0): 

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **56 isolated node(s):** `conversation_store.py --------------------- Persistent conversation storage back`, `Neo4j-backed conversation storage.`, `[Review #24] Single atomic DETACH DELETE — no orphan risk.`, `[Review #25] Uses fulltext index — no longer a full property scan.`, `Factory. Pass a shared driver to avoid multiple pool instances.` (+51 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 15`** (2 nodes): `layout.tsx`, `RootLayout()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (2 nodes): `page.tsx`, `HomePage()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (2 nodes): `page.tsx`, `ChatPage()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (2 nodes): `utils.ts`, `cn()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `next-env.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `next.config.mjs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `postcss.config.mjs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `tailwind.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GraphStore` connect `Community 2` to `Community 1`, `Community 3`, `Community 4`, `Community 5`, `Community 9`, `Community 11`?**
  _High betweenness centrality (0.251) - this node is a cross-community bridge._
- **Why does `RAGEngine` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 9`, `Community 11`?**
  _High betweenness centrality (0.235) - this node is a cross-community bridge._
- **Why does `RAGEngine` connect `Community 3` to `Community 0`, `Community 2`, `Community 11`, `Community 4`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `RAGEngine` (e.g. with `SignInRequest` and `SignUpRequest`) actually correct?**
  _`RAGEngine` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 36 inferred relationships involving `GraphStore` (e.g. with `paper_ingest.py --------------- Incremental PDF ingest: extend Neo4j + optiona` and `Append new chunk records (without embeddings) to data/chunks.json.`) actually correct?**
  _`GraphStore` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `PCCMemory` (e.g. with `RAGEngine` and `rag_engine.py ------------- Advanced GraphRAG pipeline with session-aware PCC me`) actually correct?**
  _`PCCMemory` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `ConversationStore` (e.g. with `SignInRequest` and `SignUpRequest`) actually correct?**
  _`ConversationStore` has 21 INFERRED edges - model-reasoned connections that need verification._