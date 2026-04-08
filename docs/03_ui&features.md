# HybdRAG UI & Features
## Streamlit UX, product backlog, and security hardening

> **Focus:** User experience, feature roadmap, and gaps vs production hardening  
> **Primary UI:** `UI/streamlit_app.py` (decifur.ai)  
> **Related API:** `api.py` (FastAPI — conversation persistence in Neo4j when using HTTP clients)

---

## 1. Current UI Assessment

### Implemented today (`UI/streamlit_app.py`)

| Area | Status | Notes |
|------|--------|--------|
| Streaming chat | Done | Real Mistral token stream |
| Sources expander | Done | Optional “Show sources” |
| Sidebar metrics | Done | Paper count, message count, source history length |
| Local chat archive | Done | `streamlit_archive` — bucketed “Past conversations”, load without server DB |
| Session controls | Done | New conversation (archives current), clear, PCC reset on switch |
| Session rate limiting | Done | Min 2s between queries; max 100 queries per session |
| Query timeout | Done | 60s guard during streaming |
| Engine init | Done | Cached `RAGEngine`, Mistral + Neo4j required |

### What works well

- Clean chat layout with streaming and typing feedback
- Source chunk preview with expandable sections
- Local persistence of threads without requiring the FastAPI stack

### Pain points / gaps

- No **server-backed** multi-user chat list in Streamlit (Neo4j conversations exist for **API** clients only)
- No in-UI **conversation search** against a shared store (API has `/api/conversations/search`)
- No **+ Papers** / live corpus CRUD from the UI (see supplementary themes below)
- **Security:** no auth, no row-level isolation, no API rate limits by identity — see [Section 11](#11-security--access-control-planned)

---

## 2. Supplementary product themes (`supplementary_material.md`)

The following ideas are captured from `docs/supplementary_material.md` and merged into a single backlog. They are **not** all implemented; they inform prioritization.

### 2.1 Paper interaction workflow

- **Vision:** Add paper → ask → delete → repeat (dynamic KB).
- **Technical direction:** Real-time CRUD on chunks/graph (today: batch `build_index.py` + Neo4j); future could use vector store + KG updates per document.
- **UI:** “+ Papers” or upload flow with progress and re-index job.

### 2.2 Guardrails for professional tone

- **Vision:** Redirect off-topic or inappropriate prompts; keep answers in-domain (research corpus).
- **Technical direction:** Keyword/heuristic layers first; optional moderation API; align with refusal path in `rag_engine.py` for **grounding**, not social moderation (extend separately).

### 2.3 Adaptive conversational depth

- **Vision:** Toggle “simple” vs “technical” explanations (e.g. ELI10 vs graduate).
- **Technical direction:** System prompt variant or user preference in session (and later per-user store).

### 2.4 “+ Papers” and KG updates

- **Vision:** Button to add corpora and refresh retrieval.
- **Technical direction:** Webhook or upload → embed → `graph_store` ingest; UI shows build status.

### 2.5 Toggle for deep reasoning

- **Vision:** User-visible “fast vs deep” mode.
- **Note:** Backend already routes `mistral-small-latest` vs `mistral-large-latest` by **intent** (`rag_engine.py`). A **manual override** in the UI is still a backlog item.

### 2.6 Memory management

- **Vision:** Preferences and history across sessions.
- **Current:** PCC short/long-term in Neo4j for memory; Streamlit uses **local** archive + session state. Full cross-device memory needs authenticated user binding.

### 2.7 UI/UX improvements (from supplement)

- Reduce redundant entity metadata in the main thread
- Richer **chat history** (sidebar timeline) when wired to API or shared store
- **Context window** copy: e.g. “Last N messages used” surfaced in Settings

### 2.8 Evaluation & quality loop

- **Vision:** Automated metrics + human thumbs + optional `evaluation/llm_judge.py` batch runs.
- **Backlog:** In-app feedback (thumbs) and export of conversations for eval datasets.

### 2.9 Guardrails consolidation

- Merge overlapping rules (tone + off-topic) into one moderation path where possible.

### 2.10 Context / token limits

- Document model context limits (~3k–10k+ tokens depending on model); PCC summarization already reduces drift — surface limits in UI help text.

---

## 3. Proposed UI Architecture

### 3.1 Overall Layout

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ HybdRAG Research Assistant                                    [Settings] [?]  │
├─────────────────┬───────────────────────────────────────────────────────────────┤
│                 │                                                               │
│  SIDEBAR        │  MAIN CHAT AREA                                               │
│  ────────       │  ──────────────                                               │
│                 │                                                               │
│  ┌───────────┐  │  ┌─────────────────────────────────────────────────────────┐ │
│  │ Conversations │  │                                                         │ │
│  │ ──────────│  │  │  User: What role does cAMP play in chemotaxis?           │ │
│  │           │  │  │                                                         │ │
│  │ ▸ Today   │  │  │  Assistant: cAMP acts as a key second messenger...      │ │
│  │   • Query about PTEN  │  │  [S1] Devreotes 2006 [S2] Janetopoulos 2014     │ │
│  │   • cAMP signaling    │  │  ───────────────────────────────────────────── │ │
│  │   • Comparison        │  │  Sources (2) | Memory: 2 msgs | Intent: entity│ │
│  │                       │  │  ┌─────────────────────────────────────────┐   │ │
│  │ ▸ Yesterday           │  │  │ 📄 Devreotes & Janetopoulos, 2006      │   │ │
│  │   • LEGI model        │  │  │    "During aggregation, cAMP binds..." │   │ │
│  │   • PI3K pathway      │  │  └─────────────────────────────────────────┘   │ │
│  │                       │  │                                                         │
│  │ ▸ This Week           │  │  ┌─────────────────────────────────────────────────┐ │
│  │   • Paper search      │  │  │ [Type your question here...]          [Send]  │ │
│  │                       │  │  └─────────────────────────────────────────────────┘ │
│  └───────────────────────┘  │                                                         │
│                             │                                                         │
│  ┌───────────────────────┐  │  ┌─────────────────────────────────────────────────┐ │
│  │ Memory Controls       │  │  │ Quick Actions                                   │ │
│  │ ─────────────────     │  │  │ • [Clear Chat] [Export PDF] [Share Link]        │ │
│  │ • Short-term: 3 msgs  │  │  └─────────────────────────────────────────────────┘ │
│  │ • Long-term: 5 episodes│  │                                                         │
│  │ [Clear Memory]        │  │                                                         │
│  └───────────────────────┘  │                                                         │
│                             │                                                         │
└─────────────────────────────┴───────────────────────────────────────────────────────┘
```

---

## 4. Feature Specifications

### 4.1 Persistent Conversation Sidebar

#### Data Model
```python
@dataclass
class ConversationMeta:
    conversation_id: str
    title: str              # Auto-generated from first query
    created_at: datetime
    updated_at: datetime
    message_count: int
    primary_topics: list    # Extracted from conversation
    user_id: str
```

#### Sidebar Components

**Conversation List (Grouped by Date)**
```
▸ Today
  • "PTEN regulation of PI3K" (2 min ago)
  • "Compare chemotaxis methods" (1 hr ago)
  
▸ Yesterday  
  • "LEGI model explanation"
  • "Protein phosphorylation"
  
▸ This Week
  • "cAMP pathway overview"
  • "Paper recommendations"
  
▸ Older
  • "Initial exploration" (2 weeks ago)
```

**Conversation Actions (Right-click/Menu)**
- Rename conversation
- Delete conversation
- Export as PDF
- Duplicate conversation
- Pin to top

#### Storage Implementation

```python
# Neo4j Schema Addition
"""
CREATE CONSTRAINT conversation_id IF NOT EXISTS
FOR (c:Conversation) REQUIRE c.conversation_id IS UNIQUE

CREATE (conv:Conversation {
    conversation_id: $id,
    title: $title,
    created_at: datetime(),
    updated_at: datetime(),
    user_id: $user_id
})

CREATE (msg:Message {
    message_id: $msg_id,
    role: $role,
    content: $content,
    timestamp: datetime(),
    intent: $intent,
    memory_info: $memory_info
})

CREATE (conv)-[:HAS_MESSAGE {index: $idx}]->(msg)
"""
```

---

### 4.2 Enhanced Chat Interface

#### Message Enhancements

**Assistant Message Card**
```
┌─────────────────────────────────────────────────────────────────┐
│ 🤖 Assistant                                    [Copy] [Regen] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ cAMP plays a crucial role in Dictyostelium chemotaxis by       │
│ acting as both a cell-cell signaling molecule and an            │
│ intracellular second messenger...                               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ 📊 Metadata                                                    │
│ • Intent: entity_query | Model: mistral-small | Latency: 1.8s  │
│ • Memory: 3 short-term, 1 long-term episode                    │
│ • Confidence: 0.87 (based on retrieval scores)                  │
├─────────────────────────────────────────────────────────────────┤
│ 📚 Sources (2)                                    [Expand All] │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ [S1] Devreotes & Janetopoulos, 2006                         │ │
│ │      "During aggregation, cAMP binds to receptors..."      │ │
│ │      Score: 0.82 | [View Full Chunk]                        │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ [S2] Parent & Devreotes, 2009                               │ │
│ │      "The cAMP signaling pathway regulates..."             │ │
│ │      Score: 0.76 | [View Full Chunk]                        │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

#### User Message Actions
- **Edit & Resubmit**: Modify previous query
- **Branch**: Create alternate conversation path
- **Bookmark**: Save important exchange

#### Quick Actions Bar
- **Clear Chat**: Reset current conversation
- **Export PDF**: Download conversation as PDF
- **Share Link**: Generate shareable URL (if deployed)
- **New Conversation**: Start fresh

---

### 4.3 Memory Visualization Panel

#### Current State Display
```
┌─────────────────────────────────────────────────┐
│ 🧠 PCC Memory Status                            │
├─────────────────────────────────────────────────┤
│                                                 │
│ Short-Term (Active Window)                      │
│ ████████░░ 8/10 messages                        │
│                                                 │
│ Last Compressed: 2 minutes ago                  │
│ "Discussion about cAMP signaling in             │
│  Dictyostelium chemotaxis..."                   │
│                                                 │
├─────────────────────────────────────────────────┤
│ Long-Term Episodes                              │
│ ─────────────────                               │
│ • Episode 1 (2 days ago)                        │
│   Topics: PTEN, PI3K, Chemotaxis                │
│   [View] [Delete]                               │
│                                                 │
│ • Episode 2 (1 week ago)                        │
│   Topics: LEGI model, Feedback                  │
│   [View] [Delete]                               │
│                                                 │
│ [Clear All Long-Term] [Export Memory]           │
└─────────────────────────────────────────────────┘
```

#### Memory Search
```
┌─────────────────────────────────────────────────┐
│ 🔍 Search Past Conversations                    │
├─────────────────────────────────────────────────┤
│ [Search memory...]                    [Search]  │
│                                                 │
│ Results:                                        │
│ • "cAMP in chemotaxis" - 2 days ago             │
│ • "PTEN regulation" - 1 week ago                │
│ • "PI3K pathway" - 2 weeks ago                  │
└─────────────────────────────────────────────────┘
```

---

### 4.4 Context Management Controls

#### Conversation Settings Panel
```
┌─────────────────────────────────────────────────┐
│ ⚙️ Conversation Settings                        │
├─────────────────────────────────────────────────┤
│                                                 │
│ Multi-Turn Context                              │
│ [✓] Include previous messages in context        │
│     Window size: [5] messages                   │
│                                                 │
│ Memory Settings                                 │
│ [✓] Enable PCC Memory                          │
│     Auto-compress after [4] messages            │
│     Long-term retention: [30] days              │
│                                                 │
│ Response Settings                               │
│ [✓] Show source chunks                         │
│ [✓] Show intent classification                 │
│ [✓] Show memory info                           │
│ [ ] Stream responses                           │
│                                                 │
│ Advanced                                        │
│ Retrieval count: [6] chunks                     │
│ Relevance threshold: [0.45]                     │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 5. Additional UI Enhancements

### 5.1 Research Workflow Features

#### Paper Collection
```
┌─────────────────────────────────────────────────┐
│ 📁 Saved Papers Collection                      │
├─────────────────────────────────────────────────┤
│                                                 │
│ • Devreotes & Janetopoulos, 2006 ★              │
│   "cAMP-mediated adaptation..."                 │
│   [View] [Add to Queue]                         │
│                                                 │
│ • Parent & Devreotes, 2009                      │
│   "Self-organization of..."                     │
│   [View] [Add to Queue]                         │
│                                                 │
│ [+ Add Current Paper]                           │
└─────────────────────────────────────────────────┘
```

#### Citation Manager
- Copy formatted citations (APA, MLA, etc.)
- Export bibliography
- Link to DOI

### 5.2 Collaboration Features

#### Shared Conversations
- Generate shareable link
- Export/import conversation JSON
- Team workspace (future)

#### Annotations
- Highlight important passages
- Add personal notes to chunks
- Flag incorrect answers

### 5.3 Accessibility Features

- Keyboard shortcuts
- High contrast mode
- Font size controls
- Screen reader support

---

## 6. Technical Implementation Plan

> **Note:** `conversation_store.py` + `api.py` already persist conversations in Neo4j for HTTP clients. Streamlit still uses **local** archive (`streamlit_archive`); aligning the web UI with the API store is a natural Phase 1 follow-up.

### Phase 1: Conversation Persistence (Week 1)
- [x] Conversation and Message nodes in Neo4j (API path)
- [ ] Wire Streamlit (or a Next.js client) to the same backend for shared history
- [ ] Build conversation list sidebar component (server-backed)
- [ ] Add conversation CRUD operations with **auth + IDOR checks** (Section 11)

### Phase 2: Enhanced Chat UI (Week 2)
- [ ] Redesign message cards with metadata
- [ ] Add message actions (copy, regenerate, edit)
- [ ] Implement streaming with visual feedback
- [ ] Add confidence score display

### Phase 3: Memory Visualization (Week 3)
- [ ] Build memory status panel
- [ ] Implement memory search
- [ ] Add episode viewer/detail modal
- [ ] Memory export functionality

### Phase 4: Advanced Features (Week 4)
- [ ] Conversation search
- [ ] Export to PDF/Markdown
- [ ] Paper collection management
- [ ] Keyboard shortcuts

---

## 7. Component Library

### Reusable Streamlit Components

```python
# conversation_card.py
def render_conversation_card(conversation: ConversationMeta, is_active: bool):
    """Render a conversation item in the sidebar."""
    pass

# message_bubble.py  
def render_message(message: dict, show_metadata: bool = False):
    """Render a chat message with optional metadata."""
    pass

# source_expander.py
def render_source_chunk(chunk: dict, index: int):
    """Render a source chunk with citation info."""
    pass

# memory_status.py
def render_memory_status(memory_info: dict):
    """Render PCC memory status panel."""
    pass
```

---

## 8. Mockup Screens

### 8.1 Empty State
```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                      Welcome to HybdRAG                         │
│                  Research Assistant for Devreotes Lab            │
│                                                                 │
│         ┌─────────────────────────────────────────────┐         │
│         │                                             │         │
│         │  Ask questions about research papers on:    │         │
│         │  • cAMP signaling                           │         │
│         │  • Chemotaxis mechanisms                    │         │
│         │  • PI3K/PTEN pathways                       │         │
│         │  • Dictyostelium models                     │         │
│         │                                             │         │
│         └─────────────────────────────────────────────┘         │
│                                                                 │
│    Suggested Questions:                                         │
│    ┌────────────────────────────────────────────────────────┐   │
│    │ "What role does PTEN play in chemotaxis?"              │   │
│    │ "Compare cAMP and PI3K signaling pathways"             │   │
│    │ "What methods are used to study cell migration?"       │   │
│    └────────────────────────────────────────────────────────┘   │
│                                                                 │
│         [Type your question here...]                   [Send]   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Error State
```
┌─────────────────────────────────────────────────────────────────┐
│ ⚠️ Unable to process query                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ The available papers do not contain sufficient information      │
│ to answer this question.                                        │
│                                                                 │
│ Suggestions:                                                    │
│ • Rephrase your question with specific terms                    │
│ • Ask about topics covered in the corpus (chemotaxis, cAMP)     │
│ • Try a simpler question first                                  │
│                                                                 │
│ [Try Again]  [Clear Memory]  [New Conversation]                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Responsive Design Considerations

| Screen Size | Layout Adaptation |
|-------------|-------------------|
| Desktop (>1024px) | Full sidebar + chat |
| Tablet (768-1024px) | Collapsible sidebar |
| Mobile (<768px) | Bottom sheet for sidebar, full-width chat |

---

## 10. Performance Considerations

- Lazy load conversation history
- Paginate long conversations
- Cache rendered components
- Debounce search inputs
- Use session state efficiently

---

## 11. Security & access control (planned)

Production deployment must not rely on “trusted local users” only. The following items are **leftover** relative to a hardened multi-tenant product.

### 11.1 Row-level security (RLS) — not everyone can query the whole DB

- **Risk:** Today, Neo4j and API patterns often assume a single `user_id` default (e.g. `"default"`). Any client that can reach the database or API could **read or mutate another tenant’s graph** if identifiers are guessable and there is no server-side check.
- **Direction:**
  - **Neo4j:** enforce tenant scoping in every Cypher path (e.g. `WHERE c.user_id = $uid`) or use **Neo4j Fabric / separate DBs** per tenant for strict isolation.
  - **Application:** bind every `Conversation`, `Message`, and PCC artifact to an authenticated **subject** (JWT/OAuth sub), never to a client-supplied string alone.
  - **Principle:** the database should not be “wide open” to arbitrary queries from the UI; only the **backend** should query with least privilege.

### 11.2 Rate-limited endpoints

- **Today:** Streamlit applies **session** throttling (seconds between queries, max queries per session). `api.py` uses a **request timeout** for chat, but that is not the same as abuse protection.
- **Needed for API:**
  - Per-IP and per-user (or per-API-key) limits on `POST /api/chat`, `/api/chat/stream`, and expensive reads.
  - Return `429` with `Retry-After`; optional queue for streaming.
  - Align limits with Mistral/Neo4j capacity and cost envelopes.

### 11.3 Insecure direct object references (IDOR)

- **Risk:** If `conversation_id` (or other IDs) are passed from the client without verifying **ownership**, an attacker could read/delete another user’s conversation (`GET/DELETE /api/conversations/{id}`).
- **Direction:**
  - After auth, resolve the resource and assert `resource.owner_id == current_user.id` (or equivalent) on **every** mutating and sensitive read.
  - Use opaque, unguessable IDs **plus** server-side ownership checks (IDs alone are not enough).
  - Audit list endpoints (`GET /api/conversations`) to ensure they filter by authenticated user only.

### 11.4 Related hardening (short list)

- CORS locked to known frontends in production (already narrowed in dev for `api.py`).
- Secrets only via env / secret manager; never in client bundles.
- Optional: WAF, bot protection, and audit logs for admin actions.

---

## 12. Future Enhancements (Post-MVP)

1. **AI-Generated Conversation Titles**: Auto-summarize first exchange
2. **Conversation Tags**: User-created tags for organization
3. **Smart Suggestions**: Context-aware follow-up questions
4. **Voice Input**: Audio transcription for queries
5. **Multi-language**: Support for non-English queries
6. **Diff View**: Show changes when regenerating answers

---

*This document serves as the UI/UX and feature specification for HybdRAG / decifur.ai, including backlog themes from `supplementary_material.md` and security gaps for production.*
