# HybdRAG System Architecture
## Devreotes Research Explorer — Consolidated Technical Documentation

> **Project:** Conversational AI for interrogating the research corpus of Prof. Peter Devreotes (JHU)  
> **Domain:** Cell Biology → Signal Transduction & Chemotaxis  
> **Approach:** Hybrid GraphRAG with Neo4j, SciSpaCy, Mistral AI, Surya OCR  
> **Version:** 2.0 (Post-PCC Memory Integration)

---

## 1. Executive Summary

HybdRAG is a production-grade Hybrid Graph RAG (Retrieval-Augmented Generation) system designed specifically for biomedical research literature. It combines vector similarity search with knowledge graph traversal to provide accurate, grounded answers to complex scientific queries.

**Key Innovations:**
- **PCC Memory:** Personal Context Compression for persistent conversation memory
- **Tiered LLM Routing:** Cost-optimized query classification
- **Hybrid Retrieval:** Vector + Graph + Entity fusion scoring
- **Independent Evaluation:** LLM-as-Judge with cross-model validation

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HybdRAG System Architecture                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐ │
│  │   Frontend Layer     │    │   RAG Engine         │    │   Storage Layer     │ │
│  │                     │    │                     │    │                     │ │
│  │  • Streamlit UI     │◄──►│  • Intent Router    │◄──►│  • Neo4j Desktop    │ │
│  │  • CLI Chatbot      │    │  • Entity Extractor │    │  • Vector Index     │ │
│  │  • Chat History     │    │  • Context Builder  │    │  • Knowledge Graph  │ │
│  └─────────────────────┘    │  • PCC Memory       │    └─────────────────────┘ │
│                              └──────────┬──────────┘                            │
│                                         │                                       │
│                                         ▼                                       │
│              ┌──────────────────────────────────────────────────────────┐       │
│              │                    LLM Layer                              │       │
│              │  ┌────────────────┐  ┌────────────────┐  ┌────────────┐  │       │
│              │  │ Mistral-Small  │  │ Mistral-Medium │  │Llama-Large │  │       │
│              │  │ (Fast/Intent)  │  │ (Standard)     │  │ (Deep)     │  │       │
│              │  └────────────────┘  └────────────────┘  └────────────┘  │       │
│              └──────────────────────────────────────────────────────────┘       │
│                                         │                                       │
│                                         ▼                                       │
│              ┌──────────────────────────────────────────────────────────┐       │
│              │               Evaluation Layer (LLM Judge)                │       │
│              │  • Faithfulness Score (0.0-1.0)                          │       │
│              │  • Relevance Score (0.0-1.0)                             │       │
│              │  • Weighted Verdict = 0.4×Faithful + 0.6×Relevance       │       │
│              └──────────────────────────────────────────────────────────┘       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

### 3.1 Core Components

| Layer | Technology | Purpose | Notes |
|-------|-----------|---------|-------|
| **Language** | Python 3.11+ | Core runtime | GPU-accelerated |
| **Frontend** | Streamlit | Web UI | Port 8506 |
| **CLI** | Python CLI | Terminal interface | `chatbot.py` |
| **Graph DB** | Neo4j Desktop | Knowledge Graph | Local instance |
| **Vector DB** | Neo4j Vector Index | Semantic search | 768-dim, cosine |

### 3.2 AI/ML Pipeline

| Component | Technology | Purpose | GPU Required |
|-----------|-----------|---------|--------------|
| **Embeddings** | `sentence-transformers` (all-mpnet-base-v2) | Chunk embeddings | Yes |
| **Biomedical NER** | SciSpaCy (`en_core_sci_lg`) | Entity extraction | No |
| **Primary LLM** | Mistral AI (small/large) | Generation | No |
| **Evaluation** | OpenAI (gpt-4o-mini) | Judge scoring | No |
| **PDF OCR** | Surya OCR | Scanned papers | Yes |

### 3.3 Dependencies Summary

```
# Core
mistralai>=1.0.0
neo4j>=5.0.0
python-dotenv

# Embeddings & NLP
sentence-transformers
scispacy
en_core_sci_lg-0.5.4

# OCR (GPU)
torch==2.11.0+cu128
surya-ocr

# UI
streamlit
```

---

## 4. Data Flow Architecture

### 4.1 Ingestion Pipeline (One-Time Build)

```
PDFs ──► PyMuPDF/Surya OCR ──► Text Extraction ──► Chunking (400 words, 20 overlap)
         │
         ▼
SciSpaCy NER ──► Entities (Proteins, Organisms, Concepts)
         │
         ▼
SentenceTransformer ──► 768-dim Embeddings
         │
         ▼
Mistral-Small ──► Relationship Discovery (REGULATES, INTERACTS_WITH, etc.)
         │
         ▼
Neo4j Batch UNWIND ──► Knowledge Graph + Vector Index
```

### 4.2 Query Processing Pipeline

```
User Query ──► Local Intent Classifier ──► Route by Intent
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
   Simple Lookup  Entity Query  Cross-Paper Synthesis
         │           │           │
         ▼           ▼           ▼
   Vector Search  Graph+Vector  Multi-hop Graph
         │           │           │
         └───────────┼───────────┘
                     ▼
            Context Fusion (RRF)
                     │
                     ▼
            Relevance Threshold Check (0.45)
                     │
                     ▼
            Mistral-Large Generation
                     │
                     ▼
            Response + Citations [S1], [S2], etc.
```

### 4.3 PCC Memory Flow

```
Short-Term (Sliding Window, 10 messages)
    │
    ▼ (Every 4 turns)
LLM Compression (Mistral-Small)
    │
    ▼
Long-Term Storage (Neo4j Vector Index)
    │
    ▼ (On next query)
Vector Similarity Retrieval ──► Context Injection
```

---

## 5. Neo4j Knowledge Graph Schema

### 5.1 Node Types

| Node Label | Properties | Purpose |
|------------|-----------|---------|
| `Paper` | paper_id, title, year, journal, doi | Research papers |
| `Chunk` | chunk_id, text, embedding[], chunk_index | Text segments |
| `Author` | author_id, name | Paper authors |
| `Topic` | topic_id, name | Research topics |
| `Method` | method_id, name, type | Experimental methods |
| `Molecule` | molecule_id, type | Proteins, genes |
| `Organism` | organism_id, taxonomy | Species |
| `Concept` | concept_id, definition | General concepts |
| `MemoryEpisode` | episode_id, content, embedding[], topics | PCC long-term memory |
| `PCCUser` | user_id | User identity |

### 5.2 Relationships

```
Paper ──[AUTHORED_BY]──► Author
Paper ──[COVERS_TOPIC]──► Topic
Paper ──[USES_METHOD]──► Method
Paper ──[HAS_CHUNK]──► Chunk
Chunk ──[STUDIES_MOLECULE]──► Molecule
Chunk ──[STUDIES_ORGANISM]──► Organism
Chunk ──[DISCUSSES_CONCEPT]──► Concept
Molecule/Organism/Concept ──[RELATED_TO]──► Molecule/Organism/Concept
PCCUser ──[HAS_EPISODE]──► MemoryEpisode
```

### 5.3 Vector Indexes

| Index Name | Node | Property | Dimensions | Similarity |
|------------|------|----------|------------|------------|
| `chunk_embeddings` | Chunk | embedding | 768 | Cosine |
| `memory_episode_embeddings` | MemoryEpisode | embedding | 768 | Cosine |

---

## 6. PCC Memory System Architecture

### 6.1 Overview

The Personal Context Compression (PCC) system provides both short-term and long-term conversation memory:

```
┌─────────────────────────────────────────────────────────────────┐
│                      PCC Memory Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Short-Term Memory (In-Memory)                          │   │
│  │  • Sliding window: 10 messages                          │   │
│  │  • Compressed summary (LLM-based every 4 turns)         │   │
│  │  • Instant access, no Neo4j query                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼ (Flush every 4 turns)              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Long-Term Memory (Neo4j Vector Index)                  │   │
│  │  • MemoryEpisode nodes with 768-dim embeddings          │   │
│  │  • Cosine similarity retrieval                          │   │
│  │  • 30-day expiry window                                 │   │
│  │  • Topic extraction for filtering                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Memory Injection: System prompt level (separate from context)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 PCC Configuration

```python
SHORT_TERM_WINDOW = 10      # Messages in sliding window
LONG_TERM_EXPIRY_DAYS = 30  # Episode retention
LLM_COMPRESS_EVERY_N = 4    # Compress after N messages
EMBEDDING_DIM = 768         # Vector dimension
```

---

## 7. LLM Routing Strategy

### 7.1 Intent Classification (Local, No API Call)

| Intent | Keywords/Pattern | Model Used |
|--------|-----------------|------------|
| `simple_lookup` | Default, 0-1 entities | Mistral-Small |
| `entity_query` | 2+ named entities | Mistral-Small |
| `cross_paper_synthesis` | "compare", "across", "synthesize" | Mistral-Large |
| `topic_evolution` | "over time", "history", "trend" | Mistral-Large |
| `recommendation` | "recommend", "suggest", "should I read" | Mistral-Small |

### 7.2 Tiered Generation

| Tier | Query Complexity | Model | Avg Latency | Cost/Query |
|------|-----------------|-------|-------------|------------|
| Tier 1 (Fast) | Simple factual | Mistral-Small | ~1.5s | ~$0.001 |
| Tier 2 (Standard) | Entity-based | Mistral-Small | ~2.0s | ~$0.002 |
| Tier 3 (Deep) | Cross-paper | Mistral-Large | ~4.0s | ~$0.01 |

---

## 8. File Structure

```
HybdRAG_bot/
├── Core Engine
│   ├── rag_engine.py          # Main RAG pipeline with PCC integration
│   ├── graph_store.py         # Neo4j vector + graph operations
│   ├── pcc_memory.py          # PCC Memory module
│   └── extract.py             # PDF extraction + SciSpaCy NER
│
├── Ingestion
│   ├── build_index.py         # Batch ingestion pipeline
│   ├── ingest_entities.py     # Entity-only ingestion
│   └── ingest_neo4j_batch.py  # Batch Neo4j operations
│
├── Evaluation
│   ├── llm_judge.py           # LLM-as-Judge evaluation
│   ├── ragas_eval.py          # RAGAS metrics evaluation
│   └── evaluation/            # Evaluation scripts directory
│
├── Interfaces
│   ├── chatbot.py             # CLI chatbot
│   └── ui/
│       └── streamlit_app.py   # Streamlit web UI
│
├── Utilities
│   ├── neo4j_queries.py       # Basic Cypher queries
│   └── neo4j_complex_queries.py # Advanced multi-hop queries
│
├── Data
│   ├── data/
│   │   ├── chunks.json        # 7,597 chunks
│   │   └── marker_output/     # OCR text outputs
│   └── .env                   # API keys & Neo4j config
│
├── Configuration
│   ├── requirements.txt       # Dependencies
│   └── setup_indexes.py       # Index creation script
│
└── Documentation
    └── Docs/
        ├── 01_system_architecture.md  (this file)
        ├── 02_walkthrough_status.md
        ├── 03_ui_suggestions.md
        └── 04_cloud_deployment_roadmap.md
```

---

## 9. Configuration Reference

### 9.1 Environment Variables (.env)

```bash
# Neo4j
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678

# LLM APIs
MISTRAL_API_KEY=your-mistral-key
OPENAI_API_KEY=your-openai-key  # For evaluation only
```

### 9.2 Key Constants (rag_engine.py)

```python
FAST_MODEL = "mistral-small-latest"
DEEP_MODEL = "mistral-large-latest"
TOP_K_RETRIEVAL = 6
MAX_TOKENS = 1200
RELEVANCE_THRESHOLD = 0.45
```

---

## 10. Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Papers indexed | 226 | Devreotes lab corpus |
| Chunks stored | 7,597 | ~400 words each |
| Entity nodes | ~84,000 | Proteins, organisms, concepts |
| Avg query latency | 2.18s | Tier 1 queries |
| Embedding dimension | 768 | all-mpnet-base-v2 |
| GPU VRAM usage | ~4GB | Shared embedder model |

---

## 11. Security Considerations

1. **No Web Search:** System is structurally isolated from internet
2. **API Keys:** Stored in `.env`, never committed
3. **Grounded Responses:** Refusal message for out-of-context queries
4. **LLM Judge:** Independent validation prevents hallucination

---

*Last Updated: April 2026*
