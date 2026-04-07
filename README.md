# HybdRAG - Devreotes Research Explorer

> **Conversational AI for interrogating the research corpus of Prof. Peter Devreotes (Johns Hopkins University)**  
> **Domain:** Cell Biology → Signal Transduction & Chemotaxis  
> **Approach:** Hybrid GraphRAG with Neo4j, SciSpaCy, Mistral AI, and Surya OCR

---

## Quick Start

```powershell
# 1. Setup environment
python -m venv .venv_gpu
.\.venv_gpu\Scripts\activate
pip install -r requirements.txt

# 2. Ensure Neo4j Desktop is running

# 3. Launch Streamlit UI (port 8506)
.\.venv_gpu\Scripts\streamlit.exe run ui/streamlit_app.py --server.port 8506

# 4. Or launch CLI chatbot
.\.venv_gpu\Scripts\python.exe chatbot.py
```

---

## Documentation

Comprehensive documentation is available in the `Docs/` folder:

| Document | Description |
|----------|-------------|
| [01_system_architecture.md](docs/01_system_architecture.md) | Complete system architecture, tech stack, data flow, and Neo4j schema |
| [02_walkthrough_status.md](docs/02_walkthrough_status.md) | Tasks completed, in progress, and pending with status |
| [03_ui_suggestions.md](docs/03_ui_suggestions.md) | UI enhancement suggestions for conversational interface with chat history |
| [04_cloud_deployment_roadmap.md](docs/04_cloud_deployment_roadmap.md) | Cloud deployment roadmap from local to production |

---

## Project Structure

```
HybdRAG_bot/
├── Core Engine
│   ├── rag_engine.py          # Main RAG pipeline with PCC Memory
│   ├── graph_store.py         # Neo4j vector + graph operations
│   ├── pcc_memory.py          # PCC Memory module
│   └── extract.py             # PDF extraction + SciSpaCy NER
│
├── Interfaces
│   ├── chatbot.py             # CLI chatbot
│   └── ui/streamlit_app.py    # Streamlit web UI
│
├── Ingestion
│   └── build_index.py         # Batch ingestion pipeline
│
├── Evaluation
│   ├── llm_judge.py           # LLM-as-Judge evaluation
│   └── ragas_eval.py          # RAGAS metrics
│
├── Data
│   └── data/chunks.json       # 7,597 chunks from 226 papers
│
├── Docs/
│   ├── 01_system_architecture.md
│   ├── 02_walkthrough_status.md
│   ├── 03_ui_suggestions.md
│   └── 04_cloud_deployment_roadmap.md
│
├── requirements.txt
└── .env                       # API keys (not committed)
```

---

## Key Features

- **Hybrid GraphRAG**: Vector similarity + knowledge graph traversal
- **PCC Memory**: Personal Context Compression for persistent conversations
- **Tiered LLM Routing**: Cost-optimized query classification
- **Biomedical NER**: SciSpaCy entity extraction
- **GPU Acceleration**: Surya OCR and embeddings

---

## Current Status

| Component | Status |
|-----------|--------|
| Core Infrastructure | ✅ Complete |
| Document Ingestion | ✅ Complete |
| RAG Pipeline | ✅ Complete |
| PCC Memory | ✅ Complete |
| Frontend UI | ✅ Complete |
| Cloud Deployment | ⏳ Pending |

---

## Configuration

### Environment Variables (.env)

```bash
BIOPORTAL_API_KEY=your-bioportal-api-key
NEO4J_URI=neo4j:// URL
NEO4J_USER=neo4j
NEO4J_PASSWORD=123456789
MISTRAL_API_KEY=your-mistral-key
OPENAI_API_KEY=your-openai-key  # For evaluation/llm judge only
```

---

## Contact

For questions or issues, please refer to the documentation or contact the development team.
