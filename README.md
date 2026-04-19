# Decifur.ai - Devreotes Research Explorer

> **Conversational AI for interrogating the research corpus of Prof. Peter Devreotes (Johns Hopkins University)**  
> **Domain:** Cell Biology → Signal Transduction & Chemotaxis  
> **Approach:** Hybrid GraphRAG with Neo4j, SciSpaCy, Mistral AI, and Surya OCR

---

## Quick Start

```powershell
# 1. Setup environment
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Ensure Neo4j Desktop is running

# 3. Launch Streamlit UI (port 8506)
.\.venv\Scripts\streamlit.exe run ui/streamlit_app.py --server.port 8506

# 4. Or launch CLI chatbot
.\.venv\Scripts\python.exe chatbot.py
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
NovaAI_v2/
│
├── core/                          # Core RAG + Memory + Graph Engine
│   ├── rag_engine.py              # Main RAG pipeline
│   ├── graph_store.py             # Neo4j vector + graph ops
│   ├── pcc_memory.py              # PCC Memory module
│   ├── extract.py                 # PDF extraction + SciSpaCy NER
│   ├── biomedical_normalizer.py   # UMLS/BioPortal normalization
│   ├── conversation_store.py      # Chat history + session memory
│
├── interfaces/                    # User-facing interfaces
│   ├── chatbot.py                 # CLI chatbot
│   ├── ui/
│   │   ├── streamlit_app.py       # Streamlit UI
│   │   ├── components/            # UI widgets, layouts
│   │   └── assets/                # Images, CSS, JS
│
├── ingestion/                     # Data ingestion + indexing
│   ├── build_index.py             # Batch ingestion pipeline
│   ├── refresh_embeddings.py      # Recompute embeddings
│   ├── paper_ingest.py            # PDF → chunks pipeline
│
├── evaluation/                    # Evaluation frameworks
│   ├── llm_judge.py               # LLM-as-Judge evaluation
│   ├── ragas_eval.py              # RAGAS metrics
│   ├── tests/                     # Unit tests for eval
│
├── data/                          # Data folder (LFS recommended)
│   ├── chunks.json                # 7,736 chunks from 230 papers
│   ├── raw/                       # Raw PDFs
│   ├── processed/                 # Cleaned text
│   ├── embeddings/                # Embedding vectors
│   └── README.md                  # Document data layout
│
├── docs/                          # Documentation
│   ├── 01_system_architecture.md
│   ├── 02_walkthrough_status.md
│   ├── 03_ui_suggestions.md
│   ├── 04_cloud_deployment_roadmap.md
│   └── api_reference.md
│
├── scripts/                       # Utility scripts
│   ├── start_web.bat
│   ├── start.bat
│   ├── export_graph.py
│   └── maintenance_tools.py
│
├── tests/                         # Unit tests
│   ├── test_rag_engine.py
│   ├── test_graph_store.py
│   ├── test_memory.py
│   └── test_ui.py
│
├── models/                        # Model configs + weights (ignored)
│   ├── model_config.py
│   └── bioqwen_modelfile/         #modelfile for finetuned bioqwen model
│
├── .env_sample                    # Template for environment variables
├── .gitignore
├── .gitattributes
├── requirements.txt
├── README.md
└── api.py                         # REST API entrypoint

```

---

## Key Features

- **Hybrid GraphRAG**: Vector similarity + knowledge graph traversal
- **PCC Memory**: Personal Context Compression for persistent conversations
- **Tiered LLM Routing**: Cost-optimized query classification
- **Biomedical NER**: SciSpaCy entity extraction
- **GPU Acceleration**: Surya OCR and embeddings
- **Makes use of Bioportal with SciSpacy for entity extraction and UMLS as fallback

---

## Current Status

| Component | Status |
|-----------|--------|
| Core Infrastructure | ✅ Complete |
| Document Ingestion | ✅ Complete |
| RAG Pipeline | ✅ Complete |
| PCC Memory | ✅ Complete |
| Frontend UI | ✅ Complete |
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
