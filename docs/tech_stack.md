# 💻 Technology Stack & Architecture

## Overview
Optimized for the highest answer quality while minimizing OpenAI API costs. The stack leverages local execution where compute is free (graph building, basic NLP) and delegates reasoning tasks to OpenAI's varied models to achieve a balance of insight and economy. Web search is structurally excluded.

## Core Infrastructure

| Component | Technology | Purpose & Justification |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Ecosystem standard for ML/Data tools. |
| **Frontend UI** | Streamlit | Rapid prototyping for Chat UI, Admin dashboards. |
| **Knowledge Graph** | **Neo4j Desktop** | Free local execution environment for constructing the foundational knowledge graph (nodes + relationships). Allows scaling and iteration without cloud costs before migrating to Neo4j Aura. |

## AI & Machine Learning Pipeline

### NLP & Entity Linking
- **SciSpaCy (`en_core_sci_lg`, `en_ner_bionlp13cg_md`)**: Biomedical entity extraction optimized for scientific texts. Extracts Genes, Proteins, Model Organisms, and Methods.
- **UMLS Validation (`scispacy.linker`)**: The primary baseline and benchmark for normalizing biological terms. All entities discovered must link to a UMLS Concept Unique Identifier (CUI). This resolves synonym ambiguities across decades of literature.

### Document Processing
- **PyMuPDF (`fitz`)**: For fast, layout-aware extraction of scientific notation and reference sections.
- **Sentence Transformers** or **OpenAI `text-embedding-3-small`**: For vector representations of chunked documents. The OpenAI embedding is exceptionally cheap ($0.02 / 1M tokens) and highly performant.

### LLM Orchestration
- **OpenAI API Platform** Only. No other LLM providers (Anthropic, Gemini) or local models (Ollama).
- **LangChain / LlamaIndex**: To manage document chunking, prompt execution, tool routing, and RAG pipelines.

## Cost-Balanced AI Routing Strategy

Instead of sending every request to a large reasoning model, the system introduces a **Routing Layer**:

1. **Intent Router**: Uses **`gpt-4o-mini`**. Inexpensive classification layer determining if the query is simple, relational, or complex.
2. **Tier 1 (Fast, Cheap)**: Direct Cypher traversal & cached keyword lookups using **`gpt-4o-mini`**. Perfect for explicit factual lookups (e.g., "Which paper introduced LEGI?").
3. **Tier 2 (Relational)**: Uses **Neo4j Desktop** Graph traversal + Vector DB search. The summarized context is sent to **`gpt-4o-mini`** to build the answer based strictly on UMLS unified terms.
4. **Tier 3 (Deep Synthesis)**: Requires cross-paper multi-hop deductions. Context retrieved from the graph is passed to **`gpt-4o`** for premium generation.
5. **Evaluator/Judge**: To verify faithfulness to the corpus, a two-step LLM-as-a-judge system is used. The Generator leverages `gpt-4o` / `gpt-4o-mini` while Judges only use `gpt-4o-mini` to keep costs extremely low.

## Deployment & Production
- **Milestone 1 (Local First)**: Run Neo4j Desktop locally. Embeddings, chunking, and relationship mapping occur here to keep database costs at zero during development.
- **Milestone 2 (Aura Migration)**: Once the graph is stable and QA passes the UMLS benchmark, export Neo4j Desktop to Neo4j Aura (cloud) so the frontend can be deployed globally without a local database umbilical.
- **Strict Network Bounds**: The application backend is structurally isolated from internet searches. Prompts strictly forbid out-of-bounds hallucinatory facts.
