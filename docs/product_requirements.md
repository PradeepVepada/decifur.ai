# 📋 Product Requirements Document (PRD)

## 1. Product Vision & Goals
**Name**: Devreotes Research Explorer (GraphRAG Conversational AI)  
**Objective**: Provide a highly accurate, domain-specific search and QA assistant on Prof. Peter Devreotes' closed research corpus.
**Core Constraint**: Grounded strictly in the provided scientific literature. **No web search.** No external hallucinations.

## 2. Target Audience
- **Primary**: NLP/Bio-researchers and students seeking synthesized, cross-document analysis of specific methods, proteins (like PTEN, PI3K), and pathways.
- **Secondary**: Prof. Devreotes and Lab members querying historical progress, cross-paper methodologies, and precise biological entities.

## 3. Success Metrics & Balancing Objectives
Balancing high-quality output against API costs is the central tenet of this PRD.
- **Quality**: 
  - Strict grounding in source materials.
  - Multi-hop traversal (GraphRAG) for deep synthesis.
  - Accurate entity extraction utilizing UMLS and SciSpaCy.
- **Cost Efficiency**: 
  - Dynamic routing of queries to prevent using expensive LLMs (OpenAI `gpt-4o`) for simple extraction tasks.
  - Pre-computation of relationships via Neo4j Desktop.

## 4. Key Functional Requirements

### 4.1. Closed-Domain Document Ingestion & Chunking
- Process PDF papers, extract text with layout intact.
- Chunk text semantically while retaining context (e.g., 400 words, 20-word overlap).

### 4.2. Medical Entity Extraction (UMLS + SciSpaCy)
- Treat **UMLS** as the absolute baseline/benchmark for entity normalization.
- Use **SciSpaCy** for robust named entity recognition (NER) across complex bio-medical text (genes, proteins, chemicals, methods).
- All discovered entities must map to a standardized UMLS identifier, feeding directly into the graph schema.

### 4.3. Knowledge Graph Construction (Neo4j Desktop)
- Build a structured representation of the text: `[Paper] -> [Entity] <- [Method]`.
- Start by building nodes and relationships locally in **Neo4j Desktop** (free, powerful, no immediate cloud database costs).
- Once nodes and relationships are structurally sound, the system allows pushing to Neo4j Aura if collaborative deployment is required later.

### 4.4. Tiered Query Routing Mechanism (Cost Optimization)
- **Tier 1 (Vector DB Exact Match / Simple Fetch)**: Uses purely `gpt-4o-mini` with basic semantic search. Low latency, cheapest.
- **Tier 2 (Entity + Graph Traversal)**: For questions requiring relationships. Retrieves graph paths via Cypher + vector similarities, summarized by `gpt-4o-mini`.
- **Tier 3 (Deep Reasoning & Cross-Paper Synthesis)**: For hard, multi-document synthesis queries. Engages `gpt-4o`. Expensive but high quality.

### 4.5. OpenAI Only
- Utilize `gpt-4o-mini` for routine routing, evaluation (LLM-as-a-judge), and standard summarization to save costs.
- Utilize `gpt-4o` solely for Tier 3 reasoning.

## 5. Non-Functional Requirements
- **Latency**: P95 latency under 5 seconds for standard queries.
- **Cost**: Target an average query cost of less than $0.01.
- **Extensibility**: The system should allow new papers to be ingested seamlessly without completely rebuilding the graph.

## 6. Exclusions
- **Web Search**: Absolutely no external internet access for answering questions.
- **Open-source LLMs**: Stick strictly to OpenAI endpoints per user requirement.
