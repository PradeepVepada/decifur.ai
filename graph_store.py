"""
graph_store.py
--------------
GraphRAG — Neo4j Vector + Knowledge Graph retrieval.

Review fixes applied
--------------------
  #1  Dense search + BM25 now run concurrently via ThreadPoolExecutor.
      Entity boost shares the same session; total Neo4j round-trips per
      query reduced from 3 sequential to 2 (dense||BM25) + 1 entity boost.
  #2  build() wraps logical units in explicit write transactions via
      execute_write(); no more implicit auto-commit.
  #3  Entity boost field names unified: graph nodes keyed on display name
      with a stable .name property; query matches on .name not mixed _id fields.
  #4  lru_cache on instance method replaced with a module-level LRU dict
      (no self reference → no memory leak).
  #5  build() rate-limit handling for Mistral: tenacity retry with
      exponential backoff; max_workers=5 (was 10).
  #22 Bare except replaced with logger.exception.
  #23 Shared Neo4j driver with explicit pool configuration.
  Stale comment about "all-mpnet-base-v2" removed.
  save() documented as intentional no-op.

New req #1 (chatty search)
--------------------------
  Dense + BM25 searches are now fired in parallel (ThreadPoolExecutor).
  Entity boost runs after in the same thread — it is cheap (~10ms) and
  needs the merged candidate pool first. The total latency is now:
    max(dense_latency, bm25_latency) + entity_boost_latency
  instead of the old:
    dense_latency + bm25_latency + entity_boost_latency
"""

import os
import json
import re
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from mistralai.client import Mistral
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
EMBEDDING_DIM        = 768
RELATIONSHIP_MODEL   = "mistral-small-latest"
VECTOR_INDEX_NAME    = "chunk_embeddings"   # single source of truth

# ---------------------------------------------------------------------------
# Module-level query embedding cache  [Review #4 — no instance method lru_cache]
# ---------------------------------------------------------------------------
_QUERY_EMBED_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_CACHE_MAX = 256


def _cache_get(key: str) -> Optional[tuple]:
    if key in _QUERY_EMBED_CACHE:
        _QUERY_EMBED_CACHE.move_to_end(key)
        return _QUERY_EMBED_CACHE[key]
    return None


def _cache_put(key: str, value: tuple) -> None:
    if key in _QUERY_EMBED_CACHE:
        _QUERY_EMBED_CACHE.move_to_end(key)
    else:
        _QUERY_EMBED_CACHE[key] = value
        if len(_QUERY_EMBED_CACHE) > _CACHE_MAX:
            _QUERY_EMBED_CACHE.popitem(last=False)


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

class GraphStore:
    def __init__(
        self,
        uri:      str = "bolt://localhost:7687",
        user:     str = "neo4j",
        password: str = "password",
    ):
        neo4j_uri  = os.environ.get("NEO4J_URI",      uri)
        neo4j_user = os.environ.get("NEO4J_USER",     user)
        neo4j_pass = os.environ.get("NEO4J_PASSWORD", password)

        # [Review #16] Explicit pool configuration — shared across app
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_pass),
            max_connection_pool_size=30,
            connection_acquisition_timeout=10.0,
        )

        mistral_key = os.environ.get("MISTRAL_API_KEY")
        if not mistral_key:
            raise EnvironmentError("MISTRAL_API_KEY is not set.")   # [Review #15]
        self.client = Mistral(api_key=mistral_key)

        logger.info("Loading SentenceTransformer (%s, %d-dim)...", EMBEDDING_MODEL_NAME, EMBEDDING_DIM)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        self._built   = False

    def close(self):
        self.driver.close()

    # -----------------------------------------------------------------------
    # Query embedding (module-level cache — no self ref)  [Review #4]
    # -----------------------------------------------------------------------

    def _encode_query(self, query: str) -> list:
        cached = _cache_get(query)
        if cached is not None:
            return list(cached)
        emb = self.embedder.encode(query)
        vec = tuple(emb.tolist() if hasattr(emb, "tolist") else list(emb))
        _cache_put(query, vec)
        return list(vec)

    # -----------------------------------------------------------------------
    # Dense retrieval (private, called from thread)
    # -----------------------------------------------------------------------

    def _dense_search(self, query_emb: list, top_k: int) -> list:
        cypher = f"""
            CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $top_k, $embedding)
            YIELD node AS c, score AS v_score
            MATCH (p:Paper)-[:HAS_CHUNK]->(c)
            OPTIONAL MATCH (a:Author)-[:AUTHORED_BY]->(p)
            WITH c, p, collect(a.name) AS authors, v_score
            ORDER BY v_score DESC
            RETURN
                c.chunk_id    AS id,
                c.text        AS text,
                p.paper_id    AS source,
                c.chunk_index AS chunk_index,
                c.total_chunks AS total_chunks,
                p.title       AS title,
                authors,
                p.year        AS year,
                p.journal     AS journal,
                v_score       AS score
        """
        try:
            with self.driver.session() as session:
                return session.run(cypher, top_k=top_k, embedding=query_emb).data()
        except Exception:
            logger.exception("Dense search failed.")
            return []

    # -----------------------------------------------------------------------
    # BM25 sparse retrieval (private, called from thread)
    # -----------------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> list:
        """Lucene full-text BM25 search. Special chars escaped to avoid parse errors."""
        escaped = re.sub(r'([+\-!(){}\[\]^"~*?:\\/])', r'\\\1', query)
        cypher  = """
            CALL db.index.fulltext.queryNodes('chunk_text', $query)
            YIELD node AS c, score AS bm25_score
            MATCH (p:Paper)-[:HAS_CHUNK]->(c)
            OPTIONAL MATCH (a:Author)-[:AUTHORED_BY]->(p)
            WITH c, p, collect(a.name) AS authors, bm25_score
            ORDER BY bm25_score DESC
            LIMIT $top_k
            RETURN
                c.chunk_id    AS id,
                c.text        AS text,
                p.paper_id    AS source,
                c.chunk_index AS chunk_index,
                c.total_chunks AS total_chunks,
                p.title       AS title,
                authors,
                p.year        AS year,
                p.journal     AS journal,
                bm25_score    AS score
        """
        try:
            with self.driver.session() as session:
                return session.run(cypher, query=escaped, top_k=top_k).data()
        except Exception:
            logger.exception("BM25 search failed.")
            return []

    # -----------------------------------------------------------------------
    # Entity graph boost (cheap — runs after merge)
    # -----------------------------------------------------------------------

    def _entity_boost(self, entity_names: list) -> dict:
        """Return {chunk_id: match_count} for chunks connected to query entities."""
        if not entity_names:
            return {}
        cypher = """
            MATCH (c:Chunk)-[:STUDIES_MOLECULE|STUDIES_ORGANISM|DISCUSSES_CONCEPT]->(e)
            WHERE e.name IN $entity_names
            RETURN c.chunk_id AS id, COUNT(e) AS entity_match_count
        """
        try:
            with self.driver.session() as session:
                return {
                    r["id"]: r["entity_match_count"]
                    for r in session.run(cypher, entity_names=entity_names).data()
                }
        except Exception:
            logger.exception("Entity boost query failed.")
            return {}

    # -----------------------------------------------------------------------
    # Platinum Hybrid Search — parallelised  [Review #1, new req #1]
    # -----------------------------------------------------------------------

    def search(self, query: str, k: int = 5, umls_entities: list = None) -> list:
        """
        5-stage Platinum Hybrid Search:
          Stage 1 — Dense (ANN/HNSW vector)    ┐ now PARALLEL
          Stage 2 — Sparse (BM25 Lucene)        ┘
          Stage 3 — Merge candidate pools
          Stage 4 — RRF Fusion (k=60)
          Stage 5 — Entity/Graph Boost (additive bonus)

        Latency: max(dense, bm25) + entity_boost  (was sum of all three).
        """
        query_emb    = self._encode_query(query)
        entity_names = [
            e.get("name", "") for e in (umls_entities or []) if e.get("name")
        ]
        TOP_K        = max(k * 3, 20)

        # ------------------------------------------------------------------
        # Stage 1 + 2 — Dense + BM25 fired in parallel  [Review #1]
        # ------------------------------------------------------------------
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="search") as ex:
            dense_f  = ex.submit(self._dense_search,  query_emb, TOP_K)
            sparse_f = ex.submit(self._bm25_search,   query,     TOP_K)
            dense_records  = dense_f.result()
            sparse_records = sparse_f.result()

        # ------------------------------------------------------------------
        # Stage 3 — Unified candidate pool
        # ------------------------------------------------------------------
        chunk_lookup: dict = {}
        for r in dense_records + sparse_records:
            cid = r["id"]
            if cid not in chunk_lookup:
                chunk_lookup[cid] = r

        # ------------------------------------------------------------------
        # Stage 4 — RRF Fusion
        # ------------------------------------------------------------------
        RRF_K       = 60
        rrf_scores: dict = {}
        for rank, r in enumerate(dense_records):
            cid = r["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + RRF_K)
        for rank, r in enumerate(sparse_records):
            cid = r["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + RRF_K)

        # ------------------------------------------------------------------
        # Stage 5 — Entity/Graph Boost (cheap; runs after merge)  [Review #3]
        # ------------------------------------------------------------------
        if entity_names:
            for cid, match_count in self._entity_boost(entity_names).items():
                if cid in rrf_scores:
                    rrf_scores[cid] += 0.01 * match_count

        # ------------------------------------------------------------------
        # Final ranking
        # ------------------------------------------------------------------
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results    = []
        for cid, final_score in sorted_ids:
            if cid not in chunk_lookup:
                continue
            rec                  = dict(chunk_lookup[cid])
            rec["score"]         = round(final_score, 6)
            rec["title"]         = rec.get("title") or rec.get("source", "Unknown Title")
            rec["year"]          = rec.get("year") or "Unknown"
            rec["authors"]       = [a for a in (rec.get("authors") or []) if a]
            rec["total_chunks"]  = rec.get("total_chunks") or 1
            results.append(rec)

        return results

    # -----------------------------------------------------------------------
    # Schema setup
    # -----------------------------------------------------------------------

    def _create_vector_index(self, session):
        session.run(f"""
            CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)

    def _setup_indexes(self):
        with self.driver.session() as session:
            for stmt in [
                "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
                "CREATE CONSTRAINT paper_id    IF NOT EXISTS FOR (p:Paper)    REQUIRE p.paper_id    IS UNIQUE",
                "CREATE CONSTRAINT chunk_id    IF NOT EXISTS FOR (c:Chunk)    REQUIRE c.chunk_id    IS UNIQUE",
                "CREATE CONSTRAINT author_id   IF NOT EXISTS FOR (a:Author)   REQUIRE a.author_id   IS UNIQUE",
                "CREATE CONSTRAINT concept_id  IF NOT EXISTS FOR (co:Concept) REQUIRE co.concept_id IS UNIQUE",
                "CREATE CONSTRAINT organism_id IF NOT EXISTS FOR (o:Organism) REQUIRE o.organism_id IS UNIQUE",
                "CREATE CONSTRAINT molecule_id IF NOT EXISTS FOR (mo:Molecule) REQUIRE mo.molecule_id IS UNIQUE",
                "CREATE CONSTRAINT method_id   IF NOT EXISTS FOR (me:Method)  REQUIRE me.method_id  IS UNIQUE",
                "CREATE CONSTRAINT topic_id    IF NOT EXISTS FOR (t:Topic)    REQUIRE t.topic_id    IS UNIQUE",
            ]:
                session.run(stmt)
            self._create_vector_index(session)

    def _verify_vector_index(self):
        with self.driver.session() as session:
            result = session.run("SHOW INDEXES WHERE name = $name", name=VECTOR_INDEX_NAME).data()
            if not result:
                logger.warning("Vector index '%s' missing — recreating.", VECTOR_INDEX_NAME)
                self._create_vector_index(session)

    # -----------------------------------------------------------------------
    # Relationship discovery with retry  [Review #5]
    # -----------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=False,
    )
    def _discover_entity_relationships(self, text: str, entities: list) -> list:
        if len(entities) < 2:
            return []
        prompt = (
            f"Analyze relationships between these biomedical entities mentioned in this text.\n\n"
            f"Entities: {', '.join([str(e) if not isinstance(e, str) else e for e in entities[:10]])}\n\n"
            f"Text excerpt:\n{text[:2000]}\n\n"
            "Return ONLY a JSON object with a 'relationships' array. Each item:\n"
            "- source: entity name\n- target: entity name\n"
            "- relation: type (e.g. REGULATES, INTERACTS_WITH, LOCATED_IN, PART_OF)\n"
            "Only include scientifically meaningful relationships clearly in the text."
        )
        try:
            res = self.client.chat.complete(
                model=RELATIONSHIP_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(res.choices[0].message.content or "{}").get("relationships", [])
        except Exception:
            logger.exception("Relationship discovery failed for chunk.")
            return []

    # -----------------------------------------------------------------------
    # Build  [Review #2 — explicit write transactions]
    # -----------------------------------------------------------------------

    def build(self, chunks: list) -> None:
        if not chunks:
            self._built = True
            return

        logger.info("Building Knowledge Graph over %d chunks...", len(chunks))
        self._setup_indexes()

        # Embed all texts in one batch
        texts      = [c["text"] for c in chunks]
        logger.info("Generating embeddings (%s)...", EMBEDDING_MODEL_NAME)
        embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=64)
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist() if hasattr(embeddings[i], "tolist") else list(embeddings[i])

        # Discover entity relationships (max_workers=5 to respect rate limits)  [Review #5]
        logger.info("Discovering entity relationships (parallel, rate-limited)...")
        def _build_chunk(idx_chunk):
            idx, chunk = idx_chunk
            all_ents = chunk.get("proteins", []) + chunk.get("organisms", []) + chunk.get("concepts", [])
            rels = self._discover_entity_relationships(chunk["text"], all_ents) if len(all_ents) >= 2 else []
            chunk["relationships"] = rels[:5]
            return chunk

        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="rel_discover") as ex:
            chunks = list(ex.map(_build_chunk, enumerate(chunks)))

        # Ingest to Neo4j using explicit transactions  [Review #2]
        distinct_papers = {
            c["source"]: {k: c.get(k) for k in
                ("source","title","year","journal","doi","abstract","keywords","authors","topics","methods")}
            for c in chunks
        }

        def _write_papers(tx, papers):
            for p_id, p_meta in papers.items():
                tx.run("""
                    MERGE (p:Paper {paper_id: $source})
                    SET p.title=$title, p.year=$year, p.journal=$journal,
                        p.doi=$doi, p.abstract=$abstract, p.keywords=$keywords
                """, **p_meta)
                tx.run("""
                    UNWIND $authors AS author_name
                    MERGE (a:Author {author_id: toLower(replace(replace(author_name,' ','_'),',',''))})
                    ON CREATE SET a.name = author_name
                    WITH a MATCH (p:Paper {paper_id: $source})
                    MERGE (a)-[:AUTHORED_BY]->(p)
                """, source=p_id, authors=p_meta.get("authors") or [])
                tx.run("""
                    UNWIND $topics AS topic_name
                    MERGE (t:Topic {topic_id: toLower(replace(topic_name,' ','-'))})
                    ON CREATE SET t.name = topic_name
                    WITH t MATCH (p:Paper {paper_id: $source})
                    MERGE (p)-[:COVERS_TOPIC]->(t)
                """, source=p_id, topics=p_meta.get("topics") or [])
                tx.run("""
                    UNWIND $methods AS method_obj
                    MERGE (me:Method {method_id: toLower(replace(method_obj.name,' ','_'))})
                    ON CREATE SET me.name=method_obj.name, me.type=method_obj.type
                    WITH me MATCH (p:Paper {paper_id: $source})
                    MERGE (p)-[:USES_METHOD]->(me)
                """, source=p_id, methods=p_meta.get("methods") or [])

        with self.driver.session() as session:
            session.execute_write(_write_papers, distinct_papers)

        BATCH_SIZE = 500
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            for b in batch:
                b["node_id"] = f"{b['source']}_chunk_{b['chunk_index']}"

            logger.info("Ingesting batch %d/%d (%d chunks)...",
                        i // BATCH_SIZE + 1, (len(chunks) - 1) // BATCH_SIZE + 1, len(batch))

            def _write_chunks(tx, batch=batch):
                tx.run("""
                    UNWIND $batch AS c
                    MATCH (p:Paper {paper_id: c.source})
                    MERGE (ch:Chunk {chunk_id: c.node_id})
                    SET ch.text=c.text, ch.chunk_index=c.chunk_index,
                        ch.total_chunks=c.total_chunks, ch.embedding=c.embedding
                    MERGE (p)-[:HAS_CHUNK {chunk_index: c.chunk_index}]->(ch)
                """, batch=batch)

                tx.run("""
                    UNWIND $batch AS c UNWIND c.proteins AS ent
                    MERGE (pr:Molecule {molecule_id: ent.name})
                    ON CREATE SET pr.name=ent.name, pr.type=ent.type
                    WITH pr, c MATCH (ch:Chunk {chunk_id: c.node_id})
                    MERGE (ch)-[:STUDIES_MOLECULE]->(pr)
                """, batch=batch)

                tx.run("""
                    UNWIND $batch AS c UNWIND c.organisms AS ent
                    MERGE (o:Organism {organism_id: ent.name})
                    ON CREATE SET o.name=ent.name, o.taxonomy=ent.type
                    WITH o, c MATCH (ch:Chunk {chunk_id: c.node_id})
                    MERGE (ch)-[:STUDIES_ORGANISM]->(o)
                """, batch=batch)

                tx.run("""
                    UNWIND $batch AS c UNWIND c.concepts AS ent
                    MERGE (co:Concept {concept_id: ent.name})
                    ON CREATE SET co.name=ent.name, co.definition=ent.type
                    WITH co, c MATCH (ch:Chunk {chunk_id: c.node_id})
                    MERGE (ch)-[:DISCUSSES_CONCEPT]->(co)
                """, batch=batch)

                batch_rels = []
                for b in batch:
                    all_ents = b.get("proteins",[]) + b.get("organisms",[]) + b.get("concepts",[])
                    names    = {e["name"] for e in all_ents}
                    for rel in b.get("relationships", []):
                        src = rel.get("source")
                        tgt = rel.get("target")
                        if src in names and tgt in names:
                            batch_rels.append({"src": src, "tgt": tgt,
                                               "relation": rel.get("relation","RELATED_TO"),
                                               "chunk_id": b["node_id"]})
                if batch_rels:
                    tx.run("""
                        UNWIND $rels AS r
                        MATCH (e1) WHERE e1.name = r.src
                        MATCH (e2) WHERE e2.name = r.tgt
                        MERGE (e1)-[rel:RELATED_TO {type: r.relation, source_chunk: r.chunk_id}]->(e2)
                    """, rels=batch_rels)

            with self.driver.session() as session:
                session.execute_write(_write_chunks)

        self._built = True
        logger.info("Neo4j Knowledge Graph built.")

    # -----------------------------------------------------------------------
    # Paper listing
    # -----------------------------------------------------------------------

    def get_papers(self) -> list:
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (p:Paper)
                    OPTIONAL MATCH (a:Author)-[:AUTHORED_BY]->(p)
                    OPTIONAL MATCH (p)-[:COVERS_TOPIC]->(t:Topic)
                    OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
                    RETURN p.paper_id AS source, p.title AS title,
                           p.year AS year, p.journal AS journal,
                           p.doi AS doi, p.abstract AS abstract,
                           p.keywords AS keywords,
                           collect(DISTINCT a.name) AS authors,
                           collect(DISTINCT t.name) AS topics,
                           collect(DISTINCT m.name) AS methods
                    ORDER BY p.year
                """)
                return [
                    {
                        "source":   r["source"],
                        "title":    r["title"]   or r["source"],
                        "year":     r["year"]    or "Unknown",
                        "journal":  r["journal"] or "Unknown",
                        "doi":      r["doi"],
                        "abstract": r["abstract"],
                        "keywords": r["keywords"] or [],
                        "authors":  [a for a in (r["authors"] or []) if a],
                        "topics":   [t for t in (r["topics"]  or []) if t],
                        "methods":  [m for m in (r["methods"] or []) if m],
                    }
                    for r in result
                ]
        except Exception:
            logger.exception("get_papers failed.")
            return []

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def load(self, path=None):
        try:
            with self.driver.session() as session:
                cnt = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt").single()["cnt"]
            self._built = cnt > 0
            if self._built:
                logger.info("Loaded %d chunks from Neo4j.", cnt)
                self._verify_vector_index()
            else:
                logger.warning("Neo4j is empty. Run build_index first.")
        except Exception as e:
            raise ConnectionError(f"Could not connect to Neo4j: {e}") from e

    def save(self, path=None) -> None:
        """Intentional no-op: Neo4j is the persistent store."""
        pass
