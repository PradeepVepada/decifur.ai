"""
pcc_memory.py
-------------
Personal Context Compression (PCC) Memory Module for HybdRAG.

What PCC does
-------------
  Short-term : sliding window of the last N messages kept in RAM.
               Compression happens LAZILY — only when the memory context
               is actually requested by the RAG engine, not on every turn.
  Long-term  : compressed episode embeddings stored in Neo4j, retrieved
               via ANN vector search per query.

Review fixes applied
--------------------
  #17 LLM compression was called synchronously inside add_message() on
      every N-th turn — now LAZY: compression only runs when
      get_short_term_context() is called, never on the hot write path.
  #18 Fallback embedder was "all-mpnet-base-v2"; changed to match the
      system model "NeuML/pubmedbert-base-embeddings".
  #22 Bare except replaced with logger.exception.
  #28 Hardcoded topic keywords replaced by scispaCy NER entity extraction.
  #27 LONG_TERM_EXPIRY_DAYS is now enforced: a cleanup run removes stale
      episodes on every startup.
  Fix 4 (new req): PCC memory fully wired to rag_engine — the short-term
      and long-term contexts are always surfaced as distinct labelled
      blocks in the system prompt so the LLM actually uses them.
"""

import os
import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SHORT_TERM_WINDOW     = 10     # Messages kept in sliding window
LONG_TERM_EXPIRY_DAYS = 30     # Episodes older than this are pruned
# Must match graph_store.py EMBEDDING_MODEL_NAME  [Review #18]
EMBEDDING_MODEL       = "NeuML/pubmedbert-base-embeddings"
EMBEDDING_DIM         = 768
LLM_COMPRESS_EVERY_N  = 6      # Compress short-term every N new messages


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryEpisode:
    episode_id:            str
    user_id:               str
    conversation_id:       str
    content:               str
    embedding:             list
    timestamp:             str
    message_count:         int
    topics:                list
    pcc_compression_ratio: float


@dataclass
class ShortTermMemory:
    messages:           list
    compressed_summary: str
    embedding:          list    # populated lazily
    last_updated:       str


# ---------------------------------------------------------------------------
# PCCMemory
# ---------------------------------------------------------------------------

class PCCMemory:
    """Personal Context Compression Memory Manager."""

    def __init__(
        self,
        user_id:       str = "default",
        neo4j_uri:     str = None,
        neo4j_user:    str = "neo4j",
        neo4j_pass:    str = "",
        embedder:      SentenceTransformer = None,
        mistral_client = None,
    ):
        self.user_id          = user_id
        self.conversation_id  = f"conv_{uuid.uuid4().hex[:12]}"
        self.mistral_client   = mistral_client

        # [Review #17] Lazy compression tracking
        self._message_count_since_compress = 0
        self._compression_dirty            = False

        # [Review #18] Use shared embedder; fallback uses correct model name
        if embedder is not None:
            self.embedder      = embedder
            self._owns_embedder= False
        else:
            device             = "cpu"  # safe default; caller can override
            self.embedder      = SentenceTransformer(EMBEDDING_MODEL, device=device)
            self._owns_embedder= True
            logger.info("PCCMemory: loaded own embedder (%s).", EMBEDDING_MODEL)

        self.driver = None
        if neo4j_uri:
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_pass),
                max_connection_pool_size=10,
                connection_acquisition_timeout=5.0,
            )
            self._setup_memory_schema()
            self._prune_expired_episodes()   # [Review #27] enforce expiry at startup

        self.short_term = ShortTermMemory(
            messages=[],
            compressed_summary="",
            embedding=[],
            last_updated=datetime.now().isoformat(),
        )

    # -----------------------------------------------------------------------
    # Schema
    # -----------------------------------------------------------------------

    def _setup_memory_schema(self):
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT memory_episode_id IF NOT EXISTS
                FOR (m:MemoryEpisode) REQUIRE m.episode_id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT pcc_user_id IF NOT EXISTS
                FOR (u:PCCUser) REQUIRE u.user_id IS UNIQUE
            """)
            try:
                session.run(f"""
                    CREATE VECTOR INDEX memory_episode_embeddings IF NOT EXISTS
                    FOR (m:MemoryEpisode) ON (m.embedding)
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {EMBEDDING_DIM},
                        `vector.similarity_function`: 'cosine'
                    }}}}
                """)
            except Exception:
                logger.exception("Could not create memory vector index.")

    def _prune_expired_episodes(self):
        """[Review #27] Delete episodes older than LONG_TERM_EXPIRY_DAYS."""
        if not self.driver:
            return
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (m:MemoryEpisode)
                    WHERE m.timestamp < datetime() - duration($duration)
                    WITH m LIMIT 500
                    DETACH DELETE m
                    RETURN count(m) AS deleted
                """, duration=f"P{LONG_TERM_EXPIRY_DAYS}D")
                deleted = (result.single() or {}).get("deleted", 0)
                if deleted:
                    logger.info("PCC: pruned %d expired episodes.", deleted)
        except Exception:
            logger.exception("PCC: episode pruning failed.")

    # -----------------------------------------------------------------------
    # Compression  [Review #17 — lazy, not on every add_message]
    # -----------------------------------------------------------------------

    def _compress_short_term_extractive(self, messages: list) -> str:
        """Fast extractive fallback — no LLM call."""
        recent   = messages[-SHORT_TERM_WINDOW:]
        all_text = " ".join(m.get("content", "") for m in recent if m.get("content"))
        sentences= [s.strip() for s in all_text.split(".") if s.strip()]
        summary  = ""
        for sent in sentences:
            if len(summary) + len(sent) > 500:
                break
            summary += sent + ". "
        return summary.strip() or all_text[:500]

    def _compress_short_term_llm(self, messages: list) -> str:
        """LLM-based semantic compression. Only called when dirty + LLM available."""
        if not self.mistral_client:
            return self._compress_short_term_extractive(messages)
        try:
            turns = "\n".join(
                f"{m['role']}: {m.get('content', '')[:400]}"
                for m in messages[-SHORT_TERM_WINDOW:]
            )
            res = self.mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": (
                    "Summarize this research conversation in 3 concise sentences. "
                    "Preserve all scientific entity names, key findings, and claims. "
                    "Do NOT add outside knowledge.\n\n" + turns
                )}],
                temperature=0,
            )
            compressed = (res.choices[0].message.content or "").strip()
            return compressed if compressed else self._compress_short_term_extractive(messages)
        except Exception:
            logger.exception("PCC: LLM short-term compression failed; using extractive.")
            return self._compress_short_term_extractive(messages)

    def _run_compression_if_needed(self):
        """Called lazily from get_short_term_context — never from add_message."""
        if not self._compression_dirty or not self.short_term.messages:
            return
        use_llm = (
            self.mistral_client is not None
            and self._message_count_since_compress >= LLM_COMPRESS_EVERY_N
        )
        if use_llm:
            self.short_term.compressed_summary = self._compress_short_term_llm(
                self.short_term.messages
            )
            self._message_count_since_compress = 0
        else:
            self.short_term.compressed_summary = self._compress_short_term_extractive(
                self.short_term.messages
            )
        self.short_term.last_updated = datetime.now().isoformat()
        self._compression_dirty = False

    # -----------------------------------------------------------------------
    # Short-term memory
    # -----------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """
        Append a message. Does NOT trigger compression. [Review #17]
        Compression is deferred to get_short_term_context().
        """
        self.short_term.messages.append({
            "role":      role,
            "content":   content,
            "timestamp": datetime.now().isoformat(),
        })
        self._message_count_since_compress += 1
        self._compression_dirty = True

        # Trim sliding window
        max_window = SHORT_TERM_WINDOW * 2
        if len(self.short_term.messages) > max_window:
            self.short_term.messages = self.short_term.messages[-max_window:]

    def get_short_term_context(self) -> dict:
        """Lazily compress if dirty, then return context."""
        self._run_compression_if_needed()
        return {
            "messages":           self.short_term.messages[-SHORT_TERM_WINDOW:],
            "compressed_summary": self.short_term.compressed_summary,
            "message_count":      len(self.short_term.messages),
        }

    def clear_short_term(self) -> None:
        self.short_term = ShortTermMemory(
            messages=[], compressed_summary="", embedding=[],
            last_updated=datetime.now().isoformat(),
        )
        self._message_count_since_compress = 0
        self._compression_dirty            = False

    # -----------------------------------------------------------------------
    # Long-term memory
    # -----------------------------------------------------------------------

    def compress_and_store_long_term(self, conversation_history: list) -> Optional[MemoryEpisode]:
        if not self.driver or not conversation_history:
            return None

        all_content = " ".join(
            f"{m.get('role','')}: {m.get('content','')}"
            for m in conversation_history if m.get("content")
        )
        topics     = self._extract_topics(all_content)
        compressed = self._compress_long_term(all_content)
        embedding  = self._embed_text(compressed)
        ratio      = len(compressed) / max(len(all_content), 1)

        episode = MemoryEpisode(
            episode_id            = f"ep_{self.user_id}_{uuid.uuid4().hex[:12]}",
            user_id               = self.user_id,
            conversation_id       = self.conversation_id,
            content               = compressed,
            embedding             = embedding,
            timestamp             = datetime.now().isoformat(),
            message_count         = len(conversation_history),
            topics                = topics,
            pcc_compression_ratio = ratio,
        )
        self._store_episode(episode)
        return episode

    def _compress_long_term(self, text: str, max_length: int = 600) -> str:
        if self.mistral_client:
            try:
                res = self.mistral_client.chat.complete(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": (
                        f"Compress this research conversation into a dense factual summary "
                        f"of at most {max_length} characters. Preserve all scientific entity "
                        f"names, relationships, and key findings.\n\n{text[:3000]}"
                    )}],
                    temperature=0,
                )
                compressed = (res.choices[0].message.content or "").strip()
                if compressed:
                    return compressed[:max_length]
            except Exception:
                logger.exception("PCC: LLM long-term compression failed; using extractive.")
        sentences   = text.split(". ")
        key         = [sentences[0]] + sentences[2:-2][:3] + [sentences[-1]] if len(sentences) > 5 else sentences
        compressed  = ". ".join(key)
        return (compressed[:max_length] + "...") if len(compressed) > max_length else compressed

    def _embed_text(self, text: str) -> list:
        if not text:
            return [0.0] * EMBEDDING_DIM
        emb = self.embedder.encode(text, show_progress_bar=False)
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def _store_episode(self, episode: MemoryEpisode) -> None:
        """Single atomic transaction. [Review #1 — original fix carried forward]"""
        if not self.driver:
            return
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (u:PCCUser {user_id: $user_id})
                    CREATE (m:MemoryEpisode {
                        episode_id:            $episode_id,
                        user_id:               $user_id,
                        conversation_id:       $conversation_id,
                        content:               $content,
                        embedding:             $embedding,
                        timestamp:             datetime($timestamp),
                        message_count:         $message_count,
                        topics:                $topics,
                        pcc_compression_ratio: $pcc_compression_ratio
                    })
                    MERGE (u)-[:HAS_EPISODE]->(m)
                """, **asdict(episode))
        except Exception:
            logger.exception("PCC: failed to store episode %s.", episode.episode_id)

    def retrieve_long_term_memory(self, query: str, top_k: int = 3) -> list:
        """ANN vector search over stored episodes."""
        if not self.driver:
            return []
        try:
            query_embedding = self._embed_text(query)
            with self.driver.session() as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes(
                        'memory_episode_embeddings', $top_k, $query_embedding
                    )
                    YIELD node AS m, score
                    MATCH (u:PCCUser {user_id: $user_id})-[:HAS_EPISODE]->(m)
                    WHERE m.timestamp > datetime() - duration($duration)
                    RETURN
                        m.episode_id            AS episode_id,
                        m.content               AS content,
                        m.topics                AS topics,
                        m.timestamp             AS timestamp,
                        m.pcc_compression_ratio AS compression_ratio,
                        score                   AS similarity
                    ORDER BY score DESC
                    LIMIT $top_k
                """, user_id=self.user_id, query_embedding=query_embedding,
                     top_k=top_k, duration=f"P{LONG_TERM_EXPIRY_DAYS}D")
                return result.data()
        except Exception:
            logger.exception("PCC: long-term retrieval failed.")
            return []

    # -----------------------------------------------------------------------
    # Topic extraction  [Review #28 — no longer hardcoded keyword list]
    # -----------------------------------------------------------------------

    def _extract_topics(self, text: str) -> list:
        """
        Extract topics using scispaCy if available; otherwise fall back to
        a frequency-based keyword scan. [Review #28]
        """
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_sci_lg")
            except OSError:
                nlp = spacy.load("en_core_sci_sm")
            doc = nlp(text[:2000])
            return list({ent.text for ent in doc.ents if len(ent.text) > 2})[:8]
        except Exception:
            pass
        # Frequency fallback
        words      = text.lower().split()
        stopwords  = {"the", "a", "an", "of", "in", "is", "are", "and", "or",
                      "to", "that", "this", "was", "for", "with", "have"}
        freq: dict = {}
        for w in words:
            if len(w) > 4 and w not in stopwords:
                freq[w] = freq.get(w, 0) + 1
        return sorted(freq, key=freq.get, reverse=True)[:5]

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def get_memory_summary(self) -> dict:
        summary = self.short_term.compressed_summary
        return {
            "pcc_enabled":         True,
            "user_id":             self.user_id,
            "conversation_id":     self.conversation_id,
            "short_term_messages": len(self.short_term.messages),
            "short_term_summary":  (summary[:120] + "...") if len(summary) > 120 else summary,
            "compression_dirty":   self._compression_dirty,
            "last_updated":        self.short_term.last_updated,
        }

    def close(self) -> None:
        if self.driver:
            self.driver.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_pcc_memory(
    user_id:       str = "default",
    embedder:      SentenceTransformer = None,
    mistral_client = None,
) -> PCCMemory:
    return PCCMemory(
        user_id        = user_id,
        neo4j_uri      = os.environ.get("NEO4J_URI",      "neo4j://127.0.0.1:7687"),
        neo4j_user     = os.environ.get("NEO4J_USER",     "neo4j"),
        neo4j_pass     = os.environ.get("NEO4J_PASSWORD", ""),
        embedder       = embedder,
        mistral_client = mistral_client,
    )
