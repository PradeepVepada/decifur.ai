"""
pcc_memory.py
-------------
Session-aware Personal Context Compression (PCC) Memory Module.

Features
--------
- Short-term memory with lazy compression
- Long-term episodic memory stored in Neo4j
- Rehydration from persisted chat history
- Session-aware conversation binding
- Automatic long-term storage triggers
"""

import os
import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from model_config import PCC_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SHORT_TERM_WINDOW = 10
LONG_TERM_EXPIRY_DAYS = 30
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
EMBEDDING_DIM = 768

LLM_COMPRESS_EVERY_N = 4
STORE_EPISODE_EVERY_N = 8
MIN_EPISODE_MESSAGES = 6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryEpisode:
    episode_id: str
    user_id: str
    conversation_id: str
    content: str
    embedding: list
    timestamp: str
    message_count: int
    topics: list
    pcc_compression_ratio: float


@dataclass
class ShortTermMemory:
    messages: list
    compressed_summary: str
    embedding: list
    last_updated: str


# ---------------------------------------------------------------------------
# PCCMemory
# ---------------------------------------------------------------------------

class PCCMemory:
    """Session-aware Personal Context Compression Memory Manager."""

    def __init__(
        self,
        user_id: str = "default",
        conversation_id: Optional[str] = None,
        neo4j_uri: str = None,
        neo4j_user: str = "neo4j",
        neo4j_pass: str = "",
        embedder: SentenceTransformer = None,
        openai_client=None,
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        self.openai_client = openai_client

        self._message_count_since_compress = 0
        self._message_count_since_store = 0
        self._compression_dirty = False

        if embedder is not None:
            self.embedder = embedder
            self._owns_embedder = False
        else:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
            self._owns_embedder = True
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
            self._prune_expired_episodes()

        self.short_term = ShortTermMemory(
            messages=[],
            compressed_summary="",
            embedding=[],
            last_updated=datetime.now().isoformat(),
        )

    # -----------------------------------------------------------------------
    # Session binding / hydration
    # -----------------------------------------------------------------------

    def set_conversation_id(self, conversation_id: str) -> None:
        if conversation_id:
            self.conversation_id = conversation_id

    def hydrate_from_history(self, messages: list[dict]) -> None:
        """
        Rebuild short-term memory from an existing transcript.
        """
        cleaned: list[dict] = []
        for m in messages[-(SHORT_TERM_WINDOW * 2):]:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                cleaned.append({
                    "role": role,
                    "content": content,
                    "timestamp": m.get("timestamp", datetime.now().isoformat()),
                })

        self.short_term.messages = cleaned
        self.short_term.compressed_summary = ""
        self.short_term.last_updated = datetime.now().isoformat()

        self._compression_dirty = True
        self._message_count_since_compress = len(cleaned)
        self._message_count_since_store = 0

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
        if not self.driver:
            return
        try:
            with self.driver.session() as session:
                session.run("""
                    MATCH (m:MemoryEpisode)
                    WHERE m.timestamp < datetime() - duration($duration)
                    DETACH DELETE m
                """, duration=f"P{LONG_TERM_EXPIRY_DAYS}D")
        except Exception:
            logger.exception("PCC: episode pruning failed.")

    # -----------------------------------------------------------------------
    # Short-term memory
    # -----------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return

        self.short_term.messages.append({
            "role": role,
            "content": text,
            "timestamp": datetime.now().isoformat(),
        })
        self._message_count_since_compress += 1
        self._message_count_since_store += 1
        self._compression_dirty = True

        max_window = SHORT_TERM_WINDOW * 2
        if len(self.short_term.messages) > max_window:
            self.short_term.messages = self.short_term.messages[-max_window:]

    def _compress_short_term_extractive(self, messages: list) -> str:
        recent = messages[-SHORT_TERM_WINDOW:]
        all_text = " ".join(m.get("content", "") for m in recent if m.get("content"))
        sentences = [s.strip() for s in all_text.split(".") if s.strip()]

        summary = ""
        for sent in sentences:
            if len(summary) + len(sent) > 500:
                break
            summary += sent + ". "
        return summary.strip() or all_text[:500]

    def _compress_short_term_llm(self, messages: list) -> str:
        if not self.openai_client:
            return self._compress_short_term_extractive(messages)

        try:
            turns = "\n".join(
                f"{m['role']}: {m.get('content', '')[:400]}"
                for m in messages[-SHORT_TERM_WINDOW:]
            )
            res = self.openai_client.chat.completions.create(
                model=PCC_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarize this scientific research conversation in 3 concise sentences. "
                        "Preserve scientific entities, mechanisms, claims, conclusions, and "
                        "follow-up context. Do not add outside knowledge.\n\n"
                        + turns
                    )
                }],
                temperature=0,
            )
            compressed = (res.choices[0].message.content or "").strip()
            return compressed if compressed else self._compress_short_term_extractive(messages)
        except Exception:
            logger.exception("PCC: short-term LLM compression failed; using extractive.")
            return self._compress_short_term_extractive(messages)

    def _run_compression_if_needed(self):
        if not self._compression_dirty or not self.short_term.messages:
            return

        use_llm = (
            self.openai_client is not None
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

    def get_short_term_context(self) -> dict:
        self._run_compression_if_needed()
        return {
            "messages": self.short_term.messages[-SHORT_TERM_WINDOW:],
            "compressed_summary": self.short_term.compressed_summary,
            "message_count": len(self.short_term.messages),
        }

    def clear_short_term(self) -> None:
        self.short_term = ShortTermMemory(
            messages=[],
            compressed_summary="",
            embedding=[],
            last_updated=datetime.now().isoformat(),
        )
        self._message_count_since_compress = 0
        self._message_count_since_store = 0
        self._compression_dirty = False

    # -----------------------------------------------------------------------
    # Long-term memory
    # -----------------------------------------------------------------------

    def maybe_store_long_term(self, conversation_history: list) -> Optional[str]:
        """
        Store a compressed episode when enough new interaction has accumulated.
        """
        if not self.driver or not conversation_history:
            return None
        if len(conversation_history) < MIN_EPISODE_MESSAGES:
            return None
        if self._message_count_since_store < STORE_EPISODE_EVERY_N:
            return None

        episode = self.compress_and_store_long_term(conversation_history)
        if episode:
            self._message_count_since_store = 0
            return episode.episode_id
        return None

    def compress_and_store_long_term(self, conversation_history: list) -> Optional[MemoryEpisode]:
        if not self.driver or not conversation_history:
            return None

        all_content = " ".join(
            f"{m.get('role', '')}: {m.get('content', '')}"
            for m in conversation_history
            if m.get("content")
        )

        topics = self._extract_topics(all_content)
        compressed = self._compress_long_term(all_content)
        embedding = self._embed_text(compressed)
        ratio = len(compressed) / max(len(all_content), 1)

        episode = MemoryEpisode(
            episode_id=f"ep_{self.user_id}_{uuid.uuid4().hex[:12]}",
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            content=compressed,
            embedding=embedding,
            timestamp=datetime.now().isoformat(),
            message_count=len(conversation_history),
            topics=topics,
            pcc_compression_ratio=ratio,
        )
        self._store_episode(episode)
        return episode

    def _compress_long_term(self, text: str, max_length: int = 700) -> str:
        if self.openai_client:
            try:
                res = self.openai_client.chat.completions.create(
                    model=PCC_MODEL,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Compress this scientific conversation into a dense factual memory "
                            f"of at most {max_length} characters. Preserve entities, relationships, "
                            f"findings, uncertainties, and conclusions. Do not add outside knowledge.\n\n"
                            + text[:4000]
                        )
                    }],
                    temperature=0,
                )
                compressed = (res.choices[0].message.content or "").strip()
                if compressed:
                    return compressed[:max_length]
            except Exception:
                logger.exception("PCC: long-term LLM compression failed; using fallback.")

        return text[:max_length]

    def _embed_text(self, text: str) -> list:
        if not text:
            return [0.0] * EMBEDDING_DIM
        emb = self.embedder.encode(text, show_progress_bar=False)
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def _store_episode(self, episode: MemoryEpisode) -> None:
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
                        m.conversation_id       AS conversation_id,
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
    # Topic extraction
    # -----------------------------------------------------------------------

    def _extract_topics(self, text: str) -> list:
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

        words = text.lower().split()
        stopwords = {
            "the", "a", "an", "of", "in", "is", "are", "and", "or",
            "to", "that", "this", "was", "for", "with", "have"
        }
        freq: dict[str, int] = {}
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
            "pcc_enabled": True,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "short_term_messages": len(self.short_term.messages),
            "short_term_summary": (summary[:120] + "...") if len(summary) > 120 else summary,
            "compression_dirty": self._compression_dirty,
            "last_updated": self.short_term.last_updated,
        }

    def close(self) -> None:
        if self.driver:
            self.driver.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_pcc_memory(
    user_id: str = "default",
    conversation_id: Optional[str] = None,
    embedder: SentenceTransformer = None,
    openai_client=None,
) -> PCCMemory:
    return PCCMemory(
        user_id=user_id,
        conversation_id=conversation_id,
        neo4j_uri=os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_pass=os.environ.get("NEO4J_PASSWORD", ""),
        embedder=embedder,
        openai_client=openai_client,
    )