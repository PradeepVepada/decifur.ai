"""
conversation_store.py
---------------------
Persistent conversation storage backed by Neo4j.

Review fixes applied
--------------------
  #6  add_message() called result.single() twice (cursor consumed on first
      call, second returned None → count was always 0 → every message got
      index=0 → ordering was broken). Fixed by storing the single record.
  #16 Shared Neo4j driver with explicit pool config; injected from outside
      so GraphStore, ConversationStore, and PCCMemory share one pool.
  #24 delete_conversation() now uses a single DETACH DELETE (atomic).
  #25 search_conversations() uses db.index.fulltext instead of CONTAINS
      (full scan). A fulltext index on Message.content is created at startup.
  #22 Bare except replaced with logger.exception.
  #9  print() replaced with logging.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role:        str
    content:     str
    timestamp:   str
    intent:      Optional[str]  = None
    chunks_count: int           = 0
    memory_info: Optional[Dict] = None
    message_id:  Optional[str]  = None

    def to_dict(self) -> dict:
        return {
            "message_id":   self.message_id,
            "role":         self.role,
            "content":      self.content,
            "timestamp":    self.timestamp,
            "intent":       self.intent,
            "chunks_count": self.chunks_count,
            "memory_info":  self.memory_info,
        }


@dataclass
class ConversationMeta:
    conversation_id: str
    title:           str
    created_at:      str
    updated_at:      str
    message_count:   int
    primary_topics:  List[str]
    user_id:         str

    def to_dict(self) -> dict:
        return asdict(self)


class ConversationStore:
    """Neo4j-backed conversation storage."""

    def __init__(
        self,
        neo4j_uri:  str = None,
        neo4j_user: str = "neo4j",
        neo4j_pass: str = "",
        driver=None,          # Accept an injected shared driver  [Review #16]
    ):
        if driver is not None:
            # Shared driver — do not close it in our close()
            self.driver        = driver
            self._owns_driver  = False
        else:
            uri  = neo4j_uri  or os.environ.get("NEO4J_URI",      "neo4j://127.0.0.1:7687")
            user = neo4j_user or os.environ.get("NEO4J_USER",     "neo4j")
            pw   = neo4j_pass or os.environ.get("NEO4J_PASSWORD", "")
            self.driver = GraphDatabase.driver(
                uri, auth=(user, pw),
                max_connection_pool_size=20,
                connection_acquisition_timeout=5.0,
            )
            self._owns_driver = True

        self._setup_schema()

    def _setup_schema(self):
        with self.driver.session() as session:
            for stmt in [
                "CREATE CONSTRAINT conversation_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.conversation_id IS UNIQUE",
                "CREATE CONSTRAINT message_id      IF NOT EXISTS FOR (m:Message)      REQUIRE m.message_id      IS UNIQUE",
                # [Review #25] Fulltext index for message search (replaces CONTAINS scan)
                "CREATE FULLTEXT INDEX message_content IF NOT EXISTS FOR (m:Message) ON EACH [m.content]",
            ]:
                try:
                    session.run(stmt)
                except Exception:
                    logger.exception("Schema setup stmt failed: %s", stmt[:60])

    # -----------------------------------------------------------------------
    # CRUD
    # -----------------------------------------------------------------------

    def create_conversation(
        self, user_id: str = "default", title: str = None
    ) -> ConversationMeta:
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        now     = datetime.now().isoformat()
        title   = title or f"New Conversation {now[:10]}"

        with self.driver.session() as session:
            session.run("""
                CREATE (c:Conversation {
                    conversation_id: $conv_id,
                    user_id:         $user_id,
                    title:           $title,
                    created_at:      datetime($created_at),
                    updated_at:      datetime($updated_at),
                    message_count:   0,
                    primary_topics:  []
                })
            """, conv_id=conv_id, user_id=user_id, title=title,
                 created_at=now, updated_at=now)

        return ConversationMeta(
            conversation_id=conv_id, title=title,
            created_at=now, updated_at=now,
            message_count=0, primary_topics=[], user_id=user_id,
        )

    def add_message(self, conversation_id: str, message: Message) -> None:
        if not message.message_id:
            message.message_id = f"msg_{uuid.uuid4().hex[:12]}"
        if not message.timestamp:
            message.timestamp = datetime.now().isoformat()

        with self.driver.session() as session:
            # [Review #6] Store single() result in a variable — only call once
            record = session.run("""
                MATCH (c:Conversation {conversation_id: $conv_id})
                RETURN c.message_count AS count
            """, conv_id=conversation_id).single()
            count = record["count"] if record else 0

            session.run("""
                MATCH (conv:Conversation {conversation_id: $conv_id})
                CREATE (m:Message {
                    message_id:   $msg_id,
                    role:         $role,
                    content:      $content,
                    timestamp:    datetime($timestamp),
                    intent:       $intent,
                    chunks_count: $chunks_count,
                    memory_info:  $memory_info
                })
                CREATE (conv)-[:HAS_MESSAGE {index: $index}]->(m)
            """, conv_id=conversation_id, msg_id=message.message_id,
                 role=message.role, content=message.content,
                 timestamp=message.timestamp, intent=message.intent,
                 chunks_count=message.chunks_count, memory_info=message.memory_info,
                 index=count)

            session.run("""
                MATCH (c:Conversation {conversation_id: $conv_id})
                SET c.message_count = c.message_count + 1,
                    c.updated_at    = datetime($now)
            """, conv_id=conversation_id, now=datetime.now().isoformat())

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        with self.driver.session() as session:
            conv_record = session.run("""
                MATCH (c:Conversation {conversation_id: $conv_id})
                RETURN c.conversation_id AS id, c.title AS title,
                       c.created_at AS created_at, c.updated_at AS updated_at,
                       c.message_count AS message_count,
                       c.primary_topics AS topics, c.user_id AS user_id
            """, conv_id=conversation_id).single()

            if not conv_record:
                return None

            messages = [
                Message(
                    message_id   = msg["id"],
                    role         = msg["role"],
                    content      = msg["content"],
                    timestamp    = str(msg["timestamp"]),
                    intent       = msg["intent"],
                    chunks_count = msg["chunks_count"] or 0,
                    memory_info  = msg["memory_info"],
                )
                for msg in session.run("""
                    MATCH (c:Conversation {conversation_id: $conv_id})-[r:HAS_MESSAGE]->(m:Message)
                    RETURN m.message_id AS id, m.role AS role, m.content AS content,
                           m.timestamp AS timestamp, m.intent AS intent,
                           m.chunks_count AS chunks_count, m.memory_info AS memory_info,
                           r.index AS index
                    ORDER BY r.index
                """, conv_id=conversation_id)
            ]

        return {
            "meta": ConversationMeta(
                conversation_id = conv_record["id"],
                title           = conv_record["title"],
                created_at      = str(conv_record["created_at"]),
                updated_at      = str(conv_record["updated_at"]),
                message_count   = conv_record["message_count"],
                primary_topics  = conv_record["topics"] or [],
                user_id         = conv_record["user_id"],
            ),
            "messages": messages,
        }

    def list_conversations(self, user_id: str = "default", limit: int = 50) -> List[ConversationMeta]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Conversation {user_id: $user_id})
                RETURN c.conversation_id AS id, c.title AS title,
                       c.created_at AS created_at, c.updated_at AS updated_at,
                       c.message_count AS message_count,
                       c.primary_topics AS topics, c.user_id AS user_id
                ORDER BY c.updated_at DESC
                LIMIT $limit
            """, user_id=user_id, limit=limit)
            return [
                ConversationMeta(
                    conversation_id = r["id"],    title      = r["title"],
                    created_at      = str(r["created_at"]),
                    updated_at      = str(r["updated_at"]),
                    message_count   = r["message_count"],
                    primary_topics  = r["topics"] or [],
                    user_id         = r["user_id"],
                )
                for r in result
            ]

    def delete_conversation(self, conversation_id: str) -> bool:
        """[Review #24] Single atomic DETACH DELETE — no orphan risk."""
        with self.driver.session() as session:
            session.run("""
                MATCH (c:Conversation {conversation_id: $conv_id})
                OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:Message)
                DETACH DELETE c, m
            """, conv_id=conversation_id)
        return True

    def rename_conversation(self, conversation_id: str, new_title: str) -> bool:
        with self.driver.session() as session:
            session.run("""
                MATCH (c:Conversation {conversation_id: $conv_id})
                SET c.title = $title, c.updated_at = datetime($now)
            """, conv_id=conversation_id, title=new_title, now=datetime.now().isoformat())
        return True

    def search_conversations(
        self, query: str, user_id: str = "default", limit: int = 10
    ) -> List[Dict]:
        """[Review #25] Uses fulltext index — no longer a full property scan."""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.fulltext.queryNodes('message_content', $query)
                YIELD node AS m, score
                MATCH (c:Conversation {user_id: $user_id})-[:HAS_MESSAGE]->(m)
                WITH c, count(m) AS match_count, max(score) AS best_score
                RETURN c.conversation_id AS id, c.title AS title,
                       c.updated_at AS updated_at, match_count, best_score
                ORDER BY best_score DESC, match_count DESC, c.updated_at DESC
                LIMIT $limit
            """, query=query, user_id=user_id, limit=limit)
            return [
                {
                    "conversation_id": r["id"],
                    "title":           r["title"],
                    "updated_at":      str(r["updated_at"]),
                    "match_count":     r["match_count"],
                }
                for r in result
            ]

    def update_conversation_topics(self, conversation_id: str, topics: List[str]) -> None:
        with self.driver.session() as session:
            session.run("""
                MATCH (c:Conversation {conversation_id: $conv_id})
                SET c.primary_topics = $topics
            """, conv_id=conversation_id, topics=topics)

    def get_conversation_count(self, user_id: str = "default") -> int:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Conversation {user_id: $user_id})
                RETURN count(c) AS count
            """, user_id=user_id)
            record = result.single()
            return record["count"] if record else 0

    def close(self):
        if self._owns_driver and self.driver:
            self.driver.close()


def create_conversation_store(driver=None) -> ConversationStore:
    """Factory. Pass a shared driver to avoid multiple pool instances."""
    return ConversationStore(driver=driver)
