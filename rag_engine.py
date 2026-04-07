"""
rag_engine.py
-------------
Advanced GraphRAG pipeline with fully-wired PCC Memory.

Review fixes applied
--------------------
  #4  ask() now returns a 4-tuple (answer, chunks, intent, memory_info)
      to match what api.py expects.
  #5  conversation_history is bounded to MAX_HISTORY_TURNS pairs and
      stores only the bare query + answer (NOT the full grounded context).
  #11 Single Mistral client shared between RAGEngine and PCCMemory.
  #14 store.search() now overlaps with PCC retrieval using ThreadPoolExecutor.
  #15 Missing MISTRAL_API_KEY raises EnvironmentError at startup.
  #22 Bare except replaced with logger.exception.
  #9  All print() replaced with logging.

New req #4 — PCC fully wired
-----------------------------
  Short-term context: the compressed summary of the last N messages is
  injected as a clearly labelled block BEFORE the RAG context so the LLM
  uses it for pronouns/entity resolution within a session.

  Long-term context: top-3 past episodes retrieved via ANN and injected
  as "[Related Past Research Discussions]" — distinct from paper sources
  so the LLM does not hallucinate citations from them.

  add_message() is called BEFORE the query (user turn) and AFTER the
  response (assistant turn) so every turn is captured without gaps.

  PCC compression is LAZY — triggered only on get_short_term_context()
  retrieval, never on the hot add_message() write path.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor

from mistralai.client import Mistral

from graph_store import GraphStore
from extract import load_scispacy_model
from pcc_memory import PCCMemory, create_pcc_memory
from biomedical_normalizer import create_biomedical_normalizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FAST_MODEL       = "mistral-small-latest"
DEEP_MODEL       = "mistral-large-latest"
TOP_K_RETRIEVAL  = 14
MAX_TOKENS       = 1200
# Conversation history window: number of (user, assistant) *pairs* kept in memory
MAX_HISTORY_TURNS = 8   # = 16 messages total  [Review #5]

# RRF relevance gate — chunks must exceed this to be considered on-topic
RELEVANCE_THRESHOLD = 0.015

SYSTEM_PROMPT = """You are a scientific research assistant specializing in the papers provided in the context.

You answer questions ONLY using the excerpts from the papers provided in the RESEARCH CONTEXT section below.
Do NOT use outside knowledge or web search.

Guidelines:
- Always cite which paper(s) your answer comes from, using the title and year.
- If multiple papers address the question, synthesise across them.
- Use explicit [S#] citations for every factual claim (e.g. "PI3K was shown to [S2]").
- If a question refers to something discussed earlier in the conversation, use the CONVERSATION MEMORY section.
- If the RESEARCH CONTEXT does not contain enough information, output EXACTLY the REFUSAL MESSAGE.
- Do NOT cite the CONVERSATION MEMORY section as if it were a paper source.
"""

REFUSAL_MESSAGE = (
    "The available papers do not contain sufficient information to answer this question."
)

# ---------------------------------------------------------------------------
# Local intent classifier — zero API calls
# ---------------------------------------------------------------------------
_INTENT_KEYWORDS = {
    "cross_paper_synthesis": ["compare","across","both papers","all papers","synthesize","synthesis","contrast","between"],
    "topic_evolution":       ["over time","evolution","history","trend","how has","changed","progression"],
    "recommendation":        ["recommend","suggest","should i read","what paper","which paper"],
}

def _route_intent_local(query: str, nlp) -> str:
    q = query.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    if nlp:
        doc = nlp(query)
        if len(doc.ents) >= 2:
            return "entity_query"
    return "simple_lookup"


def build_context(chunks: list) -> tuple:
    sections, tags = [], []
    for i, chunk in enumerate(chunks, 1):
        tag = f"S{i}"
        tags.append(tag)
        authors    = chunk.get("authors", [])
        author_str = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")
        header     = f"[{tag}] {chunk.get('title','Unknown')} ({chunk.get('year','Unknown')}) -- {author_str}"
        sections.append(f"{header}\n{chunk.get('text','')}")
    return "\n\n---\n\n".join(sections), tags


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------

class RAGEngine:
    def __init__(
        self,
        user_id:           str = "default",
        enable_pcc:        bool = True,
        enable_normalizer: bool = True,
    ):
        # [Review #15] Validate API key at construction time
        mistral_key = os.environ.get("MISTRAL_API_KEY")
        if not mistral_key:
            raise EnvironmentError("MISTRAL_API_KEY is not set.")

        # [Review #11] Single shared Mistral client
        self.client            = Mistral(api_key=mistral_key)
        self.store             = GraphStore()
        self.conversation_history: list = []
        self.nlp               = None
        self.enable_pcc        = enable_pcc
        self.pcc_memory: PCCMemory | None = None
        self.enable_normalizer = enable_normalizer
        self.normalizer        = None
        self.user_id           = user_id

    def load(self) -> None:
        logger.info("Connecting to Neo4j...")
        self.store.load()

        logger.info("Loading scispaCy biomedical NER model...")
        self.nlp = load_scispacy_model()

        if self.enable_normalizer:
            logger.info("Loading Biomedical Entity Normalizer (BioPortal primary / UMLS fallback)...")
            try:
                self.normalizer = create_biomedical_normalizer()
                logger.info("Biomedical Normalizer ready.")
            except Exception:
                logger.exception("Normalizer init failed — disabling.")
                self.enable_normalizer = False

        if self.enable_pcc:
            logger.info("Initialising PCC Memory...")
            try:
                # [Review #11] Pass shared Mistral client to PCC so it doesn't create its own
                self.pcc_memory = create_pcc_memory(
                    user_id=self.user_id,
                    embedder=self.store.embedder,
                    mistral_client=self.client,
                )
                s = self.pcc_memory.get_memory_summary()
                logger.info(
                    "PCC Memory ready — %d msgs in context, dirty=%s",
                    s["short_term_messages"], s["compression_dirty"]
                )
            except Exception:
                logger.exception("PCC memory init failed — disabling.")
                self.enable_pcc = False

        logger.info("Advanced GraphRAG engine ready.")

    # -----------------------------------------------------------------------
    # Entity extraction
    # -----------------------------------------------------------------------

    def _extract_entities_from_query(self, query: str) -> list:
        """Extract + normalise entities. Normaliser runs in parallel burst."""
        doc = self.nlp(query)
        entities: list[dict] = []
        protein_types  = {"GENE_OR_GENE_PRODUCT","PROTEIN","GENE","GENE_FAMILY","RNA","DNA"}
        organism_types = {"ORGANISM","SPECIES","MULTI_CELLULAR_ORGANISM","ANATOMICAL_SYSTEM"}

        for ent in doc.ents:
            name = ent.text.strip()
            if name and len(name) >= 2:
                etype = "protein" if ent.label_ in protein_types else (
                        "organism" if ent.label_ in organism_types else "concept")
                entities.append({"name": name, "type": etype, "cui": name})

        if self.enable_normalizer and self.normalizer:
            try:
                normalized = self.normalizer.normalize_text(query)
                for norm in normalized:
                    entity_name = norm.get("entity", "")
                    if entity_name and not any(e["name"] == entity_name for e in entities):
                        entities.append({
                            "name":        entity_name,
                            "type":        norm.get("entity_type", "concept").lower(),
                            "ontology":    norm.get("normalized",{}).get("primary",{}).get("ontology",""),
                            "ontology_id": norm.get("normalized",{}).get("primary",{}).get("id",""),
                            "confidence":  norm.get("confidence", 0),
                            "cui":         entity_name,
                        })
            except Exception:
                logger.exception("Entity normalization failed.")

        return list({e["name"]: e for e in entities}.values())

    # -----------------------------------------------------------------------
    # PCC context retrieval  [new req #4 — fully wired]
    # -----------------------------------------------------------------------

    def _get_pcc_context(self, query: str) -> tuple:
        """
        Returns (memory_context_str, memory_info_dict).

        memory_context_str has two clearly labelled sections:
          [Recent Conversation Summary]  — from short-term compressed summary
          [Related Past Research]        — from long-term ANN retrieval
        The LLM system prompt instructs it NOT to cite these as paper sources.
        """
        if not self.enable_pcc or not self.pcc_memory:
            return "", {"pcc_enabled": False}

        # Lazy compression happens here (not on add_message)  [Review #17]
        short_term    = self.pcc_memory.get_short_term_context()
        long_term_eps = self.pcc_memory.retrieve_long_term_memory(query, top_k=3)

        parts = []

        # Short-term: only inject if there is actual content to help the LLM
        if short_term.get("compressed_summary"):
            parts.append(
                "[Recent Conversation Summary — use for context/pronouns only]\n"
                + short_term["compressed_summary"]
            )

        # Long-term: only inject past episodes that are semantically relevant
        if long_term_eps:
            ep_lines = []
            for ep in long_term_eps:
                snippet = (ep.get("content") or "")[:200]
                topics  = ", ".join(ep.get("topics") or [])
                score   = ep.get("similarity", 0)
                ep_lines.append(
                    f"  [Past Episode | topics: {topics} | relevance: {score:.2f}]\n  {snippet}..."
                )
            parts.append("[Related Past Research Discussions — do NOT cite as paper source]\n"
                         + "\n".join(ep_lines))

        memory_info = {
            "pcc_enabled":        True,
            "user_id":            self.user_id,
            "conversation_id":    getattr(self.pcc_memory, "conversation_id", ""),
            "short_term_messages":short_term.get("message_count", 0),
            "long_term_episodes": len(long_term_eps),
            "compression_ratio":  long_term_eps[0].get("compression_ratio", 0) if long_term_eps else 0,
        }
        return "\n\n".join(parts), memory_info

    # -----------------------------------------------------------------------
    # PCC storage helper
    # -----------------------------------------------------------------------

    def _store_pcc_memory(self) -> None:
        if not self.enable_pcc or not self.pcc_memory or not self.conversation_history:
            return
        try:
            ep = self.pcc_memory.compress_and_store_long_term(self.conversation_history)
            if ep:
                logger.info("PCC: stored episode %s (ratio %.2f).",
                            ep.episode_id, ep.pcc_compression_ratio)
        except Exception:
            logger.exception("PCC storage failed.")

    # -----------------------------------------------------------------------
    # Prompt construction
    # -----------------------------------------------------------------------

    def _build_system_prompt(self, memory_context: str) -> str:
        if not memory_context:
            return SYSTEM_PROMPT
        return (
            SYSTEM_PROMPT
            + "\n\n--- CONVERSATION MEMORY (background only — do NOT cite as a paper) ---\n"
            + memory_context
        )

    def _build_messages(
        self,
        query:          str,
        context:        str,
        citation_tags:  list,
        memory_context: str,
        multi_turn:     bool,
    ) -> tuple:
        grounded = (
            f"RESEARCH CONTEXT FROM PAPERS:\n\n{context}\n\n"
            f"---\n\n"
            f"QUESTION: {query}\n\n"
            f"STRICT RULES:\n"
            f"1) Use ONLY the RESEARCH CONTEXT above. NO external knowledge.\n"
            f"2) Every factual claim must cite from: {', '.join(citation_tags)}.\n"
            f"3) Citation format: [S#].\n"
            f"4) If context is insufficient, output EXACTLY:\n{REFUSAL_MESSAGE}\n"
        )
        messages = [{"role": "system", "content": self._build_system_prompt(memory_context)}]
        if multi_turn and self.conversation_history:
            messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": grounded})
        return messages, grounded

    # -----------------------------------------------------------------------
    # Core query preparation (parallel: search overlaps PCC retrieval)  [Review #14]
    # -----------------------------------------------------------------------

    def _prepare_query(self, query: str):
        # Register the new user message in PCC immediately  [new req #4]
        if self.enable_pcc and self.pcc_memory:
            self.pcc_memory.add_message("user", query)

        # Entity extraction (CPU — do first; result feeds into search)
        extracted_entities = self._extract_entities_from_query(query)

        intent = _route_intent_local(query, self.nlp)

        # Overlap Neo4j search with PCC retrieval  [Review #14]
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="prepare") as ex:
            search_f = ex.submit(
                self.store.search, query, TOP_K_RETRIEVAL, extracted_entities
            )
            pcc_f    = ex.submit(self._get_pcc_context, query)

            final_chunks             = search_f.result()
            memory_context, memory_info = pcc_f.result()

        # Sanitise metadata
        for r in final_chunks:
            r["title"]   = r.get("title")   or r.get("source", "Unknown Title")
            r["year"]    = r.get("year")     or "Unknown"
            r["authors"] = [a for a in (r.get("authors") or []) if a]

        # Relevance gate
        if final_chunks and final_chunks[0].get("score", 0) < RELEVANCE_THRESHOLD:
            final_chunks = []

        memory_info["pcc_enabled"] = self.enable_pcc

        if not final_chunks:
            return intent, memory_context, memory_info, [], "", []

        context, citation_tags = build_context(final_chunks)
        return intent, memory_context, memory_info, final_chunks, context, citation_tags

    # -----------------------------------------------------------------------
    # Post-processing  [Review #5 — only bare query stored in history]
    # -----------------------------------------------------------------------

    def _post_process(
        self,
        query:          str,
        answer:         str,
        memory_info:    dict,
        multi_turn:     bool,
    ) -> None:
        # Add assistant response to PCC  [new req #4]
        if self.enable_pcc and self.pcc_memory:
            self.pcc_memory.add_message("assistant", answer)

        if multi_turn and answer != REFUSAL_MESSAGE:
            # [Review #5] Store ONLY the bare query, NOT the grounded context
            self.conversation_history.append({"role": "user",      "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})

            # [Review #5] Bound window to MAX_HISTORY_TURNS pairs = 2× messages
            max_msgs = MAX_HISTORY_TURNS * 2
            if len(self.conversation_history) > max_msgs:
                self.conversation_history = self.conversation_history[-max_msgs:]

            # Persist to long-term memory every 10 turns
            if len(self.conversation_history) % 20 == 0:
                self._store_pcc_memory()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def ask(self, query: str, multi_turn: bool = True) -> tuple:
        """
        Returns (answer, chunks, intent, memory_info).  [Review #4 — 4-tuple]
        """
        intent, memory_context, memory_info, final_chunks, context, citation_tags = (
            self._prepare_query(query)
        )

        if not final_chunks:
            if self.enable_pcc and self.pcc_memory:
                self.pcc_memory.add_message("assistant", REFUSAL_MESSAGE)
            return REFUSAL_MESSAGE, [], intent, memory_info

        gen_model = DEEP_MODEL if intent in ("cross_paper_synthesis","topic_evolution") else FAST_MODEL
        messages, _ = self._build_messages(query, context, citation_tags, memory_context, multi_turn)

        response = self.client.chat.complete(
            model=gen_model, max_tokens=MAX_TOKENS, messages=messages, temperature=0.2
        )
        answer = (response.choices[0].message.content or "").strip()
        self._post_process(query, answer, memory_info, multi_turn)
        return answer, final_chunks, intent, memory_info

    def ask_stream(self, query: str, multi_turn: bool = True):
        """
        Streaming ask using real Mistral SSE.

        Yields:
          (token: str,  None,         None)         — during generation
          (None,        final_chunks, memory_info)  — once complete
        """
        intent, memory_context, memory_info, final_chunks, context, citation_tags = (
            self._prepare_query(query)
        )

        if not final_chunks:
            if self.enable_pcc and self.pcc_memory:
                self.pcc_memory.add_message("assistant", REFUSAL_MESSAGE)
            yield REFUSAL_MESSAGE, [], memory_info
            return

        gen_model   = DEEP_MODEL if intent in ("cross_paper_synthesis","topic_evolution") else FAST_MODEL
        messages, _ = self._build_messages(query, context, citation_tags, memory_context, multi_turn)
        full_answer = ""

        with self.client.chat.stream(
            model=gen_model, max_tokens=MAX_TOKENS, messages=messages, temperature=0.2
        ) as stream:
            for chunk in stream:
                delta = chunk.data.choices[0].delta.content
                if delta:
                    full_answer += delta
                    yield delta, None, None

        self._post_process(query, full_answer, memory_info, multi_turn)
        yield None, final_chunks, memory_info

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def get_paper_list(self) -> list:
        return self.store.get_papers()

    def get_memory_status(self) -> dict:
        if not self.enable_pcc or not self.pcc_memory:
            return {"pcc_enabled": False}
        try:
            return self.pcc_memory.get_memory_summary()
        except Exception:
            logger.exception("get_memory_status failed.")
            return {"pcc_enabled": False}

    def reset_conversation(self) -> None:
        if self.enable_pcc and self.conversation_history:
            self._store_pcc_memory()
        self.conversation_history = []
        if self.enable_pcc and self.pcc_memory:
            self.pcc_memory.clear_short_term()
        logger.info("Conversation history cleared.")

    def close(self) -> None:
        if self.enable_pcc and self.conversation_history:
            self._store_pcc_memory()
        self.store.close()
        if self.pcc_memory:
            self.pcc_memory.close()
        if self.normalizer:
            self.normalizer.close()
        logger.info("RAG Engine closed.")
