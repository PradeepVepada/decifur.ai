"""
rag_engine.py
-------------
Advanced GraphRAG pipeline with session-aware PCC memory,
follow-up query rewriting, and paper-level recommendation handling.
"""

from __future__ import annotations

import os
import re
import logging
import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

from openai import OpenAI

from graph_store import GraphStore
from extract import load_scispacy_model
from pcc_memory import PCCMemory, create_pcc_memory
from biomedical_normalizer import create_biomedical_normalizer
from model_config import (
    DEEP_REASONING_BACKEND,
    FAST_MODEL,
    OPENAI_API_KEY,
    OPENAI_CORPUS_MODEL,
    OPENAI_DEEP_MODEL,
    OLLAMA_DEEP_MODEL,
    OLLAMA_RETRY_TEMPERATURE,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    RAG_MAX_OUTPUT_TOKENS,
    REWRITE_MODEL,
    get_ollama_base_url,
    ollama_api_extra_body,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOP_K_RETRIEVAL = 12
TOP_K_PAPER_DISCOVERY = 16
MAX_TOKENS = RAG_MAX_OUTPUT_TOKENS
MAX_HISTORY_TURNS = 8
RELEVANCE_THRESHOLD = 0.015


def _rag_generation_kwargs(ollama_extras: bool, *, retry: bool = False) -> dict:
    """Sampling params aligned with `ollama/Modelfile.qwen35-biomedical` (Ollama + OpenAI deep path)."""
    kw: dict = {
        "temperature": OLLAMA_RETRY_TEMPERATURE if retry else OLLAMA_TEMPERATURE,
        "top_p": OLLAMA_TOP_P,
    }
    if ollama_extras:
        kw["extra_body"] = ollama_api_extra_body()
    return kw


SYSTEM_PROMPT = """You are a PhD-level biomedical research assistant with deep expertise in cell signaling, molecular biology, biochemistry, and systems biology. You provide precise, rigorous answers grounded in the evidence given in the conversation (the research excerpts in the user message). Explain WHY mechanistically: name pathways, effectors, and substrates where relevant; cite quantitative values when the context provides them.

Formatting (strict): Do NOT use markdown headings (no #, ##, ###, or numbered heading lines). Do NOT paste the section outline or template text into your answer. Structure the main answer as **2–4 paragraphs** only: each paragraph several sentences on one sub-topic, with **one blank line** between paragraphs (or a short bullet list where it clearly helps). **Never** put the whole answer in a single uninterrupted paragraph or one mega-block.

Math: Use LaTeX only in forms that render cleanly: inline math as `$...$` and display equations as `$$ ... $$` on their own lines (not glued to prose). Prefer simple symbols ($K_d$, $V_{max}$) over huge multi-line blocks unless the excerpts justify them.

Citations: When the user message includes excerpt tags like [S1], cite them immediately after the supported claim. Use only tags that appear in that message.

Grounding rules (non-negotiable): Use only facts supported by the numbered excerpts ([S1], [S2], …) in the user message. Every [S#] in your answer must refer to one of those tags. Do not name specific people, labs, or institutions as having done work unless that name appears in the excerpt you cite for that claim. If the excerpts omit an author list or detail, say so instead of inventing it.

Length: Aim for roughly 220–450 words when the excerpts support it—always split across those 2–4 paragraphs. If the question is narrow, fewer words but still at least **2** short paragraphs unless the refusal applies."""

REFUSAL_MESSAGE = (
    "The available papers do not contain sufficient information to answer this question."
)


_INTENT_KEYWORDS = {
    "cross_paper_synthesis": ["compare", "across", "both papers", "all papers", "synthesize", "synthesis", "contrast", "between"],
    "topic_evolution": ["over time", "evolution", "history", "trend", "how has", "changed", "progression"],
    "recommendation": ["recommend", "suggest", "should i read", "what paper", "which paper"],
}

_FOLLOWUP_PHRASES = {
    "that", "this", "it", "those", "these", "them",
    "explain that", "explain this", "explain in detail", "simpler words", "simpler",
    "summarize that", "summarize this", "go deeper", "elaborate",
    "tell me more", "continue", "what about that", "what about this",
    "the first one", "the second one", "the third one",
}

# User asked for depth — extra instruction in the grounded prompt
_ELABORATION_TERMS = (
    "in detail", "more detail", "more details", "get into more", "get into detail",
    "further detail", "further details", "dig deeper", "deeper dive", "expand on",
    "elaborate", "go deeper", "tell me more", "explain more",
)


def _wants_elaboration(query: str) -> bool:
    q = (query or "").lower()
    return any(t in q for t in _ELABORATION_TERMS)

_PAPER_DISCOVERY_TERMS = {
    "foundational", "earliest", "first paper", "first established",
    "best papers", "which papers should i read", "start with",
    "intro papers", "key papers", "seminal", "recommended papers"
}

# Author × gene / co-authorship style questions → graph analytics + optional retrieval relax
_AUTHOR_GENE_TERMS = (
    "co-author", "coauthor", "co-authored", "coauthored",
    "who authored", "who wrote", "who worked on", "authors who",
    "researchers who", "investigators who", "which authors",
    "what authors", "which scientists", "who contributed",
    "list authors", "all authors",
)

_GENE_SYMBOL_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,15})\b")
# Mixed-case gene/protein symbols (e.g. YakA, mTor) missed by all-caps-only regex
_GENE_MIXED_RE = re.compile(r"\b([A-Z][a-z]{1,12}[A-Z][A-Za-z0-9\-]*)\b")
_P_DIGIT_GENE_RE = re.compile(r"\b(p\d{2})\b", re.IGNORECASE)


def _all_gene_symbol_candidates(query: str) -> list[str]:
    """Uppercase symbols (PI3K) plus mixed-case (YakA) and p21-style tokens."""
    ordered: list[str] = []
    seen: set[str] = set()

    def _push(x: str) -> None:
        x = (x or "").strip()
        if len(x) < 2 or x in seen:
            return
        seen.add(x)
        ordered.append(x)

    for x in _gene_symbol_candidates(query):
        _push(x)
    if not query:
        return ordered
    for m in _GENE_MIXED_RE.finditer(query):
        _push(m.group(1))
    for m in _P_DIGIT_GENE_RE.finditer(query):
        _push(m.group(1))
    return ordered


def _collect_protein_needles_from_query(
    query: str,
    retrieval_query: str,
    extracted_entities: list,
) -> list[str]:
    """Gene/protein strings to require in chunk text+title when filtering (case-insensitive)."""
    needles: list[str] = []
    for e in extracted_entities or []:
        if (e.get("type") or "").lower() == "protein" and e.get("name"):
            needles.append(str(e["name"]).strip())
    for blob in (query, retrieval_query):
        needles.extend(_all_gene_symbol_candidates(blob or ""))
    seen: set[str] = set()
    out: list[str] = []
    for n in needles:
        n = (n or "").strip()
        if len(n) < 2:
            continue
        k = n.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(n)
    return out


def _filter_chunks_matching_protein_needles(
    chunks: list,
    query: str,
    retrieval_query: str,
    extracted_entities: list,
) -> list:
    """
    Prefer chunks whose text/title mention at least one extracted protein/gene token from the query.
    If that would remove everything, return the original list (retrieval still available).
    """
    needles = _collect_protein_needles_from_query(query, retrieval_query, extracted_entities)
    if not needles or not chunks:
        return chunks
    filtered: list = []
    for ch in chunks:
        blob = ((ch.get("text") or "") + "\n" + (ch.get("title") or "")).lower()
        if any(n.lower() in blob for n in needles):
            filtered.append(ch)
    if not filtered:
        logger.info(
            "Protein needle filter skipped: no chunks mention %s — keeping full retrieval.",
            needles[:6],
        )
        return chunks
    if len(filtered) < len(chunks):
        logger.info(
            "Protein needle filter: %d → %d chunks (needles=%s)",
            len(chunks),
            len(filtered),
            needles[:8],
        )
    return filtered


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


def _looks_like_context_dependent_followup(query: str) -> bool:
    q = query.strip().lower()
    if len(q.split()) <= 6:
        return True
    return any(phrase in q for phrase in _FOLLOWUP_PHRASES)


def _looks_like_paper_discovery_query(query: str) -> bool:
    q = query.strip().lower()
    return any(term in q for term in _PAPER_DISCOVERY_TERMS)


def _wants_author_gene_analytics(q: str) -> bool:
    s = (q or "").lower()
    return any(t in s for t in _AUTHOR_GENE_TERMS)


def _gene_symbol_candidates(query: str) -> list[str]:
    return list(dict.fromkeys(_GENE_SYMBOL_RE.findall(query or "")))


_PROTEIN_ENTITY_LABELS = frozenset({
    "GENE_OR_GENE_PRODUCT", "PROTEIN", "GENE", "GENE_FAMILY", "RNA", "DNA",
})


def _ner_protein_like_names(text: str, nlp) -> list[str]:
    """Gene/protein-like entity strings from SciSpaCy (for retrieval anchoring)."""
    if not nlp or not (text or "").strip():
        return []
    out: list[str] = []
    seen: set[str] = set()
    try:
        doc = nlp(text[:8000])
    except Exception:
        return []
    for ent in doc.ents:
        if ent.label_ not in _PROTEIN_ENTITY_LABELS:
            continue
        name = ent.text.strip()
        if len(name) < 2:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _retrieval_anchor_suffix(retrieval_query: str, history: list[dict], nlp, max_symbols: int = 5) -> str:
    """
    Append invisible retrieval context (gene symbols / protein entities) from recent
    dialogue so vague follow-ups do not drift away from the active topic.
    """
    lines: list[str] = []
    for m in (history or [])[-(MAX_HISTORY_TURNS * 2) :]:
        c = (m.get("content") or "").strip()
        if c:
            lines.append(c)
    blob = "\n".join(lines)
    if not blob:
        return ""
    symbols: list[str] = []
    for sym in _gene_symbol_candidates(blob):
        if sym not in symbols:
            symbols.append(sym)
    for name in _ner_protein_like_names(blob, nlp):
        if name not in symbols:
            symbols.append(name)
    rq_lower = (retrieval_query or "").lower()
    missing: list[str] = []
    for s in symbols:
        if len(missing) >= max_symbols:
            break
        if s.lower() not in rq_lower:
            missing.append(s)
    if not missing:
        return ""
    return f" (context: {', '.join(missing)})"


def _format_recent_history_for_rewrite(history: list[dict], max_messages: int = 6) -> str:
    parts = []
    for m in history[-max_messages:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content[:500]}")
    return "\n".join(parts)


import re as _re_module

# Very specific patterns the fine-tuned model leaks — all start with a digit
# and contain formatting/meta-instruction language with NO citation brackets.
# We do NOT filter lines containing [S#] citations or biomedical entity names.
_LEAKED_PATTERNS = [
    # "5.  Answer in 1-2 sentences."
    _re_module.compile(r"^\d+[\.)]\s+Answer\s+in\s+\d", _re_module.IGNORECASE),
    # "5.  Do not use phrases like ..."
    _re_module.compile(r"^\d+[\.)]\s+Do\s+not\s+use\s+phrases", _re_module.IGNORECASE),
    # "5.  Use phrases like ..." / "5. Use direct language"
    _re_module.compile(r"^\d+[\.)]\s+Use\s+(phrases|direct|active|passive|concise|clear)\b", _re_module.IGNORECASE),
    # "5.  Be concise." / "5. Be direct."
    _re_module.compile(r"^\d+[\.)]\s+Be\s+(concise|direct|precise|brief|specific|clear)\b", _re_module.IGNORECASE),
    # "5.  Output only the answer."
    _re_module.compile(r"^\d+[\.)]\s+Output\s+only", _re_module.IGNORECASE),
    # "5.  Keep the answer to ..."
    _re_module.compile(r"^\d+[\.)]\s+Keep\s+the\s+answer", _re_module.IGNORECASE),
    # "5.  Avoid phrases like ..."
    _re_module.compile(r"^\d+[\.)]\s+Avoid\s+(phrases|saying|using|stating)", _re_module.IGNORECASE),
    # "5.  Do not start with ..."
    _re_module.compile(r"^\d+[\.)]\s+Do\s+not\s+(start|begin|use|output|include|add|repeat)", _re_module.IGNORECASE),
    # "5.  Never start your answer ..."
    _re_module.compile(r"^\d+[\.)]\s+Never\s+(start|begin|use|say|output)", _re_module.IGNORECASE),
    # "5.  Always cite ..." — BUT only if line has no [S#] pattern (real citations are fine)
    _re_module.compile(r"^\d+[\.)]\s+Always\s+(cite\s+sources|include\s+citation|use\s+citation)", _re_module.IGNORECASE),
    # "5.  Format your answer ..."
    _re_module.compile(r"^\d+[\.)]\s+Format\s+(your|the)\s+answer", _re_module.IGNORECASE),
]


def _is_leaked_instruction(line: str) -> bool:
    """Return True only if the line precisely matches a known leaked instruction pattern."""
    stripped = line.strip()
    if not stripped:
        return False
    # Never filter lines that contain citation brackets — those are real answers
    if _re_module.search(r"\[S\d+\]", stripped):
        return False
    for pat in _LEAKED_PATTERNS:
        if pat.match(stripped):
            return True
    return False


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks and precisely matched leaked instruction lines."""
    # Remove explicit think/reasoning blocks
    text = _re_module.sub(r"<think>[\s\S]*?</think>", "", text, flags=_re_module.DOTALL).strip()
    lines = text.splitlines()
    filtered = [line for line in lines if not _is_leaked_instruction(line)]
    return "\n".join(filtered).strip()


def build_context(chunks: list) -> tuple[str, list[str]]:
    sections: list[str] = []
    tags: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        tag = f"S{i}"
        tags.append(tag)
        authors = chunk.get("authors", [])
        author_str = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")
        header = f"[{tag}] {chunk.get('title', 'Unknown')} ({chunk.get('year', 'Unknown')}) -- {author_str}"
        sections.append(f"{header}\n{chunk.get('text', '')}")

    return "\n\n---\n\n".join(sections), tags


def _strip_invalid_citation_tags(answer: str, num_sources: int) -> str:
    """
    Remove [S#] markers where # is outside 1..num_sources (model hallucination).
    Keeps tags aligned with build_context / the sources list sent to the UI.
    """
    if not answer or num_sources <= 0:
        return answer

    def _sub(m: _re_module.Match[str]) -> str:
        try:
            n = int(m.group(1))
        except ValueError:
            return m.group(0)
        if 1 <= n <= num_sources:
            return m.group(0)
        return ""

    out = _re_module.sub(r"\[S(\d+)\]", _sub, answer, flags=_re_module.IGNORECASE)
    out = _re_module.sub(r"  +", " ", out)
    out = _re_module.sub(r" \n", "\n", out)
    return out.strip()


def _strip_markdown_headings(answer: str) -> str:
    """
    Remove markdown heading lines (# …) and trailing '### N. Title' junk often echoed from prompts.
    """
    if not answer:
        return answer
    lines_out: list[str] = []
    for line in answer.splitlines():
        st = line.strip()
        if _re_module.match(r"^#{1,6}\s", st):
            continue
        line = _re_module.sub(r"\s+#{1,6}\s+\d*[\.)]?\s*[^\n#]+$", "", line)
        lines_out.append(line)
    text = "\n".join(lines_out)
    text = _re_module.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _inject_citations_post(answer: str, chunks: list, tags: list[str]) -> str:
    """
    Deterministically inject [S#] citation markers into the LLM answer
    by matching each sentence to the most lexically similar chunk via keyword overlap.
    Called after the full answer is buffered, so it works regardless of whether
    the model itself follows citation instructions.
    """
    _STOP = {
        "that", "this", "with", "from", "have", "been", "were", "they", "their",
        "which", "also", "into", "when", "than", "more", "after", "other", "some",
        "these", "through", "between", "would", "could", "should", "about", "over",
        "such", "both", "each", "cell", "cells", "protein", "proteins", "gene",
        "genes", "study", "studies", "found", "show", "shows", "showed", "known",
    }

    def _kws(text: str) -> set:
        words = _re_module.findall(r"\b[a-z]{4,}\b", text.lower())
        return {w for w in words if w not in _STOP}

    chunk_kws = [_kws(c.get("text", "") + " " + c.get("title", "")) for c in chunks]

    sentences = _re_module.split(r"(?<=[.!?])\s+", answer.strip())
    result = []

    for sent in sentences:
        if len(sent.split()) < 5:
            result.append(sent)
            continue

        sent_kws = _kws(sent)
        if not sent_kws:
            result.append(sent)
            continue

        best_tag = None
        best_score = 0.0
        for i, ckws in enumerate(chunk_kws):
            overlap = len(sent_kws & ckws)
            score = overlap / len(sent_kws)
            if score > best_score:
                best_score = score
                best_tag = tags[i]

        if best_tag and best_score >= 0.3:
            last = sent[-1] if sent else ""
            if last in ".!?":
                sent = sent[:-1] + f" [{best_tag}]" + last
            else:
                sent = sent + f" [{best_tag}]"

        result.append(sent)

    return " ".join(result)


class RAGEngine:
    def __init__(
        self,
        user_id: str = "default",
        enable_pcc: bool = True,
        enable_normalizer: bool = True,
    ):
        self.client = OpenAI(base_url=f"{get_ollama_base_url()}/v1", api_key="ollama")
        self._openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        if DEEP_REASONING_BACKEND == "openai" and not self._openai_client:
            logger.warning(
                "DEEP_REASONING_BACKEND=openai but OPENAI_API_KEY is missing; "
                "using Ollama-compatible endpoint %s model=%s for deep reasoning.",
                get_ollama_base_url(),
                OLLAMA_DEEP_MODEL,
            )
        self.store = GraphStore()
        self.conversation_history: list[dict] = []
        self.nlp = None
        self.enable_pcc = enable_pcc
        self.pcc_memory: PCCMemory | None = None
        self.enable_normalizer = enable_normalizer
        self.normalizer = None
        self.user_id = user_id
        self.active_conversation_id = "default"

    def load(self) -> None:
        logger.info("Connecting to Neo4j...")
        self.store.load()

        logger.info("Loading scispaCy biomedical NER model...")
        self.nlp = load_scispacy_model()

        if self.enable_normalizer:
            logger.info("Loading Biomedical Entity Normalizer...")
            try:
                self.normalizer = create_biomedical_normalizer()
                logger.info("Biomedical Normalizer ready.")
            except Exception:
                logger.exception("Normalizer init failed — disabling.")
                self.enable_normalizer = False

        if self.enable_pcc:
            logger.info("Initialising PCC Memory...")
            try:
                self.pcc_memory = create_pcc_memory(
                    user_id=self.user_id,
                    conversation_id=self.active_conversation_id,
                    embedder=self.store.embedder,
                    openai_client=self.client,
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
    # Session binding / hydration
    # -----------------------------------------------------------------------

    def set_conversation_id(self, conversation_id: str) -> None:
        self.active_conversation_id = conversation_id or "default"
        if self.enable_pcc and self.pcc_memory:
            self.pcc_memory.set_conversation_id(self.active_conversation_id)

    def hydrate_memory_from_messages(self, messages: list[dict]) -> None:
        cleaned: list[dict] = []
        for m in messages:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                cleaned.append({
                    "role": role,
                    "content": content,
                    "timestamp": m.get("timestamp"),
                })

        if self.enable_pcc and self.pcc_memory:
            self.pcc_memory.hydrate_from_history(cleaned)

        self.conversation_history = [
            {"role": m["role"], "content": m["content"]}
            for m in cleaned[-(MAX_HISTORY_TURNS * 2):]
        ]

    # -----------------------------------------------------------------------
    # Generation (Ollama vs OpenAI for deep reasoning)
    # -----------------------------------------------------------------------

    def _resolve_generation(self, intent: str) -> tuple[OpenAI, str, bool]:
        """
        Returns (client, model_id, use_ollama_extras).
        Deep intents use OpenAI when DEEP_REASONING_BACKEND=openai and a key is present.
        """
        is_deep = intent in ("cross_paper_synthesis", "topic_evolution")
        if is_deep and DEEP_REASONING_BACKEND == "openai" and self._openai_client:
            return self._openai_client, OPENAI_DEEP_MODEL, False
        if is_deep:
            return self.client, OLLAMA_DEEP_MODEL, True
        return self.client, FAST_MODEL, True

    def _resolve_corpus_generation(
        self, intent: str, corpus_generation_model: str | None
    ) -> tuple[OpenAI, str, bool]:
        """
        Main RAG answer path only. Honors UI-selected OpenAI models when API key is set;
        otherwise falls back to _resolve_generation(intent).
        """
        raw = (corpus_generation_model or "").strip().lower()
        if raw in (
            "gpt-5-nano",
            "gpt-5 nano",
            "gpt5-nano",
            "5-nano",
        ):
            if not self._openai_client:
                raise RuntimeError(
                    "OPENAI_API_KEY is required to answer with the OpenAI corpus model. "
                    "Add it to .env or choose the default corpus model."
                )
            return self._openai_client, OPENAI_CORPUS_MODEL, False
        if raw in ("gpt-4o-mini", "gpt-4o mini", "4o-mini", "4o mini"):
            if not self._openai_client:
                raise RuntimeError(
                    "OPENAI_API_KEY is required to answer with gpt-4o-mini. "
                    "Add it to .env or choose the default corpus model."
                )
            return self._openai_client, "gpt-4o-mini", False
        if raw:
            logger.warning("Unknown corpus_generation_model %r — using default routing.", raw)
        return self._resolve_generation(intent)

    # -----------------------------------------------------------------------
    # Query rewriting
    # -----------------------------------------------------------------------

    def _rewrite_query_with_context(self, query: str) -> str:
        if not _looks_like_context_dependent_followup(query):
            return query

        if not self.conversation_history:
            return query

        history_block = _format_recent_history_for_rewrite(self.conversation_history)

        prior_user_turns: list[str] = []
        for m in reversed(self.conversation_history):
            if m.get("role") != "user":
                continue
            text = (m.get("content") or "").strip()
            if not text or text.lower() == query.lower():
                continue
            prior_user_turns.append(text)
            if len(prior_user_turns) >= 2:
                break

        fallback = query
        if len(prior_user_turns) == 1:
            fallback = f"{prior_user_turns[0]}. Follow-up request: {query}"
        elif len(prior_user_turns) >= 2:
            older, newer = prior_user_turns[1], prior_user_turns[0]
            fallback = f"{older} {newer}. Follow-up request: {query}"

        try:
            res = self.client.chat.completions.create(
                model=REWRITE_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        "Rewrite the follow-up question into a standalone scientific retrieval query. "
                        "Use the recent conversation only to resolve references like 'that', 'it', "
                        "'this', 'those', or vague follow-ups. Preserve the user's intent. "
                        "Include important gene names, proteins, and pathway terms from the conversation "
                        "in the rewritten query when they are implied. Return only the rewritten query.\n\n"
                        f"Recent conversation:\n{history_block}\n\n"
                        f"Follow-up question:\n{query}"
                    )
                }],
                temperature=0,
            )
            rewritten = (res.choices[0].message.content or "").strip()
            if rewritten:
                logger.info("Rewrote follow-up query '%s' -> '%s'", query, rewritten)
                return rewritten
        except Exception:
            logger.exception("Query rewrite failed; using fallback.")

        return fallback

    def _rewrite_paper_discovery_query(self, query: str) -> str:
        """
        Rephrase recommendation / chronology questions into retrieval-friendly queries.
        """
        if not _looks_like_paper_discovery_query(query):
            return query

        try:
            res = self.client.chat.completions.create(
                model=REWRITE_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        "Rewrite this scientific paper-discovery question into a retrieval-friendly query. "
                        "Keep the biological topic, remove recommendation wording like 'best' or 'foundational' "
                        "if needed, and focus on the scientific topic. Return only the rewritten query.\n\n"
                        f"Question:\n{query}"
                    )
                }],
                temperature=0,
            )
            rewritten = (res.choices[0].message.content or "").strip()
            if rewritten:
                logger.info("Rewrote paper-discovery query '%s' -> '%s'", query, rewritten)
                return rewritten
        except Exception:
            logger.exception("Paper-discovery rewrite failed; using original query.")

        return query

    # -----------------------------------------------------------------------
    # Entity extraction
    # -----------------------------------------------------------------------

    def _extract_entities_from_query(self, query: str) -> list:
        doc = self.nlp(query)
        entities: list[dict] = []

        protein_types = {"GENE_OR_GENE_PRODUCT", "PROTEIN", "GENE", "GENE_FAMILY", "RNA", "DNA"}
        organism_types = {"ORGANISM", "SPECIES", "MULTI_CELLULAR_ORGANISM", "ANATOMICAL_SYSTEM"}

        for ent in doc.ents:
            name = ent.text.strip()
            if name and len(name) >= 2:
                etype = (
                    "protein" if ent.label_ in protein_types
                    else "organism" if ent.label_ in organism_types
                    else "concept"
                )
                entities.append({"name": name, "type": etype, "cui": name})

        if self.enable_normalizer and self.normalizer:
            try:
                normalized = self.normalizer.normalize_text(query)
                for norm in normalized:
                    entity_name = norm.get("entity", "")
                    if entity_name and not any(e["name"] == entity_name for e in entities):
                        entities.append({
                            "name": entity_name,
                            "type": norm.get("entity_type", "concept").lower(),
                            "ontology": norm.get("normalized", {}).get("primary", {}).get("ontology", ""),
                            "ontology_id": norm.get("normalized", {}).get("primary", {}).get("id", ""),
                            "confidence": norm.get("confidence", 0),
                            "cui": entity_name,
                        })
            except Exception:
                logger.exception("Entity normalization failed.")

        dedup: dict[str, dict] = {}
        for e in entities:
            dedup[e["name"]] = e
        return list(dedup.values())

    # -----------------------------------------------------------------------
    # PCC context retrieval
    # -----------------------------------------------------------------------

    def _get_pcc_context(self, query: str) -> tuple[str, dict]:
        if not self.enable_pcc or not self.pcc_memory:
            return "", {"pcc_enabled": False}

        short_term = self.pcc_memory.get_short_term_context()
        long_term_eps = self.pcc_memory.retrieve_long_term_memory(query, top_k=3)

        parts: list[str] = []

        if short_term.get("compressed_summary"):
            parts.append(
                "[Recent Conversation Summary — use for context/pronouns only]\n"
                + short_term["compressed_summary"]
            )

        if long_term_eps:
            ep_lines = []
            for ep in long_term_eps:
                snippet = (ep.get("content") or "")[:220]
                topics = ", ".join(ep.get("topics") or [])
                score = ep.get("similarity", 0)
                conv_id = ep.get("conversation_id", "")
                ep_lines.append(
                    f"  [Past Episode | conversation: {conv_id} | topics: {topics} | relevance: {score:.2f}]\n"
                    f"  {snippet}..."
                )

            parts.append(
                "[Related Past Research Discussions — do NOT cite as paper source]\n"
                + "\n".join(ep_lines)
            )

        memory_info = {
            "pcc_enabled": True,
            "user_id": self.user_id,
            "conversation_id": getattr(self.pcc_memory, "conversation_id", ""),
            "short_term_messages": short_term.get("message_count", 0),
            "long_term_episodes": len(long_term_eps),
            "compression_ratio": long_term_eps[0].get("compression_ratio", 0) if long_term_eps else 0,
        }
        return "\n\n".join(parts), memory_info

    # -----------------------------------------------------------------------
    # Paper-level aggregation for recommendation / chronology queries
    # -----------------------------------------------------------------------

    def _aggregate_papers_from_chunks(self, chunks: list) -> list[dict]:
        grouped: dict[str, dict] = {}

        for ch in chunks:
            source = ch.get("source") or ch.get("title") or "unknown"
            title = ch.get("title") or source
            year = ch.get("year") or "Unknown"
            journal = ch.get("journal") or "Unknown"
            authors = [a for a in (ch.get("authors") or []) if a]
            score = float(ch.get("score", 0) or 0)

            if source not in grouped:
                grouped[source] = {
                    "source": source,
                    "title": title,
                    "year": year,
                    "journal": journal,
                    "authors": authors,
                    "supporting_chunks": 0,
                    "max_score": score,
                    "avg_score_total": score,
                    "examples": [ch.get("text", "")[:240]],
                }
            else:
                grouped[source]["max_score"] = max(grouped[source]["max_score"], score)
                grouped[source]["avg_score_total"] += score
                if len(grouped[source]["examples"]) < 2:
                    grouped[source]["examples"].append(ch.get("text", "")[:240])

            grouped[source]["supporting_chunks"] += 1

        papers = []
        for item in grouped.values():
            item["avg_score"] = item["avg_score_total"] / max(item["supporting_chunks"], 1)
            del item["avg_score_total"]
            papers.append(item)

        return papers

    def _rank_papers_for_discovery(self, papers: list[dict], query: str) -> list[dict]:
        q = query.lower()
        wants_earliest = any(term in q for term in ["earliest", "first", "foundational", "seminal"])

        def safe_year(y):
            try:
                return int(y)
            except Exception:
                return 9999 if wants_earliest else -1

        if wants_earliest:
            return sorted(
                papers,
                key=lambda p: (
                    safe_year(p.get("year")),
                    -p.get("supporting_chunks", 0),
                    -p.get("max_score", 0),
                ),
            )
        return sorted(
            papers,
            key=lambda p: (
                -p.get("supporting_chunks", 0),
                -p.get("max_score", 0),
                safe_year(p.get("year")),
            ),
        )

    def _answer_paper_discovery_query(self, query: str, chunks: list, memory_info: dict) -> tuple[str, list]:
        papers = self._aggregate_papers_from_chunks(chunks)
        if not papers:
            return REFUSAL_MESSAGE, []

        ranked = self._rank_papers_for_discovery(papers, query)
        top = ranked[:3]

        lines = []
        lines.append("Based on the most relevant papers retrieved from the corpus, these are the strongest starting points:")
        for i, p in enumerate(top, 1):
            authors = ", ".join((p.get("authors") or [])[:2])
            if len(p.get("authors") or []) > 2:
                authors += " et al."
            year = p.get("year", "Unknown")
            journal = p.get("journal", "Unknown")
            lines.append(
                f"{i}. {p.get('title', p.get('source'))} ({year})"
                + (f" — {authors}" if authors else "")
                + (f" — {journal}" if journal else "")
            )
            lines.append(
                f"   Reason: retrieved in {p.get('supporting_chunks', 0)} supporting chunk(s), "
                f"max relevance {p.get('max_score', 0):.4f}."
            )

        q = query.lower()
        if any(term in q for term in ["first", "earliest", "foundational", "seminal"]):
            lines.append(
                "Note: this ranking uses the earliest and most relevant papers retrieved from the corpus. "
                "It is a corpus-grounded recommendation, not a guaranteed claim about the entire external literature."
            )

        answer = "\n".join(lines)
        memory_info["paper_discovery_mode"] = True
        memory_info["ranked_papers"] = [
            {
                "title": p.get("title"),
                "year": p.get("year"),
                "supporting_chunks": p.get("supporting_chunks"),
                "max_score": round(p.get("max_score", 0), 6),
            }
            for p in top
        ]
        return answer, chunks

    # -----------------------------------------------------------------------
    # Prompt construction
    # -----------------------------------------------------------------------

    def _build_system_prompt(self, memory_context: str) -> str:
        return SYSTEM_PROMPT

    def _build_messages(
        self,
        query: str,
        context: str,
        citation_tags: list[str],
        memory_context: str,
        multi_turn: bool,
    ) -> tuple[list[dict], str]:
        memory_section = (
            f"\n\nConversation memory (background context only — do not cite as a paper source):\n{memory_context}"
            if memory_context else ""
        )
        tags_inline = ", ".join(f"[{t}]" for t in citation_tags)
        elaboration_note = ""
        if _wants_elaboration(query):
            elaboration_note = (
                "The user asked for depth: add one extra short section or a few more bullets—still readable, "
                "no markdown headings, no pasted outline.\n\n"
            )
        grounded = (
            f"The following excerpts are from primary research papers. "
            f"Each excerpt is labelled with a citation tag such as [S1], [S2], etc."
            f"{memory_section}"
            f"\n\n---\n\nResearch excerpts:\n\n{context}"
            f"\n\n---\n\n"
            f"Question: {query}\n\n"
            f"{elaboration_note}"
            f"Write a clear answer using only the excerpts above. Use **2–4 paragraphs** (blank line between each); "
            f"typical flow: (1) what it is / why it matters, (2) mechanism and key players if the text supports it, "
            f"(3) regulation or context if relevant, (4) only if needed—numbers/equations from excerpts using `$...$` or `$$...$$` on their own lines. "
            f"Do not merge everything into one paragraph. Do not dump a numbered markdown outline or ### headings into your reply.\n\n"
            f"After every factual claim cite the source tag (e.g. 'PTEN dephosphorylates PIP3 [S3].'). "
            f"Available tags only: {tags_inline}.\n\n"
            f"If the excerpts lack enough information, respond with exactly: {REFUSAL_MESSAGE}"
        )

        messages = [{"role": "system", "content": self._build_system_prompt(memory_context)}]
        if multi_turn and self.conversation_history:
            messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": grounded})
        return messages, grounded

    # -----------------------------------------------------------------------
    # Query preparation
    # -----------------------------------------------------------------------

    def _prepare_query(self, query: str):
        if self.enable_pcc and self.pcc_memory:
            self.pcc_memory.add_message("user", query)

        followup_rewritten = self._rewrite_query_with_context(query)
        retrieval_query = self._rewrite_paper_discovery_query(followup_rewritten)
        anchor = _retrieval_anchor_suffix(retrieval_query, self.conversation_history, self.nlp)
        if anchor:
            retrieval_query = retrieval_query + anchor

        extracted_entities = self._extract_entities_from_query(retrieval_query)
        intent = _route_intent_local(retrieval_query, self.nlp)
        paper_discovery = _looks_like_paper_discovery_query(query)

        k = TOP_K_PAPER_DISCOVERY if paper_discovery else TOP_K_RETRIEVAL

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="prepare") as ex:
            search_f = ex.submit(self.store.search, retrieval_query, k, extracted_entities)
            pcc_f = ex.submit(self._get_pcc_context, query)

            final_chunks = search_f.result()
            memory_context, memory_info = pcc_f.result()

        for r in final_chunks:
            r["title"] = r.get("title") or r.get("source", "Unknown Title")
            r["year"] = r.get("year") or "Unknown"
            r["authors"] = [a for a in (r.get("authors") or []) if a]

        final_chunks = _filter_chunks_matching_protein_needles(
            final_chunks, query, retrieval_query, extracted_entities
        )

        author_gene_relaxed = (
            _wants_author_gene_analytics(query)
            or _wants_author_gene_analytics(retrieval_query)
        ) and (
            any((e.get("type") or "").lower() == "protein" for e in extracted_entities)
            or bool(_all_gene_symbol_candidates(query))
            or bool(_all_gene_symbol_candidates(retrieval_query))
        )

        # Soft gating for paper discovery, rewritten follow-ups, and author×gene questions
        if final_chunks and final_chunks[0].get("score", 0) < RELEVANCE_THRESHOLD:
            if paper_discovery or retrieval_query != query or author_gene_relaxed:
                logger.info(
                    "Allowing low-score retrieval (paper-discovery / follow-up / author-gene): %.6f",
                    final_chunks[0].get("score", 0),
                )
            else:
                final_chunks = []

        memory_info["pcc_enabled"] = self.enable_pcc
        memory_info["retrieval_query"] = retrieval_query
        memory_info["rewritten_followup"] = followup_rewritten != query
        memory_info["paper_discovery_mode"] = paper_discovery

        if not final_chunks:
            return intent, memory_context, memory_info, [], "", [], paper_discovery

        context, citation_tags = build_context(final_chunks)
        return intent, memory_context, memory_info, final_chunks, context, citation_tags, paper_discovery

    # -----------------------------------------------------------------------
    # Confidence scoring (heuristic from rag_engine_sreshta)
    # -----------------------------------------------------------------------

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _count_supported_claims(self, answer: str) -> tuple[int, int]:
        """
        Rough groundedness heuristic:
        - total_claims: number of non-empty answer lines/sentences
        - cited_claims: how many contain [S#] style citations
        """
        if not answer or answer.strip() == REFUSAL_MESSAGE:
            return 0, 0

        parts = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
        parts = [p.strip() for p in parts if p.strip()]

        total_claims = len(parts)
        cited_claims = sum(1 for p in parts if re.search(r"\[S\d+\]", p))

        return cited_claims, total_claims

    def _compute_confidence_score(
        self,
        query: str,
        answer: str,
        final_chunks: list[dict],
        memory_info: dict | None = None,
    ) -> dict:
        """
        Weighted combination of retrieval, evidence, and answer heuristics.
        Returns scores in [0, 1] plus a human-readable label.
        """
        if not final_chunks or not answer or answer.strip() == REFUSAL_MESSAGE:
            return {
                "retrieval_conf": 0.0,
                "evidence_conf": 0.0,
                "answer_conf": 0.0,
                "final_conf": 0.0,
                "label": "Very Low",
            }

        top_chunks = final_chunks[:5]
        scores = [float(c.get("score", 0.0) or 0.0) for c in top_chunks]

        top1_score = scores[0] if scores else 0.0
        avg_top3 = sum(scores[:3]) / max(len(scores[:3]), 1)

        max_rrf_like = 0.06

        norm_top1 = self._clamp01(top1_score / max_rrf_like)
        norm_avg_top3 = self._clamp01(avg_top3 / max_rrf_like)

        support_ratio = (
            sum(1 for s in scores if s >= RELEVANCE_THRESHOLD) / max(len(top_chunks), 1)
        )

        retrieval_conf = self._clamp01(
            0.5 * norm_top1 +
            0.3 * norm_avg_top3 +
            0.2 * support_ratio
        )

        distinct_sources = len({
            (c.get("source") or c.get("title") or "unknown")
            for c in top_chunks
        })

        source_agreement = self._clamp01(distinct_sources / 3.0)

        cited_claims, total_claims = self._count_supported_claims(answer)
        citation_coverage = (
            cited_claims / max(total_claims, 1)
            if total_claims > 0 else 0.0
        )

        chunk_support = support_ratio

        evidence_conf = self._clamp01(
            0.4 * chunk_support +
            0.3 * source_agreement +
            0.3 * citation_coverage
        )

        answer_lower = answer.lower()

        uncertainty_terms = [
            "may",
            "might",
            "possibly",
            "unclear",
            "not enough evidence",
            "insufficient information",
            "suggests",
            "appears to",
            "likely",
            "could",
        ]
        uncertainty_hits = sum(1 for t in uncertainty_terms if t in answer_lower)
        uncertainty_penalty = min(uncertainty_hits / 6.0, 1.0)

        groundedness = citation_coverage
        consistency = 1.0
        completeness = 0.8 if len(answer.strip()) > 80 else 0.4

        answer_conf = self._clamp01(
            0.4 * groundedness +
            0.3 * consistency +
            0.3 * completeness
        )
        answer_conf = self._clamp01(answer_conf * (1.0 - 0.25 * uncertainty_penalty))

        final_conf = self._clamp01(
            0.45 * retrieval_conf +
            0.30 * evidence_conf +
            0.25 * answer_conf
        )

        if final_conf >= 0.85:
            label = "High"
        elif final_conf >= 0.65:
            label = "Good"
        elif final_conf >= 0.45:
            label = "Moderate"
        elif final_conf >= 0.25:
            label = "Low"
        else:
            label = "Very Low"

        return {
            "retrieval_conf": round(retrieval_conf, 3),
            "evidence_conf": round(evidence_conf, 3),
            "answer_conf": round(answer_conf, 3),
            "final_conf": round(final_conf, 3),
            "label": label,
        }

    # -----------------------------------------------------------------------
    # Post-processing
    # -----------------------------------------------------------------------

    def _post_process(
        self,
        query: str,
        answer: str,
        memory_info: dict,
        multi_turn: bool,
    ) -> None:
        if self.enable_pcc and self.pcc_memory:
            self.pcc_memory.add_message("assistant", answer)

        if multi_turn and answer != REFUSAL_MESSAGE:
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})

            max_msgs = MAX_HISTORY_TURNS * 2
            if len(self.conversation_history) > max_msgs:
                self.conversation_history = self.conversation_history[-max_msgs:]

            if self.enable_pcc and self.pcc_memory:
                try:
                    self.pcc_memory.maybe_store_long_term(self.conversation_history)
                except Exception:
                    logger.exception("PCC auto-store failed.")

    # -----------------------------------------------------------------------
    # Graph analytics: authors × molecule (corpus KG)
    # -----------------------------------------------------------------------

    def try_author_gene_graph_answer(self, query: str) -> tuple[str, list, str, dict] | None:
        """
        If the question asks for authors/co-authors linked to a gene in the corpus graph,
        answer from Neo4j aggregation (no chunk relevance gate).
        """
        rq = self._rewrite_paper_discovery_query(self._rewrite_query_with_context(query))
        if not (_wants_author_gene_analytics(query) or _wants_author_gene_analytics(rq)):
            return None

        entities = self._extract_entities_from_query(rq)
        candidates: list[str] = []
        for e in entities:
            if (e.get("type") or "").lower() == "protein" and e.get("name"):
                candidates.append(e["name"])
        candidates.extend(_all_gene_symbol_candidates(query))
        candidates.extend(_all_gene_symbol_candidates(rq))
        # de-dupe preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for c in candidates:
            k = c.strip()
            if k and k not in seen:
                seen.add(k)
                ordered.append(k)

        memory_info: dict = {
            "pcc_enabled": self.enable_pcc,
            "gene_author_analytics": True,
            "retrieval_query": rq,
        }

        for gene_try in ordered:
            data = self.store.get_authors_for_molecule(gene_try)
            if data.get("author_count", 0) == 0 and data.get("paper_count", 0) == 0:
                continue

            gene = data.get("gene", gene_try)
            lines = [
                f"From the indexed corpus knowledge graph, the following **authors** are linked to "
                f"work involving **{gene}** (via extracted gene–chunk associations). "
                "This is corpus-grounded metadata, not a guarantee of every real-world collaborator.",
                "",
            ]
            for i, row in enumerate(data.get("authors", [])[:80], 1):
                nm = row.get("name", "")
                papers = row.get("papers") or []
                bits = []
                for p in papers[:6]:
                    t = p.get("title") or p.get("source", "")
                    y = p.get("year", "")
                    bits.append(f"{t} ({y})" if y else str(t))
                more = f" (+{len(papers) - 6} more)" if len(papers) > 6 else ""
                lines.append(f"{i}. **{nm}** — {', '.join(bits)}{more}")

            if data.get("author_count", 0) > 80:
                lines.append(f"\n… and {data['author_count'] - 80} additional author(s) in the graph.")

            if self.enable_pcc and self.pcc_memory:
                self.pcc_memory.add_message("user", query)

            answer = "\n".join(lines)
            sources = [
                {
                    "source": "graph_analytics",
                    "title": f"KG: authors × {gene}",
                    "text": answer[:1200],
                    "year": "",
                    "authors": [],
                    "score": 1.0,
                }
            ]
            memory_info["analytics_gene"] = gene
            memory_info["analytics_author_count"] = data.get("author_count", 0)
            memory_info["analytics_paper_count"] = data.get("paper_count", 0)
            memory_info["confidence"] = self._compute_confidence_score(
                query, answer, sources, memory_info
            )
            return answer, sources, "gene_author_analytics", memory_info

        return None

    def ingest_pdf_path(self, path: str | Path, *, copy_to_pdf_dir: bool = True) -> dict:
        """Incremental ingest of one PDF into Neo4j and chunks.json."""
        from paper_ingest import incremental_ingest_pdf

        return incremental_ingest_pdf(Path(path), self.store, copy_to_pdf_dir=copy_to_pdf_dir)

    def refresh_store(self) -> None:
        """Reload store state from Neo4j after external ingest."""
        self.store.load()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def ask(
        self,
        query: str,
        multi_turn: bool = True,
        corpus_generation_model: str | None = None,
    ) -> tuple:
        graph_ans = self.try_author_gene_graph_answer(query)
        if graph_ans:
            answer, final_chunks, intent, memory_info = graph_ans
            self._post_process(query, answer, memory_info, multi_turn)
            return answer, final_chunks, intent, memory_info

        intent, memory_context, memory_info, final_chunks, context, citation_tags, paper_discovery = (
            self._prepare_query(query)
        )

        if not final_chunks:
            memory_info["confidence"] = self._compute_confidence_score(
                query=query,
                answer=REFUSAL_MESSAGE,
                final_chunks=[],
                memory_info=memory_info,
            )
            if self.enable_pcc and self.pcc_memory:
                self.pcc_memory.add_message("assistant", REFUSAL_MESSAGE)
                try:
                    self.conversation_history.append({"role": "user", "content": query})
                    self.conversation_history.append({"role": "assistant", "content": REFUSAL_MESSAGE})
                    self.conversation_history = self.conversation_history[-(MAX_HISTORY_TURNS * 2):]
                    self.pcc_memory.maybe_store_long_term(self.conversation_history)
                except Exception:
                    logger.exception("PCC auto-store after refusal failed.")
            return REFUSAL_MESSAGE, [], intent, memory_info

        if paper_discovery:
            answer, used_chunks = self._answer_paper_discovery_query(query, final_chunks, memory_info)
            memory_info["confidence"] = self._compute_confidence_score(
                query=query,
                answer=answer,
                final_chunks=used_chunks,
                memory_info=memory_info,
            )
            self._post_process(query, answer, memory_info, multi_turn)
            return answer, used_chunks, intent, memory_info

        gen_client, gen_model, ollama_extras = self._resolve_corpus_generation(
            intent, corpus_generation_model
        )
        messages, _ = self._build_messages(query, context, citation_tags, memory_context, multi_turn)

        _uses_openai = self._openai_client is not None and gen_client is self._openai_client
        logger.info(
            "ask LLM call: intent=%s model=%s -> %s",
            intent,
            gen_model,
            "OpenAI API (RunPod will not see this request)" if _uses_openai else f"Ollama {get_ollama_base_url()}",
        )

        create_kw = dict(
            model=gen_model,
            max_tokens=MAX_TOKENS,
            messages=messages,
            **_rag_generation_kwargs(ollama_extras),
        )
        response = gen_client.chat.completions.create(**create_kw)
        answer = _strip_thinking((response.choices[0].message.content or "").strip())

        if answer and answer != REFUSAL_MESSAGE:
            answer = _inject_citations_post(answer, final_chunks, citation_tags)
            answer = _strip_invalid_citation_tags(answer, len(final_chunks))
            answer = _strip_markdown_headings(answer)

        memory_info["confidence"] = self._compute_confidence_score(
            query=query,
            answer=answer,
            final_chunks=final_chunks,
            memory_info=memory_info,
        )
        self._post_process(query, answer, memory_info, multi_turn)
        return answer, final_chunks, intent, memory_info

    def ask_stream(
        self,
        query: str,
        multi_turn: bool = True,
        corpus_generation_model: str | None = None,
    ):
        graph_ans = self.try_author_gene_graph_answer(query)
        if graph_ans:
            answer, used_chunks, _intent, memory_info = graph_ans
            self._post_process(query, answer, memory_info, multi_turn)
            yield answer, None, None
            yield None, used_chunks, memory_info
            return

        # _prepare_query can take a long time (Ollama rewrites + Neo4j + PCC). Yield heartbeats
        # so the Streamlit UI does not freeze on "Searching…" with no updates.
        yield None, None, None
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag_prepare") as _prep_pool:
            _fut = _prep_pool.submit(self._prepare_query, query)
            while True:
                try:
                    (
                        intent,
                        memory_context,
                        memory_info,
                        final_chunks,
                        context,
                        citation_tags,
                        paper_discovery,
                    ) = _fut.result(timeout=2.0)
                    break
                except FutureTimeoutError:
                    yield None, None, None

        if not final_chunks:
            memory_info["confidence"] = self._compute_confidence_score(
                query=query,
                answer=REFUSAL_MESSAGE,
                final_chunks=[],
                memory_info=memory_info,
            )
            if self.enable_pcc and self.pcc_memory:
                self.pcc_memory.add_message("assistant", REFUSAL_MESSAGE)
                try:
                    self.conversation_history.append({"role": "user", "content": query})
                    self.conversation_history.append({"role": "assistant", "content": REFUSAL_MESSAGE})
                    self.conversation_history = self.conversation_history[-(MAX_HISTORY_TURNS * 2):]
                    self.pcc_memory.maybe_store_long_term(self.conversation_history)
                except Exception:
                    logger.exception("PCC auto-store after refusal failed.")
            yield REFUSAL_MESSAGE, [], memory_info
            yield None, [], memory_info
            return

        if paper_discovery:
            answer, used_chunks = self._answer_paper_discovery_query(query, final_chunks, memory_info)
            memory_info["confidence"] = self._compute_confidence_score(
                query=query,
                answer=answer,
                final_chunks=used_chunks,
                memory_info=memory_info,
            )
            self._post_process(query, answer, memory_info, multi_turn)
            yield answer, None, None
            yield None, used_chunks, memory_info
            return

        gen_client, gen_model, ollama_extras = self._resolve_corpus_generation(
            intent, corpus_generation_model
        )
        messages, _ = self._build_messages(query, context, citation_tags, memory_context, multi_turn)
        full_answer = ""

        _uses_openai = self._openai_client is not None and gen_client is self._openai_client
        logger.info(
            "ask_stream LLM call: intent=%s model=%s -> %s",
            intent,
            gen_model,
            "OpenAI API (RunPod will not see this request)" if _uses_openai else f"Ollama {get_ollama_base_url()}",
        )

        # Let the UI timeout loop run while the remote stream blocks (no tokens until end otherwise).
        yield None, None, None

        stream_kw = dict(
            model=gen_model,
            max_tokens=MAX_TOKENS,
            messages=messages,
            stream=True,
            **_rag_generation_kwargs(ollama_extras),
        )
        stream = gen_client.chat.completions.create(**stream_kw)

        chunk_q: queue.Queue = queue.Queue()
        stream_err: list[BaseException] = []

        def _pump_stream() -> None:
            try:
                for c in stream:
                    chunk_q.put(c)
            except BaseException as exc:
                stream_err.append(exc)
            finally:
                chunk_q.put(None)

        threading.Thread(target=_pump_stream, daemon=True).start()

        in_think_block = False
        raw_buffer = ""

        # --- Phase 1: consume stream from queue with heartbeats so Streamlit can enforce timeouts
        _heartbeat_sec = 2.0
        while True:
            try:
                chunk = chunk_q.get(timeout=_heartbeat_sec)
            except queue.Empty:
                yield None, None, None
                continue
            if chunk is None:
                if stream_err:
                    raise stream_err[0]
                break

            ch0 = chunk.choices[0] if chunk.choices else None
            if not ch0 or not ch0.delta:
                continue
            delta = ch0.delta.content
            if not delta:
                continue

            if "<think>" in delta:
                in_think_block = True
            if in_think_block:
                if "</think>" in delta:
                    in_think_block = False
                    delta = delta.split("</think>", 1)[-1]
                else:
                    continue

            if delta:
                raw_buffer += delta
                yield delta, None, None

        # --- Phase 2: clean leaked instructions line-by-line
        cleaned_lines = []
        for line in raw_buffer.splitlines():
            line_clean = _re_module.sub(r"<think>[\s\S]*?</think>", "", line).rstrip()
            if not _is_leaked_instruction(line_clean):
                cleaned_lines.append(line_clean)
        full_answer = "\n".join(cleaned_lines).strip()

        logger.info("RAW model output (first 600 chars): %r", full_answer[:600])

        # --- Phase 3: fallback retry if answer is empty/leaked
        if len(full_answer.strip()) < 20:
            logger.warning("Empty/leaked answer detected — retrying non-streaming.")
            retry_messages = [
                {"role": "system", "content": "You are a biomedical scientist. Answer the question using only the provided research context. Start your answer immediately with the scientific content."},
                {"role": "user", "content": f"Research context:\n\n{context}\n\nQuestion: {query}\n\nProvide a direct scientific answer:"},
            ]
            try:
                retry_kw = dict(
                    model=gen_model,
                    max_tokens=MAX_TOKENS,
                    messages=retry_messages,
                    **_rag_generation_kwargs(ollama_extras, retry=True),
                )
                retry_resp = gen_client.chat.completions.create(**retry_kw)
                raw_retry = (retry_resp.choices[0].message.content or "").strip()
                logger.info("RAW retry output (first 600 chars): %r", raw_retry[:600])
                full_answer = _re_module.sub(r"<think>[\s\S]*?</think>", "", raw_retry, flags=_re_module.DOTALL).strip()
            except Exception as exc:
                logger.exception("Retry after empty answer failed: %s", exc)

        # --- Phase 4: inject citations; replace_full lets UIs swap streamed text for cited final
        if full_answer:
            full_answer = _inject_citations_post(full_answer, final_chunks, citation_tags)
            full_answer = _strip_invalid_citation_tags(full_answer, len(final_chunks))
            full_answer = _strip_markdown_headings(full_answer)
            yield full_answer, None, {"replace_full": True}

        memory_info["confidence"] = self._compute_confidence_score(
            query=query,
            answer=full_answer,
            final_chunks=final_chunks,
            memory_info=memory_info,
        )
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
        if self.enable_pcc and self.pcc_memory and self.conversation_history:
            try:
                self.pcc_memory.compress_and_store_long_term(self.conversation_history)
            except Exception:
                logger.exception("PCC store on reset failed.")

        self.conversation_history = []

        if self.enable_pcc and self.pcc_memory:
            self.pcc_memory.clear_short_term()

        logger.info("Conversation history cleared.")

    def close(self) -> None:
        if self.enable_pcc and self.pcc_memory and self.conversation_history:
            try:
                self.pcc_memory.compress_and_store_long_term(self.conversation_history)
            except Exception:
                logger.exception("PCC store on close failed.")

        self.store.close()

        if self.pcc_memory:
            self.pcc_memory.close()

        if self.normalizer:
            self.normalizer.close()

        logger.info("RAG Engine closed.")