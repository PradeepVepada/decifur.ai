"""
web_search.py
-------------
Low-latency web search pipeline:
  1. DuckDuckGo search  (~1 s)
  2. Parallel scrape top-3 pages with trafilatura + httpx  (~2 s total)
  3. Snippet fallback when scraping fails
  4. LRU cache (64 entries) to skip repeat searches
  5. OpenAI API streaming summarisation (default **gpt-4o-mini** — not your hosted Ollama/RunPod)

Public API matches RAGEngine.ask_stream yield protocol exactly:
    Yields (token,  None,    None)              during generation.
    Yields (None,   sources, {"mode": "web"})   when done.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import httpx
import trafilatura
from openai import OpenAI

from model_config import WEB_SEARCH_OPENAI_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_SEARCH_RESULTS  = 5
MAX_SCRAPE_WORKERS  = 5         # one worker per result
SCRAPE_TIMEOUT_S    = 5.0       # per-page timeout
MAX_CHARS_PER_PAGE  = 5000      # chars kept per scraped page
MAX_CONTEXT_CHARS   = 14_000    # total context passed to LLM (~5 k tokens)
_CACHE_MAX          = 64

WEB_SYSTEM_PROMPT = """\
You are a knowledgeable biomedical research assistant. You have been given live web search results below.

INSTRUCTIONS:
1. Prioritise the WEB SEARCH CONTEXT — cite sources as [1], [2], etc. for every key claim drawn from it.
2. If the search context is partial or thin, supplement with your own scientific knowledge to give a
   complete, well-structured answer, but make clear which parts come from the sources and which from
   general knowledge.
3. Always provide a substantive answer — never refuse solely because the snippets are short.
4. Be concise, accurate, and scientifically precise.
5. Format the answer as **2–4 paragraphs** with a blank line between paragraphs — never one wall-of-text paragraph.
"""

# ---------------------------------------------------------------------------
# Module-level LRU cache  (query.lower().strip() → (context, sources))
# ---------------------------------------------------------------------------
_CACHE: OrderedDict[str, tuple] = OrderedDict()


def _cache_get(query: str) -> Optional[tuple]:
    key = query.lower().strip()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    return None


def _cache_put(query: str, value: tuple) -> None:
    key = query.lower().strip()
    if key in _CACHE:
        _CACHE.move_to_end(key)
    else:
        _CACHE[key] = value
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


# ---------------------------------------------------------------------------
# Step 1 — DuckDuckGo search
# ---------------------------------------------------------------------------

def _ddg_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """Return [{title, href, body}, …]. Empty list on any failure."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results, region="wt-wt"))
    except Exception:
        logger.exception("DuckDuckGo search failed.")
        return []


# ---------------------------------------------------------------------------
# Step 2 — Parallel scraping with trafilatura + httpx
# ---------------------------------------------------------------------------

def _scrape_one(url: str) -> Optional[str]:
    """
    Fetch URL with httpx (3 s timeout), extract main text with trafilatura.
    Returns None on paywall / non-HTML / timeout / empty extraction.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
        resp = httpx.get(
            url,
            timeout=SCRAPE_TIMEOUT_S,
            headers=headers,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        if "text/html" not in resp.headers.get("content-type", ""):
            return None
        text = trafilatura.extract(
            resp.text,
            no_fallback=True,
            include_comments=False,
        )
        return text if text and len(text) > 100 else None
    except Exception:
        logger.debug("Scrape failed for %s", url)
        return None


def _scrape_parallel(urls: list[str]) -> dict[str, Optional[str]]:
    """Scrape urls concurrently. Returns {url: text | None}."""
    out: dict[str, Optional[str]] = {}
    with ThreadPoolExecutor(
        max_workers=MAX_SCRAPE_WORKERS, thread_name_prefix="ws_scrape"
    ) as ex:
        futures = {ex.submit(_scrape_one, url): url for url in urls}
        for fut in as_completed(futures, timeout=SCRAPE_TIMEOUT_S + 1):
            url = futures[fut]
            try:
                out[url] = fut.result()
            except Exception:
                out[url] = None
    return out


# ---------------------------------------------------------------------------
# Step 3 — Build LLM context + source dicts
# ---------------------------------------------------------------------------

def _build_web_context(
    results: list[dict], scraped: dict[str, Optional[str]]
) -> tuple[str, list[dict]]:
    """
    Merge search results with scraped content.
    Prefers scraped text; falls back to DDG snippet.
    Returns (context_string, sources_list).
    sources_list matches the shape render_sources() expects.
    """
    sources: list[dict] = []
    parts:   list[str]  = []
    total   = 0

    for i, r in enumerate(results, 1):
        url     = r.get("href", "")
        title   = r.get("title", url)
        snippet = r.get("body", "")

        page_text = (scraped.get(url) or snippet or "").strip()
        if not page_text:
            continue

        page_text = page_text[:MAX_CHARS_PER_PAGE]
        remaining = MAX_CONTEXT_CHARS - total
        if remaining <= 0:
            break

        chunk  = page_text[:remaining]
        total += len(chunk)

        parts.append(f"[{i}] {title}\nURL: {url}\n\n{chunk}")
        sources.append(
            {
                "source":   url,
                "title":    title,
                "text":     snippet[:500],
                "year":     "",
                "authors":  [],
                "score":    1.0,
                "is_web":   True,
            }
        )

    return "\n\n---\n\n".join(parts), sources


# ---------------------------------------------------------------------------
# Public API — streaming generator
# ---------------------------------------------------------------------------

def web_search_answer_stream(query: str, client: OpenAI):
    """
    Full pipeline: search → scrape → LLM stream.

    Yields (token, None, None) per token during generation,
    then   (None, sources_list, {"mode": "web"}) once complete.
    """
    # ---- Cache hit ----
    cached = _cache_get(query)
    if cached:
        context, sources = cached
        logger.info("Web cache hit: %s", query[:60])
    else:
        # ---- Search ----
        results = _ddg_search(query)
        if not results:
            yield (
                "Web search is temporarily unavailable or returned no results. "
                "Please try again or switch to Corpus mode.",
                None,
                None,
            )
            yield None, [], {"mode": "web"}
            return

        # ---- Parallel scrape all results ----
        top_urls = [r.get("href", "") for r in results if r.get("href")]
        scraped = _scrape_parallel(top_urls) if top_urls else {}

        # ---- Build context ----
        context, sources = _build_web_context(results, scraped)
        if not context.strip():
            yield (
                "No usable content was retrieved from web search. "
                "Please try rephrasing or switch to Corpus mode.",
                None,
                None,
            )
            yield None, [], {"mode": "web"}
            return

        _cache_put(query, (context, sources))

    # ---- Stream LLM completion ----
    messages = [
        {"role": "system", "content": WEB_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"WEB SEARCH CONTEXT:\n\n{context}\n\n---\n\nQUESTION: {query}"
            ),
        },
    ]

    try:
        stream = client.chat.completions.create(
            model=WEB_SEARCH_OPENAI_MODEL,
            messages=messages,
            max_tokens=1200,
            temperature=0.3,
            top_p=0.9,
            stream=True,
        )
        for chunk in stream:
            ch0 = chunk.choices[0] if chunk.choices else None
            if not ch0 or not ch0.delta:
                continue
            delta = ch0.delta.content
            if delta:
                yield delta, None, None
    except Exception as exc:
        logger.exception("Web search LLM streaming failed.")
        yield f"\n\n*Web search answer failed: {exc}*", None, None

    yield None, sources, {"mode": "web", "sources_count": len(sources)}
