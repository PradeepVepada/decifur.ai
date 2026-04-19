"""
Ollama-compatible API + optional OpenAI config for the RAG stack.

OLLAMA_BASE_URL  — host root only, no `/v1` suffix (the OpenAI client appends `/v1`).
  Example RunPod: `https://…-11434.proxy.runpod.net`. If `.env` mistakenly includes
  `/v1`, requests can hit the wrong path and return HTTP 405 Method Not Allowed.
  Defaults to http://localhost:11434 when unset (local Ollama).
OLLAMA_MODEL     — model tag on that server (e.g. qwen35-biomedical).

Fast path, rewrite, PCC, extraction, and graph relationship calls use the
Ollama-compatible endpoint at OLLAMA_BASE_URL.

Deep reasoning (cross-paper synthesis / topic evolution) can use OpenAI when
DEEP_REASONING_BACKEND=openai and OPENAI_API_KEY is set; otherwise it falls
back to the same remote Ollama model (OLLAMA_MODEL).

OPENAI_CORPUS_MODEL — chat “GPT-5 nano” / OpenAI corpus answers (override id if your key uses a snapshot).
"""

import os


def _env(name: str, default: str) -> str:
    v = (os.environ.get(name) or "").strip()
    return v if v else default


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _normalize_ollama_base_url(url: str) -> str:
    """Strip trailing slashes and a mistaken `/v1` suffix; code adds `/v1` for the OpenAI shim."""
    u = (url or "").strip().rstrip("/")
    if u.lower().endswith("/v1"):
        u = u[:-3].rstrip("/")
    return u


def get_ollama_base_url() -> str:
    """
    Current Ollama host root (read from the environment each call).

    Streamlit and other long-lived processes must not rely on a module-level
    constant here: `import model_config` runs once, so a frozen URL would
    ignore later `.env` edits until process restart.
    """
    return _normalize_ollama_base_url(_env("OLLAMA_BASE_URL", "http://localhost:11434"))


OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen35-biomedical")

# openai | ollama — openai uses OPENAI_DEEP_MODEL on the OpenAI API for deep intents only
# (cross_paper_synthesis, topic_evolution). With openai + OPENAI_API_KEY, RunPod sees no GPU
# load for those questions; set to ollama to keep all corpus generation on OLLAMA_BASE_URL.
DEEP_REASONING_BACKEND = _env("DEEP_REASONING_BACKEND", "openai").lower()
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_DEEP_MODEL = _env("OPENAI_DEEP_MODEL", "gpt-5-nano")
# Corpus answers when the UI selects the OpenAI path (see rag_engine._resolve_corpus_generation).
OPENAI_CORPUS_MODEL = _env("OPENAI_CORPUS_MODEL", "gpt-5-nano")
# Streamlit "Web" mode: DuckDuckGo + this model on the OpenAI API (not RunPod).
WEB_SEARCH_OPENAI_MODEL = _env("WEB_SEARCH_OPENAI_MODEL", "gpt-4o-mini")

FAST_MODEL = OLLAMA_MODEL
OLLAMA_DEEP_MODEL = OLLAMA_MODEL
# Backward compatibility: deep generation uses OPENAI_DEEP_MODEL or Ollama depending on backend
DEEP_MODEL = OPENAI_DEEP_MODEL if DEEP_REASONING_BACKEND == "openai" and OPENAI_API_KEY else OLLAMA_DEEP_MODEL
PCC_MODEL = OLLAMA_MODEL
REWRITE_MODEL = OLLAMA_MODEL
RELATIONSHIP_MODEL = OLLAMA_MODEL
METADATA_MODEL = OLLAMA_MODEL

# RAG generation — keep in sync with `ollama/Modelfile.qwen35-biomedical` PARAMETER lines.
OLLAMA_NUM_CTX = max(2048, _env_int("OLLAMA_NUM_CTX", 8192))
OLLAMA_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.55)
OLLAMA_TOP_P = _env_float("OLLAMA_TOP_P", 0.85)
OLLAMA_TOP_K = max(1, _env_int("OLLAMA_TOP_K", 40))
OLLAMA_REPEAT_PENALTY = _env_float("OLLAMA_REPEAT_PENALTY", 1.1)
# Stream retry after empty/leaked output (slightly lower than main for stability).
OLLAMA_RETRY_TEMPERATURE = _env_float("OLLAMA_RETRY_TEMPERATURE", 0.35)
# Matches Modelfile `num_predict`; OpenAI-compat `max_tokens`.
RAG_MAX_OUTPUT_TOKENS = max(256, _env_int("RAG_MAX_OUTPUT_TOKENS", 2816))


def ollama_api_extra_body() -> dict:
    """Ollama OpenAI-compatible `extra_body` for chat.completions (ignored by OpenAI.com)."""
    return {
        "think": False,
        "num_ctx": OLLAMA_NUM_CTX,
        "top_k": OLLAMA_TOP_K,
        "repeat_penalty": OLLAMA_REPEAT_PENALTY,
    }
