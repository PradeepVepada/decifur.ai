"""Minimal env so `rag_engine` and related modules import in CI without a live Neo4j/Ollama."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _default_env() -> None:
    os.environ.setdefault("NEO4J_URI", "neo4j://127.0.0.1:7687")
    os.environ.setdefault("NEO4J_PASSWORD", "test")
    os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
