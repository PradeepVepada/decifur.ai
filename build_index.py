"""
build_index.py
--------------
One-time script: extracts text from all PDFs and builds the Hybrid Vector
Knowledge Graph in Neo4j.

    python build_index.py

Set PDF_DIR env var to point at your papers folder before running.
"""

import time
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from extract import build_chunks
from graph_store import GraphStore
import torch


def main():
    start = time.time()
    logger.info("=" * 60)
    logger.info("  GraphRAG — Neo4j Index Builder")
    logger.info("=" * 60)

    cuda_status = "ENABLED (NVIDIA)" if torch.cuda.is_available() else "DISABLED (CPU)"
    logger.info("GPU Acceleration: %s", cuda_status)

    # Step 1: Extract and chunk
    logger.info("[1/2] Extracting and chunking PDFs...")
    chunks = build_chunks()
    logger.info("Total semantic chunks produced: %d", len(chunks))

    Path("data").mkdir(exist_ok=True)
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info("Saved chunks.json (%d chunks)", len(chunks))

    # Step 2: Build Knowledge Graph
    logger.info("[2/2] Building Knowledge Graph in Neo4j...")
    store = GraphStore()
    store.build(chunks)
    store.close()

    elapsed = time.time() - start
    logger.info("Done in %.1fs. Ready to chat — run: python chatbot.py", elapsed)


if __name__ == "__main__":
    main()
