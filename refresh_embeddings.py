"""
refresh_embeddings.py
---------------------
Safe, idempotent embedding refresh pipeline for HybdRAG.

Swaps the embedding model to NeuML/pubmedbert-base-embeddings.

Usage:
    python refresh_embeddings.py           # interactive confirmation
    python refresh_embeddings.py --yes     # non-interactive (CI/CD)

Set EMBEDDING_DEVICE=cuda to use GPU (default: cpu for stability).
[Review: EMBEDDING_DEVICE env var replaces hardcoded CPU override]
[Review: --yes flag for non-interactive automation]
"""

import os
import sys
import time
import argparse
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import torch

NEW_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
EMBEDDING_DIM  = 768
VECTOR_INDEX   = "chunk_embeddings"
ENCODE_BATCH   = 128
NEO4J_BATCH    = 500


def fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s"


def load_chunks(driver) -> list:
    logger.info("Loading chunk texts from Neo4j...")
    t0 = time.time()
    with driver.session() as session:
        records = session.run(
            "MATCH (c:Chunk) RETURN c.chunk_id AS id, c.text AS text ORDER BY c.chunk_id"
        ).data()
    logger.info("Loaded %d chunks in %s.", len(records), fmt_time(time.time() - t0))
    if not records:
        logger.error("No chunks found. Is Neo4j populated?")
        sys.exit(1)
    missing = [r["id"] for r in records if not r.get("text")]
    if missing:
        logger.warning("%d chunk(s) have empty text — will receive zero-vectors.", len(missing))
    return records


def generate_embeddings(chunks: list) -> list:
    # [Review] EMBEDDING_DEVICE env var — no hardcoded CPU override
    device = os.environ.get("EMBEDDING_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("EMBEDDING_DEVICE=cuda but CUDA not available; falling back to cpu.")
        device = "cpu"
    logger.info("Generating embeddings with %s on %s...", NEW_MODEL_NAME, device)

    t_model = time.time()
    model   = SentenceTransformer(NEW_MODEL_NAME, device=device)
    logger.info("Model loaded in %s.", fmt_time(time.time() - t_model))

    texts = [c["text"] or "" for c in chunks]
    t_enc = time.time()
    vecs  = model.encode(
        texts, batch_size=ENCODE_BATCH,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
    )
    logger.info("Encoded %d chunks → shape %s in %s.", len(texts), vecs.shape, fmt_time(time.time() - t_enc))

    if vecs.shape[1] != EMBEDDING_DIM:
        logger.error("Expected %d-dim, got %d.", EMBEDDING_DIM, vecs.shape[1])
        sys.exit(1)

    return [v.tolist() for v in vecs]


def push_embeddings(driver, chunks, embeddings) -> None:
    logger.info("Pushing updated embeddings to Neo4j...")
    total   = len(chunks)
    t0      = time.time()
    updated = 0

    for start in range(0, total, NEO4J_BATCH):
        payload = [
            {"id": c["id"], "embedding": e}
            for c, e in zip(chunks[start:start + NEO4J_BATCH], embeddings[start:start + NEO4J_BATCH])
        ]
        with driver.session() as session:
            rec = session.run("""
                UNWIND $batch AS row
                MATCH (c:Chunk {chunk_id: row.id})
                SET c.embedding = row.embedding
                RETURN count(c) AS updated_count
            """, batch=payload).single()
            updated += rec["updated_count"] if rec else 0

        end = min(start + NEO4J_BATCH, total)
        logger.info("Progress: %d/%d (%.1f%%) — %s elapsed", end, total,
                    end / total * 100, fmt_time(time.time() - t0))

    logger.info("Updated %d/%d Chunk nodes in %s.", updated, total, fmt_time(time.time() - t0))
    if updated < total:
        logger.warning("%d chunk(s) not matched — chunk_id mismatch?", total - updated)


def rebuild_vector_index(driver) -> None:
    logger.info("Rebuilding vector index '%s'...", VECTOR_INDEX)
    t0 = time.time()
    with driver.session() as session:
        session.run(f"DROP INDEX {VECTOR_INDEX} IF EXISTS")
        session.run(f"""
            CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)
        logger.info("Index created. Waiting for ONLINE state...")
        state = "unknown"
        for _ in range(60):
            row   = session.run("SHOW INDEXES WHERE name = $name", name=VECTOR_INDEX).single()
            state = row["state"] if row else "unknown"
            if state == "ONLINE":
                break
            time.sleep(1)

    if state != "ONLINE":
        logger.warning("Index state is '%s' after 60s — may still be building.", state)
    else:
        logger.info("Index ONLINE in %s.", fmt_time(time.time() - t0))


def main():
    parser = argparse.ArgumentParser(description="Refresh HybdRAG embeddings.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip interactive confirmation (for CI/CD)")
    args = parser.parse_args()

    neo4j_uri  = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER",     "neo4j")
    neo4j_pass = os.environ.get("NEO4J_PASSWORD", "password")

    logger.info("HybdRAG — Embedding Refresh")
    logger.info("  New model : %s", NEW_MODEL_NAME)
    logger.info("  Index     : %s", VECTOR_INDEX)
    logger.info("  Neo4j URI : %s", neo4j_uri)
    logger.info("  ⚠  embedding property on ALL Chunk nodes will be updated.")
    logger.info("  ⚠  vector index will be DROPPED and RECREATED.")

    if not args.yes:
        answer = input("\nProceed? [yes/no]: ").strip().lower()
        if answer != "yes":
            logger.info("Aborted.")
            sys.exit(0)

    driver   = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    timings  = {}

    try:
        with driver.session() as session:
            cnt = session.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
        logger.info("Connected to Neo4j — %d Chunk nodes found.", cnt)

        t = time.time(); chunks     = load_chunks(driver);              timings["1. Load chunks"]     = time.time() - t
        t = time.time(); embeddings = generate_embeddings(chunks);      timings["2. Generate embeds"] = time.time() - t
        t = time.time(); push_embeddings(driver, chunks, embeddings);   timings["3. Push to Neo4j"]   = time.time() - t
        t = time.time(); rebuild_vector_index(driver);                  timings["4. Rebuild index"]   = time.time() - t

        logger.info("=" * 50)
        logger.info("SUMMARY — %d chunks processed", len(chunks))
        for label, secs in timings.items():
            logger.info("  %-30s %s", label, fmt_time(secs))
        logger.info("  %-30s %s", "Total", fmt_time(sum(timings.values())))
        logger.info("Embedding refresh complete!")
        logger.info("NEXT: update EMBEDDING_MODEL_NAME in graph_store.py to '%s'", NEW_MODEL_NAME)

    except KeyboardInterrupt:
        logger.warning("Interrupted. Batches already written are persisted; re-run to complete.")
    except Exception:
        logger.exception("Embedding refresh failed.")
        raise
    finally:
        driver.close()


if __name__ == "__main__":
    main()
