"""
paper_ingest.py
---------------
Incremental PDF ingest: extend Neo4j + optional data/chunks.json without full rebuild.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from pathlib import Path

from graph_store import GraphStore
from extract import build_chunks_for_pdf, PDF_DIR, OUTPUT_FILE

logger = logging.getLogger(__name__)


def _strip_embeddings_for_json(chunk: dict) -> dict:
    out = {k: v for k, v in chunk.items() if k not in ("embedding", "relationships", "node_id")}
    return out


def _max_numeric_chunk_id(chunks_path: Path) -> int:
    if not chunks_path.exists():
        return -1
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return -1
        best = -1
        for c in data:
            if isinstance(c, dict) and c.get("id") is not None:
                try:
                    best = max(best, int(c["id"]))
                except (TypeError, ValueError):
                    continue
        return best
    except Exception:
        logger.exception("Could not read %s for max chunk id.", chunks_path)
        return -1


def merge_chunks_json(new_chunks: list[dict], chunks_path: Path | None = None) -> None:
    """Append new chunk records (without embeddings) to data/chunks.json."""
    path = chunks_path or OUTPUT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            logger.exception("Resetting chunks.json — could not parse.")
            existing = []

    slim = [_strip_embeddings_for_json(c) for c in new_chunks]
    existing.extend(slim)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    logger.info("Merged %d chunk(s) into %s (total %d).", len(slim), path, len(existing))


def safe_upload_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^\w.\- ]", "_", base, flags=re.UNICODE).strip()
    return (base or "upload.pdf")[:240]


def incremental_ingest_pdf(
    pdf_path: Path,
    store: GraphStore,
    *,
    chunks_json_path: Path | None = None,
    copy_to_pdf_dir: bool = True,
) -> dict:
    """
    Ingest one PDF: chunk → Neo4j MERGE → merge chunks.json.
    ``pdf_path`` may be a temp file; if copy_to_pdf_dir, file is copied into PDF_DIR first.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        return {"ok": False, "error": f"File not found: {pdf_path}"}

    dest_dir = Path(PDF_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    name = safe_upload_filename(pdf_path.name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"

    dest = dest_dir / name
    if copy_to_pdf_dir and pdf_path.resolve() != dest.resolve():
        if dest.exists():
            return {"ok": False, "error": f"A PDF named “{name}” already exists in the papers folder."}
        shutil.copy2(pdf_path, dest)
        work_path = dest
    else:
        work_path = pdf_path

    if store.paper_exists(work_path.name):
        return {"ok": False, "error": f"The corpus already includes “{work_path.name}”."}

    try:
        file_sha256 = hashlib.sha256(work_path.read_bytes()).hexdigest().lower()
    except OSError as e:
        return {"ok": False, "error": f"Could not read PDF: {e}"}

    existing_same_bytes = store.find_paper_id_by_file_hash(file_sha256)
    if existing_same_bytes:
        if copy_to_pdf_dir and dest.exists() and pdf_path.resolve() != dest.resolve():
            try:
                dest.unlink()
            except OSError:
                pass
        return {
            "ok": False,
            "error": (
                f'This file is already in the corpus under the name “{existing_same_bytes}” '
                "(same file content)."
            ),
        }

    cpath = chunks_json_path or OUTPUT_FILE
    start_id = _max_numeric_chunk_id(cpath) + 1

    try:
        new_chunks, _cache = build_chunks_for_pdf(work_path, start_chunk_id=start_id)
    except Exception as e:
        logger.exception("build_chunks_for_pdf failed.")
        if copy_to_pdf_dir and dest.exists() and pdf_path.resolve() != dest.resolve():
            try:
                dest.unlink()
            except OSError:
                pass
        return {"ok": False, "error": str(e)}

    if not new_chunks:
        if copy_to_pdf_dir and dest.exists() and pdf_path.resolve() != dest.resolve():
            try:
                dest.unlink()
            except OSError:
                pass
        return {"ok": False, "error": "No text or chunks produced from PDF."}

    try:
        store.ingest_chunk_batch(new_chunks)
    except Exception as e:
        logger.exception("ingest_chunk_batch failed.")
        if copy_to_pdf_dir and dest.exists() and pdf_path.resolve() != dest.resolve():
            try:
                dest.unlink()
            except OSError:
                pass
        return {"ok": False, "error": str(e)}

    try:
        merge_chunks_json(new_chunks, cpath)
    except Exception:
        logger.exception("chunks.json merge failed — Neo4j ingest succeeded.")

    store.load()
    return {
        "ok": True,
        "paper": work_path.name,
        "chunks": len(new_chunks),
        "path": str(work_path),
    }
