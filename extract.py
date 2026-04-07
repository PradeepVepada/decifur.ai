"""
extract.py
----------
Extracts text from PDFs using hybrid approach:
  1. PyMuPDF for fast text extraction from digital PDFs.
  2. Surya OCR (GPU-accelerated) for scanned documents.
  3. scispaCy biomedical NER for entity extraction.

Review fixes applied
--------------------
  #24 Hardcoded Windows absolute path removed; PDF_DIR read from env var.
  #20 ThreadPoolExecutor replaced with sequential processing for CPU-bound
      scispaCy NER (GIL prevents true threading parallelism for CPU work).
  #22 Bare except replaced with logger.exception.
  #9  print() replaced with logging.
  Metadata cache persisted to disk so re-runs don't re-call Mistral API.
"""

import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import tiktoken
from mistralai.client import Mistral
import spacy
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# [Review #24] Read from environment variable — no more hardcoded Windows path
PDF_DIR         = Path(os.environ.get("PDF_DIR", "data/papers"))
OUTPUT_FILE     = Path("data/chunks.json")
MARKER_OUT_DIR  = Path("data/marker_output")
METADATA_CACHE_FILE = Path("data/metadata_cache.json")  # persistent cache

CHUNK_SIZE = 500

# ---------------------------------------------------------------------------
# Surya OCR (lazy init)
# ---------------------------------------------------------------------------
_surya_foundation   = None
_surya_recognition  = None
_surya_detection    = None


def get_surya_predictors():
    global _surya_foundation, _surya_recognition, _surya_detection
    if _surya_recognition is None:
        import torch
        from surya.recognition import RecognitionPredictor
        from surya.detection  import DetectionPredictor
        from surya.foundation  import FoundationPredictor

        logger.info("Initialising Surya OCR (GPU Mode with FlashAttention-2)...")
        _surya_foundation   = FoundationPredictor()
        _surya_detection    = DetectionPredictor()
        _surya_recognition  = RecognitionPredictor(_surya_foundation)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("FlashAttention-2 enabled.")
    return _surya_recognition, _surya_detection


def extract_pdf_surya(pdf_path: Path) -> str:
    rec_predictor, det_predictor = get_surya_predictors()
    doc       = fitz.open(str(pdf_path))
    ocr_text  = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix  = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        results = rec_predictor([img], det_predictor=det_predictor)
        if results and results[0]:
            for line in results[0].text_lines:
                if line.text and line.text.strip():
                    ocr_text.append(line.text.strip())
    doc.close()
    return "\n".join(ocr_text)


def extract_pdf_text(pdf_path: Path) -> str:
    doc      = fitz.open(str(pdf_path))
    all_text = [page.get_text().strip() for page in doc if page.get_text().strip()]
    doc.close()
    if all_text:
        return "\n".join(all_text)
    logger.info("No text via PyMuPDF for %s — using Surya OCR.", pdf_path.name)
    return extract_pdf_surya(pdf_path)


# ---------------------------------------------------------------------------
# Tokenizer + scispaCy
# ---------------------------------------------------------------------------

def get_encoder():
    try:
        return tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def load_scispacy_model():
    try:
        return spacy.load("en_core_sci_lg")
    except OSError as e:
        raise RuntimeError(
            "scispaCy model not found. Install via:\n"
            "  pip install scispacy\n"
            "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            "releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
        ) from e


def chunk_text_spacy(text: str, nlp, encoder, chunk_size: int) -> list[str]:
    doc              = nlp(text)
    chunks           = []
    current_sents    = []
    current_tokens   = 0

    for sent in doc.sents:
        sentence   = sent.text.strip()
        if not sentence:
            continue
        sent_tokens = len(encoder.encode(sentence))

        if current_sents and current_tokens + sent_tokens > chunk_size:
            chunks.append(" ".join(current_sents))
            current_sents  = []
            current_tokens = 0

        if sent_tokens > chunk_size:
            token_ids = encoder.encode(sentence)
            for i in range(0, len(token_ids), chunk_size):
                piece = encoder.decode(token_ids[i:i + chunk_size]).strip()
                if piece:
                    chunks.append(piece)
            continue

        current_sents.append(sentence)
        current_tokens += sent_tokens

    if current_sents:
        chunks.append(" ".join(current_sents))
    return chunks


# ---------------------------------------------------------------------------
# Metadata extraction (with persistent on-disk cache)
# ---------------------------------------------------------------------------

def _load_metadata_cache() -> dict:
    if METADATA_CACHE_FILE.exists():
        try:
            with open(METADATA_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("Could not load metadata cache; starting fresh.")
    return {}


def _save_metadata_cache(cache: dict) -> None:
    METADATA_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def extract_paper_metadata(pdf_filename: str, extracted_text: str, client: Mistral) -> dict:
    prompt = (
        "Extract metadata from this scientific paper. Return ONLY valid JSON with:\n"
        "- title, authors (array), year (int), journal, doi (or null),\n"
        "- abstract (or null), keywords (array), methods (array of {name, type}),\n"
        "- topics (3-5 key topic strings)\n\n"
        f"Filename: {pdf_filename}\n\nText (first 4000 chars):\n{extracted_text[:4000]}"
    )
    try:
        res = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(res.choices[0].message.content or "{}")
    except Exception:
        logger.exception("Metadata extraction failed for %s.", pdf_filename)
        return {
            "title": pdf_filename, "authors": ["Unknown"], "year": 0,
            "journal": "Unknown", "doi": None, "abstract": None,
            "keywords": [], "methods": [], "topics": [],
        }


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_entities_from_chunk(nlp, text: str) -> dict:
    doc           = nlp(text)
    proteins      = []
    organisms     = []
    concepts      = []
    protein_types = {"GENE_OR_GENE_PRODUCT","PROTEIN","GENE","GENE_FAMILY","RNA","DNA"}
    organism_types= {"ORGANISM","SPECIES","MULTI_CELLULAR_ORGANISM","ANATOMICAL_SYSTEM",
                     "IMMATERIAL_ANATOMICAL_ENTITY","DEVELOPING_ANATOMICAL_STRUCTURE"}

    for ent in doc.ents:
        obj = {"name": ent.text.strip(), "type": ent.label_}
        if not obj["name"] or len(obj["name"]) < 2:
            continue
        if ent.label_ in protein_types:
            proteins.append(obj)
        elif ent.label_ in organism_types:
            organisms.append(obj)
        else:
            concepts.append(obj)

    def dedup(lst):
        return list({e["name"]: e for e in lst}.values())

    return {"proteins": dedup(proteins), "organisms": dedup(organisms), "concepts": dedup(concepts)}


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_chunks() -> list[dict]:
    all_chunks  = []
    chunk_id    = 0
    encoder     = get_encoder()
    nlp         = load_scispacy_model()
    api_key     = os.environ.get("MISTRAL_API_KEY", "")
    client      = Mistral(api_key=api_key)
    paper_cache = _load_metadata_cache()   # load persistent cache

    MARKER_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
        filename = pdf_file.name
        paper_id = filename.removesuffix(".pdf").removesuffix(".PDF")

        # Try loading cached OCR output
        marker_path = MARKER_OUT_DIR / paper_id / f"{paper_id}.md"
        if marker_path.exists():
            logger.info("Loading cached OCR for %s...", filename)
            raw_text = marker_path.read_text(encoding="utf-8")
        else:
            logger.info("Extracting: %s...", filename)
            raw_text = extract_pdf_text(pdf_file)
            if raw_text.strip():
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.write_text(raw_text, encoding="utf-8")
                logger.info("Saved %d chars to marker_output.", len(raw_text))

        if not raw_text.strip():
            logger.warning("No text extracted from %s — skipping.", filename)
            continue

        # Persistent metadata cache — no repeat Mistral calls on re-runs
        if filename not in paper_cache:
            logger.info("Extracting metadata for %s via Mistral...", filename)
            paper_cache[filename] = extract_paper_metadata(filename, raw_text, client)
            _save_metadata_cache(paper_cache)

        meta  = paper_cache[filename]
        title = (meta.get("title") or filename)[:60]
        logger.info("Chunking '%s' (target %d tokens/chunk)...", title, CHUNK_SIZE)

        chunks = chunk_text_spacy(raw_text, nlp, encoder, CHUNK_SIZE)

        # [Review #20] Sequential processing for CPU-bound scispaCy NER
        # (GIL means threads give no speedup for CPU work)
        chunk_results = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text) < 50:
                continue
            entities = extract_entities_from_chunk(nlp, chunk_text)
            chunk_results.append({
                "id":           None,
                "text":         chunk_text,
                "source":       filename,
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "title":        meta.get("title", filename),
                "authors":      meta.get("authors", []),
                "year":         meta.get("year", 0),
                "journal":      meta.get("journal", "Unknown"),
                "doi":          meta.get("doi"),
                "abstract":     meta.get("abstract"),
                "keywords":     meta.get("keywords", []),
                "methods":      meta.get("methods", []),
                "topics":       meta.get("topics", []),
                "proteins":     entities["proteins"],
                "organisms":    entities["organisms"],
                "concepts":     entities["concepts"],
            })

        for res in chunk_results:
            res["id"] = chunk_id
            all_chunks.append(res)
            chunk_id += 1

        logger.info("Produced %d semantic chunks from %s.", len(chunks), filename)

    return all_chunks


def main():
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    Path("data").mkdir(exist_ok=True)

    if not PDF_DIR.exists():
        logger.error(
            "PDF_DIR '%s' does not exist. "
            "Set the PDF_DIR environment variable to your papers folder.", PDF_DIR
        )
        sys.exit(1)

    logger.info("Extracting text from PDFs in %s...", PDF_DIR)
    chunks = build_chunks()
    logger.info("Total chunks: %d", len(chunks))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info("Saved to %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
