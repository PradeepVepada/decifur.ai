"""
biomedical_normalizer.py
------------------------
Biomedical Entity Normalization Engine.

Architecture
------------
  1. SciSpaCy for fast local NER (CPU, no network).
  2. BioPortal as PRIMARY lookup (richer ontology coverage).
  3. UMLS as FALLBACK when BioPortal returns nothing.
  4. Full parallel burst via ThreadPoolExecutor + as_completed:
       ALL entities are fired simultaneously; results processed
       the instant any service responds (not after the slowest).

Review fixes applied
--------------------
  #2  BioPortal promoted to primary; UMLS used as fallback only.
  #21 normalize_text/normalize_batch were sequential — now a true
      parallel burst across all entities simultaneously.
  #22 Bare except replaced with logger.exception throughout.
  #23 urllib3 Retry policy on all HTTP calls (3 retries, backoff).
  #30 requests.Session pool configured explicitly.
      Duplicate "PATHWAY" key in _map_entity_type removed.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass
from pathlib import Path

import spacy
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ONTOLOGY_PRIORITY: dict[str, dict] = {
    "Disease":   {"primary": ["SNOMEDCT", "ICD10", "DOID"],          "secondary": ["MeSH", "UMLS"],      "fallback": "UMLS"},
    "Drug":      {"primary": ["ChEBI", "RxNorm", "MeSH"],            "secondary": ["UMLS"],              "fallback": "ChEBI"},
    "Protein":   {"primary": ["PRO", "UniProt", "GO"],               "secondary": ["Ensembl"],           "fallback": "UniProt"},
    "Gene":      {"primary": ["NCBI Gene", "HGNC"],                  "secondary": ["Ensembl", "UniProt"],"fallback": "NCBI Gene"},
    "Organism":  {"primary": ["NCBI Taxonomy"],                      "secondary": ["UMLS"],              "fallback": "NCBI Taxonomy"},
    "Anatomy":   {"primary": ["FMA", "Uberon"],                      "secondary": ["UMLS", "MeSH"],      "fallback": "Uberon"},
    "CellType":  {"primary": ["CL"],                                 "secondary": ["UMLS"],              "fallback": "CL"},
    "Pathway":   {"primary": ["Reactome", "GO Biological Process"],  "secondary": ["KEGG"],              "fallback": "Reactome"},
    "Chemical":  {"primary": ["ChEBI", "MeSH"],                      "secondary": ["RxNorm", "UMLS"],   "fallback": "ChEBI"},
}

# Cap parallelism to stay within BioPortal/UMLS rate limits.
MAX_PARALLEL_ENTITIES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(pool_connections: int = 5, pool_maxsize: int = 10) -> requests.Session:
    """
    Build a requests.Session with explicit pool sizing and automatic
    retry on transient errors (429, 5xx). [Review #23, #30]
    """
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


def load_scispacy_model():
    """Load the best available SciSpaCy biomedical NER model."""
    for model_name in ("en_core_sci_lg", "en_core_sci_md", "en_core_sci_sm"):
        try:
            return spacy.load(model_name)
        except OSError:
            logger.warning("SciSpaCy model '%s' not found, trying next.", model_name)
    raise RuntimeError(
        "No SciSpaCy model found. Install via:\n"
        "  pip install scispacy\n"
        "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
        "releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
    )


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class NormalizedEntity:
    entity: str
    entity_type: str
    normalized: dict
    confidence: float
    notes: str
    sources: list

    def to_dict(self) -> dict:
        return {
            "entity":      self.entity,
            "entity_type": self.entity_type,
            "normalized":  self.normalized,
            "confidence":  self.confidence,
            "notes":       self.notes,
            "sources":     self.sources,
        }


# ---------------------------------------------------------------------------
# BioPortal Service  (PRIMARY)
# ---------------------------------------------------------------------------

class BioPortalService:
    """
    BioPortal API — primary ontology source.
    Richer ontology coverage for most biomedical entity types.
    """

    def __init__(self, api_key: str = None):
        self.api_key  = api_key or os.environ.get("BIOPORTAL_API_KEY", "")
        self.base_url = "https://data.bioontology.org"
        self.session  = _make_session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"apikey token={self.api_key}",
                "Accept":        "application/json",
            })

    def search(self, term: str, max_results: int = 5) -> list:
        if not self.api_key:
            return []
        try:
            resp = self.session.get(
                f"{self.base_url}/search",
                params={"q": term, "pagesize": max_results, "include": "prefLabel,synonym,definition"},
                timeout=12,
            )
            if resp.status_code == 200:
                results = []
                for item in resp.json().get("collection", []):
                    ont = item.get("links", {}).get("ontology", "").split("/")[-1]
                    results.append({
                        "id":       item.get("@id", ""),
                        "label":    item.get("prefLabel", ""),
                        "ontology": ont,
                        "synonyms": item.get("synonym", []) if isinstance(item.get("synonym"), list) else [],
                        "source":   "BioPortal",
                    })
                return results
        except Exception:
            logger.exception("BioPortal search failed for term '%s'", term)
        return []


# ---------------------------------------------------------------------------
# UMLS Service  (FALLBACK)
# ---------------------------------------------------------------------------

class UMLSService:
    """UMLS API — fallback when BioPortal returns nothing."""

    def __init__(self, api_key: str = None):
        self.api_key  = api_key or os.environ.get("UMLS_API_KEY", "")
        self.base_url = "https://uts-ws.nlm.nih.gov/rest"
        self.session  = _make_session()

    def search_concept(self, term: str) -> list:
        if not self.api_key:
            return []
        try:
            resp = self.session.get(
                f"{self.base_url}/search/current",
                params={"string": term, "apiKey": self.api_key, "pageNumber": 0, "pageSize": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                return [
                    {
                        "cui":      item.get("ui", ""),
                        "name":     item.get("name", ""),
                        "ontology": "UMLS",
                        "source":   "UMLS",
                    }
                    for item in resp.json().get("result", {}).get("results", [])
                ]
        except Exception:
            logger.exception("UMLS search failed for term '%s'", term)
        return []


# ---------------------------------------------------------------------------
# BiomedicalNormalizer
# ---------------------------------------------------------------------------

class BiomedicalNormalizer:
    """
    Full-parallel biomedical entity normalization engine.

    Per-entity strategy
    -------------------
      1. Query BioPortal (primary).
      2. If BioPortal empty → fall back to UMLS.
      3. Apply ontology selection rules.

    Batch strategy (normalize_text / normalize_batch)
    --------------------------------------------------
      ALL entities submitted to thread pool at once.
      Results consumed via as_completed() — fastest entity wins;
      no waiting for the slowest. [Review #21, #3]
      Expected speedup vs sequential: 5–10×.
    """

    # De-duplicated label map [Review: duplicate PATHWAY key removed]
    _LABEL_MAP: dict[str, str] = {
        "CHEMICAL":             "Chemical",
        "DISEASE":              "Disease",
        "DRUG":                 "Drug",
        "GENE":                 "Gene",
        "PROTEIN":              "Protein",
        "GENE_OR_GENE_PRODUCT": "Protein",
        "ORGANISM":             "Organism",
        "SPECIES":              "Organism",
        "ANATOMY":              "Anatomy",
        "CELL":                 "CellType",
        "PATHWAY":              "Pathway",
        "BIOLOGICAL_PROCESS":   "Pathway",
        "MOLECULAR_FUNCTION":   "Protein",
        "MUTATION":             "Gene",
    }

    def __init__(self, umls_api_key: str = None, bioportal_api_key: str = None):
        self.nlp       = None
        self.bioportal = BioPortalService(bioportal_api_key)
        self.umls      = UMLSService(umls_api_key)
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_PARALLEL_ENTITIES,
            thread_name_prefix="normalizer",
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self):
        if self.nlp is None:
            logger.info("Loading SciSpaCy NER model...")
            self.nlp = load_scispacy_model()
            logger.info("SciSpaCy model loaded.")

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def _map_entity_type(self, label: str) -> str:
        return self._LABEL_MAP.get(label.upper(), "Entity")

    def extract_entities(self, text: str) -> list:
        if self.nlp is None:
            self.load_models()
        doc = self.nlp(text)
        return [
            {"text": ent.text, "type": self._map_entity_type(ent.label_),
             "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

    # ------------------------------------------------------------------
    # Single-entity normalisation
    # ------------------------------------------------------------------

    def _select_ontology(self, entity_type: str, mappings: list) -> tuple:
        priority     = ONTOLOGY_PRIORITY.get(entity_type, ONTOLOGY_PRIORITY["Chemical"])
        primary_set  = set(priority.get("primary", []))
        secondary_set= set(priority.get("secondary", []))
        primary, secondary = None, []
        for m in mappings:
            ont = m.get("ontology", "")
            if ont in primary_set and primary is None:
                primary = m
            elif ont in secondary_set:
                secondary.append(m)
            elif primary is None:
                primary = m
        if primary is None and mappings:
            primary = mappings[0]
        return primary, secondary[:3]

    def normalize_entity(self, entity_text: str, entity_type: str) -> NormalizedEntity:
        """BioPortal first; UMLS only if BioPortal is empty. [Review #2]"""
        mappings     = self.bioportal.search(entity_text)
        sources_used = ["BioPortal"] if mappings else []

        if not mappings:
            logger.debug("BioPortal empty for '%s' — trying UMLS fallback.", entity_text)
            mappings     = self.umls.search_concept(entity_text)
            sources_used = ["UMLS"] if mappings else []

        primary, secondary = self._select_ontology(entity_type, mappings)

        normalized: dict = {"primary": None, "secondary": []}
        if primary:
            normalized["primary"] = {
                "ontology": primary.get("ontology", "Unknown"),
                "id":       primary.get("id", primary.get("cui", "")),
                "label":    primary.get("label", primary.get("name", entity_text)),
                "synonyms": list(set(primary.get("synonyms", []))),
            }
        else:
            normalized["primary"] = {"ontology": "none", "id": "none",
                                     "label": entity_text, "synonyms": []}

        normalized["secondary"] = [
            {"ontology": s.get("ontology", ""), "id": s.get("id", s.get("cui", "")),
             "label": s.get("label", s.get("name", ""))}
            for s in secondary
        ]

        confidence = 0.0
        if primary and primary.get("ontology") != "none":
            confidence = 0.8
            if len(mappings) > 1:
                confidence = min(0.95, confidence + 0.1)

        if primary and secondary:
            notes = f"{primary.get('ontology')} selected as primary per domain rules."
        elif mappings:
            notes = f"Single mapping found via {sources_used}."
        else:
            notes = "No match found in BioPortal or UMLS."

        return NormalizedEntity(entity=entity_text, entity_type=entity_type,
                                normalized=normalized, confidence=confidence,
                                notes=notes, sources=sources_used)

    # ------------------------------------------------------------------
    # Batch normalisation — FULL PARALLEL BURST  [Review #21]
    # ------------------------------------------------------------------

    def normalize_batch(self, entities: list) -> list:
        """
        Submit ALL entities to the thread pool simultaneously.
        Consume results via as_completed — fastest wins, no straggler wait.
        """
        if not entities:
            return []

        future_to_entity: dict[Future, dict] = {
            self._executor.submit(self.normalize_entity, e["text"], e["type"]): e
            for e in entities
        }

        results = []
        for future in as_completed(future_to_entity, timeout=30):
            original = future_to_entity[future]
            try:
                results.append(future.result().to_dict())
            except Exception:
                logger.exception("Normalization failed for entity '%s'", original.get("text"))
                results.append(NormalizedEntity(
                    entity=original.get("text", ""), entity_type=original.get("type", "Entity"),
                    normalized={"primary": {"ontology": "none", "id": "none",
                                            "label": original.get("text", ""), "synonyms": []},
                                "secondary": []},
                    confidence=0.0, notes="Error during normalization.", sources=[],
                ).to_dict())
        return results

    def normalize_text(self, text: str) -> list:
        """Extract entities then normalize all in one parallel burst."""
        entities = self.extract_entities(text)
        if not entities:
            return []
        logger.debug("Launching parallel burst for %d entities.", len(entities))
        return self.normalize_batch(entities)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_biomedical_normalizer() -> BiomedicalNormalizer:
    normalizer = BiomedicalNormalizer(
        umls_api_key=os.environ.get("UMLS_API_KEY", ""),
        bioportal_api_key=os.environ.get("BIOPORTAL_API_KEY", ""),
    )
    normalizer.load_models()
    return normalizer
