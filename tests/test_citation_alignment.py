"""Citation tags, retrieval list length, and gene-token helpers must stay aligned."""

from conversation_store import _title_from_first_message
from rag_engine import (
    _all_gene_symbol_candidates,
    _filter_chunks_matching_protein_needles,
    _strip_invalid_citation_tags,
    _strip_markdown_headings,
    build_context,
)


def test_build_context_tags_match_chunk_count():
    chunks = [
        {"title": "A", "year": "2001", "authors": ["X"], "text": "one"},
        {"title": "B", "year": "2002", "authors": ["Y"], "text": "two"},
        {"title": "C", "year": "2003", "authors": ["Z"], "text": "three"},
    ]
    ctx, tags = build_context(chunks)
    assert tags == ["S1", "S2", "S3"]
    assert "[S1]" in ctx and "[S3]" in ctx
    assert len(tags) == len(chunks)


def test_strip_invalid_citation_tags_removes_out_of_range():
    ans = "Claim one [S1]. Claim bad [S99]."
    out = _strip_invalid_citation_tags(ans, num_sources=3)
    assert "[S1]" in out
    assert "[S99]" not in out


def test_all_gene_symbol_candidates_mixed_case():
    q = "What authors worked on YakA and PI3K?"
    got = _all_gene_symbol_candidates(q)
    assert "YakA" in got
    assert "PI3K" in got


def test_title_from_first_message_truncates():
    long_q = "Q " * 80
    t = _title_from_first_message(long_q, max_len=40)
    assert len(t) <= 40
    assert t.endswith("...")


def test_filter_chunks_matching_protein_needles_keeps_yaka():
    chunks = [
        {"text": "YakA regulates signaling.", "title": "Paper A", "source": "a.pdf"},
        {"text": "General cell growth control.", "title": "Paper B", "source": "b.pdf"},
    ]
    ents = [{"type": "protein", "name": "YakA", "cui": "YakA"}]
    out = _filter_chunks_matching_protein_needles(
        chunks, "What is the role of YakA?", "What is the role of YakA?", ents
    )
    assert len(out) == 1
    assert out[0]["source"] == "a.pdf"


def test_strip_markdown_headings_drops_hash_lines():
    raw = "## Bad\nKeep this.\n### Also bad\nMore."
    out = _strip_markdown_headings(raw)
    assert "##" not in out
    assert "Keep this" in out
    assert "More" in out


def test_strip_markdown_headings_trailing_section_echo():
    line = "Some prose ### 4. Quantitative / math framework"
    out = _strip_markdown_headings(line)
    assert "###" not in out
    assert "Some prose" in out


def test_filter_chunks_matching_protein_needles_falls_back_when_no_match():
    chunks = [
        {"text": "Nothing here.", "title": "X", "source": "x.pdf"},
    ]
    out = _filter_chunks_matching_protein_needles(
        chunks, "YakA role?", "YakA role?", [{"type": "protein", "name": "YakA", "cui": "YakA"}]
    )
    assert len(out) == 1
