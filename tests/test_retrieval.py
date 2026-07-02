"""Citation grounding: only sources the answer actually cites should be shown.

Regression test for the gap where every top-k retrieved doc was surfaced as a
citation regardless of whether the model's answer referenced it.
"""
from langchain_core.documents import Document

from app.graph.retrieval import ground_citations, parse_cited_sources


def _doc(source: str, page: int, text: str = "content") -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_parse_cited_sources_extracts_inline_tags():
    answer = "Revenue grew [report_a.pdf p.1]. Risks were noted [report_b.pdf p.3]."
    assert parse_cited_sources(answer) == {("report_a.pdf", "1"), ("report_b.pdf", "3")}


def test_parse_cited_sources_empty_when_no_tags():
    assert parse_cited_sources("No citations here.") == set()


def test_ground_citations_drops_uncited_docs():
    docs = [_doc("report_a.pdf", 1), _doc("report_b.pdf", 3), _doc("report_c.pdf", 5)]
    answer = "Revenue grew [report_a.pdf p.1]."

    grounded = ground_citations(answer, docs)

    assert [d.metadata["source"] for d in grounded] == ["report_a.pdf"]


def test_ground_citations_falls_back_when_model_cites_nothing():
    docs = [_doc("report_a.pdf", 1), _doc("report_b.pdf", 3)]
    answer = "Revenue grew, according to the documents."

    grounded = ground_citations(answer, docs)

    assert grounded == docs
