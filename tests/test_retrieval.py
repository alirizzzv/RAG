"""Citation grounding: only sources the answer actually cites should be shown.

Regression test for the gap where every top-k retrieved doc was surfaced as a
citation regardless of whether the model's answer referenced it.
"""
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.graph import retrieval
from app.graph.retrieval import ground_citations, parse_cited_sources, standalone_query


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


def test_standalone_query_skips_llm_on_first_turn(monkeypatch):
    # No history → nothing to resolve, and we must NOT spend an LLM call.
    def _boom(*a, **k):
        raise AssertionError("get_llm must not be called when there is no history")

    monkeypatch.setattr(retrieval, "get_llm", _boom)

    assert standalone_query("What was Q4 revenue?", None) == "What was Q4 revenue?"
    assert standalone_query("What was Q4 revenue?", "") == "What was Q4 revenue?"


def test_standalone_query_resolves_followup_against_history(monkeypatch):
    # A follow-up with a dangling reference gets rewritten into a standalone query.
    fake_llm = RunnableLambda(
        lambda _pv: AIMessage(content="What was Initech's Q4 revenue?")
    )
    monkeypatch.setattr(retrieval, "get_llm", lambda *a, **k: fake_llm)

    history = "Q: What was Acme's Q4 revenue?\nA: Acme's Q4 revenue was $10M."
    out = standalone_query("what about the other company?", history)

    assert out == "What was Initech's Q4 revenue?"
