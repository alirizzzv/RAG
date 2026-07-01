"""Retrieval agent node (Bullet 1: semantic retrieval + citation grounding).

TODO(P2/P3): similarity-search ChromaDB, build a grounded answer, and attach
Citation objects (source + page + snippet) pulled from doc metadata.
"""
from app.graph.state import GraphState


def retrieve(state: GraphState) -> GraphState:
    raise NotImplementedError("Implemented in P2")


def answer_with_citations(state: GraphState) -> GraphState:
    raise NotImplementedError("Implemented in P3")
