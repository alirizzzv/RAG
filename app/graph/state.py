"""Shared state that flows through every LangGraph node.

LangGraph merges each node's returned dict into this TypedDict, so a field set
by the retrieval node (e.g. `context`) is visible to the code / answer nodes.
"""
from typing import Optional, TypedDict

from langchain_core.documents import Document

from app.models.schemas import Citation, Intent


class GraphState(TypedDict, total=False):
    # input
    question: str
    history: Optional[str]   # formatted prior-turn context from SessionMemory

    # router output
    intent: Intent

    # retrieval output
    search_query: str        # question after history-aware rewrite (may == question)
    context: list[Document]
    citations: list[Citation]

    # code-agent output
    generated_code: Optional[str]
    artifact_path: Optional[str]
    retries: int
    error: Optional[str]

    # final
    answer: str
