"""Retrieval agent (Bullet 1: semantic retrieval + citation grounding).

`retrieve` runs for every question so both the answer path and the code path
get grounded context. `answer_with_citations` produces the final grounded text
plus structured Citation objects the UI can render.
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.graph.state import GraphState
from app.llm.provider import get_llm
from app.models.schemas import Citation
from app.vectorstore import get_vectorstore

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using ONLY the provided context. Cite the source of "
     "each claim inline as [source p.PAGE]. If the context does not contain the "
     "answer, say you don't have enough information — do not invent facts.\n\n"
     "{history}"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def retrieve(state: GraphState) -> GraphState:
    docs = get_vectorstore().similarity_search(state["question"], k=settings.top_k)
    return {"context": docs}


def format_context(docs: list[Document]) -> str:
    """Render retrieved docs into a prompt block tagged with source + page."""
    return "\n\n".join(
        f"[{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}] "
        f"{d.page_content}"
        for d in docs
    )


def docs_to_citations(docs: list[Document]) -> list[Citation]:
    """Turn retrieved docs into structured citations for the UI."""
    return [
        Citation(
            source=d.metadata.get("source", "?"),
            page=d.metadata.get("page"),
            snippet=d.page_content[:200].strip(),
        )
        for d in docs
    ]


def answer_with_citations(state: GraphState) -> GraphState:
    docs = state.get("context", [])
    answer = (_ANSWER_PROMPT | get_llm()).invoke({
        "context": format_context(docs),
        "question": state["question"],
        "history": state.get("history") or "",
    }).content
    return {"answer": answer, "citations": docs_to_citations(docs)}
