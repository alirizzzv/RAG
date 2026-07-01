"""Intent router node (Bullet 1: dynamic intent-based routing).

Classifies each question into RETRIEVAL (answer from docs) or CODE (compute /
plot from retrieved data) with a single cheap LLM call.
"""
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import GraphState
from app.llm.provider import get_llm
from app.models.schemas import Intent

_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You route a user question to one of two handlers. Reply with EXACTLY one "
     "word, no punctuation:\n"
     "  code       - the user wants a number computed, data aggregated, or a "
     "chart/plot/table generated from figures in the documents.\n"
     "  retrieval  - a factual or explanatory question answered by quoting the "
     "documents.\n"
     "Examples:\n"
     "  'What was the company strategy in 2024?' -> retrieval\n"
     "  'Plot the quarterly revenue' -> code\n"
     "  'What is total segment revenue?' -> code"),
    ("human", "{question}"),
])


def route(state: GraphState) -> GraphState:
    resp = (_ROUTER_PROMPT | get_llm()).invoke({"question": state["question"]})
    decision = resp.content.strip().lower()
    intent = Intent.CODE if "code" in decision else Intent.RETRIEVAL
    return {"intent": intent}


def route_condition(state: GraphState) -> str:
    """Edge selector: branch on the intent the router stored in state."""
    return state.get("intent", Intent.RETRIEVAL).value
