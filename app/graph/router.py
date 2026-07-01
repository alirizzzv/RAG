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
     "word — either 'retrieval' or 'code' — no punctuation, no explanation.\n\n"
     "  retrieval  - the answer is a fact, explanation, or quote that exists "
     "in the documents (strategy, risks, descriptions, comparisons).\n"
     "  code       - the user explicitly wants a chart, plot, graph, calculation, "
     "or aggregation computed from numbers in the documents.\n\n"
     "When in doubt choose 'retrieval'.\n\n"
     "Examples:\n"
     "  'What was the company strategy in 2024?'          -> retrieval\n"
     "  'Which segment grew fastest?'                     -> retrieval\n"
     "  'What were the key risks?'                        -> retrieval\n"
     "  'Plot quarterly revenue as a bar chart'           -> code\n"
     "  'Show a pie chart of segment breakdown'           -> code\n"
     "  'Calculate total annual revenue across companies' -> code"),
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
