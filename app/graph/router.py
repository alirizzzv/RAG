"""Intent router node (Bullet 1: dynamic intent-based routing).

TODO(P2): classify the question into Intent.RETRIEVAL vs Intent.CODE using the
LLM with a tight few-shot prompt, and return {"intent": ...}.
"""
from app.graph.state import GraphState
from app.models.schemas import Intent


def route(state: GraphState) -> GraphState:
    raise NotImplementedError("Implemented in P2")


def route_condition(state: GraphState) -> str:
    """Edge selector used by the graph to branch on intent."""
    return state.get("intent", Intent.RETRIEVAL).value
