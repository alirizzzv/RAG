"""Assemble the LangGraph orchestrator (Bullet 1: multi-agent pipeline).

    START -> router -> retrieve -> (intent?) -> answer  -> END
                                             -> code    -> END

Router sets the intent; retrieval always runs so both branches have grounded
context; then we fork to the plain-answer path or the code-execution agent.
"""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph.code_agent import run_code_agent
from app.graph.retrieval import answer_with_citations, retrieve
from app.graph.router import route, route_condition
from app.graph.state import GraphState
from app.models.schemas import Intent


@lru_cache(maxsize=1)
def get_graph():
    g = StateGraph(GraphState)
    g.add_node("router", route)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer_with_citations)
    g.add_node("code", run_code_agent)

    g.add_edge(START, "router")
    g.add_edge("router", "retrieve")
    g.add_conditional_edges("retrieve", route_condition, {
        Intent.RETRIEVAL.value: "answer",
        Intent.CODE.value: "code",
    })
    g.add_edge("answer", END)
    g.add_edge("code", END)
    return g.compile()
