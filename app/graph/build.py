"""Assemble the LangGraph orchestrator (Bullet 1: multi-agent pipeline).

TODO(P2): wire router -> {retrieval | code_agent} -> answer, compile, and expose
a `get_graph()` singleton the FastAPI layer invokes.
"""


def get_graph():
    raise NotImplementedError("Implemented in P2")
