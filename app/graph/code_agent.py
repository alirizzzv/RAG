"""Code-exec agent node (Bullet 2: sandboxed execution + self-correcting retry).

TODO(P4): generate Python from the question + retrieved data, run it in the
sandbox, and on error feed the traceback back to the LLM to retry — up to
settings.code_agent_max_retries times.
"""
from app.graph.state import GraphState


def run_code_agent(state: GraphState) -> GraphState:
    raise NotImplementedError("Implemented in P4")
