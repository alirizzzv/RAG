"""FastAPI entrypoint (Bullet 3: containerized FastAPI service).

For now exposes /health and a stub /chat. P2 wires /chat to the LangGraph
orchestrator; Chainlit UI is mounted in P6.
"""
from fastapi import FastAPI

from app.models.schemas import ChatRequest, ChatResponse, Intent

app = FastAPI(title="Agentic Knowledge Retrieval System")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # TODO(P2): graph = get_graph(); result = graph.invoke({"question": req.question})
    return ChatResponse(answer="Scaffold ready — orchestrator lands in P2.",
                        intent=Intent.RETRIEVAL)
