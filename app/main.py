"""FastAPI entrypoint (Bullet 3: containerized FastAPI service).

/chat invokes the LangGraph orchestrator and returns a typed, citation-carrying
response. Chainlit UI is mounted in P6.
"""
from fastapi import FastAPI

from app.graph.build import get_graph
from app.models.schemas import ChatRequest, ChatResponse, Intent

app = FastAPI(title="Agentic Knowledge Retrieval System")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = get_graph().invoke({"question": req.question})
    return ChatResponse(
        answer=result.get("answer", ""),
        intent=result.get("intent", Intent.RETRIEVAL),
        citations=result.get("citations", []),
        generated_code=result.get("generated_code"),
        artifact_path=result.get("artifact_path"),
        retries=result.get("retries", 0),
    )
