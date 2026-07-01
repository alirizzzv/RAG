"""FastAPI entrypoint (Bullet 3: containerized FastAPI service).

/chat invokes the LangGraph orchestrator and returns a typed, citation-carrying
response. Chainlit UI is mounted in P6.
"""
import logging

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.graph.build import get_graph
from app.models.schemas import ChatRequest, ChatResponse, Intent
from app.ratelimit import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Knowledge Retrieval System")

_session_limiter = SlidingWindowRateLimiter(
    settings.rate_limit_per_session, settings.rate_limit_window_seconds)
_global_limiter = SlidingWindowRateLimiter(
    settings.rate_limit_global, settings.rate_limit_window_seconds)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not _global_limiter.allow():
        raise HTTPException(status_code=429, detail="Global rate limit exceeded; retry shortly.")
    if not _session_limiter.allow(req.session_id or "anon"):
        raise HTTPException(status_code=429, detail="Per-session rate limit exceeded; slow down.")

    try:
        result = get_graph().invoke({"question": req.question})
    except Exception:
        logger.exception("graph invocation failed")
        raise HTTPException(status_code=500, detail="Failed to process the question.")

    return ChatResponse(
        answer=result.get("answer", ""),
        intent=result.get("intent", Intent.RETRIEVAL),
        citations=result.get("citations", []),
        generated_code=result.get("generated_code"),
        artifact_path=result.get("artifact_path"),
        retries=result.get("retries", 0),
    )
