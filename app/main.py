"""FastAPI entrypoint (Bullet 3: containerized FastAPI service).

/chat invokes the LangGraph orchestrator and returns a typed, citation-carrying
response. Chainlit UI is mounted in P6.
"""
import logging
import threading

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.graph.build import get_graph
from app.memory import SessionMemory
from app.models.schemas import ChatRequest, ChatResponse, Intent
from app.ratelimit import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Knowledge Retrieval System")

_session_limiter = SlidingWindowRateLimiter(
    settings.rate_limit_per_session, settings.rate_limit_window_seconds)
_global_limiter = SlidingWindowRateLimiter(
    settings.rate_limit_global, settings.rate_limit_window_seconds)

# Per-session conversation memory, keyed by the caller's session_id. Anonymous
# callers (no session_id) stay stateless so their turns never bleed together.
# In-process only — swap for a shared store (Redis) if scaling horizontally.
_memories: dict[str, SessionMemory] = {}
_memories_lock = threading.Lock()


def _get_memory(session_id: str) -> SessionMemory:
    with _memories_lock:
        mem = _memories.get(session_id)
        if mem is None:
            mem = SessionMemory()
            _memories[session_id] = mem
        return mem


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not _global_limiter.allow():
        raise HTTPException(status_code=429, detail="Global rate limit exceeded; retry shortly.")
    if not _session_limiter.allow(req.session_id or "anon"):
        raise HTTPException(status_code=429, detail="Per-session rate limit exceeded; slow down.")

    memory = _get_memory(req.session_id) if req.session_id else None
    history = memory.as_prompt_block() if memory else None

    try:
        result = get_graph().invoke({"question": req.question, "history": history})
    except Exception:
        logger.exception("graph invocation failed")
        raise HTTPException(status_code=500, detail="Failed to process the question.")

    intent = result.get("intent", Intent.RETRIEVAL)
    answer = result.get("answer", "")
    if memory:
        memory.add(question=req.question, answer=answer, intent=intent.value)

    return ChatResponse(
        answer=answer,
        intent=intent,
        citations=result.get("citations", []),
        generated_code=result.get("generated_code"),
        artifact_path=result.get("artifact_path"),
        retries=result.get("retries", 0),
    )
