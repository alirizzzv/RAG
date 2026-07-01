"""Pydantic schemas — the typed contract at every boundary (API, graph, UI)."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """What the router decides a question needs."""
    RETRIEVAL = "retrieval"   # answer from the knowledge base
    CODE = "code"             # compute / plot from retrieved data


class Citation(BaseModel):
    source: str = Field(..., description="Source document filename")
    page: Optional[int] = Field(None, description="Page number if known")
    snippet: str = Field(..., description="The grounding text chunk")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(None, description="Multi-turn memory key")


class ChatResponse(BaseModel):
    answer: str
    intent: Intent
    citations: list[Citation] = Field(default_factory=list)
    generated_code: Optional[str] = None      # populated for CODE intent
    artifact_path: Optional[str] = None        # e.g. produced chart image
    retries: int = 0                           # self-correction attempts used
