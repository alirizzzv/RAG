"""Chainlit chat UI (Bullet 3: multi-turn conversational interface).

Each browser session gets its own SessionMemory; prior Q&A turns are injected
into the answer prompt so follow-up questions ("what about the other company?")
resolve correctly. Charts from the code agent are rendered inline.
"""
import logging
import uuid

import chainlit as cl

import app.chainlit_patch  # noqa: F401  — rebinds local_steps to carry a default
from app.config import settings
from app.graph.build import get_graph
from app.memory import SessionMemory
from app.models.schemas import Intent
from app.ratelimit import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

GRAPH = get_graph()

_session_limiter = SlidingWindowRateLimiter(
    settings.rate_limit_per_session, settings.rate_limit_window_seconds)
_global_limiter = SlidingWindowRateLimiter(
    settings.rate_limit_global, settings.rate_limit_window_seconds)

WELCOME = """**Agentic Knowledge Retrieval System**

Ask anything about the loaded knowledge base. I can:
- Answer factual questions with **cited sources** (doc + page)
- **Generate and run Python** to compute or chart data — with self-correcting retries if the code fails

*Try: "What was Northwind Robotics total revenue?" or "Plot quarterly revenue for all three companies"*
"""


@cl.on_chat_start
async def start():
    cl.user_session.set("memory", SessionMemory())
    cl.user_session.set("rl_key", uuid.uuid4().hex)
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message):
    memory: SessionMemory = cl.user_session.get("memory") or SessionMemory()

    # --- guardrails ---------------------------------------------------------
    if len(message.content) > settings.max_question_chars:
        await cl.Message(content=(
            f"⚠️ That question is too long "
            f"(max {settings.max_question_chars} characters). Please shorten it."
        )).send()
        return

    rl_key = cl.user_session.get("rl_key") or "anon"
    if not _global_limiter.allow():
        await cl.Message(content=(
            "⚠️ The public demo is busy right now. Please try again in a minute."
        )).send()
        return
    if not _session_limiter.allow(rl_key):
        await cl.Message(content=(
            "⚠️ You're sending messages a bit too quickly — give it a few seconds."
        )).send()
        return

    # Stream a placeholder while the graph runs (graph is synchronous)
    msg = cl.Message(content="")
    await msg.send()

    try:
        result = GRAPH.invoke({
            "question": message.content,
            "history": memory.as_prompt_block(),
        })
    except Exception:
        logger.exception("graph invocation failed")
        msg.content = (
            "⚠️ Sorry — something went wrong answering that. "
            "Please try rephrasing, or check that the LLM API key is configured."
        )
        await msg.update()
        return

    intent: Intent = result.get("intent", Intent.RETRIEVAL)
    answer: str = result.get("answer", "")
    citations = result.get("citations", [])
    artifact = result.get("artifact_path")
    retries = result.get("retries", 0)

    # --- build response elements ---
    elements = []

    # inline chart image
    if artifact:
        elements.append(cl.Image(path=artifact, name="chart", display="inline"))

    # citations as a side panel
    if citations:
        cite_lines = []
        seen = set()
        for c in citations:
            key = (c.source, c.page)
            if key in seen:
                continue
            seen.add(key)
            cite_lines.append(f"**{c.source}** — p.{c.page}")
            cite_lines.append(f"> {c.snippet[:160]}…")
        elements.append(cl.Text(
            name="Sources",
            content="\n".join(cite_lines),
            display="side",
        ))

    # footer: intent badge + retry count
    meta_parts = [f"`{intent.value}`"]
    if retries:
        meta_parts.append(f"self-corrected {retries}×")
    footer = "  \n*" + " · ".join(meta_parts) + "*"

    msg.content = answer + footer
    msg.elements = elements
    await msg.update()

    # persist this turn in session memory
    memory.add(
        question=message.content,
        answer=answer,
        intent=intent.value,
    )
