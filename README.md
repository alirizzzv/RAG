# Agentic Knowledge Retrieval System

A multi-agent Retrieval-Augmented Generation (RAG) pipeline that answers
questions over a document knowledge base with **citation-grounded** responses,
and can **write & run its own Python** (sandboxed, with self-correcting retries)
to compute or plot answers from retrieved data.

Built with **LangGraph · ChromaDB · FastAPI · Chainlit · Pydantic · Docker**.

---

## Architecture

```
User (Chainlit UI)
   │
   ▼
FastAPI /chat  ──►  LangGraph orchestrator
                         │
              ┌──────────┴───────────┐
        [Router node]  ─ intent ─►   │
              │                      │
      ┌───────┴────────┐             │
      ▼                ▼             │
 [Retrieval agent]  [Code-exec agent]│
      │                │             │
      ▼                ▼             │
  ChromaDB        Sandbox (Docker /  │
 (local embeds)   subprocess) + ≤3   │
      │           self-correct retries
      ▼                │
 Citation grounding ◄──┘
      │
      ▼
 answer + sources  ──►  UI
```

## Design highlights

- **Swappable LLM backend** — change one env var (`LLM_MODEL`) to move between
  Gemini / Groq / Claude / GPT. No other code changes. (`app/llm/provider.py`)
- **Local, free embeddings** — `sentence-transformers`, runs offline, no API cost.
- **Two sandbox backends** — Docker (strong isolation, local) or a
  resource-limited subprocess (portable, for hosts without Docker).
- **Typed everywhere** — Pydantic schemas at the API boundary, a TypedDict
  `GraphState` threaded through every node.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your provider API key
python -m app.ingest.loader   # build the ChromaDB index from data/*.pdf
uvicorn app.main:app --reload # http://localhost:8000/docs
```

## Build roadmap

- [x] **P0** — scaffold, config, swappable LLM interface, typed schemas
- [ ] **P1** — PDF → ChromaDB ingestion, semantic retrieval
- [ ] **P2** — LangGraph orchestrator: router + retrieval agent
- [ ] **P3** — citation grounding (source + page threaded to the answer)
- [ ] **P4** — sandboxed code-exec agent + self-correcting retry loop
- [ ] **P5** — Pydantic validation + multi-turn session memory
- [ ] **P6** — Chainlit UI, Dockerize, deploy to Hugging Face Spaces
- [ ] **P7** — README demo GIF, smoke test, (optional) eval metric

## Layout

| Path | Responsibility |
|------|----------------|
| `app/graph/`   | LangGraph nodes: router, retrieval, code agent, assembly |
| `app/llm/`     | Provider-agnostic LLM + embeddings |
| `app/ingest/`  | PDF loading, chunking, ChromaDB persistence |
| `app/sandbox/` | Isolated Python executor (docker / subprocess) |
| `app/models/`  | Pydantic request/response/citation schemas |
| `app/main.py`  | FastAPI app |
