---
title: Agentic Knowledge Retrieval System
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Agentic Knowledge Retrieval System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-0.6-orange?logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-1.5-green?logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

<p align="center">
  <a href="https://alirizzv-agentic-rag.hf.space">
    <img src="https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face%20Spaces-yellow" />
  </a>
</p>

<p align="center">
  <img src="docs/demo.gif" width="780" alt="Retrieval with citations, multi-turn memory, and a code agent that generates and runs charts" />
</p>

A multi-agent Retrieval-Augmented Generation system that answers questions over a
document knowledge base with **citation-grounded** responses, and writes and
executes **sandboxed Python** — with self-correcting retries — to compute or chart
answers directly from retrieved data.

---

## Features

- **Intent-based routing** — a router node classifies each question and dispatches it to the retrieval agent or the code-execution agent.
- **Citation grounding** — every retrieval answer carries its source document and page number; nothing is asserted without a reference.
- **Sandboxed code execution** — generated Python runs in an isolated process with a hard timeout and CPU cap; a Docker backend adds network and memory isolation for production.
- **Self-correcting retries** — when generated code fails, the traceback is fed back to the model and the code is regenerated, up to a configurable retry limit.
- **Provider-agnostic LLM** — switch between Gemini, Groq, Claude, or GPT by changing a single environment variable.
- **Local embeddings** — `sentence-transformers` runs offline with no API cost.
- **Multi-turn memory** — per-session conversation history resolves follow-up questions.

---

## Architecture

```
User (Chainlit UI)
       │
       ▼
 FastAPI /chat
       │
       ▼
 LangGraph Orchestrator
       │
   [Router node]  ── single LLM call to classify intent
       │
  ┌────┴────────────────────────┐
  ▼                             ▼
[Retrieval node]           (always runs — grounds both paths)
  │
  ├─ intent = retrieval ──► [Answer node] ──► grounded answer + citations
  │
  └─ intent = code ────────► [Code-exec node]
                                  │
                             LLM generates Python
                                  │
                             Sandbox (subprocess / Docker)
                             timeout · CPU cap · isolated dir
                                  │
                             ┌────┴────┐
                           ok?     error? ── feed traceback back to LLM
                                  │          and retry (≤ N times)
                                  ▼
                             chart / computed result
```

**Design principles**

- **Retrieval runs before branching**, so the code agent operates on grounded numbers rather than hallucinated ones.
- **Typed state throughout** — a `TypedDict` graph state flows through every node, so no field is silently dropped.
- **Single source of truth for the vector store** — ingestion (writes) and retrieval (reads) share one accessor and can never disagree on collection, embeddings, or path.
- **Pluggable backends** — both the LLM provider and the sandbox executor are swappable behind a stable interface.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (local) |
| LLM backend | Gemini / Groq / Claude / GPT (swappable) |
| API | FastAPI |
| UI | Chainlit |
| Validation | Pydantic v2 |
| Sandboxing | subprocess + Docker |
| Deployment | Docker · Hugging Face Spaces |

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/alirizzzv/RAG.git && cd RAG

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure the LLM provider
cp .env.example .env          # set LLM_MODEL and the matching API key

# 4. Generate sample documents (or drop your own PDFs into data/)
python scripts/generate_sample_docs.py

# 5. Build the knowledge base
python -m app.ingest.loader   # PDF → chunks → embeddings → ChromaDB

# 6. Run
chainlit run chainlit_app.py  # Chat UI  → http://localhost:8000
uvicorn app.main:app --reload # REST API → http://localhost:8000/docs
```

### Docker

```bash
docker build -t agentic-rag .
docker run -p 7860:7860 -e GROQ_API_KEY=... -e LLM_MODEL=groq:llama-3.3-70b-versatile agentic-rag
```

---

## Configuration

All settings are read from the environment (see `.env.example`). Switch the entire
LLM backend by changing one line:

```bash
LLM_MODEL=google_genai:gemini-2.0-flash      # Google AI Studio (free tier)
LLM_MODEL=groq:llama-3.3-70b-versatile       # Groq (free, fast)
LLM_MODEL=anthropic:claude-3-5-haiku-latest  # Anthropic
LLM_MODEL=openai:gpt-4o-mini                 # OpenAI
```

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `google_genai:gemini-1.5-flash` | Provider and model, as `provider:model` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `SANDBOX_BACKEND` | `subprocess` | `subprocess` or `docker` |
| `SANDBOX_TIMEOUT_SECONDS` | `10` | Hard wall-clock limit per execution |
| `CODE_AGENT_MAX_RETRIES` | `3` | Max self-correction attempts |

---

## Usage

```
User:  What was Northwind Robotics' Q4 revenue?

Agent: Q4 revenue was $138 million [northwind_robotics_2024_annual_report.pdf p.1].
       Full-year revenue was $425 million, up 37% from $310 million in 2023.

       Sources
       • northwind_robotics_2024_annual_report.pdf — p.1


User:  Plot quarterly revenue for all three companies as grouped bars.

Agent: [generates Python → executes in sandbox → returns chart]
       📊 Chart generated  ·  retries: 0
```

---

## Evaluation

Measured on a 10-question set (`eval/qa_set.json`) spanning factual retrieval,
numerical reasoning, and chart generation:

| Metric | Score | Notes |
|---|---|---|
| Citation recall | **7 / 7 (100%)** | Correct source in top-*k* for every retrieval question |
| Router accuracy | **9 / 10 (90%)** | Intent correctly classified |
| Answer accuracy | **9 / 10 (90%)** | Expected facts present in the answer |

```bash
python eval/run_eval.py
```

---

## Project Structure

```
RAG/
├── chainlit_app.py              # Chat UI entry point
├── app/
│   ├── main.py                  # FastAPI app
│   ├── config.py                # Settings (pydantic-settings)
│   ├── vectorstore.py           # Shared ChromaDB accessor
│   ├── memory.py                # Per-session conversation memory
│   ├── graph/
│   │   ├── state.py             # Typed graph state
│   │   ├── router.py            # Intent classifier node
│   │   ├── retrieval.py         # Semantic search + citation node
│   │   ├── code_agent.py        # Code-gen + sandbox + retry node
│   │   └── build.py             # LangGraph assembly
│   ├── llm/provider.py          # Swappable LLM + embeddings
│   ├── ingest/loader.py         # PDF → chunks → ChromaDB
│   ├── sandbox/executor.py      # subprocess + Docker backends
│   └── models/schemas.py        # Pydantic schemas
├── scripts/
│   ├── generate_sample_docs.py  # Reproducible sample corpus
│   └── make_demo_gif.py         # Demo GIF generator
├── eval/                        # Evaluation set + harness
├── data/                        # PDFs + ChromaDB store
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## License

Released under the MIT License.
