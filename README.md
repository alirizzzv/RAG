# Agentic Knowledge Retrieval System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-0.6-orange?logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-1.5-green?logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-swappable-blueviolet?logo=openai&logoColor=white" />
</p>

<p align="center">
  <b>Ask questions. Get cited answers. Have Python written, executed, and self-corrected — all grounded in your documents.</b>
</p>

---

## What it does

Upload any PDF knowledge base. The system routes each question to one of two agents:

| Question type | Agent | What happens |
|---|---|---|
| *"What was the company strategy in 2024?"* | **Retrieval agent** | Semantic search → cited answer (source + page) |
| *"Plot quarterly revenue as a bar chart"* | **Code agent** | Python generated → sandboxed execution → self-correcting retry if it fails → chart returned |

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
   [Router node]  ← single LLM call to classify intent
       │
  ┌────┴────────────────────────┐
  ▼                             ▼
[Retrieval node]           (always runs — grounds both paths)
  │
  ├─ intent = retrieval ──► [Answer node]
  │                              │
  │                         grounded answer
  │                         + citations
  │
  └─ intent = code ────────► [Code-exec node]
                                  │
                             LLM generates Python
                                  │
                             Sandbox (subprocess / Docker)
                             timeout · CPU cap · isolated dir
                                  │
                             ┌────┴────┐
                           ok?     error?
                                  │
                             feed traceback back to LLM
                             retry ≤ N times  ◄── self-correction loop
                                  │
                             chart / computed result
```

**Key design decisions worth defending in interviews:**

- **Retrieval always runs before branching** — so the code agent always has grounded numbers, not hallucinated ones.
- **Provider abstraction** — swap the entire LLM backend (Gemini / Groq / Claude / GPT) by changing one env var, zero code changes.
- **Typed state throughout** — a `TypedDict GraphState` flows through every LangGraph node; impossible to silently drop a field.
- **Two sandbox backends** — `subprocess` (portable, works on Hugging Face Spaces) or `docker` (hard network/memory/CPU isolation for production).

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Agent orchestration | **LangGraph** | Typed graph, conditional edges, easy to extend |
| Vector store | **ChromaDB** | Persistent, local, no infra needed |
| Embeddings | **sentence-transformers** (local) | Free, offline, no API cost |
| LLM backend | **Gemini / Groq / Claude / GPT** (swappable) | One env-var switch |
| API | **FastAPI** | Auto-generated docs, Pydantic-native |
| UI | **Chainlit** | Multi-turn chat with file attachment, zero frontend code |
| Validation | **Pydantic v2** | Typed schemas at every boundary |
| Sandboxing | **subprocess + Docker** | Dual-backend untrusted code execution |
| Deploy | **Hugging Face Spaces** (Docker SDK) | Free public URL |

---

## Quickstart

```bash
# 1. Clone and enter
git clone https://github.com/alirizzzv/RAG.git && cd RAG

# 2. Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure LLM provider (one key, one model string)
cp .env.example .env
#  → edit .env: set LLM_MODEL and paste the matching API key

# 4. Add your PDFs (or use the generated samples)
python scripts/generate_sample_docs.py   # creates 3 fictional annual reports in data/

# 5. Build the knowledge base
python -m app.ingest.loader              # PDF → chunks → embeddings → ChromaDB

# 6. Run
uvicorn app.main:app --reload            # REST API  → http://localhost:8000/docs
# or
chainlit run app/chainlit_app.py         # Chat UI   → http://localhost:8000
```

---

## Swapping the LLM (one line)

Edit `LLM_MODEL` in `.env` — nothing else changes:

```bash
LLM_MODEL=google_genai:gemini-2.0-flash      # Google AI Studio — free tier
LLM_MODEL=groq:llama-3.3-70b-versatile       # Groq — free, very fast
LLM_MODEL=anthropic:claude-3-5-haiku-latest  # Anthropic (paid)
LLM_MODEL=openai:gpt-4o-mini                 # OpenAI (paid)
```

This works because `app/llm/provider.py` wraps `init_chat_model` — a provider-agnostic factory that parses the `"provider:model"` string at runtime.

---

## Sample interaction

```
User:  What was Northwind Robotics' Q4 revenue?

Agent: Q4 revenue was $138 million [northwind_robotics_2024_annual_report.pdf p.1].
       Full-year revenue was $425 million, up 37% from $310 million in 2023.

Sources:
  • northwind_robotics_2024_annual_report.pdf — p.1
    "...Q4 revenue of $138M, reflecting strong holiday-season..."

---

User:  Plot quarterly revenue for all three companies as grouped bars.

Agent: [runs self-contained Python in sandbox, produces chart]
       📊 Chart generated → artifacts/a3f1c2.png
       [retries: 0]
```

---

## Project layout

```
RAG/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # All settings via pydantic-settings
│   ├── vectorstore.py       # Shared ChromaDB handle
│   ├── graph/
│   │   ├── state.py         # TypedDict GraphState
│   │   ├── router.py        # Intent classifier node
│   │   ├── retrieval.py     # Semantic search + citation node
│   │   ├── code_agent.py    # Code-gen + sandbox + retry node
│   │   └── build.py         # LangGraph assembly
│   ├── llm/provider.py      # Swappable LLM + embeddings
│   ├── ingest/loader.py     # PDF → chunks → ChromaDB
│   ├── sandbox/executor.py  # subprocess + Docker backends
│   └── models/schemas.py    # Pydantic request/response schemas
├── scripts/
│   └── generate_sample_docs.py   # Generates fictional annual reports
├── data/                    # PDFs + ChromaDB store (gitignored)
├── eval/                    # Q&A evaluation set
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## Build roadmap

- [x] **P0** — scaffold, config, swappable LLM interface, typed schemas
- [x] **P1** — PDF ingestion into ChromaDB, selective retrieval verified
- [x] **P2/P3** — LangGraph orchestrator: router + retrieval agent + citation grounding
- [x] **P4** — sandboxed code-exec agent + self-correcting retry loop (4/4 sandbox tests pass)
- [ ] **P5** — multi-turn session memory + Chainlit UI
- [ ] **P6** — Dockerize + deploy to Hugging Face Spaces
- [ ] **P7** — eval set, demo GIF, README polish

---

## Resume bullets this implements

> *Architected a **multi-agent RAG pipeline** using **LangGraph** with dynamic intent-based routing, semantic retrieval via **ChromaDB**, and citation-grounded responses.*

> *Developed a **sandboxed Python execution framework** with automated **self-correcting retry loops** for reliable chart, table, and artifact generation.*

> *Deployed a containerized **FastAPI** and **Chainlit** application with **Pydantic**-based validation, multi-turn conversational memory, and scalable backend orchestration.*

---

<p align="center">Built by <a href="https://github.com/alirizzzv">alirizzzv</a></p>
