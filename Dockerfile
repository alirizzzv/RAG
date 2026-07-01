FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-warm matplotlib font cache so the first chart doesn't timeout
RUN python -c "import matplotlib.pyplot"

COPY . .

# Build the ChromaDB index from the bundled sample PDFs at image-build time.
# Local sentence-transformers embeddings — no API key needed here.
RUN python -m app.ingest.loader

# HF Spaces requires port 7860
EXPOSE 7860

# GROQ_API_KEY (or any other LLM key) must be set as a Space Secret in the
# HF Spaces UI — it is injected as an env var at runtime, never baked in.
CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "7860"]
