"""PDF ingestion pipeline (Bullet 1: ChromaDB knowledge base).

TODO(P1): load PDFs from data/, split into chunks (settings.chunk_size), embed
with the local model, and persist to a Chroma collection. Runnable as:
    python -m app.ingest.loader
"""
from app.config import settings


def ingest(data_dir: str = "./data") -> int:
    """Ingest all PDFs under data_dir into ChromaDB. Returns chunk count."""
    raise NotImplementedError("Implemented in P1")


if __name__ == "__main__":
    n = ingest()
    print(f"Ingested {n} chunks into {settings.chroma_dir}")
