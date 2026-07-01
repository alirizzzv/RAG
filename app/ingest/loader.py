"""PDF ingestion pipeline (Bullet 1: ChromaDB knowledge base).

Loads every PDF under data/, splits into overlapping chunks, embeds with the
local model, and persists to a Chroma collection. Run as:
    python -m app.ingest.loader
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.vectorstore import get_vectorstore


def _load_pdfs(data_dir: str) -> list:
    """Load each PDF page as a Document (carries source + page metadata)."""
    docs = []
    for pdf in sorted(Path(data_dir).glob("*.pdf")):
        docs.extend(PyPDFLoader(str(pdf)).load())
    return docs


def ingest(data_dir: str = "./data") -> int:
    """Ingest all PDFs under data_dir into ChromaDB. Returns chunk count."""
    raw = _load_pdfs(data_dir)
    if not raw:
        print(f"No PDFs found in {data_dir} — add some and re-run.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(raw)

    # Normalise `source` to a bare filename so citations read cleanly.
    for c in chunks:
        c.metadata["source"] = Path(c.metadata.get("source", "")).name
        # PyPDF pages are 0-indexed; present them 1-indexed to humans.
        if "page" in c.metadata:
            c.metadata["page"] = int(c.metadata["page"]) + 1

    vs = get_vectorstore()
    # Idempotent: clear any prior contents so re-running never duplicates.
    existing = vs.get()
    if existing and existing.get("ids"):
        vs.delete(ids=existing["ids"])
    vs.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    n = ingest()
    print(f"Ingested {n} chunks into {settings.chroma_dir}")
