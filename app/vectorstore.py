"""Single source of truth for the ChromaDB vector-store handle.

Both the ingestion pipeline (writes) and the retrieval agent (reads) go through
get_vectorstore(), so they can never disagree on collection / embeddings / path.
"""
from functools import lru_cache

from langchain_chroma import Chroma

from app.config import settings
from app.llm.provider import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )
