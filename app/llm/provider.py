"""Swappable LLM + embeddings backend.

The whole point: switch providers by editing LLM_MODEL in .env, with zero
changes anywhere else. init_chat_model parses the "provider:model" string and
returns a uniform LangChain chat interface.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()  # must run before any provider client is imported
from app.config import settings


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0):
    """Return the configured chat model (provider chosen via LLM_MODEL)."""
    return init_chat_model(settings.llm_model, temperature=temperature)


@lru_cache(maxsize=1)
def get_embeddings():
    """Local, free embeddings — no API key, runs offline."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
