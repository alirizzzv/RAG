"""Central configuration, loaded from environment / .env file.

Every tunable lives here so the rest of the code never reads os.environ directly.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_model: str = "google_genai:gemini-1.5-flash"

    # Embeddings (local sentence-transformers model)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    chroma_dir: str = "./data/chroma"
    collection_name: str = "documents"

    # Retrieval
    top_k: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Sandbox
    sandbox_backend: str = "subprocess"  # "docker" | "subprocess"
    sandbox_timeout_seconds: int = 10
    code_agent_max_retries: int = 3
    artifact_dir: str = "./artifacts"

    # Guardrails (protect the public demo's shared LLM quota)
    max_question_chars: int = 2000
    rate_limit_per_session: int = 15     # requests per window, per session
    rate_limit_global: int = 60          # requests per window, all sessions
    rate_limit_window_seconds: int = 60


settings = Settings()
