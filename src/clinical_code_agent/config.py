"""Application configuration using Pydantic Settings.

All config is read from environment variables with the CODE_AGENT_ prefix.
This keeps secrets (API keys) out of code and allows different behavior
in local dev vs Streamlit Cloud vs production.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration for the Clinical Code Agent.

    Environment variables are prefixed with CODE_AGENT_.
    Example: CODE_AGENT_OPENAI_API_KEY=sk-...
    """

    # LLM Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # Data paths
    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/processed/codes.duckdb")
    chroma_persist_dir: Path = Path("chroma_db")

    # Retrieval settings
    semantic_search_top_k: int = 10
    similarity_threshold: float = 0.4

    # Agent settings
    max_tool_calls: int = 3  # Max tools the agent can call per query
    abstention_threshold: float = 0.3  # Below this, say "I don't know"

    # MLflow
    mlflow_experiment_name: str = "/clinical-code-agent/runs"

    model_config = {"env_prefix": "CODE_AGENT_"}


# Module-level singleton — import this everywhere
settings = Settings()
