"""Central configuration (12-factor style: every value can be overridden by env vars or a .env file)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_prefix="NETOPS_", extra="ignore")

    # ---- Data ---------------------------------------------------------------
    dataset_uri: str = "hf://datasets/fenar/5g_netops/5G_netops_data_100K.csv.xz"
    raw_csv: Path = ROOT_DIR / "data" / "raw" / "5G_netops_data_100K.csv"
    artifacts_dir: Path = ROOT_DIR / "data" / "artifacts"
    index_rows: int = 100_000          # how many records to index (all 100K by default)
    random_seed: int = 42

    # ---- Retrieval ----------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "auto"    # auto | sentence-transformers | tfidf
    top_k: int = 5
    hybrid_alpha: float = 0.6          # weight of dense (semantic) vs. sparse (BM25) in fusion
    rrf_k: int = 60                    # Reciprocal-Rank-Fusion constant
    candidate_pool: int = 200          # candidates pulled from each retriever before fusion

    # ---- LLM ----------------------------------------------------------------
    llm_provider: str = "auto"         # auto | openai | azure | gemini | anthropic | none
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.5-flash"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2500
    llm_timeout_s: float = 45.0

    # ---- Guardrails ---------------------------------------------------------
    max_query_chars: int = 1000
    min_query_chars: int = 8

    # ---- Feedback / storage --------------------------------------------------
    feedback_db: Path = ROOT_DIR / "data" / "artifacts" / "feedback.db"
    feedback_weight: float = 0.15      # max re-ranking boost/penalty from user feedback

    # ---- API ----------------------------------------------------------------
    api_url: str = "http://localhost:8000"   # used by the Streamlit front-end
    api_key: str = ""                        # optional: if set, required as X-API-Key header


def _load_dotenv(path: Path) -> None:
    """Export .env entries (e.g. OPENAI_API_KEY) to os.environ without overriding real env vars."""
    import os
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@lru_cache
def get_settings() -> Settings:
    _load_dotenv(ROOT_DIR / ".env")
    s = Settings()
    s.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return s
