from __future__ import annotations

from functools import lru_cache
from pydantic import HttpUrl, SecretStr
from pydantic_settings import BaseSettings




class Settings(BaseSettings):
    # Core
    app_name: str = "sec-10q-analyst"
    environment: str = "local"
    debug: bool = True

    # SEC / EDGAR
    sec_user_agent: str
    sec_max_rps: int = 8
    sec_base_url: HttpUrl = "https://www.sec.gov"
    sec_data_base_url: HttpUrl = "https://data.sec.gov"

    # OpenAI
    openai_api_key: SecretStr
    llm_model: str = "openai:gpt-5"
    embedding_model: str = "text-embedding-3-large"

    # Vector DB
    vector_db_url: str = "postgresql://user:pass@localhost:5432/sec_vectors"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
