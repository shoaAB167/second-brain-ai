from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings."""

    app_name: str = "Second Brain AI"
    app_env: str = "development"
    debug: bool = True

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_v1_str: str = "/api/v1"

    # CORS Settings
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    # LLM Settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    ollama_api_base: Optional[str] = "http://localhost:11434"

    # Database Settings
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/second_brain_ai"

    # Experience Classifier Settings
    experience_classifier_min_confidence: float = 0.70
    experience_classifier_min_importance: float = 0.50

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
