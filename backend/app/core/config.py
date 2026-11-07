from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and the project .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_model: str = "qwen3.8:27b-mlx"
    ollama_api_key: SecretStr = SecretStr("ollama")

    agent_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    joke_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=300.0, gt=0)
    graph_recursion_limit: int = Field(default=12, ge=4, le=50)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached settings instance."""
    return Settings()