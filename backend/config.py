from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = False

    # LLM Configuration
    LLM_PROVIDER: str = "openai"  # "openai" or "ollama"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.1

    # OpenAI Configuration (if using OpenAI)
    OPENAI_API_KEY: Optional[str] = None

    # Ollama Configuration (if using Ollama)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./market_research.db"

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour

    # Task Configuration
    MAX_CONCURRENT_TASKS: int = 5
    TASK_TIMEOUT_MINUTES: int = 30

    # Data Collection Limits
    MAX_SOURCES_PER_QUERY: int = 20
    MAX_DATA_POINTS_PER_SOURCE: int = 100

    # Compliance Settings
    ENABLE_STRICT_COMPLIANCE: bool = True
    COMPLIANCE_LOG_LEVEL: str = "INFO"

    # Cache Configuration
    ENABLE_CACHING: bool = True
    CACHE_TTL_HOURS: int = 24

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "logs/market_research.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def get_llm_config(self) -> dict:
        """Get LLM configuration based on provider"""
        if self.LLM_PROVIDER.lower() == "ollama":
            return {
                "provider": "ollama",
                "model": self.OLLAMA_MODEL,
                "base_url": self.OLLAMA_BASE_URL,
                "temperature": self.LLM_TEMPERATURE,
            }
        else:
            return {
                "provider": "openai",
                "model": self.LLM_MODEL,
                "api_key": self.OPENAI_API_KEY,
                "temperature": self.LLM_TEMPERATURE,
            }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings():
    """Reload settings from environment"""
    global _settings
    _settings = Settings()
    return _settings
