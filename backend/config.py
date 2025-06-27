from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with Gemini and Ollama support (no OpenAI)"""

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = False

    # LLM Configuration - Gemini or Ollama only
    LLM_PROVIDER: str = "gemini"  # "gemini" or "ollama"
    LLM_MODEL: str = "gemini-pro"
    LLM_TEMPERATURE: float = 0.1

    # Google Gemini Configuration
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None  # Alternative name
    GEMINI_MODEL: str = "gemini-pro"

    # Ollama Configuration (local LLM)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./data/market_research.db"

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600

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

    def get_gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key from either variable"""
        return self.GOOGLE_API_KEY or self.GEMINI_API_KEY

    def get_llm_config(self) -> dict:
        """Get LLM configuration based on provider (Gemini or Ollama only)"""
        provider = self.LLM_PROVIDER.lower()

        if provider == "gemini":
            api_key = self.get_gemini_api_key()
            if not api_key:
                print(
                    "⚠️ Warning: No Gemini API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY"
                )
                print("🔄 Falling back to Ollama if available...")
                return self._get_ollama_config()

            return {
                "provider": "gemini",
                "model": self.LLM_MODEL or self.GEMINI_MODEL,
                "api_key": api_key,
                "temperature": self.LLM_TEMPERATURE,
            }

        elif provider == "ollama":
            return self._get_ollama_config()

        else:
            # Default to Gemini, fallback to Ollama
            print(f"⚠️ Unknown provider: {provider}. Defaulting to Gemini...")
            return self.get_llm_config()  # Recursive call with default

    def _get_ollama_config(self) -> dict:
        """Get Ollama configuration"""
        return {
            "provider": "ollama",
            "model": (
                self.LLM_MODEL
                if self.LLM_PROVIDER.lower() == "ollama"
                else self.OLLAMA_MODEL
            ),
            "base_url": self.OLLAMA_BASE_URL,
            "temperature": self.LLM_TEMPERATURE,
        }

    def get_available_providers(self) -> list:
        """Get list of available LLM providers"""
        providers = []

        # Check Gemini
        if self.get_gemini_api_key():
            providers.append("gemini")

        # Ollama is always available (local)
        providers.append("ollama")

        return providers

    def validate_configuration(self) -> dict:
        """Validate current configuration"""
        validation = {
            "provider": self.LLM_PROVIDER,
            "model": self.LLM_MODEL,
            "valid": False,
            "warnings": [],
            "errors": [],
        }

        if self.LLM_PROVIDER.lower() == "gemini":
            if self.get_gemini_api_key():
                validation["valid"] = True
            else:
                validation["errors"].append("Gemini API key not configured")
                validation["warnings"].append("Will fallback to Ollama")

        elif self.LLM_PROVIDER.lower() == "ollama":
            validation["valid"] = True
            if self.OLLAMA_BASE_URL == "http://localhost:11434":
                validation["warnings"].append(
                    "Using default Ollama URL - ensure Ollama is running"
                )

        else:
            validation["errors"].append(f"Unknown provider: {self.LLM_PROVIDER}")

        return validation


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


def print_llm_status():
    """Print current LLM configuration status"""
    settings = get_settings()
    validation = settings.validate_configuration()

    print("🤖 LLM Configuration Status:")
    print(f"   Provider: {validation['provider']}")
    print(f"   Model: {validation['model']}")
    print(f"   Valid: {'✅' if validation['valid'] else '❌'}")

    if validation["warnings"]:
        for warning in validation["warnings"]:
            print(f"   ⚠️ {warning}")

    if validation["errors"]:
        for error in validation["errors"]:
            print(f"   ❌ {error}")

    available = settings.get_available_providers()
    print(f"   Available providers: {', '.join(available)}")
