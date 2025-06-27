"""
Fixed backend/config.py with corrected Gemini model names and better error handling
Replace your backend/config.py with this version
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with fixed Gemini configuration"""

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = False

    # LLM Configuration - Gemini or Ollama only
    LLM_PROVIDER: str = "gemini"  # "gemini" or "ollama"
    LLM_MODEL: str = "gemini-1.5-flash"  # Fixed to working model
    LLM_TEMPERATURE: float = 0.1

    # Google Gemini Configuration
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None  # Alternative name
    GEMINI_MODEL: str = "gemini-1.5-flash"  # Updated to working model

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

    # Web Scraping Configuration (Crawl4AI)
    ENABLE_WEB_SCRAPING: bool = True
    CRAWL_HEADLESS: bool = True
    CRAWL_TIMEOUT: int = 30
    CRAWL_MAX_PAGES: int = 1
    CRAWL_USER_AGENT: str = "Hybrid-Research-Bot/1.0"

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
        """Get LLM configuration based on provider (fixed for Gemini errors)"""
        provider = self.LLM_PROVIDER.lower()

        if provider == "gemini":
            api_key = self.get_gemini_api_key()
            if not api_key:
                print(
                    "⚠️ Warning: No Gemini API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY"
                )
                print("🔄 Falling back to Ollama if available...")
                return self._get_ollama_config()

            # Use fixed working model
            model = self.LLM_MODEL
            if model in ["gemini-2.0-flash", "gemini-pro"]:
                model = "gemini-1.5-flash"  # Fix problematic models
                print(f"🔧 Updated model to working version: {model}")

            return {
                "provider": "gemini",
                "model": model,
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
        """Validate current configuration with fixes"""
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

                # Check for problematic model names
                if self.LLM_MODEL in ["gemini-2.0-flash", "gemini-pro"]:
                    validation["warnings"].append(
                        f"Model {self.LLM_MODEL} may cause issues - recommend gemini-1.5-flash"
                    )
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

    def get_working_gemini_models(self) -> list:
        """Get list of known working Gemini models"""
        return [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]

    def fix_model_name(self) -> str:
        """Fix problematic model names"""
        model_fixes = {
            "gemini-2.0-flash": "gemini-1.5-flash",
            "gemini-pro": "gemini-1.5-flash",
            "gemini-1.0-pro": "gemini-1.5-flash",
            "gemini-pro-latest": "gemini-1.5-flash",
        }

        fixed_model = model_fixes.get(self.LLM_MODEL, self.LLM_MODEL)
        if fixed_model != self.LLM_MODEL:
            print(f"🔧 Fixed model name: {self.LLM_MODEL} -> {fixed_model}")

        return fixed_model


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
    """Print current LLM configuration status with fixes"""
    settings = get_settings()
    validation = settings.validate_configuration()

    print("🤖 LLM Configuration Status:")
    print(f"   Provider: {validation['provider']}")
    print(f"   Model: {validation['model']}")
    print(f"   Fixed Model: {settings.fix_model_name()}")
    print(f"   Valid: {'✅' if validation['valid'] else '❌'}")

    if validation["warnings"]:
        for warning in validation["warnings"]:
            print(f"   ⚠️ {warning}")

    if validation["errors"]:
        for error in validation["errors"]:
            print(f"   ❌ {error}")

    available = settings.get_available_providers()
    print(f"   Available providers: {', '.join(available)}")
    print(
        f"   Working Gemini models: {', '.join(settings.get_working_gemini_models())}"
    )


def create_fixed_env_example():
    """Create a fixed .env.example with working configuration"""
    content = """# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# LLM Configuration - FIXED SETTINGS
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
LLM_TEMPERATURE=0.1

# Google Gemini Configuration (RECOMMENDED)
GOOGLE_API_KEY=your-gemini-api-key-here
# Alternative name for the API key
GEMINI_API_KEY=your-gemini-api-key-here

# Ollama Configuration (if using local LLM)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Database Configuration
DATABASE_URL=sqlite:///./data/market_research.db

# Security
SECRET_KEY=your-secret-key-change-in-production-make-it-long-and-random

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Task Configuration
MAX_CONCURRENT_TASKS=5
TASK_TIMEOUT_MINUTES=30

# Data Collection Limits
MAX_SOURCES_PER_QUERY=20
MAX_DATA_POINTS_PER_SOURCE=100

# Web Scraping Configuration
ENABLE_WEB_SCRAPING=true
CRAWL_HEADLESS=true
CRAWL_TIMEOUT=30
CRAWL_USER_AGENT=Hybrid-Research-Bot/1.0

# Compliance Settings
ENABLE_STRICT_COMPLIANCE=true
COMPLIANCE_LOG_LEVEL=INFO

# Cache Configuration
ENABLE_CACHING=true
CACHE_TTL_HOURS=24

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/market_research.log

# WORKING MODEL RECOMMENDATIONS:
# - gemini-1.5-flash (fast, reliable)
# - gemini-1.5-pro (more capable, slower)
# 
# AVOID THESE MODELS (cause errors):
# - gemini-2.0-flash
# - gemini-pro
# - gemini-1.0-pro
"""

    try:
        with open(".env.example", "w") as f:
            f.write(content)
        print("✅ Created fixed .env.example file")
    except Exception as e:
        print(f"❌ Could not create .env.example: {e}")


if __name__ == "__main__":
    print_llm_status()
    create_fixed_env_example()
