from typing import Optional

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    OPENAI_API_KEY: str = Field(..., description="OpenAI API key", min_length=20)
    ANTHROPIC_API_KEY: str = Field(..., description="Anthropic API key", min_length=20)
    DEFAULT_MODEL: str = Field(
        default="openai", description="Default model provider to use"
    )
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    @field_validator("DEFAULT_MODEL")
    @classmethod
    def validate_default_model(cls, v: str) -> str:
        """Validate the default model setting."""
        if v.lower() not in ["openai", "anthropic"]:
            raise ValueError("DEFAULT_MODEL must be either 'openai' or 'anthropic'")
        return v.lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate the logging level setting."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
