"""API configuration and dependency injection."""

from functools import lru_cache

from src.config.settings import Settings
from src.processors.document import DocumentProcessor
from src.services.model_manager import ModelManager
from src.services.summary import SummaryGenerator


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@lru_cache
def get_model_manager() -> ModelManager:
    """Get cached model manager instance."""
    settings = get_settings()
    return ModelManager(
        openai_api_key=settings.OPENAI_API_KEY,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
    )


@lru_cache
def get_document_processor() -> DocumentProcessor:
    """Get cached document processor instance."""
    return DocumentProcessor()


@lru_cache
def get_summary_generator() -> SummaryGenerator:
    """Get cached summary generator instance."""
    return SummaryGenerator(get_model_manager())
