"""Model manager for handling different LLM providers."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

# Import the Chat models
from langchain_openai import ChatOpenAI

# Tenacity for retry logic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.models.schemas import ModelProvider

# Local imports
from src.utils.logging import LoggerMixin, log_execution_time


class ModelError(Exception):
    """Base class for model-related errors."""

    pass


class APIKeyError(ModelError):
    """Raised when there are issues with API keys."""

    pass


class RateLimitError(ModelError):
    """Raised when rate limits are hit."""

    pass


class ModelManager(LoggerMixin):
    """Manages different LLM providers and handles model interactions."""

    def __init__(
        self,
        openai_api_key: str,
        anthropic_api_key: str,
        default_provider: ModelProvider = ModelProvider.OPENAI,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize the model manager.

        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            default_provider: Default model provider to use
            max_retries: Maximum number of retries for failed calls
            timeout: Timeout in seconds for API calls
        """
        self._validate_api_keys(openai_api_key, anthropic_api_key)

        self.max_retries = max_retries
        self.timeout = timeout
        self._current_provider: ModelProvider = default_provider

        # Initialize rate limiting
        self._call_history: Dict[str, List[float]] = {"openai": [], "anthropic": []}
        self._rate_limits = {
            "openai": {"requests_per_min": 60, "tokens_per_min": 90000},
            "anthropic": {"requests_per_min": 50, "tokens_per_min": 100000},
        }

        # Initialize models
        self._models: Dict[str, BaseChatModel] = {
            "openai": ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0,
                max_retries=max_retries,
                timeout=timeout,
            ),
            "anthropic": ChatAnthropic(
                api_key=anthropic_api_key,
                model="claude-3-sonnet-20240229",
                temperature=0,
                max_retries=max_retries,
                timeout=timeout,
            ),
        }

    def _validate_api_keys(self, openai_key: str, anthropic_key: str) -> None:
        """Validate API keys."""
        if not openai_key or len(openai_key) < 20:
            raise APIKeyError("Invalid OpenAI API key")
        if not anthropic_key or len(anthropic_key) < 20:
            raise APIKeyError("Invalid Anthropic API key")

    def _check_rate_limit(self, provider: str) -> None:
        """
        Check if we're within rate limits.

        Args:
            provider: The provider to check

        Raises:
            RateLimitError: If rate limit would be exceeded
        """
        now = time.time()
        history = self._call_history[provider]

        # Remove calls older than 60 seconds
        updated_history = [t for t in history if (now - t) <= 60]
        self._call_history[provider] = updated_history

        # Check if we're at the limit
        if len(updated_history) >= self._rate_limits[provider]["requests_per_min"]:
            raise RateLimitError(f"Rate limit exceeded for {provider}")

        # Add current call
        updated_history.append(now)

    def _handle_rate_limit(self, provider: str) -> None:
        """Handle rate limit by waiting if necessary."""
        try:
            self._check_rate_limit(provider)
        except RateLimitError:
            self.logger.warning(
                f"Rate limit reached for {provider}, waiting 2s before retrying..."
            )
            time.sleep(2)
            self._check_rate_limit(provider)

    def switch_provider(self, provider: ModelProvider | str) -> None:
        """Switch to a different model provider."""
        # If user passed a string, try to map it to a ModelProvider
        if isinstance(provider, str):
            # Either do a direct check of known strings, or attempt to parse
            provider_lower = provider.lower()
            if provider_lower == "openai":
                provider = ModelProvider.OPENAI
            elif provider_lower == "anthropic":
                provider = ModelProvider.ANTHROPIC
            else:
                raise ValueError(f"Unsupported provider string: {provider}")

        if not isinstance(provider, ModelProvider):
            raise ValueError(f"Unsupported provider type: {provider}")

        self._current_provider = provider
        self.logger.info(f"Switched to provider: {provider.value}")

    def get_current_provider(self) -> ModelProvider:
        """Get the current model provider."""
        return self._current_provider

    @property
    def available_models(self) -> Dict[str, Any]:
        """Get information about available models."""
        return {
            "openai": {"model": "gpt-3.5-turbo", "max_tokens": 4096},
            "anthropic": {"model": "claude-3-sonnet-20240229", "max_tokens": 100000},
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
    )
    @log_execution_time()
    async def generate(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        """
        Generate a response from the current model.

        Args:
            messages: List of messages to send
            **kwargs: Additional arguments to pass to the model

        Returns:
            ChatResult containing the generations

        Raises:
            ModelError: If there's an error generating the response
        """
        provider = self._current_provider.value
        self._handle_rate_limit(provider)

        try:
            model = self._models[provider]
            return await self._call_model_agenerate(model, messages, **kwargs)
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise ModelError(f"Failed to generate response: {str(e)}") from e

    async def _call_model_agenerate(
        self, model: BaseChatModel, messages: List[BaseMessage], **kwargs: Any
    ) -> ChatResult:
        """
        Helper method so we can patch this method easily in tests
        without fighting Pydantic's dynamic attribute protections.
        """
        return await model.agenerate([messages], **kwargs)
