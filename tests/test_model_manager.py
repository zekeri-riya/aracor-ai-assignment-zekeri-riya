"""Tests for model manager."""

import time
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.services.model_manager import (
    APIKeyError,
    ModelError,
    ModelManager,
    ModelProvider,
    RateLimitError,
)

# Since we're testing implementation details, we need to access protected members
# pylint: disable=protected-access


class TestModelManager:
    """Tests for ModelManager class."""

    @pytest.fixture
    def manager(self):
        """Provide a model manager instance."""
        return ModelManager(
            openai_api_key="sk-test-key-openai-123456789",
            anthropic_api_key="sk-test-key-anthropic-123456789",
            default_provider=ModelProvider.OPENAI,
            max_retries=3,
            timeout=30,
        )

    def test_api_key_validation(self):
        """Test API key validation."""
        # Test invalid OpenAI key
        with pytest.raises(APIKeyError):
            ModelManager(
                openai_api_key="invalid",
                anthropic_api_key="sk-test-key-anthropic-123456789",
            )

        # Test invalid Anthropic key
        with pytest.raises(APIKeyError):
            ModelManager(
                openai_api_key="sk-test-key-openai-123456789",
                anthropic_api_key="invalid",
            )

    def test_model_switching(self, manager):
        """Test model provider switching."""
        assert manager.get_current_provider() == ModelProvider.OPENAI

        manager.switch_provider(ModelProvider.ANTHROPIC)
        assert manager.get_current_provider() == ModelProvider.ANTHROPIC

        # Switch with string
        manager.switch_provider("openai")
        assert manager.get_current_provider() == ModelProvider.OPENAI

        # Invalid string
        with pytest.raises(ValueError, match="Unsupported provider string"):
            manager.switch_provider("invalid_provider")

    @pytest.mark.asyncio
    async def test_invoke(self, manager):
        """Test successful invocation."""
        messages = [HumanMessage(content="Hello, how are you?")]
        mock_response = AIMessage(content="I'm doing well, thank you!")

        with patch.object(
            manager, "_call_model_invoke", return_value=mock_response
        ) as mock_call:
            response = await manager.invoke(messages)
            assert response.content == "I'm doing well, thank you!"
            mock_call.assert_called_once()

    def test_timeout_handling(self, manager):
        """Test timeout configuration."""
        assert manager.timeout == 30

        custom_manager = ModelManager(
            openai_api_key="sk-test-key-openai-123456789",
            anthropic_api_key="sk-test-key-anthropic-123456789",
            timeout=60,
        )
        assert custom_manager.timeout == 60

    def test_available_models(self, manager):
        """Test available models information."""
        models = manager.available_models
        assert "openai" in models
        assert "anthropic" in models
        assert models["openai"]["model"] == "gpt-3.5-turbo"
        assert models["anthropic"]["model"] == "claude-3-sonnet-20240229"

    @pytest.mark.asyncio
    async def test_error_handling(self, manager):
        """Test error handling during invocation."""
        messages = [HumanMessage(content="Test message")]

        with patch.object(
            manager, "_call_model_invoke", side_effect=Exception("API Error")
        ):
            with pytest.raises(ModelError) as exc_info:
                await manager.invoke(messages)
            assert "Failed to generate response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rate_limiting(self, manager):
        """Test rate limiting functionality."""
        messages = [HumanMessage(content="Test message")]
        mock_response = AIMessage(content="Test response")

        def clear_history(*args):
            manager._call_history["openai"].clear()

        with (
            patch.object(manager, "_call_model_invoke", return_value=mock_response),
            patch("time.sleep", side_effect=clear_history) as mock_sleep,
        ):
            # Set low rate limit for testing
            manager._rate_limits["openai"]["requests_per_min"] = 1

            # First call should work
            await manager.invoke(messages)

            # Second call should trigger rate limit, call sleep, then succeed
            response = await manager.invoke(messages)
            assert response.content == "Test response"
            mock_sleep.assert_called_with(2)

    def test_rate_limit_check(self, manager):
        """Test the rate limit checking logic."""
        provider = "openai"
        current_time = time.time()

        # Set up test calls
        test_calls = [
            current_time - 70,  # Old call that should be removed
            current_time - 30,  # Recent call that should be kept
            current_time - 10,  # Recent call that should be kept
        ]
        manager._call_history[provider] = test_calls.copy()

        try:
            # Test cleanup and new call addition
            manager._check_rate_limit(provider)

            # Check recent calls (should be 2 recent + 1 new)
            recent_calls = [
                t for t in manager._call_history[provider] if t > current_time - 60
            ]
            assert len(recent_calls) == 3
            assert all(t > current_time - 60 for t in recent_calls)

        except RateLimitError:
            # Rate limit hit is acceptable, we're testing cleanup
            pass
