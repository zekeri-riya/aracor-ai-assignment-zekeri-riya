"""Tests for model manager."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from src.services.model_manager import (
    APIKeyError,
    ModelError,
    ModelManager,
    ModelProvider,
    RateLimitError,
)


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
    async def test_generation(self, manager):
        """Test successful generation."""
        messages = [HumanMessage(content="Hello, how are you?")]

        # Mock the helper method instead of patching .agenerate directly
        mock_response = MagicMock()
        mock_response.generations[0][0].text = "I'm doing well, thank you!"

        with patch.object(
            manager, "_call_model_agenerate", return_value=mock_response
        ) as mock_call:
            response = await manager.generate(messages)
            assert response.generations[0][0].text == "I'm doing well, thank you!"
            mock_call.assert_called_once()

    def test_timeout_handling(self, manager):
        """Test timeout configuration."""
        assert manager.timeout == 30

        # Create manager with custom timeout
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
        """Test error handling during generation."""
        messages = [HumanMessage(content="Test message")]

        with patch.object(
            manager, "_call_model_agenerate", side_effect=Exception("API Error")
        ):
            with pytest.raises(ModelError) as exc_info:
                await manager.generate(messages)
            assert "Failed to generate response" in str(exc_info.value)
