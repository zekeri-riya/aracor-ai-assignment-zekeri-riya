"""Tests for summary generator."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.models.schemas import DocumentMetadata, ModelProvider, SummaryType
from src.services.model_manager import ModelManager
from src.services.summary import SummaryGenerationError, SummaryGenerator


@pytest.fixture
def model_manager():
    """Provide a mock model manager."""
    manager = MagicMock(spec=ModelManager)
    manager.get_current_provider.return_value = ModelProvider.OPENAI
    return manager


@pytest.fixture
def generator(model_manager):
    """Provide a summary generator instance."""
    return SummaryGenerator(model_manager)


@pytest.fixture
def sample_metadata():
    """Provide sample document metadata."""
    return DocumentMetadata(
        filename="test.txt",
        file_type="text/plain",
        file_size=1000,
        created_at=datetime.utcnow(),
    )


def create_mock_response(content: str) -> ChatResult:
    """Helper to create mock response."""
    ai_message = AIMessage(content=content)
    generation = ChatGeneration(message=ai_message)
    return ChatResult(generations=[generation])


@pytest.mark.asyncio
class TestSummaryGenerator:
    """Tests for SummaryGenerator class."""

    async def test_brief_summary(self, generator, sample_metadata):
        """Test brief summary generation."""
        text = "This is a test document that needs to be summarized briefly."
        mock_response = create_mock_response("Brief summary of the test document.")

        generator.model_manager.generate = AsyncMock(return_value=mock_response)

        response = await generator.generate_summary(
            text, sample_metadata, summary_type=SummaryType.BRIEF
        )

        assert response.summary == "Brief summary of the test document."
        assert response.summary_type == SummaryType.BRIEF
        assert response.provider == ModelProvider.OPENAI
        assert response.processing_time > 0
        assert response.token_count > 0

    async def test_detailed_summary(self, generator, sample_metadata):
        """Test detailed summary generation."""
        text = "This is a longer document that requires a detailed summary with multiple key points."
        mock_response = create_mock_response("Detailed multi-paragraph summary...")

        generator.model_manager.generate = AsyncMock(return_value=mock_response)

        response = await generator.generate_summary(
            text, sample_metadata, summary_type=SummaryType.DETAILED
        )

        assert "Detailed" in response.summary
        assert response.summary_type == SummaryType.DETAILED

    async def test_bullet_points(self, generator, sample_metadata):
        """Test bullet point summary generation."""
        text = "Multiple points to be summarized in bullet format."
        mock_response = create_mock_response("• Point 1\n• Point 2\n• Point 3")

        generator.model_manager.generate = AsyncMock(return_value=mock_response)

        response = await generator.generate_summary(
            text, sample_metadata, summary_type=SummaryType.BULLET_POINTS
        )

        assert "•" in response.summary
        assert response.summary_type == SummaryType.BULLET_POINTS

    async def test_long_input(self, generator, sample_metadata):
        """Test handling of very long input."""
        long_text = "Test sentence. " * 1000  # Create long text
        mock_response = create_mock_response("Summary of long text")

        generator.model_manager.generate = AsyncMock(return_value=mock_response)

        response = await generator.generate_summary(long_text, sample_metadata)

        assert response.summary == "Summary of long text"
        assert response.token_count > 0

    async def test_short_input(self, generator, sample_metadata):
        """Test handling of very short input."""
        short_text = "Brief text."
        mock_response = create_mock_response("Very brief summary.")

        generator.model_manager.generate = AsyncMock(return_value=mock_response)

        response = await generator.generate_summary(short_text, sample_metadata)

        assert len(response.summary) > 0
        assert response.token_count > 0

    async def test_special_characters(self, generator, sample_metadata):
        """Test handling of special characters."""
        special_text = "Text with special characters: ∑πΩ≈☺★♠♣"
        mock_response = create_mock_response("Summary with ∑πΩ symbols")

        generator.model_manager.generate = AsyncMock(return_value=mock_response)

        response = await generator.generate_summary(special_text, sample_metadata)

        assert "∑πΩ" in response.summary

    async def test_multiple_languages(self, generator, sample_metadata):
        """Test handling of multiple languages."""
        text = "Este es un texto en español."
        mock_response = create_mock_response("Resumen en español.")

        generator.model_manager.generate = AsyncMock(return_value=mock_response)

        response = await generator.generate_summary(
            text, sample_metadata, language="es"
        )

        assert response.summary == "Resumen en español."

    async def test_technical_content(self, generator, sample_metadata):
        """Test handling of technical content."""
        technical_text = """
        def example_function(x: int) -> int:
            return x * 2

        class ExampleClass:
            def __init__(self):
                self.value = 42
        """
        mock_response = create_mock_response("Summary of technical code.")

        generator.model_manager.generate = AsyncMock(return_value=mock_response)

        response = await generator.generate_summary(technical_text, sample_metadata)

        assert len(response.summary) > 0

    async def test_error_handling(self, generator, sample_metadata):
        """Test error handling during summary generation."""
        generator.model_manager.generate = AsyncMock(side_effect=Exception("API Error"))

        with pytest.raises(SummaryGenerationError) as exc_info:
            await generator.generate_summary("Test text", sample_metadata)
        assert "Failed to generate summary" in str(exc_info.value)

    async def test_chunking_behavior(self, generator, sample_metadata):
        """Test text chunking behavior."""
        # Create text that will be split into multiple chunks
        long_text = "Chunk test. " * 500
        mock_response = create_mock_response("Chunk summary")

        generator.model_manager.generate = AsyncMock(return_value=mock_response)

        response = await generator.generate_summary(long_text, sample_metadata)

        assert response.summary == "Chunk summary"
        # Verify generate was called multiple times (for chunks)
        assert generator.model_manager.generate.call_count > 1

    async def test_empty_text(self, generator, sample_metadata):
        """Test handling of empty text."""
        response = await generator.generate_summary("", sample_metadata)

        assert response.summary == ""
        assert response.token_count == 0

    def test_invalid_chunk_size(self):
        """Test validation of chunk size."""
        with pytest.raises(ValueError):
            SummaryGenerator(MagicMock(), chunk_size=-1)
