"""Tests for summary generator."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.models.schemas import DocumentMetadata, ModelProvider, SummaryType
from src.services.model_manager import ModelManager
from src.services.summary import SummaryGenerator


@pytest.fixture
def model_manager():
    """Provide a mock model manager."""
    manager = MagicMock(spec=ModelManager)
    manager.get_current_provider.return_value = ModelProvider.OPENAI
    manager.invoke = AsyncMock()
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


@pytest.mark.asyncio
class TestSummaryGenerator:
    """Tests for SummaryGenerator class."""

    async def test_brief_summary(
        self,
        generator,  # pylint: disable=redefined-outer-name
        sample_metadata,  # pylint: disable=redefined-outer-name
    ):
        """Test brief summary generation."""
        text = "This is a test document that needs to be summarized briefly."
        mock_response = AIMessage(content="Brief summary of the test document.")
        generator.model_manager.invoke.return_value = mock_response

        response = await generator.generate_summary(
            text,
            sample_metadata,
            summary_type=SummaryType.BRIEF
        )

        assert response.summary == "Brief summary of the test document."
        assert response.summary_type == SummaryType.BRIEF
        assert response.provider == ModelProvider.OPENAI
        assert response.processing_time > 0
        assert response.token_count > 0

    async def test_detailed_summary(
        self,
        generator,  # pylint: disable=redefined-outer-name
        sample_metadata,  # pylint: disable=redefined-outer-name
    ):
        """Test detailed summary generation."""
        text = (
            "This is a longer document that requires a detailed summary with "
            "multiple key points."
        )
        mock_response = AIMessage(content="Detailed multi-paragraph summary...")
        generator.model_manager.invoke.return_value = mock_response

        response = await generator.generate_summary(
            text,
            sample_metadata,
            summary_type=SummaryType.DETAILED
        )

        assert "Detailed" in response.summary
        assert response.summary_type == SummaryType.DETAILED

    async def test_bullet_points(
        self,
        generator,  # pylint: disable=redefined-outer-name
        sample_metadata,  # pylint: disable=redefined-outer-name
    ):
        """Test bullet point summary generation."""
        text = "Multiple points to be summarized in bullet format."
        mock_response = AIMessage(content="• Point 1\n• Point 2\n• Point 3")
        generator.model_manager.invoke.return_value = mock_response

        response = await generator.generate_summary(
            text,
            sample_metadata,
            summary_type=SummaryType.BULLET_POINTS
        )

        assert "•" in response.summary
        assert response.summary_type == SummaryType.BULLET_POINTS

    async def test_long_input(
        self,
        generator,  # pylint: disable=redefined-outer-name
        sample_metadata,  # pylint: disable=redefined-outer-name
    ):
        """Test handling of very long input."""
        long_text = "Test sentence. " * 1000
        mock_response = AIMessage(content="Summary of long text")
        generator.model_manager.invoke.return_value = mock_response

        response = await generator.generate_summary(long_text, sample_metadata)

        assert response.summary == "Summary of long text"
        assert response.token_count > 0

    async def test_short_input(
        self,
        generator,  # pylint: disable=redefined-outer-name
        sample_metadata,  # pylint: disable=redefined-outer-name
    ):
        """Test handling of very short input."""
        short_text = "Brief text."
        mock_response = AIMessage(content="Very brief summary.")
        generator.model_manager.invoke.return_value = mock_response

        response = await generator.generate_summary(short_text, sample_metadata)

        assert response.summary == "Very brief summary."
        assert response.token_count > 0

    async def test_special_characters(
        self,
        generator,  # pylint: disable=redefined-outer-name
        sample_metadata,  # pylint: disable=redefined-outer-name
    ):
        """Test handling of special characters."""
        special_text = "Text with special characters: ∑πΩ≈☺★♠♣"
        mock_response = AIMessage(content="Summary with ∑πΩ symbols")
        generator.model_manager.invoke.return_value = mock_response

        response = await generator.generate_summary(special_text, sample_metadata)

        assert "∑πΩ" in response.summary

    async def test_multiple_languages(
        self,
        generator,  # pylint: disable=redefined-outer-name
        sample_metadata,  # pylint: disable=redefined-outer-name
    ):
        """Test handling of multiple languages."""
        text = "Este es un texto en español."
        mock_response = AIMessage(content="Resumen en español.")
        generator.model_manager.invoke.return_value = mock_response

        response = await generator.generate_summary(
            text,
            sample_metadata,
            language="es"
        )

        assert response.summary == "Resumen en español."

    async def test_technical_content(
        self,
        generator,  # pylint: disable=redefined-outer-name
        sample_metadata,  # pylint: disable=redefined-outer-name
    ):
        """Test handling of technical content."""
        technical_text = """
        def example_function(x: int) -> int:
            return x * 2

        class ExampleClass:
            def __init__(self):
                self.value = 42
        """
        mock_response = AIMessage(content="Summary of technical code.")
        generator.model_manager.invoke.return_value = mock_response

        response = await generator.generate_summary(technical_text, sample_metadata)

        assert response.summary == "Summary of technical code."

    async def test_chunking_behavior(
        self,
        generator,  # pylint: disable=redefined-outer-name
        sample_metadata,  # pylint: disable=redefined-outer-name
    ):
        """Test text chunking behavior."""
        long_text = "Chunk test. " * 500
        mock_response = AIMessage(content="Chunk summary")
        generator.model_manager.invoke.return_value = mock_response

        response = await generator.generate_summary(long_text, sample_metadata)

        # Verify that invoke was called multiple times (for chunks + combining)
        assert generator.model_manager.invoke.call_count > 1
        assert response.summary == "Chunk summary"

    def test_invalid_chunk_size(self):
        """Test validation of chunk size."""
        with pytest.raises(ValueError):
            SummaryGenerator(MagicMock(), chunk_size=-1)

    async def test_empty_text(
        self,
        generator,  # pylint: disable=redefined-outer-name
        sample_metadata,  # pylint: disable=redefined-outer-name
    ):
        """Test handling of empty text."""
        response = await generator.generate_summary("", sample_metadata)
        assert response.summary == ""
        assert response.token_count == 0
