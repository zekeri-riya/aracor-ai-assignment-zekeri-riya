# pylint: disable=no-member
"""Tests for schema models."""

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    DocumentMetadata,
    DocumentRequest,
    ErrorResponse,
    ModelProvider,
    ProcessingOptions,
    SummaryResponse,
    SummaryType,
)


class TestSummaryType:
    """Tests for SummaryType enumeration."""

    def test_valid_values(self):
        """Test valid summary types."""
        assert SummaryType.BRIEF.value == "brief"
        assert SummaryType.DETAILED.value == "detailed"
        assert SummaryType.BULLET_POINTS.value == "bullet_points"

    def test_max_tokens(self):
        """Test max token values for each summary type."""
        assert SummaryType.BRIEF.get_max_tokens() == 150
        assert SummaryType.DETAILED.get_max_tokens() == 500
        assert SummaryType.BULLET_POINTS.get_max_tokens() == 300

    def test_invalid_type(self):
        """Test handling of invalid summary type."""
        with pytest.raises(ValueError):
            SummaryType("invalid_type")


class TestModelProvider:
    """Tests for ModelProvider enumeration."""

    def test_valid_providers(self):
        """Test valid model providers."""
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.ANTHROPIC.value == "anthropic"

    def test_default_models(self):
        """Test default model selection for providers."""
        assert ModelProvider.OPENAI.get_default_model() == "gpt-3.5-turbo"
        assert "claude" in ModelProvider.ANTHROPIC.get_default_model()

    def test_invalid_provider(self):
        """Test handling of invalid provider."""
        with pytest.raises(ValueError):
            ModelProvider("invalid_provider")


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_valid_metadata(self):
        """Test creation of valid metadata."""
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_type="application/pdf",
            file_size=1024,
            page_count=5,
        )
        assert metadata.filename == "test.pdf"
        assert metadata.file_size == 1024
        assert metadata.page_count == 5
        assert isinstance(metadata.created_at, datetime)

    def test_invalid_file_type(self):
        """Test validation of file type."""
        with pytest.raises(ValidationError):
            DocumentMetadata(
                filename="test.invalid", file_type="application/invalid", file_size=1024
            )

    def test_optional_fields(self):
        """Test handling of optional fields."""
        metadata = DocumentMetadata(
            filename="test.txt", file_type="text/plain", file_size=100
        )
        assert metadata.page_count is None


class TestProcessingOptions:
    """Tests for ProcessingOptions model."""

    def test_default_values(self):
        """Test default processing options."""
        options = ProcessingOptions()
        assert options.chunk_size == 2000
        assert options.chunk_overlap == 200
        assert options.include_metadata is True
        assert options.extract_tables is True
        assert options.language is None

    def test_custom_values(self):
        """Test custom processing options."""
        options = ProcessingOptions(
            chunk_size=1000,
            chunk_overlap=100,
            include_metadata=False,
            extract_tables=False,
            language="en",
        )
        assert options.chunk_size == 1000
        assert options.chunk_overlap == 100
        assert options.include_metadata is False
        assert options.extract_tables is False
        assert options.language == "en"

    def test_validation(self):
        """Test validation of processing options."""
        with pytest.raises(ValidationError):
            ProcessingOptions(chunk_size=50)  # Too small

        with pytest.raises(ValidationError):
            ProcessingOptions(chunk_size=10000)  # Too large

        with pytest.raises(ValidationError):
            ProcessingOptions(chunk_overlap=2000)  # Too large


class TestDocumentRequest:
    """Tests for DocumentRequest model."""

    def test_valid_request(self, sample_pdf, cleanup_files):
        """Test creation of valid document request."""
        cleanup_files(sample_pdf)

        request = DocumentRequest(
            path=sample_pdf,
            summary_type=SummaryType.BRIEF,
            provider=ModelProvider.OPENAI,
        )
        assert request.summary_type == SummaryType.BRIEF
        assert request.provider == ModelProvider.OPENAI
        assert request.options.chunk_size == 2000

    def test_invalid_path(self):
        """Test validation of invalid file path."""
        with pytest.raises(ValidationError):
            DocumentRequest(
                path=Path("/nonexistent/file.txt"), summary_type=SummaryType.BRIEF
            )

    def test_custom_options(self, sample_pdf, cleanup_files):
        """Test request with custom processing options."""
        cleanup_files(sample_pdf)

        request = DocumentRequest(
            path=sample_pdf,
            summary_type=SummaryType.DETAILED,
            provider=ModelProvider.ANTHROPIC,
            options=ProcessingOptions(chunk_size=1000, language="en"),
        )
        assert request.options.chunk_size == 1000
        assert request.options.language == "en"


class TestSummaryResponse:
    """Tests for SummaryResponse model."""

    def test_valid_response(self):
        """Test creation of valid summary response."""
        metadata = DocumentMetadata(
            filename="test.pdf", file_type="application/pdf", file_size=1024
        )

        response = SummaryResponse(
            summary="Test summary",
            metadata=metadata,
            provider=ModelProvider.OPENAI,
            summary_type=SummaryType.BRIEF,
            processing_time=1.5,
            token_count=100,
        )

        assert response.summary == "Test summary"
        assert response.processing_time == 1.5
        assert response.token_count == 100


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_error_creation(self):
        """Test creation of error response."""
        error = ErrorResponse(
            error="Test error", code="TEST_001", detail="Detailed error message"
        )

        assert error.error == "Test error"
        assert error.code == "TEST_001"
        assert error.detail == "Detailed error message"
        assert isinstance(error.timestamp, datetime)

    def test_error_without_detail(self):
        """Test error response without detail."""
        error = ErrorResponse(error="Test error", code="TEST_002")

        assert error.error == "Test error"
        assert error.code == "TEST_002"
        assert error.detail is None
