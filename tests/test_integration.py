"""Integration tests for the document analysis system."""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.endpoints import app
from src.models.schemas import ModelProvider, SummaryType
from src.services.model_manager import ModelManager
from src.services.summary import SummaryGenerator


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


class TestIntegration:
    """Integration tests for the complete document analysis workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_format_handling(
        self,
        test_client: TestClient,
        sample_pdf: Path,
        sample_txt: Path,
        cleanup_files: callable,
    ):
        """Test end-to-end processing of different file formats."""
        cleanup_files(sample_pdf)
        cleanup_files(sample_txt)

        # Test PDF processing
        with open(sample_pdf, "rb") as f:
            pdf_response = test_client.post(
                "/analyze/",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"summary_type": SummaryType.BRIEF.value},
            )

        assert pdf_response.status_code == 200
        pdf_result = pdf_response.json()
        assert pdf_result["metadata"]["file_type"] == "application/pdf"

        # Test TXT processing
        with open(sample_txt, "rb") as f:
            txt_response = test_client.post(
                "/analyze/",
                files={"file": ("test.txt", f, "text/plain")},
                data={"summary_type": SummaryType.BRIEF.value},
            )

        assert txt_response.status_code == 200
        txt_result = txt_response.json()
        assert txt_result["metadata"]["file_type"] == "text/plain"

    @pytest.mark.asyncio
    async def test_error_handling_workflow(
        self,
        test_client: TestClient,
        corrupted_pdf: Path,
        cleanup_files: callable,
    ):
        """Test error handling in the complete workflow."""
        cleanup_files(corrupted_pdf)

        # Test corrupted file handling
        with open(corrupted_pdf, "rb") as f:
            response = test_client.post(
                "/analyze/",
                files={"file": ("corrupted.pdf", f, "application/pdf")},
                data={"summary_type": SummaryType.BRIEF.value},
            )

        assert response.status_code == 400
        error = response.json()
        assert "detail" in error
        assert "Failed to process file" in error["detail"]

    @pytest.mark.asyncio
    async def test_provider_switching_workflow(
        self,
        test_client: TestClient,
        sample_txt: Path,
        cleanup_files: callable,
    ):
        """Test the complete workflow with different providers."""
        cleanup_files(sample_txt)

        # Test text content
        test_content = "This is a test document for model switching."
        sample_txt.write_text(test_content)

        async def test_provider(provider: ModelProvider) -> dict:
            with open(sample_txt, "rb") as f:
                response = test_client.post(
                    "/analyze/",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={
                        "model": provider.value,
                        "summary_type": SummaryType.BRIEF.value,
                    },
                )
            assert response.status_code == 200
            return response.json()

        # Test both providers
        openai_result = await test_provider(ModelProvider.OPENAI)
        anthropic_result = await test_provider(ModelProvider.ANTHROPIC)

        assert openai_result["provider"] == "openai"
        assert anthropic_result["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_api_endpoints_workflow(self, test_client: TestClient):
        """Test the complete API endpoints workflow."""
        # Test models endpoint
        models_response = test_client.get("/models/")
        assert models_response.status_code == 200
        models = models_response.json()
        assert "openai" in models
        assert "anthropic" in models

        # Test summary types endpoint
        types_response = test_client.get("/summary-types/")
        assert types_response.status_code == 200
        summary_types = types_response.json()
        assert "brief" in summary_types
        assert "detailed" in summary_types
        assert "bullet_points" in summary_types

        # Test health check
        health_response = test_client.get("/health/")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
