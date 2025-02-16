"""Test configuration and shared fixtures."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.endpoints import app


# Environment setup fixtures
@pytest.fixture(autouse=True)
def env_setup():
    """Set up test environment variables with valid test keys."""
    os.environ.update(
        {
            "OPENAI_API_KEY": "sk-1234567890abcdefghijklmnopqrstuvwxyz1234",
            "ANTHROPIC_API_KEY": "sk-ant-1234567890abcdefghijklmnopqrstuvwxyz",
            "DEFAULT_MODEL": "openai",
            "LOG_LEVEL": "DEBUG",
        }
    )
    yield


@pytest.fixture
def test_client() -> TestClient:
    """Provide a test client for the FastAPI application."""
    return TestClient(app)


# File handling fixtures
@pytest.fixture
def test_files_dir(tmp_path) -> Path:
    """Create and provide a directory for test files."""
    test_dir = tmp_path / "test_files"
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def cleanup_files():
    """Register files for cleanup after tests."""
    temp_files = []

    def _register(file_path: Path):
        temp_files.append(file_path)

    yield _register

    for file_path in temp_files:
        try:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Warning: Failed to clean up {file_path}: {e}")


# Test file fixtures
@pytest.fixture
def sample_pdf(test_files_dir) -> Path:
    """Create a sample PDF file for testing."""
    pdf_path = test_files_dir / "test.pdf"

    # Minimal valid PDF with extractable text
    pdf_content = b"""%PDF-1.7
1 0 obj
<</Type/Catalog/Pages 2 0 R>>
endobj
2 0 obj
<</Type/Pages/Kids[3 0 R]/Count 1>>
endobj
3 0 obj
<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>
endobj
4 0 obj
<</Length 68>>
stream
BT
/F1 12 Tf
72 720 Td
(Test PDF Content) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000015 00000 n
0000000061 00000 n
0000000111 00000 n
0000000254 00000 n
trailer
<</Root 1 0 R/Size 5>>
startxref
372
%%EOF"""

    with open(pdf_path, "wb") as f:
        f.write(pdf_content)

    return pdf_path


@pytest.fixture
def corrupted_pdf(test_files_dir) -> Path:
    """Create a corrupted PDF file for testing."""
    pdf_path = test_files_dir / "corrupted.pdf"
    corrupted_content = b"%PDF-1.7\nThis is not a valid PDF file\n%%EOF"
    with open(pdf_path, "wb") as f:
        f.write(corrupted_content)
    return pdf_path


@pytest.fixture
def sample_txt(test_files_dir) -> Path:
    """Create a sample text file for testing."""
    txt_path = test_files_dir / "test.txt"
    txt_path.write_text(
        "This is a test document.\nIt has multiple lines.\nEnd of document."
    )
    return txt_path
