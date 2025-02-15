"""Tests for document processor."""

from pathlib import Path

import pytest

from src.models.schemas import ProcessingOptions
from src.processors.document import (
    DocumentProcessor,
    ExtractError,
    FileTypeError,
    ProcessingError,
)


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""

    @pytest.fixture
    def processor(self):
        """Provide a document processor instance."""
        return DocumentProcessor()

    def test_process_txt_file(self, processor, sample_txt):
        """Test processing of text files."""
        text, metadata = processor.process_file(sample_txt)

        assert "test document" in text.lower()
        assert metadata.filename == "test.txt"
        assert metadata.file_type == "text/plain"
        assert metadata.file_size > 0
        assert metadata.page_count is None

    def test_process_pdf_file(self, processor, sample_pdf):
        """Test processing of PDF files."""
        text, metadata = processor.process_file(sample_pdf)

        assert "Test PDF Content" in text
        assert metadata.filename == "test.pdf"
        assert metadata.file_type == "application/pdf"
        assert metadata.file_size > 0
        assert metadata.page_count == 1

    def test_unsupported_format(self, processor, test_files_dir):
        """Test handling of unsupported file formats."""
        unsupported_file = test_files_dir / "test.xyz"
        unsupported_file.write_text("test content")

        with pytest.raises(FileTypeError) as exc_info:
            processor.process_file(unsupported_file)
        assert "Unsupported file type" in str(exc_info.value)

    def test_nonexistent_file(self, processor):
        """Test handling of nonexistent files."""
        with pytest.raises(FileNotFoundError) as exc_info:
            processor.process_file(Path("nonexistent.txt"))
        assert "File not found" in str(exc_info.value)

    def test_corrupted_pdf(self, processor, corrupted_pdf):
        """Test handling of corrupted PDF files."""
        with pytest.raises(ProcessingError) as exc_info:
            processor.process_file(corrupted_pdf)
        assert "Failed to process file" in str(exc_info.value)

    def test_custom_options(self, test_files_dir):
        """Test processor with custom options."""
        options = ProcessingOptions(
            chunk_size=1000, chunk_overlap=100, include_metadata=False
        )
        processor = DocumentProcessor(options=options)

        # Create a test file
        test_file = test_files_dir / "test.txt"
        test_file.write_text("Test content")

        text, metadata = processor.process_file(test_file)
        assert "Test content" in text

    def test_empty_file(self, processor, test_files_dir):
        """Test processing of empty files."""
        empty_file = test_files_dir / "empty.txt"
        empty_file.write_text("")

        text, metadata = processor.process_file(empty_file)
        assert text == ""
        assert metadata.file_size == 0

    def test_large_file_handling(self, processor, test_files_dir):
        """Test handling of large files."""
        large_file = test_files_dir / "large.txt"
        # Create a 1MB file
        large_file.write_text("test\n" * 250000)

        text, metadata = processor.process_file(large_file)
        assert len(text) > 1000000  # Should be > 1MB
        assert metadata.file_size > 1000000

    def test_metadata_accuracy(self, processor, sample_pdf):
        """Test accuracy of metadata extraction."""
        _, metadata = processor.process_file(sample_pdf)

        assert metadata.filename == "test.pdf"
        assert metadata.file_type == "application/pdf"
        assert metadata.file_size > 0
        assert metadata.page_count == 1  # Our test PDF has exactly 1 page
