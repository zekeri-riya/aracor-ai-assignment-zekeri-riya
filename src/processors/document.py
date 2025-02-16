"""Document processor implementation for handling different file types."""

from pathlib import Path
from typing import Optional, Tuple

import docx
from pypdf import PdfReader

from src.models.schemas import DocumentMetadata, ProcessingOptions
from src.utils.logging import LoggerMixin, log_execution_time


class ProcessingError(Exception):
    """Base class for document processing errors."""


class FileTypeError(ProcessingError):
    """Error raised when file type is not supported."""


class ExtractError(ProcessingError):
    """Error raised when text extraction fails."""


class DocumentProcessor(LoggerMixin):
    """Handles document processing and text extraction for different file formats."""

    SUPPORTED_TYPES = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".docx": (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
    }

    def __init__(self, options: Optional[ProcessingOptions] = None):
        """Initialize document processor."""
        self.options = options or ProcessingOptions()
        self.logger.info(
            "Initialized DocumentProcessor with options: %s",
            self.options.model_dump()
        )

    def _process_pdf(self, file_path: Path) -> str:
        """Process a PDF file."""
        try:
            text_content = []
            self.logger.debug("Starting PDF processing for file: %s", file_path)
            with open(file_path, "rb") as f:
                reader = PdfReader(f, strict=False)
                for i, page in enumerate(reader.pages):
                    content = page.extract_text()
                    if content:
                        content = content.strip()
                        text_content.append(content)
                        self.logger.debug(
                            "Extracted text from page %d: %.100s",
                            i + 1,
                            content
                        )
                    else:
                        self.logger.warning(
                            "No text found on page %d of %s",
                            i + 1,
                            file_path
                        )

            if not text_content:
                self.logger.warning("No text content extracted from PDF: %s", file_path)

            final_text = "\n".join(text_content)
            self.logger.info(
                "Completed PDF processing. Total extracted characters: %d",
                len(final_text),
            )
            return final_text
        except (IOError, ValueError) as e:
            self.logger.error("PDF processing error: %s", str(e))
            raise ExtractError(f"Failed to extract text from PDF: {str(e)}") from e

    def _create_metadata(self, file_path: Path, file_type: str) -> DocumentMetadata:
        """Create metadata for a document."""
        file_size = file_path.stat().st_size
        page_count = None

        if file_type == "application/pdf":
            try:
                with open(file_path, "rb") as f:
                    reader = PdfReader(f, strict=False)
                    page_count = len(reader.pages)
                self.logger.debug("PDF page count for %s: %d", file_path, page_count)
            except (IOError, ValueError) as e:
                self.logger.warning(
                    "Could not get PDF page count for %s: %s",
                    file_path,
                    str(e)
                )

        metadata = DocumentMetadata(
            filename=file_path.name,
            file_type=file_type,
            file_size=file_size,
            page_count=page_count,
        )
        self.logger.debug("Created metadata: %s", metadata.model_dump())
        return metadata

    @log_execution_time()
    def process_file(self, file_path: Path) -> Tuple[str, DocumentMetadata]:
        """Process a file and extract its text content with metadata."""
        self.logger.info("Processing file: %s", file_path)
        if not file_path.exists():
            self.logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file type
        file_type = self._get_file_type(file_path)
        self.logger.debug("Determined MIME type: %s", file_type)
        if file_type not in self.SUPPORTED_TYPES.values():
            self.logger.error("Unsupported file type: %s", file_type)
            raise FileTypeError(f"Unsupported file type: {file_type}")

        try:
            # Create metadata
            metadata = self._create_metadata(file_path, file_type)

            # Extract text based on file type
            if file_type == "application/pdf":
                text = self._process_pdf(file_path)
            elif file_type == "text/plain":
                text = self._process_txt(file_path)
            elif file_type == self.SUPPORTED_TYPES[".docx"]:
                text = self._process_docx(file_path)
            else:
                self.logger.error("No processor available for file type: %s", file_type)
                raise FileTypeError(f"No processor for file type: {file_type}")

            self.logger.info("Successfully processed file: %s", file_path)
            self.logger.debug(
                "Extracted text preview (first 200 chars): %.200s",
                text
            )
            return text, metadata

        except (IOError, ValueError) as e:
            self.logger.error("Error processing file %s: %s", file_path, str(e))
            raise ProcessingError(f"Failed to process file: {str(e)}") from e

    def _get_file_type(self, file_path: Path) -> str:
        """Get the MIME type of a file."""
        extension = file_path.suffix.lower()
        file_type = self.SUPPORTED_TYPES.get(extension, "application/octet-stream")
        self.logger.debug(
            "File extension: %s mapped to MIME type: %s",
            extension,
            file_type
        )
        return file_type

    def _process_txt(self, file_path: Path) -> str:
        """Process a text file."""
        try:
            text = file_path.read_text(encoding="utf-8")
            self.logger.debug("Extracted text from TXT file (length %d)", len(text))
            return text
        except IOError as e:
            self.logger.error("Text file processing error: %s", str(e))
            raise ExtractError(f"Failed to read text file: {str(e)}") from e

    def _process_docx(self, file_path: Path) -> str:
        """Process a DOCX file."""
        try:
            self.logger.debug("Starting DOCX processing for file: %s", file_path)
            doc = docx.Document(file_path)
            paragraphs = [
                paragraph.text
                for paragraph in doc.paragraphs
                if paragraph.text.strip()
            ]
            text = "\n".join(paragraphs)
            self.logger.info("Extracted %d paragraphs from DOCX file", len(paragraphs))
            self.logger.debug(
                "DOCX extracted text preview (first 200 chars): %.200s",
                text
            )
            return text
        except (IOError, ValueError) as e:
            self.logger.error("DOCX processing error: %s", str(e))
            raise ExtractError(f"Failed to extract text from DOCX: {str(e)}") from e