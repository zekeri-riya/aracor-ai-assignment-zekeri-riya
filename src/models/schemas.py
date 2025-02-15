from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SummaryType(str, Enum):
    """Type of summary to generate."""

    BRIEF = "brief"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"

    def get_max_tokens(self) -> int:
        """Get maximum tokens for each summary type."""
        return {self.BRIEF: 150, self.DETAILED: 500, self.BULLET_POINTS: 300}[self]


class ModelProvider(str, Enum):
    """Supported model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"

    def get_default_model(self) -> str:
        """Get default model for each provider."""
        return {
            self.OPENAI: "gpt-3.5-turbo",
            self.ANTHROPIC: "claude-3-sonnet-20240229",
        }[self]


class DocumentMetadata(BaseModel):
    """Metadata for processed documents."""

    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File MIME type")
    file_size: int = Field(..., description="File size in bytes")
    page_count: Optional[int] = Field(
        None, description="Number of pages (if applicable)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="forbid")

    @field_validator("file_type")
    @classmethod
    def validate_file_type(cls, v: str) -> str:
        """Validate that the file type is supported."""
        supported_types = {
            "application/pdf",
            "text/plain",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        if v not in supported_types:
            raise ValueError(f"Unsupported file type: {v}")
        return v


class ProcessingOptions(BaseModel):
    """Options for document processing."""

    chunk_size: int = Field(
        default=2000, ge=100, le=8000, description="Size of text chunks for processing"
    )
    chunk_overlap: int = Field(
        default=200, ge=0, le=1000, description="Overlap between chunks"
    )
    include_metadata: bool = Field(
        default=True, description="Whether to include document metadata in processing"
    )
    extract_tables: bool = Field(
        default=True, description="Whether to extract tables from documents"
    )
    language: Optional[str] = Field(
        default=None, description="Target language for processing"
    )

    model_config = ConfigDict(extra="forbid")


class DocumentRequest(BaseModel):
    """Request model for document processing."""

    path: Path = Field(..., description="Path to the document")
    summary_type: SummaryType = Field(
        default=SummaryType.BRIEF, description="Type of summary to generate"
    )
    provider: ModelProvider = Field(
        default=ModelProvider.OPENAI, description="Model provider to use"
    )
    options: ProcessingOptions = Field(
        default_factory=ProcessingOptions, description="Processing options"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="after")
    def validate_path(self) -> "DocumentRequest":
        """Validate that the path exists and is a file."""
        if not self.path.exists():
            raise ValueError(f"File does not exist: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"Path is not a file: {self.path}")
        return self


class SummaryResponse(BaseModel):
    """Response model for summary generation."""

    summary: str = Field(..., description="Generated summary")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    provider: ModelProvider = Field(..., description="Model provider used")
    summary_type: SummaryType = Field(..., description="Type of summary generated")
    processing_time: float = Field(..., description="Processing time in seconds")
    token_count: int = Field(..., description="Number of tokens in the summary")

    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "summary": self.summary,
            "metadata": self.metadata.model_dump(),
            "provider": self.provider.value,
            "summary_type": self.summary_type.value,
            "processing_time": self.processing_time,
            "token_count": self.token_count,
        }


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "File processing failed",
                "code": "PROC_ERR_001",
                "detail": "Unable to extract text from corrupted PDF",
                "timestamp": "2024-02-15T12:00:00Z",
            }
        }
    )
