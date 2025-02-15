"""API endpoints for document analysis system."""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.config import (
    get_document_processor,
    get_model_manager,
    get_settings,
    get_summary_generator,
)
from src.config.settings import Settings
from src.models.schemas import ModelProvider, SummaryType
from src.processors.document import DocumentProcessor
from src.services.model_manager import ModelManager
from src.services.summary import SummaryGenerator
from src.utils.logging import get_logger

# Create a logger for the endpoints module
logger = get_logger("Endpoints", level="DEBUG")

app = FastAPI(
    title="Document Analysis API",
    description="API for analyzing documents using different LLM providers",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/analyze/")
async def analyze_document(
    file: UploadFile = File(...),
    summary_type: SummaryType = Form(SummaryType.BRIEF),
    model: ModelProvider = Form(ModelProvider.OPENAI),
    language: Optional[str] = Form(None),
    processor: DocumentProcessor = Depends(get_document_processor),
    model_manager: ModelManager = Depends(get_model_manager),
    generator: SummaryGenerator = Depends(get_summary_generator),
):
    """
    Analyze a document and generate a summary.

    Args:
        file: Uploaded document (PDF, DOCX, or TXT)
        summary_type: Type of summary to generate
        model: Model provider to use
        language: Optional target language

    Returns:
        Summary response

    Raises:
        HTTPException: If processing fails
    """
    logger.info("Received analysis request for file: %s", file.filename)
    try:
        # Validate file type
        suffix = Path(file.filename).suffix.lower()
        logger.debug("File suffix determined as: %s", suffix)
        if suffix not in [".pdf", ".txt", ".docx"]:
            logger.error("Unsupported file type: %s", suffix)
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}. Supported types are: .pdf, .txt, .docx",
            )

        # Create a temporary file to store the uploaded document
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            file_path = Path(tmp_file.name)
        logger.info("Temporary file created at: %s", file_path)

        try:
            # Process document to extract text and metadata
            logger.info("Processing file: %s", file_path)
            text, metadata = processor.process_file(file_path)
            logger.info("File processed. Extracted text length: %d", len(text))
            logger.debug("Metadata extracted: %s", metadata)

            logger.info(
                "Text extraction complete. Generating summary... Full text: %s\nMetadata: %s",
                text,
                metadata,
            )

            # Switch to requested model provider
            logger.info("Switching model provider to: %s", model.value)
            model_manager.switch_provider(model)

            # Generate summary using the extracted text
            logger.info(
                "Generating summary with type '%s' and language '%s'",
                summary_type.value,
                language,
            )
            response = await generator.generate_summary(
                text, metadata, summary_type=summary_type, language=language
            )
            logger.info(
                "Summary generation complete. Summary length: %d", len(response.summary)
            )
            logger.debug(
                "Generated summary: %.300s", response.summary
            )  # Log first 300 chars

            return response.to_dict()

        finally:
            # Clean up temporary file
            logger.info("Cleaning up temporary file: %s", file_path)
            try:
                file_path.unlink()
                logger.debug("Temporary file deleted successfully.")
            except Exception as cleanup_error:
                logger.error("Error cleaning up temporary file: %s", cleanup_error)

    except Exception as e:
        logger.exception("Error in analyze_document endpoint: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models/")
def list_models(model_manager: ModelManager = Depends(get_model_manager)):
    """Get available models and their capabilities."""
    logger.debug("Listing available models.")
    return model_manager.available_models


@app.get("/summary-types/")
def list_summary_types():
    """Get available summary types."""
    logger.debug("Listing available summary types.")
    return {
        summary_type.value: {
            "description": summary_type.name.replace("_", " ").title(),
            "max_tokens": summary_type.get_max_tokens(),
        }
        for summary_type in SummaryType
    }


@app.get("/health/")
def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint."""
    logger.info("Health check requested.")
    return {
        "status": "healthy",
        "version": "1.0.0",
        "config": {
            "default_model": settings.DEFAULT_MODEL,
            "log_level": settings.LOG_LEVEL,
        },
    }
