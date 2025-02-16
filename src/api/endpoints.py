"""API endpoints for document analysis system."""

import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware

try:
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
except ImportError as e:
    raise ImportError(
        f"Failed to import required modules: {e}. "
        "Please ensure all dependencies are installed."
    ) from e

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def handle_document_analysis(
    file_path: Path,
    text: str,
    metadata: Dict[str, Any],
    model: ModelProvider,
    summary_type: SummaryType,
    language: Optional[str],
    model_manager: ModelManager,
    generator: SummaryGenerator,
) -> Dict[str, Any]:
    """Handle the core document analysis logic."""
    model_manager.switch_provider(model)
    logger.info(
        "Generating summary with type '%s' and language '%s'",
        summary_type.value,
        language,
    )

    response = await generator.generate_summary(
        text,
        metadata,
        summary_type=summary_type,
        language=language,
    )

    logger.info("Summary generation complete. Length: %d", len(response.summary))
    logger.debug("Generated summary preview: %.88s...", response.summary)

    return response.to_dict()


@app.post("/analyze/")
async def analyze_document(
    file: UploadFile = File(...),
    summary_type: SummaryType = Form(SummaryType.BRIEF),
    model: ModelProvider = Form(ModelProvider.OPENAI),
    language: Optional[str] = Form(None),
    processor: DocumentProcessor = Depends(get_document_processor),
    model_manager: ModelManager = Depends(get_model_manager),
    generator: SummaryGenerator = Depends(get_summary_generator),
) -> Dict[str, Any]:
    """
    Analyze a document and generate a summary.

    Args:
        file: Uploaded document (PDF, DOCX, or TXT)
        summary_type: Type of summary to generate
        model: Model provider to use
        language: Optional target language
        processor: Document processor dependency
        model_manager: Model manager dependency
        generator: Summary generator dependency

    Returns:
        Dict containing the summary response

    Raises:
        HTTPException: If processing fails
    """
    logger.info("Received analysis request for file: %s", file.filename)

    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    logger.debug("File suffix determined as: %s", suffix)
    if suffix not in {".pdf", ".txt", ".docx"}:
        logger.error("Unsupported file type: %s", suffix)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported types are: .pdf, .txt, .docx",
        )

    # Create temporary file
    file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            file_path = Path(tmp_file.name)
        logger.info("Temporary file created at: %s", file_path)

        # Process document
        logger.info("Processing file: %s", file_path)
        text, metadata = processor.process_file(file_path)
        logger.info("File processed. Extracted text length: %d", len(text))
        logger.debug("Metadata extracted: %s", metadata)

        # Generate summary
        result = await handle_document_analysis(
            file_path,
            text,
            metadata,
            model,
            summary_type,
            language,
            model_manager,
            generator,
        )

        return result

    except Exception as e:
        logger.exception("Error in analyze_document endpoint: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e

    finally:
        if file_path:
            try:
                file_path.unlink()
                logger.debug("Temporary file deleted successfully")
            except Exception as cleanup_error:
                logger.error("Error cleaning up temporary file: %s", cleanup_error)


@app.get("/models/")
def list_models(
    model_manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, Any]:
    """Get available models and their capabilities."""
    logger.debug("Listing available models")
    return model_manager.available_models


@app.get("/summary-types/")
def list_summary_types() -> Dict[str, Dict[str, Any]]:
    """Get available summary types."""
    logger.debug("Listing available summary types")
    return {
        summary_type.value: {
            "description": summary_type.name.replace("_", " ").title(),
            "max_tokens": summary_type.get_max_tokens(),
        }
        for summary_type in SummaryType
    }


@app.get("/health/")
def health_check(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    """Health check endpoint."""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "version": "1.0.0",
        "config": {
            "default_model": settings.DEFAULT_MODEL,
            "log_level": settings.LOG_LEVEL,
        },
    }
